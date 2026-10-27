"""Prepare a real-edit/control episode, reconstruct its input ROI, and capture NG.

Run with `python -m benchmark.proofreading --help`. No network on import.
Researcher and harness artifacts are private; only solver/ is a public export.
This is a scene preparation/debugging utility, not an isolated solver runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading
import time

import numpy as np


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bounds(center, size, resolution):
    c, s, r = (np.asarray(v, dtype=float) for v in (center, size, resolution))
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (c, s, r)):
        raise ValueError("Expected finite XYZ triples")
    if np.any(s <= 0) or np.any(r <= 0):
        raise ValueError("Size and resolution must be positive")
    return np.floor((c - s / 2) / r).astype(int), np.ceil((c + s / 2) / r).astype(int)


def map_roots(raw, graph, timestamp, batch_size=32768):
    if raw.ndim != 3 or raw.dtype != np.uint64:
        raise ValueError("Expected XYZ uint64 SV volume")
    ids, inverse = np.unique(raw, return_inverse=True)
    mapped = np.zeros(len(ids), dtype=np.uint64)
    positions = np.flatnonzero(ids)
    for start in range(0, len(positions), batch_size):
        part = positions[start:start + batch_size]
        roots = np.asarray(graph.get_roots(ids[part], timestamp=timestamp), dtype=np.uint64)
        if roots.shape != part.shape or np.any(roots == 0):
            raise ValueError("Missing historical root assignments; no current-root fallback")
        mapped[part] = roots
    return mapped[inverse].reshape(raw.shape)


def neutralize(roots, candidates):
    """Preserve every neighboring object but replace global IDs with local labels."""
    ids, inverse = np.unique(roots, return_inverse=True)
    present = set(map(int, ids)) - {0}
    candidates = list(map(int, candidates))
    if len(set(candidates)) != len(candidates) or not set(candidates) <= present:
        raise ValueError("Every distinct candidate must be visible in the historical ROI")
    ordered = candidates + sorted(present - set(candidates))
    mapping = {root: i + 1 for i, root in enumerate(ordered)}
    lut = np.asarray([mapping.get(int(root), 0) for root in ids], dtype=np.uint32)
    return lut[inverse].reshape(roots.shape), {str(k): str(v) for k, v in mapping.items()}


def prepare(args):
    source = load_json(args.source)
    m = source["metadata"]
    task = {0: "split", 1: "merge"}[source["task_id"]]
    expected_type = "merge_edit" if task == "split" else "split_edit"
    controls = {('synapse_control', 'synapse_pair'), ('adjacent_control', 'control'), ('junction_control', 'control')}
    is_edit = source['sample_type'] == expected_type and m['source_type'] == 'operation'
    is_control = (source['sample_type'], m['source_type']) in controls
    if is_edit:
        if int(m['operation_id']) <= 0 or source['label'] != 1:
            raise ValueError('Real edit provenance/label mismatch')
        before = list(map(str, m['before_root_ids']))
    elif is_control:
        if source['label'] != 0:
            raise ValueError('Control provenance/label mismatch')
        before = list(map(str, source['root_ids']))
        if m.get('root_ids') is not None and list(map(str, m['root_ids'])) != before:
            raise ValueError('Control input roots disagree with metadata')
    else:
        raise ValueError('Unsupported sample provenance')
    if (before != list(map(str, source["root_ids"])) or len(before) != (2 if task == "split" else 1)
            or len(set(before)) != len(before) or any(not 0 < int(x) < 2**64 for x in before)):
        raise ValueError("Original input/before root mismatch")
    timestamp = m.get('cutout_timestamp') if is_control and m.get('cutout_timestamp') is not None else m['timestamp']
    if is_control and m.get('timestamp') is not None and float(timestamp) != float(m['timestamp']):
        raise ValueError('Control timestamp/cutout timestamp disagree')
    if not np.isfinite(timestamp) or timestamp <= 0:
        raise ValueError('Invalid recorded timestamp')
    center = m.get("edit_point_nm") or m.get("interface_point_nm")
    bounds(center, args.size_nm, [1, 1, 1])
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in ("researcher", "harness", "solver"):
        (args.output / name).mkdir(mode=0o700)
    research = {
        "schema": "proofreading.researcher.v1", "case_id": args.case_id,
        "sample_id": source["sample_id"], "task": task,
        "ground_truth": "error" if is_edit else "no_error",
        "label_basis": "historical corrective edit" if is_edit else "dataset-derived control; not independently adjudicated",
        "expert_review": "not_performed", "difficulty": "not_validated",
        "operation_id": str(m["operation_id"]), "input_root_ids": before,
        "provenance_kind": 'real_edit' if is_edit else 'recorded_snapshot',
        "operation_id_is_queryable": bool(is_edit),
        "source_metadata_path": str(args.source.resolve()), "source_sha256": digest(args.source),
        "status": "offline_metadata_checked",
    }
    if is_edit:
        research.update(before_root_ids=before, after_root_ids=list(map(str, m['after_root_ids'])))
    harness = {
        "schema": "proofreading.harness.v1", "case_id": args.case_id, "task": task,
        "datastack": args.datastack, "center_nm_xyz": center,
        "center_source": "edit_point_nm" if m.get("edit_point_nm") else "interface_point_nm",
        "input_root_ids": before, "timestamp_hint_unix_s": timestamp,
        "provenance_kind": 'real_edit' if is_edit else 'recorded_snapshot',
        "roi_size_nm_xyz": args.size_nm, "segmentation_mip": args.seg_mip,
        "em_mip": args.em_mip, "max_voxels_per_volume": args.max_voxels,
        "objects": {chr(65 + i): {"input_root_id": r, "local_label": str(i + 1)}
                    for i, r in enumerate(before)},
        "status": "not_fetched", "mode": "bounded_historical_volume",
    }
    solver = {
        "schema": "proofreading.solver.v1", "case_id": args.case_id, "task": task,
        "instruction": ("判断候选 A 与 B 是否存在错误断裂。" if task == "split"
                        else "判断候选 A 是否包含错误连接。"),
        "coordinate_system": "XYZ nanometers", "center_nm_xyz": center,
        "objects": [{"handle": chr(65 + i), "local_label": str(i + 1)} for i in range(len(before))],
        "decisions": ["error", "no_error", "abstain"],
        "observations": [], "status": "not_rendered",
        "scope_note": "观察范围有限；局部体边界截断不代表真实神经元末端。",
    }
    write_json(args.output / "researcher/config.json", research)
    write_json(args.output / "harness/config.json", harness)
    write_json(args.output / "solver/config.json", solver)
    print(f"Three-layer configuration prepared: {args.output}", flush=True)


def validate_input_history(h, r, graph):
    """Controls never call operation APIs, even with a positive synthetic ID."""
    if h.get('provenance_kind', 'real_edit') == 'recorded_snapshot':
        when = datetime.fromtimestamp(h['timestamp_hint_unix_s'], tz=timezone.utc)
        roots = list(map(int, h['input_root_ids']))
        births = list(graph.get_root_timestamps(roots))
        valid = list(map(bool, graph.is_latest_roots(roots, timestamp=when)))
        if len(births) != len(roots) or any(t > when for t in births) or len(valid) != len(roots) or not all(valid):
            raise ValueError('Input control roots are not valid at the recorded snapshot time')
        return when, {'provenance_kind': 'recorded_snapshot', 'snapshot_timestamp_utc': when.isoformat(),
            'timestamp_basis': 'recorded cutout/input time, not a corrective edit',
            'root_creation_timestamps_utc': [t.isoformat() for t in births],
            'roots_valid_at_snapshot': valid, 'input_root_ids': h['input_root_ids'],
            'operation_api_queried': False, 'snapshot_identity_verified': True}
    op = int(r["operation_id"])
    detail = graph.get_operation_details([op])[str(op)]
    if set(map(str, detail["roots"])) != set(r["after_root_ids"]):
        raise ValueError("Operation after roots do not match source metadata")
    births = list(graph.get_root_timestamps(list(map(int, r["after_root_ids"]))))
    if len(births) != len(r['after_root_ids']) or not births or any(b != births[0] for b in births):
        raise ValueError('After roots do not share one exact creation timestamp')
    timestamp_field = next((key for key in ('operation_ts', 'timestamp') if detail.get(key) is not None), None)
    commit = datetime.fromisoformat(str(detail[timestamp_field])) if timestamp_field else births[0]
    if any(abs((b - commit).total_seconds()) > 1e-6 for b in births):
        raise ValueError("Operation commit and after-root birth times disagree")
    before, after = commit - timedelta(milliseconds=1), commit + timedelta(milliseconds=1)
    edge_key = "added_edges" if h["task"] == "split" else "removed_edges"
    witnesses = np.unique(np.asarray(detail[edge_key], dtype=np.uint64))
    if witnesses.size == 0:
        raise ValueError("Missing witness SVs")
    audit = {"commit_timestamp_utc": commit.isoformat(), "witness_sv_ids": list(map(str, witnesses)),
        'commit_timestamp_source': timestamp_field or 'shared_after_root_creation_time',
        'after_root_creation_timestamps_utc': [b.isoformat() for b in births]}
    for phase, ts in (("before", before), ("after", after)):
        actual = set(map(str, graph.get_roots(witnesses, timestamp=ts)))
        if actual != set(r[f"{phase}_root_ids"]):
            raise ValueError(f"Witness roots disagree with {phase} input identity")
        audit[f"{phase}_timestamp_utc"] = ts.isoformat()
        audit[f"{phase}_witness_roots"] = sorted(actual)
    coords = np.concatenate([detail["source_coords"], detail["sink_coords"]], axis=0)
    center = coords.mean(axis=0) * np.asarray(graph.base_resolution)
    if not np.allclose(center, h["center_nm_xyz"], rtol=0, atol=1e-3):
        raise ValueError("API edit center and source physical center disagree")
    audit["edit_center_verified"] = True
    audit["graph_resolution_nm"] = list(map(float, graph.base_resolution))
    return before, audit


def fetch(args):
    from caveclient import CAVEclient
    from cloudvolume import CloudVolume
    from cloudvolume.lib import Bbox

    root = args.output
    hp, rp = root / "harness/config.json", root / "researcher/config.json"
    h, r = load_json(hp), load_json(rp)
    h["status"] = "fetching"
    write_json(hp, h)
    client = CAVEclient(h["datastack"], write_server_cache=False, max_retries=1)
    graph = client.chunkedgraph
    before, audit = validate_input_history(h, r, graph)
    write_json(root / "researcher/history_audit.json", audit)
    h['snapshot_timestamp_utc'] = before.isoformat()
    if h.get('provenance_kind', 'real_edit') == 'real_edit':
        h["before_timestamp_utc"] = before.isoformat()
        h["history_witness_verified"] = True
    else:
        h['snapshot_identity_verified'] = True
    h["volumes"] = {}
    seg_source, em_source = client.info.segmentation_source(), client.info.image_source()
    print("Historical input identity verified", flush=True)
    for name, source, mip in (("segmentation", seg_source, h["segmentation_mip"]),
                              ("em", em_source, h["em_mip"])):
        kwargs = dict(mip=mip, bounded=True, fill_missing=False, cache=False, use_https=True,
                      progress=False, provenance={}, parallel=1)
        if name == "segmentation":
            kwargs.update(agglomerate=False, secrets={"token": client.auth.token})
        cv = CloudVolume(source, **kwargs)
        if name == "segmentation":
            cv.config.secrets = None  # Graphene header retains bearer; GCS is not a service account.
        lo, hi = bounds(h["center_nm_xyz"], h["roi_size_nm_xyz"], cv.resolution)
        if np.any(lo < cv.bounds.minpt) or np.any(hi > cv.bounds.maxpt):
            raise ValueError(f"{name} ROI exceeds source bounds")
        voxels = int(np.prod(hi - lo))
        if voxels > h["max_voxels_per_volume"]:
            raise ValueError(f"{name}: {voxels} voxels exceed configured limit")
        print(f"Downloading {name}: mip={mip}, resolution={list(cv.resolution)}, shape={list(hi-lo)}", flush=True)
        raw = np.asarray(cv.download(Bbox(lo, hi), **({"agglomerate": False} if name == "segmentation" else {})))
        if raw.shape != tuple(hi - lo) + (1,):
            raise ValueError(f"Unexpected {name} shape {raw.shape}")
        raw = raw[..., 0]
        if name == "segmentation":
            if raw.dtype != np.uint64:
                raise ValueError("SV source is not uint64")
            np.save(root / "harness/supervoxels.npy", raw, allow_pickle=False)
            print(f"Mapping {len(np.unique(raw))} unique SV labels to pre-edit roots", flush=True)
            historical = map_roots(raw, graph, before)
            np.save(root / "harness/historical_roots.npy", historical, allow_pickle=False)
            audit['candidate_volume_witnesses'] = {}
            for handle, obj in h['objects'].items():
                svs = np.unique(raw[historical == int(obj['input_root_id'])])[:3]
                if not len(svs):
                    raise ValueError('Candidate has no historical SV witness in the ROI')
                mapped = list(map(str, graph.get_roots(svs, timestamp=before)))
                if set(mapped) != {obj['input_root_id']}:
                    raise ValueError('Candidate volume witness mismatch')
                audit['candidate_volume_witnesses'][handle] = {'sv_ids': list(map(str, svs)), 'mapped_root_ids': mapped}
            write_json(root / 'researcher/history_audit.json', audit)
            raw, mapping = neutralize(historical, h["input_root_ids"])
            write_json(root / "harness/root_to_local_label.json", mapping)
            h["historical_object_count"] = len(mapping)
            h["candidate_voxel_counts"] = {obj: int(np.count_nonzero(raw == int(value["local_label"])))
                                            for obj, value in h["objects"].items()}
        elif raw.dtype != np.uint8:
            raise ValueError("Expected uint8 EM source; refuse implicit intensity rescaling")
        path = root / f"harness/{name}.npy"
        np.save(path, raw, allow_pickle=False)
        h["volumes"][name] = {"source": source, "mip": mip,
            "resolution_nm_xyz": list(map(float, cv.resolution)), "voxel_offset_xyz": lo.tolist(),
            "shape_xyz": list(raw.shape), "dtype": str(raw.dtype), "sha256": digest(path),
            "bounds_nm_xyz": [(lo * cv.resolution).tolist(), (hi * cv.resolution).tolist()]}
        write_json(hp, h)
    h["status"] = "snapshot_verified"
    h["segmentation_note"] = "Raw SV volume mapped to validated historical input roots at snapshot_timestamp_utc; no current root meshes used"
    h["em_note"] = "Read-only static EM source; no temporal edit is applied to EM"
    h["mesh_note"] = "Meshes generated locally from the bounded historical labels; ROI boundary truncates them"
    h["versions"] = {p: importlib.metadata.version(p) for p in ("neuroglancer", "cloud-volume", "caveclient", "numpy")}
    write_json(hp, h)
    print("Snapshot complete; all neighboring historical objects retained", flush=True)


def make_viewer(root, port):
    import neuroglancer as ng
    h = load_json(root / "harness/config.json")
    if h["status"] != "snapshot_verified":
        raise ValueError("A verified snapshot is required")
    ng.set_server_bind_address(bind_address="127.0.0.1", bind_port=port)
    viewer = ng.Viewer()
    with viewer.txn() as s:
        s.dimensions = ng.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=[1, 1, 1])
        s.position = h["center_nm_xyz"]
        s.layout = "4panel"
        s.cross_section_scale = max(h["roi_size_nm_xyz"][:2]) / 650
        s.projection_scale = max(h["roi_size_nm_xyz"]) * 1.5
        s.show_slices = False
        s.show_axis_lines = False
        s.show_default_annotations = False
        for name in ("em", "segmentation"):
            meta = h["volumes"][name]
            path = root / f"harness/{name}.npy"
            if digest(path) != meta["sha256"]:
                raise ValueError(f"Cached {name} checksum mismatch")
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(data.shape) != meta["shape_xyz"] or str(data.dtype) != meta["dtype"]:
                raise ValueError(f"Cached {name} metadata mismatch")
            local = ng.LocalVolume(data=data,
                dimensions=ng.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=meta["resolution_nm_xyz"]),
                voxel_offset=meta["voxel_offset_xyz"],
                volume_type="image" if name == "em" else "segmentation")
            if name == "em":
                s.layers["EM"] = ng.ImageLayer(source=local)
            else:
                s.layers["Objects"] = ng.SegmentationLayer(source=local,
                    segments=[int(v["local_label"]) for v in h["objects"].values()],
                    segment_colors={"1": "#ff6f59", "2": "#43c6db"},
                    selected_alpha=0.45, not_selected_alpha=0.0, object_alpha=0.85)
    return viewer


def screenshot(viewer, timeout, width, height, *, return_action=False):
    """Use NG's screenshot request/ack, which waits for required render chunks."""
    event = threading.Event()
    result = []
    prior = viewer.config_state.state
    try:
        with viewer.config_state.txn() as c:
            c.show_ui_controls = False
            c.show_panel_borders = False
            c.viewer_size = [width, height]
        viewer.async_screenshot(lambda action: (result.append(action), event.set()))
        if not event.wait(timeout):
            raise TimeoutError("Neuroglancer screenshot acknowledgement timed out")
        shot = result[0].screenshot
        if (shot.width, shot.height) != (width, height):
            raise ValueError("Screenshot dimensions disagree with requested canvas")
        return result[0] if return_action else shot.image
    finally:
        viewer.config_state.set_state(prior)


def render(args):
    public = load_json(args.output / "solver/config.json")
    public["status"] = "rendering"
    public["observations"] = []
    if not args.serve_only:
        write_json(args.output / "solver/config.json", public)
    viewer = make_viewer(args.output, args.port)
    url = viewer.get_viewer_url()
    write_json(args.output / "harness/viewer.json", {"url": url, "scope": "loopback debug viewer; not a solver sandbox"})
    print(f"Neuroglancer: {url}", flush=True)
    if args.serve_only:
        print("Serving cached before scene; keep this process alive", flush=True)
        threading.Event().wait()
        return
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, executable_path=args.chromium,
            args=["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        try:
            page = browser.new_page(viewport={"width": args.width, "height": args.height})
            failures = []
            page.on("pageerror", lambda error: failures.append(type(error).__name__))
            page.goto(url, wait_until="domcontentloaded", timeout=args.timeout * 1000)
            h = load_json(args.output / "harness/config.json")
            presets = [("4panel", "4panel", max(h["roi_size_nm_xyz"]), False)]
            presets += [(axis, axis, args.view_extent_nm, False) for axis in ("xy", "xz", "yz")]
            presets += [("candidates_3d", "3d", args.view_extent_nm, False)]
            presets += [(axis + "_em", axis, args.view_extent_nm, True) for axis in ("xy", "xz", "yz")]
            for name, layout, extent, em_only in presets:
                panel_pixels = min(args.width, args.height) / (2 if layout == "4panel" else 1)
                section_scale = extent / panel_pixels
                with viewer.txn() as state:
                    state.layout = layout
                    state.cross_section_scale = section_scale
                    state.projection_scale = extent * 1.5
                    state.layers["Objects"].visible = not em_only
                # Browser events must be pumped while Python waits for NG's ack.
                outcome = {}
                def capture():
                    try:
                        outcome["image"] = screenshot(viewer, args.timeout, args.width, args.height)
                    except Exception as exc:
                        outcome["error"] = exc
                thread = threading.Thread(target=capture, daemon=True)
                thread.start()
                deadline = time.monotonic() + args.timeout + 5
                while thread.is_alive() and time.monotonic() < deadline:
                    page.wait_for_timeout(100)
                if thread.is_alive():
                    raise TimeoutError("Capture worker did not finish")
                if "error" in outcome:
                    raise outcome["error"]
                path = args.output / f"solver/observations/{name}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(outcome["image"])
                from PIL import Image
                pixels = np.asarray(Image.open(path).convert("RGB"))
                if pixels.std() < 2:
                    raise ValueError("Screenshot is effectively blank")
                state_path = args.output / f"harness/states/{name}.json"
                write_json(state_path, viewer.state.to_json())
                public["observations"].append({"observation_id": f"initial_{name}",
                    "layout": layout, "em_only": em_only,
                    "cross_section_scale_nm_per_pixel": section_scale,
                    "projection_scale": extent * 1.5,
                    "image": f"observations/{name}.png", "sha256": digest(path),
                    "center_nm_xyz": public["center_nm_xyz"], "canvas_size_px": [args.width, args.height]})
                write_json(args.output / "solver/config.json", public)
                print(f"Saved {path}", flush=True)
            if failures:
                raise RuntimeError("Browser reported page errors; screenshots require investigation")
            public["status"] = "captured_pending_human_review"
            write_json(args.output / "solver/config.json", public)
            write_json(args.output / "harness/render_report.json", {
                "status": "captured", "render_completion": "Neuroglancer screenshot acknowledgement",
                "human_review": "pending", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "browser_version": browser.version,
                "scope": "Local bounded pre-edit scene; not a full-neuron or blinded benchmark validation"})
        finally:
            browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare", help="Write private researcher/harness and public solver configs")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--datastack", required=True)
    p.add_argument("--size-nm", type=float, nargs=3, default=[10000, 10000, 10000])
    p.add_argument("--seg-mip", type=int, required=True)
    p.add_argument("--em-mip", type=int, required=True)
    p.add_argument("--max-voxels", type=int, default=32_000_000)
    p.set_defaults(func=prepare)
    p = subs.add_parser("fetch", help="Read-only historical validation and bounded volume download")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=fetch)
    p = subs.add_parser("render", help="Serve local volumes and capture using headless Chromium")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=0, help="Loopback port; 0 selects a free port")
    p.add_argument("--serve-only", action="store_true")
    p.add_argument("--chromium", help="Optional existing Chromium executable")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--view-extent-nm", type=float, default=4000,
                   help="Focused 2D vertical extent and 3D scale basis; overview keeps full ROI")
    p.set_defaults(func=render)
    args = parser.parse_args()
    try:
        args.func(args)
        failure_path = args.output / f"harness/{args.command}_failure.json"
        failure_path.unlink(missing_ok=True)
    except Exception as exc:
        # Exception bodies can contain authenticated URLs. Keep public diagnostics minimal.
        if args.output.exists():
            write_json(args.output / f"harness/{args.command}_failure.json", {
                "status": "failed", "stage": args.command, "exception_type": type(exc).__name__,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        print(f"{args.command} failed ({type(exc).__name__}); no current-state fallback", flush=True)
        # Stack locations aid debugging without printing API exception bodies/tokens.
        import traceback
        traceback.print_tb(exc.__traceback__, limit=8)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
