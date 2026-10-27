"""Model-independent, bounded proofreading tools. No CAVE/NG/browser on import."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import threading
import time
from typing import Annotated, Literal, Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from benchmark.proofreading import digest, load_json, write_json

Number = Annotated[float, Field(allow_inf_nan=False)]
Position = Annotated[list[Number], Field(min_length=3, max_length=3)]
Handle = Annotated[str, Field(pattern=r"^(?:[A-Z]|N[1-9][0-9]*)$")]


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Empty(Arguments):
    pass


class SetView(Arguments):
    layout: Literal["4panel", "3d", "xy", "xz", "yz"] = Field(description="4panel=three orthogonal slices plus 3D; xy/xz/yz=one EM slice; 3d=meshes.")


class MoveTo(Arguments):
    position_nm_xyz: Position = Field(description="Absolute global [x,y,z] in nanometers, inside the reported observation bounds. Not voxels or screen pixels.")


class RotateView(Arguments):
    axis: Literal["x", "y", "z"] = Field(description="Fixed world axis, using right-hand rule; affects the 3D camera around the current center.")
    angle_degrees: Number = Field(ge=-180, le=180, description="Signed incremental angle in degrees. E.g. 30 rotates +30 degrees; -30 reverses it.")


class ZoomView(Arguments):
    factor: Number = Field(ge=0.25, le=4, description="Magnification: 2 makes structures twice as large, 0.5 zooms out. Does not download finer-resolution data.")
    target: Literal["2d", "3d", "both"] = Field(default="both", description="Which independent camera scale to change; position stays fixed.")


class StepSlice(Arguments):
    axis: Literal["x", "y", "z"] = Field(description="Normal axis: z→XY, y→XZ, x→YZ. Uses the cached segmentation voxel spacing along this axis.")
    steps: int = Field(ge=-50, le=50, description="Signed voxel steps, e.g. z,+1 moves +40 nm if spacing is 40 nm. Returns the corresponding orthogonal slice.")


class SetLayers(Arguments):
    em_visible: bool | None = Field(default=None, description="Show/hide electron microscopy. Null keeps current value.")
    segmentation_visible: bool | None = Field(default=None, description="Show/hide segmentation overlays and meshes. Null keeps current value.")
    selected_alpha: Number | None = Field(default=None, ge=0, le=1, description="2D opacity of selected objects; 0 removes the overlay but keeps selection.")
    context_alpha: Number | None = Field(default=None, ge=0, le=1, description="2D opacity of unselected neighboring objects.")
    mesh_alpha: Number | None = Field(default=None, ge=0, le=1, description="3D opacity of selected meshes.")


class SetEMContrast(Arguments):
    black: Number = Field(ge=0, le=255, description="Raw uint8 intensity mapped to black; must be below white.")
    white: Number = Field(ge=0, le=255, description="Raw uint8 intensity mapped to white. E.g. black=70, white=170 increases contrast.")


class SelectObjects(Arguments):
    handles: list[Handle] = Field(min_length=1, max_length=16, description="Replace selected objects with episode handles, e.g. ['A','B'] or ['A','N3414']. Discover neighbor handles using inspect_position.")


class InspectPosition(MoveTo):
    pass


class Evidence(Arguments):
    observation_id: str = Field(min_length=1, max_length=64)
    position_nm_xyz: Position
    description: str = Field(min_length=1, max_length=1000, description="Brief observable evidence at this location; cite what supports the decision.")


class Finish(Arguments):
    decision: Literal["error", "no_error", "abstain"]
    confidence: Number = Field(ge=0, le=1, description="Self-assessed confidence in the submitted decision; not a calibrated error probability.")
    reason: str = Field(min_length=1, max_length=2000, description="Brief evidence-based justification or why evidence is insufficient.")
    evidence: list[Evidence] = Field(max_length=20, description="Cite successful observation IDs from this episode; may be empty only when abstaining.")
    stop_reason: Literal["sufficient_evidence", "uncertain", "budget_exhausted", "infrastructure_failure"]


TOOLS = {
    "get_task": (Empty, "Read the anonymous task, candidate handles, nm bounds, cached data resolution, current view and remaining budgets. No image; counts as one action."),
    "observe": (Empty, "Capture the current Neuroglancer view. Returns a fresh PNG, observation ID, exact rendered public state and budgets. Costs one action and one image attempt."),
    "set_view": (SetView, "Switch to three orthogonal EM slices plus 3D, one named orthogonal slice, or 3D. Immediately returns a new screenshot."),
    "move_to": (MoveTo, "Move the view center to an absolute XYZ nm position within the fixed cached ROI. Preserve layout and zoom; return a screenshot. No new remote data is fetched."),
    "rotate_view": (RotateView, "Incrementally rotate the 3D camera around a fixed world axis at the current center. Preserve 4panel, otherwise switch to 3d. Orthogonal slices remain unrotated. Return a screenshot."),
    "zoom_view": (ZoomView, "Zoom in/out by magnification factor at the same center; choose 2d, 3d or both. Return a screenshot. This changes display scale, not source resolution."),
    "step_slice": (StepSlice, "Advance through the cached volume along x/y/z by signed voxel steps, switch to the matching orthogonal slice, and return a screenshot. Reject out-of-ROI moves."),
    "set_layers": (SetLayers, "Adjust EM/segmentation visibility and selected/context/mesh opacity without changing segmentation. Return a screenshot. Both layers cannot be hidden; 3D requires visible segmentation."),
    "set_em_contrast": (SetEMContrast, "Set the grayscale window of the cached uint8 EM. Preserve underlying data and camera; return a screenshot. black must be less than white."),
    "select_objects": (SelectObjects, "Select one or more anonymous candidate/neighbor handles for 3D display and stronger 2D highlighting. Preserve camera and return a screenshot. No original root IDs are accepted."),
    "inspect_position": (InspectPosition, "Read the local segmentation voxel at an XYZ nm point. Return its anonymous handle (null for background), sampled voxel bounds and a screenshot centered there. No database query or label information."),
    "reset_view": (Empty, "Restore the initial camera, candidates, layer visibility and EM contrast, then return a screenshot. Does not reset budgets, trajectory or a submitted decision."),
    "finish": (Finish, "Submit error/no_error/abstain with brief evidence citing returned observation IDs. No new screenshot. Allowed after budget exhaustion or render failure; closes the episode permanently."),
}


def tool_specs(registry=None):
    registry = TOOLS if registry is None else registry
    return [{"name": name, "description": description, "inputSchema": model.model_json_schema()}
            for name, (model, description) in registry.items()]


def rotation(q, axis, degrees):
    """NG quaternion order is [x,y,z,w]; left-compose a fixed-world rotation."""
    v = np.zeros(3)
    v["xyz".index(axis)] = np.sin(np.deg2rad(degrees) / 2)
    w = np.cos(np.deg2rad(degrees) / 2)
    old_v, old_w = np.asarray(q[:3]), q[3]
    out = np.r_[w * old_v + old_w * v + np.cross(v, old_v), w * old_w - v @ old_v]
    return (out / np.linalg.norm(out)).tolist()


@dataclass
class Capture:
    png: bytes
    state: dict
    render_info: dict = field(default_factory=dict)


class Renderer(Protocol):
    width: int
    height: int

    def capture(self, state: dict, *, timeout_seconds: float) -> Capture: ...

    def close(self): ...


@dataclass
class ToolResult:
    metadata: dict
    image: bytes | None = None
    is_error: bool = False


class ToolError(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message


class ObservationEnvironment:
    """One process / one episode; serialized calls, no label or history reads."""

    def __init__(self, episode: Path, run_dir: Path, renderer: Renderer, *,
                 max_actions=30, max_observations=20, max_seconds=300, clock=time.monotonic, registry=None):
        self.tools = TOOLS if registry is None else registry
        if max_actions < 1 or max_observations < 1 or not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("Positive finite budgets required")
        h = load_json(episode / "harness/config.json")
        public = load_json(episode / "solver/config.json")
        if h["status"] != "snapshot_verified" or h["case_id"] != public["case_id"]:
            raise ValueError("A verified, consistent historical snapshot is required")
        self.case_id = public["case_id"]
        self.task = {k: public[k] for k in ("case_id", "task", "instruction", "center_nm_xyz", "scope_note")}
        seg = h["volumes"]["segmentation"]
        for name in ("em", "segmentation"):
            path = episode / f"harness/{name}.npy"
            if digest(path) != h["volumes"][name]["sha256"]:
                raise ValueError("Cached volume checksum mismatch")
        self.labels = np.load(episode / "harness/segmentation.npy", mmap_mode="r", allow_pickle=False)
        if self.labels.dtype != np.uint32 or list(self.labels.shape) != seg["shape_xyz"]:
            raise ValueError("Invalid neutral segmentation volume")
        self.resolution = np.asarray(seg["resolution_nm_xyz"], dtype=float)
        self.offset = np.asarray(seg["voxel_offset_xyz"], dtype=int)
        self.lo = np.max([v["bounds_nm_xyz"][0] for v in h["volumes"].values()], axis=0)
        self.hi = np.min([v["bounds_nm_xyz"][1] for v in h["volumes"].values()], axis=0)
        if np.any(self.hi <= self.lo):
            raise ValueError("EM and segmentation have no shared ROI")
        self.handle_to_label = {o["handle"]: int(o["local_label"]) for o in public["objects"]}
        present = set(map(int, np.unique(self.labels))) - {0}
        if not set(self.handle_to_label.values()) <= present:
            raise ValueError("Candidate missing from local volume")
        self.candidates = list(self.handle_to_label)
        for label in present - set(self.handle_to_label.values()):
            self.handle_to_label[f"N{label}"] = label
        self.label_to_handle = {v: k for k, v in self.handle_to_label.items()}
        extent = float(np.max(self.hi - self.lo))
        self.initial = {"position_nm_xyz": list(map(float, public["center_nm_xyz"])),
            "layout": "4panel", "projection_orientation_xyzw": [0., 0., 0., 1.],
            "cross_section_scale_nm_per_pixel": extent / (min(renderer.width, renderer.height) / 2),
            "projection_scale_nm": extent * 1.5, "selected_objects": self.candidates,
            "em_visible": True, "segmentation_visible": True,
            "selected_alpha": 0.45, "context_alpha": 0.12, "mesh_alpha": 0.85,
            "em_black": 0., "em_white": 255.}
        self.state = deepcopy(self.initial)
        self._check_position(self.state["position_nm_xyz"])
        self.renderer, self.run_dir = renderer, run_dir
        run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        (run_dir / "observations").mkdir(mode=0o700)
        self.max_actions, self.max_observations, self.max_seconds = max_actions, max_observations, max_seconds
        self.clock, self.started = clock, None
        self.actions = self.image_attempts = 0
        self.observations = {}
        self.finished = False
        self.render_failed = False
        self.lock = threading.Lock()
        write_json(run_dir / "protocol.json", {"schema": "proofreading.observation_tools.v1",
            "case_id": self.case_id, "max_actions": max_actions, "max_observations": max_observations,
            "max_seconds": max_seconds, "canvas_size_px": [renderer.width, renderer.height],
            "tools": tool_specs(self.tools), "initial_state": self.state,
            "data_resolution_nm_xyz": self.resolution.tolist(),
            "observation_bounds_nm_xyz": [self.lo.tolist(), self.hi.tolist()]})

    def _check_position(self, position):
        p = np.asarray(position, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < self.lo) or np.any(p >= self.hi):
            raise ToolError("out_of_bounds", "XYZ nm must lie inside the reported ROI (lower inclusive, upper exclusive). No expansion or silent clipping.")

    def _budget(self):
        elapsed = 0. if self.started is None else max(0., self.clock() - self.started)
        return {"actions_used": self.actions, "actions_remaining": max(0, self.max_actions - self.actions),
            "image_attempts_used": self.image_attempts,
            "images_remaining": max(0, self.max_observations - self.image_attempts),
            "successful_observations": len(self.observations), "elapsed_seconds": round(elapsed, 3),
            "seconds_remaining": round(max(0., self.max_seconds - elapsed), 3)}

    def _context(self):
        return {**self.task, "candidate_handles": self.candidates,
            "observation_bounds_nm_xyz": [self.lo.tolist(), self.hi.tolist()],
            "data_resolution_nm_xyz": self.resolution.tolist(),
            "data_limitations": "Fixed downsampled historical ROI. Zoom cannot reveal finer source data. Boundaries truncate neurons. 3D shows selected objects; 2D retains context.",
            "coordinate_system": "global XYZ nm; orthogonal slices are not rotated",
            "state": deepcopy(self.state)}

    def call(self, name: str, arguments: dict) -> ToolResult:
        with self.lock:
            if self.started is None:
                self.started = self.clock()
            self.actions += 1  # Rejections/unknown tools also count; finish has a reserved submission.
            action_id = f"action_{self.actions:04d}"
            start = self.clock()
            status, image, info = "ok", None, {}
            try:
                if self.finished:
                    raise ToolError("episode_finished", "The episode is closed; start a new trusted process for another episode.")
                if name not in self.tools:
                    raise ToolError("unknown_tool", "Tool is not on the observation allowlist.")
                try:
                    args = self.tools[name][0].model_validate(arguments).model_dump()
                except ValidationError as exc:
                    # No echo of rejected arbitrary values (paths, secrets, prompt injections).
                    fields = [".".join(map(str, e["loc"])) for e in exc.errors()][:8]
                    raise ToolError("invalid_arguments", "Invalid fields: " + ", ".join(fields)) from None
                if name == "finish":
                    info = self._finish(args)
                else:
                    if self.actions > self.max_actions or self.clock() - self.started >= self.max_seconds:
                        raise ToolError("budget_exhausted", "Action/time budget exhausted; submit finish, optionally abstain.")
                    if name == "get_task":
                        info = self._context()
                    else:
                        if self.render_failed:
                            raise ToolError("renderer_unavailable", "Prior rendering failed; no stale image will be returned. Submit finish or restart as a separate run.")
                        if self.image_attempts >= self.max_observations:
                            raise ToolError("budget_exhausted", "Image budget exhausted; submit finish.")
                        proposed, extra = self._transition(name, args)
                        self.image_attempts += 1
                        try:
                            capture = self.renderer.capture(proposed,
                                timeout_seconds=max(0., self.max_seconds - (self.clock() - self.started)))
                            pixels = np.asarray(Image.open(io.BytesIO(capture.png)).convert("RGB"))
                            if pixels.shape != (self.renderer.height, self.renderer.width, 3) or pixels.std() < 2:
                                raise ValueError("Invalid/blank screenshot")
                            if not states_match(proposed, capture.state):
                                raise ValueError("Screenshot state mismatch")
                        except Exception:
                            self.render_failed = True
                            if self.clock() - self.started >= self.max_seconds:
                                raise ToolError("budget_exhausted", "Episode deadline expired while rendering. No observation released; submit finish.") from None
                            raise ToolError("render_failed", "Screenshot failed, timed out, was blank or did not match the requested state. No image returned.") from None
                        if self.clock() - self.started >= self.max_seconds:
                            raise ToolError("budget_exhausted", "Rendering crossed the episode deadline. No observation released; submit finish.")
                        self.state = proposed
                        observation_id = f"obs_{self.image_attempts:04d}"
                        image = capture.png
                        sha = hashlib.sha256(image).hexdigest()
                        image_path = self.run_dir / f"observations/{observation_id}.png"
                        image_path.write_bytes(image)
                        record = {"observation_id": observation_id, "action_id": action_id,
                            "state": deepcopy(capture.state), "sha256": sha,
                            "state_sha256": hashlib.sha256(json.dumps(capture.state, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                            "canvas_size_px": [self.renderer.width, self.renderer.height],
                            "render_info": capture.render_info}
                        self.observations[observation_id] = record
                        write_json(self.run_dir / f"observations/{observation_id}.json", record)
                        info = {**record, **extra,
                            "data_resolution_nm_xyz": self.resolution.tolist()}
            except ToolError as exc:
                status = "error"
                info = {"error": {"code": exc.code, "message": exc.message}}
            result = {"schema": "proofreading.tool_result.v1", "case_id": self.case_id,
                "action_id": action_id, "tool": name, "status": status, **info,
                "budget": self._budget()}
            trace = {**result, "arguments": self._safe_arguments(name, arguments),
                "utc": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(max(0., self.clock() - start), 4)}
            with (self.run_dir / "trace.jsonl").open("a") as stream:
                stream.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
            return ToolResult(result, image, status != "ok")

    def _safe_arguments(self, name, arguments):
        try:
            return self.tools[name][0].model_validate(arguments).model_dump()
        except (KeyError, ValidationError):
            return {"redacted_invalid_arguments": True}

    def _transition(self, name, args):
        state, extra = deepcopy(self.state), {}
        if name == "reset_view":
            return deepcopy(self.initial), extra
        if name == "set_view":
            state["layout"] = args["layout"]
        elif name in ("move_to", "inspect_position"):
            self._check_position(args["position_nm_xyz"])
            state["position_nm_xyz"] = args["position_nm_xyz"]
            if name == "inspect_position":
                idx = np.floor(np.asarray(args["position_nm_xyz"]) / self.resolution).astype(int) - self.offset
                label = int(self.labels[tuple(idx)])
                lower = (idx + self.offset) * self.resolution
                extra["inspected_voxel"] = {"handle": self.label_to_handle.get(label),
                    "bounds_nm_xyz": [lower.tolist(), (lower + self.resolution).tolist()]}
        elif name == "rotate_view":
            state["projection_orientation_xyzw"] = rotation(state["projection_orientation_xyzw"], args["axis"], args["angle_degrees"])
            if state["layout"] != "4panel":
                state["layout"] = "3d"
        elif name == "zoom_view":
            for key, target in (("cross_section_scale_nm_per_pixel", "2d"), ("projection_scale_nm", "3d")):
                if args["target"] in (target, "both"):
                    value = state[key] / args["factor"]
                    # Fixed display limits; source spacing is still reported independently.
                    lo, hi = (0.5, 1000.) if target == "2d" else (100., 100000.)
                    if not lo <= value <= hi:
                        raise ToolError("scale_limit", f"Requested {target} display scale exceeds [{lo}, {hi}].")
                    state[key] = value
        elif name == "step_slice":
            axis = "xyz".index(args["axis"])
            state["position_nm_xyz"][axis] += args["steps"] * float(self.resolution[axis])
            self._check_position(state["position_nm_xyz"])
            state["layout"] = {"x": "yz", "y": "xz", "z": "xy"}[args["axis"]]
        elif name == "set_layers":
            keys = {"em_visible": "em_visible", "segmentation_visible": "segmentation_visible",
                    "selected_alpha": "selected_alpha", "context_alpha": "context_alpha", "mesh_alpha": "mesh_alpha"}
            for k, v in args.items():
                if v is not None:
                    state[keys[k]] = v
            if not state["em_visible"] and not state["segmentation_visible"]:
                raise ToolError("empty_view", "At least one data layer must remain visible.")
        elif name == "set_em_contrast":
            if args["black"] >= args["white"]:
                raise ToolError("invalid_window", "black must be smaller than white.")
            state["em_black"], state["em_white"] = args["black"], args["white"]
        elif name == "select_objects":
            if len(set(args["handles"])) != len(args["handles"]) or any(h not in self.handle_to_label for h in args["handles"]):
                raise ToolError("unknown_object", "Use distinct known candidate handles or neighbor handles returned by inspect_position.")
            state["selected_objects"] = args["handles"]
            state["segmentation_visible"] = True
        if state["layout"] == "3d" and not state["segmentation_visible"]:
            raise ToolError("empty_view", "3D requires visible segmentation. Use a slice layout to inspect EM only.")
        return state, extra

    def _finish(self, args):
        if args["decision"] != "abstain" and not args["evidence"]:
            raise ToolError("evidence_required", "A non-abstain decision requires a successful observation citation.")
        for evidence in args["evidence"]:
            if evidence["observation_id"] not in self.observations:
                raise ToolError("unknown_observation", "Evidence must cite an observation returned in this run.")
            self._check_position(evidence["position_nm_xyz"])
        outcome = {"case_id": self.case_id, **args, "budget": self._budget()}
        write_json(self.run_dir / "decision.json", outcome)
        self.finished = True
        return {"submitted": outcome, "episode_finished": True}

    def close(self):
        self.renderer.close()


def states_match(expected, actual):
    if set(expected) != set(actual):
        return False
    for key, value in expected.items():
        other = actual[key]
        if key == "projection_orientation_xyzw":
            # q and -q describe the same physical rotation.
            if not (np.allclose(value, other, rtol=0, atol=1e-5)
                    or np.allclose(value, -np.asarray(other), rtol=0, atol=1e-5)):
                return False
        elif key == "position_nm_xyz":
            if not np.allclose(value, other, rtol=0, atol=0.1):
                return False
        elif isinstance(value, float):
            if not np.isclose(value, other, rtol=1e-5, atol=1e-5):
                return False
        elif key == "selected_objects":
            if set(value) != set(other):
                return False
        elif value != other:
            return False
    return True
