"""Headless NG renderer: all Playwright work lives on one dedicated thread."""
from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
import queue
import threading
import time

import numpy as np

from benchmark.observation_tools import Capture
from benchmark.proofreading import load_json, make_viewer, screenshot


def em_shader(black, white):
    return ("void main() { emitGrayscale(clamp((toNormalized(getDataValue()) - "
            f"{black / 255:.12f}) / {(white - black) / 255:.12f}, 0.0, 1.0)); }}")


def source_signature(layer):
    return [(str(source.url), None if source.transform is None else source.transform.to_json())
            for source in layer.source]


def check_sources(state, expected_sources):
    if {layer.name for layer in state.layers} != set(expected_sources):
        raise ValueError("Unexpected layers in screenshot")
    for name, sources in expected_sources.items():
        if source_signature(state.layers[name]) != sources:
            raise ValueError("Data source changed during capture")


class NeuroglancerRenderer:
    def __init__(self, episode: Path, *, chromium=None, width=1000, height=800, timeout=60):
        if not (256 <= width <= 1600 and 256 <= height <= 1600 and 1 <= timeout <= 180):
            raise ValueError("Invalid bounded canvas/timeout")
        self.episode, self.chromium = episode, chromium
        self.width, self.height, self.timeout = width, height, timeout
        self.queue = queue.Queue()
        self.closed = False
        self.thread = None
        self.viewer_url = None

    def capture(self, state, *, timeout_seconds=None):
        if self.closed:
            raise RuntimeError("Renderer is closed")
        limit = self.timeout + 45 if timeout_seconds is None else min(self.timeout + 45, timeout_seconds)
        if limit <= 0:
            raise TimeoutError("Episode deadline expired")
        future = Future()
        self.queue.put((deepcopy(state), future, time.monotonic() + limit))
        if self.thread is None:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
        try:
            return future.result(timeout=limit)
        except Exception:
            future.cancel()
            self.closed = True
            self.queue.put(None)
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(None)
        if self.thread is not None:
            self.thread.join(timeout=self.timeout + 5)

    def _worker(self):
        # Imports are intentionally lazy; schema export/tests need no browser or CAVE.
        try:
            from playwright.sync_api import sync_playwright
            viewer = make_viewer(self.episode, 0)
            self.viewer_url = viewer.get_viewer_url()
            expected_sources = {name: source_signature(viewer.state.layers[name])
                                for name in ("EM", "Objects")}
            h = load_json(self.episode / "harness/config.json")
            candidates = {int(v["local_label"]): handle for handle, v in h["objects"].items()}
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True, executable_path=self.chromium,
                    args=["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
                try:
                    # Enforce localhost-only browsing: renderer never fetches remote data.
                    context = browser.new_context(viewport={"width": self.width, "height": self.height})
                    from urllib.parse import urlsplit
                    allowed = urlsplit(self.viewer_url)
                    def route(request_route):
                        target = urlsplit(request_route.request.url)
                        if target.scheme in ("data", "blob") or (target.hostname == allowed.hostname and target.port == allowed.port):
                            request_route.continue_()
                        else:
                            request_route.abort()
                    context.route("**/*", route)
                    page = context.new_page()
                    errors = []
                    page.on("pageerror", lambda error: errors.append(type(error).__name__))
                    page.goto(self.viewer_url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                    while True:
                        job = self.queue.get()
                        if job is None:
                            break
                        state, future, deadline = job
                        try:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Episode deadline expired during initialization")
                            if errors:
                                raise RuntimeError("Browser page errors")
                            self._apply(viewer, state, candidates)
                            result = {}
                            def capture():
                                try:
                                    result["action"] = screenshot(viewer, min(self.timeout, max(0., deadline - time.monotonic())),
                                                                  self.width, self.height, return_action=True)
                                except Exception as exc:
                                    result["error"] = exc
                            thread = threading.Thread(target=capture, daemon=True)
                            thread.start()
                            capture_deadline = min(deadline, time.monotonic() + self.timeout + 2)
                            while thread.is_alive() and time.monotonic() < capture_deadline:
                                page.wait_for_timeout(50)  # pump callbacks; this is not readiness detection
                            if thread.is_alive():
                                raise TimeoutError("Screenshot worker timed out")
                            if "error" in result:
                                raise result["error"]
                            if errors:
                                raise RuntimeError("Browser page errors")
                            action = result["action"]
                            check_sources(action.viewer_state, expected_sources)
                            actual = self._actual(action.viewer_state, state, candidates)
                            if not future.done():
                                future.set_result(Capture(action.screenshot.image, actual, {
                                    "completion": "neuroglancer_screenshot_ack", "request_id": action.screenshot.id,
                                    "browser_version": browser.version}))
                        except Exception as exc:
                            if not future.done():
                                future.set_exception(exc)
                finally:
                    browser.close()
        except Exception as exc:
            # Resolve queued callers even when browser launch/initialization fails.
            while True:
                try:
                    job = self.queue.get_nowait()
                except queue.Empty:
                    break
                if job is not None and not job[1].done():
                    job[1].set_exception(exc)
            self.closed = True

    @staticmethod
    def _apply(viewer, state, candidates):
        by_handle = {h: label for label, h in candidates.items()}
        labels = [by_handle[h] if h in by_handle else int(h[1:]) for h in state["selected_objects"]]
        with viewer.txn() as s:
            s.position = state["position_nm_xyz"]
            s.layout = state["layout"]
            s.projection_orientation = state["projection_orientation_xyzw"]
            s.cross_section_orientation = [0, 0, 0, 1]
            s.cross_section_scale = state["cross_section_scale_nm_per_pixel"]
            s.projection_scale = state["projection_scale_nm"]
            s.layers["EM"].visible = state["em_visible"]
            s.layers["EM"].shader = em_shader(state["em_black"], state["em_white"])
            seg = s.layers["Objects"]
            seg.visible = state["segmentation_visible"]
            seg.segments = labels
            seg.selected_alpha = state["selected_alpha"]
            seg.not_selected_alpha = state["context_alpha"]
            seg.object_alpha = state["mesh_alpha"]

    @staticmethod
    def _actual(s, expected, candidates):
        seg = s.layers["Objects"]
        # Verify non-public invariants too: the image must use the same local volumes,
        # grayscale window, orthogonal orientation and 1-nm coordinate system.
        if s.layers["EM"].shader != em_shader(expected["em_black"], expected["em_white"]):
            raise ValueError("EM shader changed during capture")
        section_q = s.cross_section_orientation
        if section_q is None:
            section_q = [0, 0, 0, 1]  # Browser omits the identity default.
        if not np.allclose(section_q, [0, 0, 0, 1], atol=1e-5):
            raise ValueError("Oblique slice state is outside protocol")
        dims = s.dimensions.to_json()
        if any(dims.get(axis) != [1e-9, "m"] for axis in "xyz"):
            raise ValueError("Coordinate system changed during capture")
        projection_q = s.projection_orientation
        if projection_q is None:
            projection_q = [0, 0, 0, 1]
        return {"position_nm_xyz": list(map(float, s.position)), "layout": s.layout.type,
            "projection_orientation_xyzw": list(map(float, projection_q)),
            "cross_section_scale_nm_per_pixel": float(s.cross_section_scale),
            "projection_scale_nm": float(s.projection_scale),
            "selected_objects": [candidates.get(int(label), f"N{int(label)}") for label in seg.segments],
            "em_visible": bool(s.layers["EM"].visible), "segmentation_visible": bool(seg.visible),
            "selected_alpha": float(seg.selected_alpha), "context_alpha": float(seg.not_selected_alpha),
            "mesh_alpha": float(seg.object_alpha), "em_black": expected["em_black"], "em_white": expected["em_white"]}
