"""Expose one cached proofreading episode as a stdio MCP tool server."""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

from benchmark.observation_tools import ObservationEnvironment, tool_specs
from benchmark.proofreading import write_json


LEGACY_INSTRUCTIONS = """You are observing a bounded pre-edit connectome scene. First call get_task.
Use anonymous object handles and global XYZ nanometers, never source root IDs.
Every view-changing tool returns a fresh PNG plus the corresponding state and observation ID.
zoom_view factor>1 magnifies; step_slice steps use the reported anisotropic voxel spacing.
Use orthogonal EM slices to check continuity, 3D rotation to inspect geometry, and layer/contrast tools to expose EM.
Data are cached at fixed resolution: display zoom does not improve source resolution, and ROI edges truncate objects.
Actions, images and wall time are limited. Invalid calls also consume actions. Reset does not reset budgets.
Submit finish with error/no_error/abstain, confidence, brief evidence citing returned observation IDs, and a stop reason.
finish is available after budget exhaustion or render failure. No editing, code execution, historical API or labels are exposed.
"""

INSTRUCTIONS = """Observe the fixed-anchor historical scene using only observe, set_view, rotate_view, step_slice.
set_view accepts mode=4panel|3d|xy|xz|yz and resets the slice to the task anchor.
rotate_view requires 3d and accepts axis=x|y|z plus angle_degrees.
step_slice requires a single 2D mode and accepts signed integer steps; positive follows the plane normal.
Every successful call returns a fresh PNG and matching state. No pan, zoom, editing or historical queries.
The host provides the task initially and collects the final structured JSON answer; neither requires a tool.
"""


def create_server(environment):
    from mcp.server import Server
    from mcp.types import CallToolResult, ImageContent, TextContent, Tool

    minimal = set(environment.tools) == {'observe', 'set_view', 'rotate_view', 'step_slice'}
    server = Server("neurscope-proofreading", instructions=INSTRUCTIONS if minimal else LEGACY_INSTRUCTIONS)

    @server.list_tools()
    async def list_tools():
        return [Tool(**spec) for spec in tool_specs(environment.tools)]

    # Own argument validation records invalid calls and enforces the same limits
    # for MCP callers and Python function callers.
    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        try:
            result = await asyncio.to_thread(environment.call, name, arguments or {})
        except Exception as exc:
            # SDK default errors echo exception strings (potential private paths).
            print(f"Observation internal failure: {type(exc).__name__}", file=sys.stderr, flush=True)
            return CallToolResult(content=[TextContent(type="text", text=json.dumps({
                "status": "error", "error": {"code": "internal_error",
                "message": "Trusted service failed; no observation released. Record this as infrastructure failure."}}))], isError=True)
        content = [TextContent(type="text", text=json.dumps(result.metadata, ensure_ascii=False, allow_nan=False))]
        if result.image is not None:
            content.append(ImageContent(type="image", mimeType="image/png",
                                        data=base64.b64encode(result.image).decode("ascii")))
        return CallToolResult(content=content, isError=result.is_error)

    return server


async def serve(environment):
    from mcp.server.stdio import stdio_server
    server = create_server(environment)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, help="Trusted directory containing verified harness/ and solver/ configs")
    parser.add_argument("--run-dir", type=Path, help="New output directory; existing directories are rejected")
    parser.add_argument("--chromium", help="Optional headless Chromium executable")
    parser.add_argument("--max-actions", type=int, default=30)
    parser.add_argument("--max-observations", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--render-timeout", type=int, default=60)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--export-tools", type=Path, help="Export JSON Schemas without loading data, MCP or browser")
    parser.add_argument("--protocol", choices=['minimal', 'legacy'], default='minimal')
    parser.add_argument("--fixed-fov-nm", type=float, default=4000)
    args = parser.parse_args()
    if args.export_tools:
        from benchmark.minimal_observation import MINIMAL_TOOLS
        write_json(args.export_tools, {"protocol": "proofreading.minimal_tools.v2" if args.protocol == 'minimal' else "proofreading.observation_tools.v1", "instructions": INSTRUCTIONS if args.protocol == 'minimal' else LEGACY_INSTRUCTIONS, "tools": tool_specs(MINIMAL_TOOLS if args.protocol == 'minimal' else None)})
        return
    if args.episode is None:
        parser.error("--episode is required unless exporting schemas")
    from benchmark.observation_renderer import NeuroglancerRenderer
    run_dir = args.run_dir or args.episode / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    renderer = NeuroglancerRenderer(args.episode, chromium=args.chromium, width=args.width,
                                   height=args.height, timeout=args.render_timeout)
    if args.protocol == 'minimal':
        from benchmark.minimal_observation import MinimalObservationEnvironment
        environment = MinimalObservationEnvironment(args.episode, run_dir, renderer,
            max_actions=args.max_actions, max_observations=args.max_observations, max_seconds=args.max_seconds,
            fixed_fov_nm=args.fixed_fov_nm)
    else:
        environment = ObservationEnvironment(args.episode, run_dir, renderer,
            max_actions=args.max_actions, max_observations=args.max_observations, max_seconds=args.max_seconds)
    print(f"Proofreading MCP ready; private trajectory: {run_dir}", file=sys.stderr, flush=True)
    try:
        asyncio.run(serve(environment))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
