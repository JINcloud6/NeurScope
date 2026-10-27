"""Real stdio-MCP/browser smoke test; no model inference or proofreading claim."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid


async def smoke(args):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from benchmark.proofreading import write_json

    run_dir = args.run_dir or args.episode / "runs" / ("mcp_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
    server_args = ["-m", "benchmark.observation_server", "--episode", str(args.episode.resolve()),
                   "--run-dir", str(run_dir.resolve()), "--max-seconds", "300", '--protocol', args.protocol]
    if args.chromium:
        server_args += ["--chromium", args.chromium]
    params = StdioServerParameters(command=sys.executable, args=server_args, env=dict(os.environ))
    async with stdio_client(params) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            available = await session.list_tools()
            assert len(available.tools) == (4 if args.protocol == 'minimal' else 13)
            print(f"Discovered {len(available.tools)} MCP tools", flush=True)
            records = []

            async def call(name, arguments, expected_error=None):
                result = await session.call_tool(name, arguments)
                meta = json.loads(next(c.text for c in result.content if c.type == "text"))
                if expected_error:
                    assert result.isError and meta["error"]["code"] == expected_error, meta
                else:
                    assert not result.isError, meta
                    if name not in ("get_task", "finish"):
                        assert any(c.type == "image" and c.mimeType == "image/png" for c in result.content)
                records.append({"tool": name, "status": meta["status"],
                    "observation_id": meta.get("observation_id"), "expected_error": expected_error})
                print(name, meta["status"], meta.get("observation_id", ""), flush=True)
                return meta

            if args.protocol == 'minimal':
                initial = await call('observe', {})
                anchor = initial['state']['position_nm_xyz']
                await call('rotate_view', {'axis': 'y', 'angle_degrees': 30}, expected_error='wrong_mode')
                await call('set_view', {'mode': '3d'})
                await call('rotate_view', {'axis': 'y', 'angle_degrees': 30})
                await call('set_view', {'mode': 'xy'})
                await call('step_slice', {'steps': 3})
                await call('step_slice', {'steps': -3})
                reset = await call('set_view', {'mode': 'xz'})
                assert reset['state']['position_nm_xyz'] == anchor
                await call('step_slice', {'steps': 2})
                await call('set_view', {'mode': 'yz'})
                await call('step_slice', {'steps': -2})
                await call('set_view', {'mode': '4panel'})
                await call('step_slice', {'steps': 1}, expected_error='wrong_mode')
                await call('zoom_view', {'factor': 2}, expected_error='unknown_tool')
                write_json(run_dir / 'smoke_report.json', {'status': 'passed', 'transport': 'stdio MCP',
                    'model_inference': False, 'tools_discovered': 4, 'protocol': 'minimal', 'calls': records})
                print(f'Smoke test passed: {run_dir}', flush=True)
                return
            task = await call("get_task", {})
            center = task["center_nm_xyz"]
            await call("observe", {})
            await call("move_to", {"position_nm_xyz": center})
            await call("set_view", {"layout": "3d"})
            await call("rotate_view", {"axis": "y", "angle_degrees": 30})
            await call("zoom_view", {"factor": 2, "target": "3d"})
            await call("set_view", {"layout": "xy"})
            await call("step_slice", {"axis": "z", "steps": 3})
            await call("set_layers", {"selected_alpha": 0, "context_alpha": 0})
            await call("set_em_contrast", {"black": 70, "white": 170})
            inspected = await call("inspect_position", {"position_nm_xyz": center})
            neighbor = inspected["inspected_voxel"]["handle"]
            handles = list(dict.fromkeys(task["candidate_handles"][:1] + ([neighbor] if neighbor else [])))
            await call("select_objects", {"handles": handles})
            await call("reset_view", {})
            outside = [task["observation_bounds_nm_xyz"][1][0], center[1], center[2]]
            await call("move_to", {"position_nm_xyz": outside}, expected_error="out_of_bounds")
            await call("finish", {"decision": "abstain", "confidence": 0,
                "reason": "Infrastructure smoke test only; no model or expert proofreading was performed.",
                "evidence": [], "stop_reason": "uncertain"})
            await call("observe", {}, expected_error="episode_finished")
            write_json(run_dir / "smoke_report.json", {"status": "passed", "transport": "stdio MCP",
                "model_inference": False, "expert_review": False, "tools_discovered": len(available.tools), "calls": records})
    print(f"Smoke test passed: {run_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--chromium")
    parser.add_argument('--protocol', choices=['minimal', 'legacy'], default='minimal')
    asyncio.run(smoke(parser.parse_args()))


if __name__ == "__main__":
    main()
