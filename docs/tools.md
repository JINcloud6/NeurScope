# Observation protocol

Current default: `proofreading.minimal_tools.v2`, display preset
`candidate_masks_only.v1`. All physical positions are XYZ nanometers. The
initial center and FOV remain fixed; no pan or zoom is exposed. The default
FOV is 4000 nm and can be configured by the trusted host before an episode.

| Function | Arguments | Behavior |
| --- | --- | --- |
| `observe` | `{}` | Capture the current scene |
| `set_view` | `{"mode":"4panel"}` | Select `4panel`, `3d`, `xy`, `xz` or `yz`; reset slicing to the anchor |
| `rotate_view` | `{"axis":"y","angle_degrees":30}` | Rotate only when the current mode is `3d` |
| `step_slice` | `{"steps":5}` | Move signed integer slices only in a single 2D mode; spacing follows the actual plane resolution |

Every successful call returns a PNG plus its observation ID, actual view state
and hashes. Invalid calls are logged and consume the action budget. Rendering
requires Neuroglancer screenshot acknowledgement and matching state, rather
than treating a fixed delay as completion. ROI and time/action/image budgets
apply. Source data resolution does not improve by changing views.

Only candidate masks have color: A/B for a split task, A for a merge task.
Neighboring raw EM remains visible. Cached historical labels remain available
to the trusted environment; they are not remotely queryable by the solver.
Legacy tools are retained for compatibility and are not the default protocol.

Task semantics:

- `split`: are A and B incorrectly separated? A true correction merges them.
- `merge`: does A contain an incorrect connection? A true correction splits it.
- Decision: `error`, `no_error` or `abstain`. Insufficient evidence is not proof
  of `no_error`.

The host gives the task and initial observation. Final JSON contains `case_id`,
`decision`, `confidence`, `reason`, `evidence` and `stop_reason`. Evidence cites
returned `observation_id` and `position_nm_xyz` with a short description. The
canonical complete task/output prompt is
[the packaged template](../benchmark/prompts/minimal_solver_prompt.md).
The runner substitutes template variables, including the attempt's response
limit. Generate the authoritative JSON schemas with:

```bash
python observation_mcp.py --export-tools docs/tool_schema.json
```

A stdio MCP server only restricts the exposed functions. It does not prevent
an agent with shell access from reading private files. Actual blind execution
requires the isolated runner or equivalent externally verified permissions.
