# Private episode input and output

Preparation currently accepts the existing ConnectomeBench-style source JSON.
This is a format dependency, not a Python dependency on another repository.
Real source files are researcher inputs and are not bundled or published.

Required top-level fields are `sample_id`, `task_id` (0=split, 1=merge),
`sample_type`, `label`, `root_ids` and `metadata`. Root/SV identifiers must be
decimal strings. Metadata includes `source_type`, `operation_id`, `timestamp`
(Unix seconds), and `edit_point_nm` or `interface_point_nm` (XYZ nm).

For real edits:

- A split task uses `sample_type=merge_edit`; a merge task uses `split_edit`.
- `metadata.source_type=operation`, a positive real operation ID and `label=1`.
- `metadata.before_root_ids` must match `root_ids` (two roots for split, one for
  merge); `metadata.after_root_ids` supplies the private operation witnesses.
- Preparation obtains precise commit time and validates historical witnesses;
  the input timestamp is a hint, not proof of the historical snapshot.

Controls have `label=0` and one supported pair of `(sample_type, source_type)`:
`(synapse_control, synapse_pair)`, `(adjacent_control, control)` or
`(junction_control, control)`. Input roots come from `root_ids`, with optional
matching `metadata.root_ids`. `metadata.cutout_timestamp`, if provided, must
match `metadata.timestamp` when both exist. A synthetic operation identifier
is stored as provenance but never queried as a real history operation.

The generated episode has three layers:

| Directory | Audience | Contents |
| --- | --- | --- |
| `researcher/` | Trusted researcher only | Source identity, provisional labels, edit history and witness audit |
| `harness/` | Trusted observation service only | Original root mapping, data sources, verified snapshot and local volumes |
| `solver/` | Solver-facing task material | Anonymous handles, task, coordinates and observation contract |

The whole episode is private and is not a source-code release. Anonymous task
material alone is not a complete runtime service; the trusted host provides
images without mounting `researcher/` or `harness/` in the solver. Even controls
are not expert-adjudicated negative examples merely because they have label 0.
