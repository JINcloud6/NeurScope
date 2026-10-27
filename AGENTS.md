# NeurScope development instructions

Build a reproducible, auditable neuron proofreading benchmark. Keep trusted
preparation, observation service, solver execution and offline scoring separate.
This standalone repository has no runtime dependency on the original
NeuroAnswer application. Do not reintroduce its global CAVE/Viewer sessions.

- `split` means incorrect separation of A/B (correction: merge); `merge` means
  incorrect connection within A (correction: split). Decisions are
  `error | no_error | abstain`. No observed error does not imply `no_error`.
- Validate exact historical input roots and SV witnesses; selecting old roots
  or a timestamp alone does not establish historical voxel/mesh consistency.
  Controls use their recorded snapshot; never query synthetic operation IDs.
- Never silently substitute current roots, meshes, another sample, or blank
  observations. Infrastructure failures and solver abstentions are distinct.
- Public coordinates are XYZ nm. Read source resolution, mip and offset
  independently for each layer. Root/SV IDs cross JSON as decimal strings.
- Remote data access is read-only; never submit edits.
- Solver inputs exclude labels, after roots, operation history, prior model
  predictions, researcher notes, source identity and credentials. Use anonymous
  episode handles. Verify filesystem/network/tool restrictions, not just prompts.
- Keep Viewer, histories, state and budgets independent per episode. Use bounded
  whitelisted tools. Log rejected actions, actual state, image/state hashes and
  render completion. A fixed sleep is not rendering verification.
- Default protocol has only observe/set_view/rotate_view/step_slice. No pan or
  zoom. Display only candidate masks while retaining surrounding raw EM.
  Update schemas, prompt and documentation together when interfaces change.
- Package resources must work after installation outside the source directory.
  No personal absolute paths, embedded account credentials or fixed browser/
  Codex installation locations. Sandbox-internal paths are intentional.
- Credentials stay in trusted local storage outside this checkout and are never
  written to request logs. Never disable TLS verification. Do not put datasets,
  researcher configs, real trajectories or authentication files in Git.
- Before release run `python scripts/check_release.py`. It is a heuristic, not
  a substitute for reviewing every file intended for publication. Do not add a
  license for third-party material without verifying its provenance and rights.
- Do not run paid/quota-consuming models or large downloads without an explicit
  experiment scope and budget. Retries share the user's overall request budget.
- Keep protocol/data/model versions and scoring/exclusion rules explicit. Seen
  development cases are not unseen test cases. Distinguish verified facts,
  hypotheses and proposed defaults; do not report unrun experiments as results.
- Offline verification: `python -m unittest discover -s tests -v`. Avoid network
  or Viewer creation on module import. Real browser/MCP tests use
  `python -m benchmark.observation_smoke` with an explicitly supplied episode;
  they do not invoke a paid model. Report actual scope and remaining limitations.
