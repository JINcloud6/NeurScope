# Standalone extraction audit — 2026-10-27

## Verified source boundaries

Inspection of Python imports and calls found that the observation environment
and runner use only local `benchmark.*` modules, the standard library and
third-party libraries. They do not import `mcp_src`, `api_server`, `mcp_main`,
`chrome_extension` or the original application's global sessions. In particular,
`NeuroglancerRenderer` calls this package's `make_viewer`, which constructs a
new Viewer directly with upstream `neuroglancer` and local cached arrays.

The original project name occurred in the MCP server name, temporary directory
prefix and isolation probe paths; these were naming/configuration artifacts,
not application API calls. They have been replaced. The preparation format
still accepts ConnectomeBench-style metadata, and remote preparation still
requires CAVEclient/CloudVolume and permission to access the chosen datastack.
The Codex runner uses the Codex subscription endpoint; that protocol dependency
is explicit and separate from the model-neutral observation/MCP interfaces.

## Included and excluded

Included: `benchmark/`, its offline tests, the minimal MCP entry, canonical
English prompt, packaging metadata, portable documentation, release checker and
offline CI. The legacy 13-tool implementation is retained because shared state,
validation and geometry are currently implemented there; four tools remain the
default exposed protocol.

Excluded: original application/server/UI code, original Git history, account
files, data and researcher handoffs, all real sample configs, ground truth,
images, model request/response traces, run logs, browser binaries, Python
environments, certificate pins and host-specific experiment documentation.
The original checkout is retained unchanged. No GitHub upload is performed.

## Portability changes

- Credentials: explicit auth-file option or `CODEX_HOME`/current-user default;
  read only by the trusted parent. No credential values copied or embedded.
- Native Codex executable: explicit option, verified as executable ELF, mounted
  into the sandbox. No global npm installation layout is assumed.
- Isolation probes: current project, user home, episode and auth-file paths plus explicit private
  paths, passed as path metadata; no host data mounts. Fail closed on visibility.
- Dataset, case ID and segmentation/EM mips: explicit preparation parameters.
- Browser executable: Playwright discovery or explicit parameter, no local
  Chromium installation path. Sandbox-internal paths and the fixed subscription
  endpoint are intentional protocol constants, not personal host locations.
- Prompt: included in the Python package; loading does not depend on checkout
  location or current working directory.
- New `.gitignore` excludes local credentials, datasets and runtime artifacts.
  Release checks flag private artifacts, personal paths and common secret-value
  patterns without printing matches. The release contains source files only;
  heuristic checks cannot guarantee safety of arbitrary future additions.

## Validation scope

Offline tests cover existing history/coordinate/ID/schema/budget/state checks
and new auth-path selection, native executable validation and sandbox command
construction. No model calls or remote data downloads are authorized or run as
part of this extraction. Results of packaging and local browser checks are
recorded below after execution. This extraction does not establish a formal
benchmark score or guarantee compatibility with every dataset or Codex build.

No top-level LICENSE was present in the original checkout. This extraction does
not invent a license grant; verify provenance and choose an appropriate license
before distributing code under an explicit open-source license.

### Executed checks

- 42 offline tests passed.
- Wheel built successfully; prompt resource and schema export worked from an
  unpacked wheel outside the source directory. Dependency installation from a
  clean remote host has not been executed.
- Real stdio MCP/browser smoke test with an existing private local cache passed:
  four tools discovered, 11 PNG observations, rotation/slicing/mode reset and
  rejection of invalid mode/legacy tools verified. No sample or smoke output is
  included in this release. The first restricted-shell attempt could not finish
  local communication and was stopped; the localhost-enabled retry passed.
- Real bubblewrap startup with `codex --version` passed all current project,
  home, private-path, credential-environment and network isolation probes. This
  did not invoke inference or read authentication. No new end-to-end model
  trial was run after the refactor.
- AST inspection found no imports of the original application modules or
  Anthropic SDK. Release-source path/secret/artifact checks passed.
