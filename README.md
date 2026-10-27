# NeurScope

Controlled Neuroglancer observations for neuron proofreading agents. Given an
anonymous candidate and a physical location, an agent inspects historical EM
and segmentation through four bounded tools and returns a structured decision.

This is a standalone extraction of the proofreading environment developed
alongside NeuroAnswer. It has no runtime dependency on NeuroAnswer's chat
server, exploration MCP, browser extension or Anthropic integration. It uses
the upstream **Neuroglancer Python package** directly.

## Install

Python 3.10+ is required. The isolated Codex runner additionally requires Linux,
bubblewrap, `/usr/bin/python3`, and a native Linux Codex executable compatible
with the app-server protocol. The original experiments used Codex CLI 0.156.1;
compatibility with other CLI builds must be checked separately.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m playwright install chromium
python -m unittest discover -s tests -v
python scripts/check_release.py
```

For trusted remote data preparation, install the optional dependencies:

```bash
python -m pip install -e '.[prepare]'
```

## Configure CAVE authentication

Remote `fetch` requires a valid CAVE token with permission for the target
services/datastack. Some underlying EM volumes may be public, but historical
root/operation queries and protected segmentation still require authentication.
Offline tests, schema export and observations of an already verified local
cache do not require a CAVE token. Model authentication is separate.

First install `.[prepare]` as above. Start an interactive Python session in the
same environment and OS user that will run `fetch`:

```bash
python
```

Then request the login/token-page instructions and save the resulting token
using a hidden prompt (paste the token at that prompt, never into source code):

```python
import os
from getpass import getpass
from caveclient.auth import AuthClient

os.umask(0o077)
auth = AuthClient()
auth.get_new_token(open=False)
# Follow the printed URL in your browser and obtain the token.
auth.save_token(token=getpass("CAVE token: "))
```

CAVEclient normally saves its default token outside the repository at
`~/.cloudvolume/secrets/cave-secret.json`, and may also maintain server-specific
credential files. NeurScope's `fetch` constructs `CAVEclient(datastack, ...)`
and lets CAVEclient load those local credentials automatically. Setting an
arbitrary `CAVE_TOKEN` environment variable alone is not currently wired into
NeurScope. `save_token` does not overwrite an existing token by default; use
`overwrite=True` only when intentionally replacing it. If your institution uses
a different authentication server, follow that service's instructions instead
of assuming a token is valid across servers.

Verify authentication and datastack metadata access without printing the token:

```python
from caveclient import CAVEclient

client = CAVEclient("YOUR_DATASTACK", write_server_cache=False)
client.info.get_datastack_info()
print("Datastack metadata access succeeded.")
```

Replace `YOUR_DATASTACK` with the actual name. This verifies metadata access;
`fetch` additionally exercises historical graph and volume permissions. A
401/403 response can mean an expired/wrong-service token or missing dataset
permissions. A token does not by itself grant access to every dataset. Check
network/proxy reachability separately. On another host, configure credentials
there using an approved private channel; do not distribute them through Git.
Requesting a replacement token can invalidate an earlier token for the same
project, so avoid regenerating one unnecessarily.

See the official [CAVEclient authentication guide](https://www.caveconnecto.me/CAVEclient/tutorials/authentication/).
Never print `auth.token`, dump credential files, or place credentials in solver
input. These setup steps are instructions only; release validation does not
read credentials or make authenticated dataset requests.

Model credentials are read only by the trusted parent from an explicit
`--codex-auth-file`, otherwise `$CODEX_HOME/auth.json`, otherwise the current
user's `~/.codex/auth.json`. They are independent of the CAVE token.

## Prepare an episode

No real samples, labels, local volumes, screenshots, trajectories or account
files are bundled. Supply a private source JSON following
[the episode format](docs/episode_format.md). Keep episode outputs outside your
Git checkout or under ignored `data/`.

```bash
python -m benchmark.proofreading prepare \
  --source /absolute/private/sample.json --output data/example \
  --case-id episode_001 --datastack YOUR_DATASTACK \
  --seg-mip SEGMENTATION_MIP --em-mip EM_MIP
python -m benchmark.proofreading fetch --output data/example
```

Replace the uppercase values with values verified for the dataset. Mip numbers
are source-specific and must not be assumed to match between EM and segmentation.
The fetch stage checks historical identity and reconstructs bounded local data;
it does not submit remote segmentation edits. It requires appropriate network
access and data permissions. Cached episode paths are resolved relative to the
episode directory, so move the entire episode rather than individual files.

## Use the four tools

```bash
neurscope-tools --export-tools docs/tool_schema.json
neurscope-tools --episode data/example --run-dir runs/mcp-example
```

The latter starts a stdio MCP server with `observe`, `set_view`, `rotate_view`
and `step_slice`. A caller must supply the task and collect the final answer;
this MCP server alone does not isolate a solver. See
[tools and boundaries](docs/tools.md). Browser location is discovered by
Playwright, or can be supplied explicitly with `--chromium`.

## Collect one isolated Codex trajectory

Authenticate Codex locally using the CLI before running. Supply the path to its
**native ELF executable**, not an npm JavaScript launcher. This is an optional
account-backed runner, not a generic OpenAI API/model adapter. It sends requests
to the Codex subscription endpoint and may consume account quota. Run only for
an explicitly authorized experiment.

```bash
neurscope-codex \
  --episode data/example --output runs/codex-example \
  --chromium /absolute/path/to/chromium \
  --codex-binary /absolute/path/to/native/codex \
  --model YOUR_AVAILABLE_MODEL --max-model-responses 20
python -m benchmark.summarize_codex_run \
  --episode data/example --run runs/codex-example
```

The trusted parent keeps credentials and private data. The child gets an empty
filesystem/PID/network namespace and only a credential-free inference socket.
Use repeated `--private-path /absolute/private/path` to test additional paths
for invisibility. Paths visible through shared system mounts cause isolation
checks to fail; do not bypass a failed probe. The native binary is mounted at a
fixed *sandbox-internal* path. The child never mounts host credentials or the
source checkout. The broker exposes only the four function schemas.

The final permitted request must return a decision and has no tools. Failed
requests consume the attempt budget. On a fresh retry, subtract already issued
requests from the overall authorization and pass the remainder with
`--max-model-responses`; the runner does not track budgets across independent
processes. Preserve the original failed run separately. Optional
`--tls-fingerprint` pins a currently verified SHA256 endpoint certificate in
addition to normal public CA validation; do not reuse an old experiment's pin.

Outputs include the actual prompt, protocol, messages, request/response events,
per-step PNG/state/hash, final JSON and audit. These outputs may contain private
observations and are excluded from the source release. Transport failures are
infrastructure failures, not solver abstentions. Visible messages and returned
service events are recorded; unexposed internal reasoning is not available.

## Code map

| File | Role |
| --- | --- |
| `benchmark/proofreading.py` | Three configuration layers, historical input validation, local volume reconstruction, Viewer and static capture |
| `benchmark/minimal_observation.py` | Current four-tool protocol, fixed anchor/FOV and candidate-only masks |
| `benchmark/observation_tools.py` | Shared validation, budgets, geometry and audit; retains legacy 13-tool implementation |
| `benchmark/observation_renderer.py` | Local Neuroglancer Viewer, Chromium rendering, completion acknowledgement and screenshot state checks |
| `benchmark/observation_server.py`, `observation_mcp.py` | Standalone stdio MCP entry |
| `benchmark/codex_isolated_run.py`, `benchmark/isolated_codex_entry.py` | Trusted inference broker and isolated Codex app-server |
| `benchmark/secure_transport.py` | Optional public CA plus certificate pin verification |
| `benchmark/summarize_codex_run.py` | Completed-run integrity audit and readable screenshot trajectory |
| `benchmark/prompts/minimal_solver_prompt.md` | Canonical English prompt template; installed as package data |
| `tests/` | Offline tests, without remote data or model calls |

Extraction scope, verification and limitations are recorded in
[the extraction audit](docs/extraction_audit.md). There is no inherited Git
history, no automatic model execution and no real benchmark dataset in this
source release. Dependency ranges are installation constraints, not a frozen
cross-platform environment; record resolved versions for each experiment.
