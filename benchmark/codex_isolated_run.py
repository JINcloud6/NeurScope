"""One fresh Codex app-server episode, four tools, <=20 model requests.

Trusted parent owns data, rendering, credentials and logs. The Codex process
has an empty filesystem/network namespace and only a credential-free broker
socket. Never use this to resume the development conversation.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import http.server
import json
import os
from pathlib import Path
import queue
import shutil
from importlib.resources import files
import socket
import subprocess
import tempfile
import threading
import time
import uuid

from pydantic import Field

from benchmark.minimal_observation import MINIMAL_TOOLS, MinimalObservationEnvironment
from benchmark.observation_tools import Finish, tool_specs
from benchmark.observation_renderer import NeuroglancerRenderer
from benchmark.proofreading import write_json


def resolve_auth_file(explicit=None):
    """Locate credentials without reading or logging them during configuration."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    codex_home = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex')).expanduser()
    return (codex_home / 'auth.json').resolve()


def validate_codex_binary(binary):
    """The sandbox mounts a native executable, never a host npm launcher."""
    binary = Path(binary).expanduser().resolve(strict=True)
    with binary.open('rb') as stream:
        if stream.read(4) != b'\x7fELF':
            raise ValueError('--codex-binary must point to a native Linux ELF executable')
    if not os.access(binary, os.X_OK):
        raise ValueError('Codex binary is not executable')
    return binary


class FinalAnswer(Finish):
    case_id: str = Field(min_length=1)


class Broker:
    """Only allow model inference; authentication stays outside the sandbox."""
    def __init__(self, path, output, max_responses=20, tls_fingerprint=None, auth_file=None):
        if not isinstance(max_responses, int) or not 1 <= max_responses <= 20:
            raise ValueError('Model response limit must be between 1 and 20')
        self.path, self.output, self.max_responses = path, output, max_responses
        self.count = 0
        self.lock = threading.Lock()
        self.errors = []
        self.usage = []
        self.seen_models = set()
        self.tls_fingerprint = tls_fingerprint
        # No headers are ever written to a log.
        auth = json.loads(resolve_auth_file(auth_file).read_text())['tokens']
        self.headers = {'Authorization': 'Bearer ' + auth['access_token'],
            'ChatGPT-Account-Id': auth['account_id'], 'User-Agent': 'codex_cli_rs/0.156.1',
            'originator': 'codex_cli_rs', 'OpenAI-Beta': 'responses=experimental'}

    def start(self):
        broker = self
        class UnixHTTP(http.server.ThreadingHTTPServer):
            address_family = socket.AF_UNIX
            def server_bind(self):
                self.socket.bind(self.server_address)
                self.server_name, self.server_port = 'local-broker', 0

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path not in ('/v1/responses', '/responses'):
                    self.send_error(403, 'Only Responses inference is allowed')
                    return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 80_000_000:
                        raise ValueError('Invalid bounded request size')
                    payload = json.loads(self.rfile.read(size))
                    with broker.lock:
                        if broker.count >= broker.max_responses:
                            self.send_error(429, 'Hard model response limit reached')
                            return
                        # Model sees exactly the four registered functions, never
                        # shell/file/web tools that the stock CLI may advertise.
                        offered = list(payload.get('tools', []))
                        additional = [x for x in payload.get('input', []) if x.get('type') == 'additional_tools']
                        for item in additional:
                            offered.extend(item.get('tools', []))
                        def flatten(tools):
                            for tool in tools:
                                if tool.get('type') == 'namespace':
                                    yield from flatten(tool.get('tools', []))
                                else:
                                    yield tool
                        offered = list(flatten(offered))
                        allowed = [t for t in offered if t.get('type') == 'function' and t.get('name') in MINIMAL_TOOLS]
                        if not allowed:
                            # This CLI build routes dynamic tools through its V8
                            # code host even when the feature is disabled. The
                            # registered names are documented in exec's schema;
                            # expose the original four native schemas to the model.
                            descriptions = '\n'.join(t.get('description', '') for t in offered)
                            if all(('### `' + name + '`') in descriptions for name in MINIMAL_TOOLS):
                                allowed = [{'type': 'function', 'name': t['name'],
                                    'description': t['description'], 'parameters': t['inputSchema'], 'strict': False}
                                    for t in tool_specs(MINIMAL_TOOLS)]
                        if {t['name'] for t in allowed} != set(MINIMAL_TOOLS):
                            broker.output.mkdir(exist_ok=True)
                            write_json(broker.output / 'rejected_request.json', payload)
                            broker.errors.append({'kind': 'tool_schema_mismatch', 'types': [t.get('type') for t in offered],
                                'names': [t.get('name') for t in offered], 'payload_keys': list(payload)})
                            self.send_error(403, 'Expected four observation function schemas')
                            return
                        text = json.dumps(payload, ensure_ascii=False)
                        for forbidden in ('after_root_ids', 'before_root_ids', 'ground_truth', 'operation_id'):
                            if forbidden in text:
                                raise ValueError('Input provenance leak detected')
                        broker.count += 1
                        index = broker.count
                    if additional:
                        additional[0]['tools'] = allowed
                        payload['input'] = [x for x in payload['input'] if x.get('type') != 'additional_tools' or x is additional[0]]
                        payload.pop('tools', None)
                    else:
                        payload['tools'] = allowed
                    payload['parallel_tool_calls'] = False
                    payload['store'] = False
                    payload['stream'] = True
                    if index == broker.max_responses:
                        if additional:
                            additional[0]['tools'] = []
                        else:
                            payload['tools'] = []
                        payload['tool_choice'] = 'none'
                        payload.setdefault('input', []).append({'role': 'developer', 'content': [
                            {'type': 'input_text', 'text': 'This is the last permitted model response. Return the final JSON now; do not call tools. Abstain if evidence is insufficient.'}]})
                    broker.output.mkdir(exist_ok=True)
                    write_json(broker.output / f'request_{index:02d}.json', payload)
                    import requests
                    if broker.tls_fingerprint:
                        from benchmark.secure_transport import pinned_session
                        session = pinned_session(broker.tls_fingerprint)
                    else:
                        session = requests.Session()
                    with session, session.post('https://chatgpt.com/backend-api/codex/responses',
                            headers={**broker.headers, 'Content-Type': 'application/json', 'Accept': 'text/event-stream'},
                            json=payload, stream=True, timeout=(30, 180), allow_redirects=False) as upstream:
                        self.send_response(upstream.status_code)
                        self.send_header('Content-Type', upstream.headers.get('Content-Type', 'text/event-stream'))
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        with (broker.output / f'response_{index:02d}.sse').open('wb') as saved:
                            for chunk in upstream.iter_content(4096):
                                if chunk:
                                    saved.write(chunk)
                                    self.wfile.write(chunk)
                                    self.wfile.flush()
                        if upstream.status_code != 200:
                            broker.errors.append({'kind': 'model_http_error', 'status': upstream.status_code, 'request': index})
                    # Extract only visible model identity/usage; keep raw stream too.
                    for line in (broker.output / f'response_{index:02d}.sse').read_text().splitlines():
                        if line.startswith('data: {'):
                            event = json.loads(line[6:])
                            response = event.get('response', {})
                            if response.get('model'):
                                broker.seen_models.add(response['model'])
                            if response.get('usage'):
                                broker.usage.append(response['usage'])
                except Exception as exc:
                    broker.errors.append({'kind': type(exc).__name__})
                    try:
                        self.send_error(502, 'Trusted broker failed')
                    except OSError:
                        pass
                finally:
                    self.close_connection = True

        self.server = UnixHTTP(str(self.path), Handler)
        os.chmod(self.path, 0o600)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def sandbox_command(socket_path, entry, model, binary, hidden_paths=()):
    binary = validate_codex_binary(binary)
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise RuntimeError('bubblewrap (bwrap) is required for solver isolation')
    command = [bwrap, '--unshare-all', '--die-with-parent', '--new-session', '--clearenv',
        '--ro-bind', '/usr', '/usr', '--ro-bind-try', '/lib', '/lib', '--ro-bind-try', '/lib64', '/lib64',
        '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--tmpfs', '/home',
        '--dir', '/home/solver', '--dir', '/work', '--chdir', '/work',
        '--dir', '/opt', '--ro-bind', str(binary), '/opt/codex',
        '--ro-bind', str(socket_path), '/broker.sock', '--ro-bind', str(entry), '/entry.py',
        '--setenv', 'HOME', '/home/solver', '--setenv', 'PATH', '/usr/bin:/bin',
        '--setenv', 'BROKER_DUMMY_KEY', 'local-no-credential',
        '--setenv', 'ISOLATION_HIDDEN_PATHS', json.dumps([str(Path(p).resolve()) for p in hidden_paths]),
        '/usr/bin/python3', '/entry.py', '/opt/codex', 'app-server', '--stdio']
    config = {
        'model_provider': 'local_broker', 'model': model,
        'model_providers.local_broker.name': 'Isolated trusted broker',
        'model_providers.local_broker.base_url': 'http://127.0.0.1:18080/v1',
        'model_providers.local_broker.env_key': 'BROKER_DUMMY_KEY',
        'model_providers.local_broker.wire_api': 'responses',
        'model_providers.local_broker.request_max_retries': 0,
        'model_providers.local_broker.stream_max_retries': 0,
        'features.shell_tool': False, 'features.unified_exec': False, 'features.multi_agent': False,
        'features.tool_suggest': False, 'features.tool_search_always_defer_mcp_tools': False,
        'features.shell_snapshot': False, 'web_search': 'disabled',
        'features.code_mode_host': False, 'features.code_mode': False,
        'features.code_mode_only': False, 'features.sleep_tool': False, 'features.apps': False,
        'model_reasoning_effort': 'medium',
    }
    for key, value in config.items():
        command += ['-c', key + '=' + json.dumps(value)]
    return command


def make_prompt(environment, initial_id):
    template = files('benchmark').joinpath('prompts/minimal_solver_prompt.md').read_text(encoding='utf-8')
    instructions = {
        'split': 'Determine whether candidates A and B are incorrectly separated parts of the same continuous neuron.',
        'merge': 'Determine whether candidate A contains an incorrect connection between structures that should remain separate.',
    }
    values = {'case_id': environment.case_id, 'task': environment.task['task'],
        'instruction': instructions[environment.task['task']], 'anchor_nm_xyz': environment.anchor,
        'fixed_fov_nm': environment.fixed_fov_nm, 'resolution_nm_xyz': environment.resolution.tolist(),
        'initial_observation_id': initial_id, 'max_model_responses': getattr(environment, 'max_model_responses', 20)}
    for key, value in values.items():
        template = template.replace('{{' + key + '}}', str(value))
    return template


def run(args):
    response_limit = getattr(args, 'max_model_responses', 20)
    if not 1 <= response_limit <= 20:
        raise ValueError('Model response limit must be between 1 and 20')
    validate_codex_binary(args.codex_binary)
    renderer = NeuroglancerRenderer(args.episode, chromium=args.chromium, width=1000, height=800, timeout=60)
    environment = MinimalObservationEnvironment(args.episode, args.output, renderer,
        max_actions=response_limit, max_observations=response_limit, max_seconds=args.timeout, fixed_fov_nm=args.fov_nm)
    environment.max_model_responses = response_limit
    initial = environment.call('observe', {})
    if initial.is_error:
        raise RuntimeError('Initial rendering failed')
    prompt = make_prompt(environment, initial.metadata['observation_id'])
    (args.output / 'initial_prompt.md').write_text(prompt)
    write_json(args.output / 'four_tool_schema.json', tool_specs(MINIMAL_TOOLS))
    messages = []
    notifications = []
    final_text = None
    process = None
    report = {'status': 'started', 'max_model_responses': response_limit, 'actual_model': None,
        'isolation': 'bwrap empty filesystem + PID/network namespaces + credential-free Unix inference broker',
        'ground_truth_access': False, 'context_forked': False}
    with tempfile.TemporaryDirectory(prefix='neurscope-codex-') as temp:
        broker = Broker(Path(temp) / 'broker.sock', args.output / 'model_wire', response_limit,
                        tls_fingerprint=getattr(args, 'tls_fingerprint', None), auth_file=args.codex_auth_file)
        broker.start()
        try:
            entry = Path(__file__).resolve().parent / 'isolated_codex_entry.py'
            stderr = (args.output / 'codex_stderr.log').open('w')
            process = subprocess.Popen(sandbox_command(broker.path, entry, args.model, args.codex_binary,
                hidden_paths=[Path(__file__).resolve().parent.parent, Path.home(), args.episode.resolve(),
                              resolve_auth_file(args.codex_auth_file), *args.private_path]), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1, env={})
            incoming = queue.Queue()
            def read_stdout():
                for line in process.stdout:
                    try:
                        incoming.put(json.loads(line))
                    except json.JSONDecodeError:
                        incoming.put({'invalid_stdout': True})
                incoming.put({'process_exited': process.poll()})
            threading.Thread(target=read_stdout, daemon=True).start()
            sequence = 0
            def log(direction, item):
                with (args.output / 'appserver_wire.jsonl').open('a') as f:
                    f.write(json.dumps({'direction': direction, 'time_utc': datetime.now(timezone.utc).isoformat(), 'message': item}, ensure_ascii=False) + '\n')
            def send(item):
                log('to_codex', item)
                process.stdin.write(json.dumps(item, ensure_ascii=False) + '\n')
                process.stdin.flush()
            def request(method, params):
                nonlocal sequence
                sequence += 1
                send({'id': sequence, 'method': method, 'params': params})
                return sequence
            def receive(deadline):
                while time.monotonic() < deadline:
                    try:
                        item = incoming.get(timeout=min(1, max(.01, deadline - time.monotonic())))
                        log('from_codex', item)
                        if 'process_exited' in item:
                            raise RuntimeError('Isolated Codex process exited')
                        return item
                    except queue.Empty:
                        pass
                raise TimeoutError('Run wall time exceeded')
            deadline = time.monotonic() + args.timeout
            init_id = request('initialize', {'clientInfo': {'name': 'proofreading_solver', 'version': '2.0'}, 'capabilities': {'experimentalApi': True}})
            while True:
                item = receive(deadline)
                if item.get('id') == init_id:
                    if 'error' in item:
                        raise RuntimeError('Codex initialization failed')
                    break
            send({'method': 'initialized'})
            thread_id = request('thread/start', {'cwd': '/work', 'model': args.model,
                'approvalPolicy': 'never', 'sandbox': 'read-only', 'ephemeral': True,
                'experimentalRawEvents': True,
                'baseInstructions': 'You are a visual proofreading solver. Use only the four supplied observation functions. No shell, files, web, editing or external information. Follow the supplied task and return final JSON.',
                'developerInstructions': f'At most {response_limit} model responses. Exactly one observation function per non-final response. The final response is JSON. Do not infer labels from provenance.',
                'dynamicTools': [{'type': 'function', 'deferLoading': False, **t} for t in tool_specs(MINIMAL_TOOLS)]})
            while True:
                item = receive(deadline)
                if item.get('id') == thread_id:
                    if 'error' in item:
                        write_json(args.output / 'startup_error.json', item)
                        raise RuntimeError('Isolated thread start failed')
                    started = item['result']
                    if started.get('instructionSources'):
                        raise RuntimeError('Unexpected instruction files loaded')
                    tid = started['thread']['id']
                    write_json(args.output / 'thread_identity.json', started)
                    break
            initial_url = 'data:image/png;base64,' + base64.b64encode(initial.image).decode()
            request('turn/start', {'threadId': tid, 'input': [{'type': 'text', 'text': prompt},
                {'type': 'image', 'url': initial_url}], 'effort': 'medium',
                'outputSchema': FinalAnswer.model_json_schema()})
            while True:
                item = receive(deadline)
                method, params = item.get('method'), item.get('params', {})
                if method == 'item/tool/call':
                    name = params.get('tool')
                    arguments = params.get('arguments', {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    result = environment.call(name, arguments)
                    metadata = {**result.metadata, 'model_responses_used': broker.count,
                        'model_responses_remaining': max(0, response_limit - broker.count)}
                    content = [{'type': 'inputText', 'text': json.dumps(metadata, ensure_ascii=False)}]
                    if result.image:
                        content.append({'type': 'inputImage', 'imageUrl': 'data:image/png;base64,' + base64.b64encode(result.image).decode()})
                    messages.append({'role': 'tool', 'name': name, 'arguments': arguments,
                        'result': metadata, 'image': None if not result.image else f"observations/{metadata['observation_id']}.png"})
                    send({'id': item['id'], 'result': {'contentItems': content, 'success': not result.is_error}})
                    print(f"model={broker.count}/{response_limit} tool={name} status={metadata['status']} observation={metadata.get('observation_id')}", flush=True)
                elif method == 'item/completed':
                    completed = params.get('item', {})
                    if completed.get('type') == 'agentMessage':
                        messages.append({'role': 'assistant', **completed})
                        if completed.get('phase') == 'final_answer' or completed.get('text', '').lstrip().startswith('{'):
                            final_text = completed.get('text')
                    elif completed.get('type') in ('commandExecution', 'fileChange', 'webSearch', 'mcpToolCall'):
                        raise RuntimeError('Unexpected non-observation capability used')
                elif method == 'turn/completed':
                    report['turn_status'] = params.get('turn', {}).get('status')
                    break
                elif 'id' in item and method:
                    # Never approve unrelated capabilities or interactive requests.
                    send({'id': item['id'], 'error': {'code': -32601, 'message': 'Capability is not available in this benchmark'}})
                notifications.append(item)
                if item.get('error'):
                    write_json(args.output / 'protocol_error.json', item)
            if final_text is not None:
                (args.output / 'final_message.txt').write_text(final_text)
                answer = FinalAnswer.model_validate_json(final_text)
                if answer.case_id != environment.case_id:
                    raise ValueError('Final case ID mismatch')
                for evidence in answer.evidence:
                    if evidence.observation_id not in environment.observations:
                        raise ValueError('Final evidence cites an unobserved image')
                    environment._check_position(evidence.position_nm_xyz)
                if answer.decision != 'abstain' and not answer.evidence:
                    raise ValueError('Missing evidence')
                write_json(args.output / 'final_answer.json', answer.model_dump())
                report['status'] = 'completed'
            else:
                report['status'] = 'no_valid_final_answer'
        except Exception as exc:
            report.update(status='infrastructure_failure', failure_type=type(exc).__name__)
            print('Isolated run stopped:', type(exc).__name__, flush=True)
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            environment.close()
            broker.close()
            report.update(model_responses=broker.count, actual_model=sorted(broker.seen_models),
                model_usage=broker.usage, broker_errors=broker.errors,
                actions=environment.actions, observations=len(environment.observations),
                price_or_cost='Not returned by the Codex subscription endpoint; token usage retained when available.')
            write_json(args.output / 'messages.json', [{'role': 'user', 'text': prompt,
                'image': 'observations/' + initial.metadata['observation_id'] + '.png'}] + messages)
            write_json(args.output / 'run_report.json', report)
    print('Report:', args.output / 'run_report.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chromium', required=True)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--fov-nm', type=float, default=4000)
    parser.add_argument('--model', required=True, help='Model available to the locally authenticated Codex account')
    parser.add_argument('--codex-binary', type=Path, required=True, help='Native Linux Codex executable, not the npm JavaScript launcher')
    parser.add_argument('--codex-auth-file', type=Path, help='Trusted parent only; defaults to CODEX_HOME/auth.json or ~/.codex/auth.json')
    parser.add_argument('--private-path', type=Path, action='append', default=[], help='Additional host path that must be invisible to the solver; repeatable')
    parser.add_argument('--max-model-responses', type=int, choices=range(1, 21), default=20,
                        help='Hard limit for this attempt; reduce it by prior attempts when sharing one 20-response budget')
    parser.add_argument('--tls-fingerprint', help='Optional SHA256 endpoint certificate pin verified before Authorization is sent')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
