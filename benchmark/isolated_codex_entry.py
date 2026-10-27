"""Stdlib-only entry inside an empty network/filesystem namespace.

Only a mounted Unix socket leads to the trusted model broker. No credentials.
"""
import os
import json
from pathlib import Path
import select
import socket
import socketserver
import subprocess
import sys
import threading


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.socket(socket.AF_UNIX) as upstream:
            upstream.connect('/broker.sock')
            sockets = [self.request, upstream]
            while True:
                readable, _, _ = select.select(sockets, [], [], 240)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is self.request else self.request).sendall(data)


def main():
    # Fail closed if host artifacts accidentally become visible.
    hidden = json.loads(os.environ.pop('ISOLATION_HIDDEN_PATHS'))
    if not hidden or not all(isinstance(p, str) and p.startswith('/') for p in hidden):
        raise RuntimeError('Explicit absolute isolation probe paths are required')
    probes = {
        'project_hidden': not Path(hidden[0]).exists(),
        'host_home_hidden': not Path(hidden[1]).exists(),
        'private_paths_hidden': all(not Path(p).exists() for p in hidden[2:]),
        'credentials_absent': not any('TOKEN' in k or k.endswith('_API_KEY') for k in os.environ if k != 'BROKER_DUMMY_KEY'),
    }
    with socket.socket() as test:
        test.settimeout(.2)
        probes['host_network_unreachable'] = test.connect_ex(('1.1.1.1', 443)) != 0
    if not all(probes.values()):
        raise RuntimeError('Isolation probe failed')
    print('ISOLATION_PROBE ' + json.dumps(probes), file=sys.stderr, flush=True)
    Path('/home/solver/.codex').mkdir(parents=True, exist_ok=True)
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 18080), Relay)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        raise SystemExit(subprocess.call(sys.argv[1:]))
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()
