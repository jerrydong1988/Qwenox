#!/usr/bin/env python3
"""Model-free protocol fixtures for the live cancellation test's pass/fail rules.

Uses only an ephemeral loopback port. This is test-client evidence, not real
inference, GPU scheduling, cancellation correctness, or performance evidence.
"""
import argparse
import json
from pathlib import Path
import socketserver
import subprocess
import sys
import threading

BASE = 910000


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(socketserver.StreamRequestHandler):
    def setup(self):
        super().setup()
        self.lock = threading.Lock()

    def send(self, text):
        with self.lock:
            self.wfile.write((text + '\n').encode())
            self.wfile.flush()

    def progress(self, req):
        if self.server.observer == 'progress':
            self.send(f'P {req} 0 1 1 6 0 1.0')

    def complete(self, req, drafter=0, cancelled=False):
        if cancelled:
            self.send(f'D {req} cancel 0 0 0.0 0.0')
        else:
            for token in [18, 24, 16]:
                self.send(f'T {req} {token}')
            self.send(f'D {req} length 6 3 1.0 1.0 {drafter} 1 2 0 {2 if drafter else 0}')

    def handle(self):
        while True:
            try:
                line = self.rfile.readline()
            except ConnectionResetError:
                return
            if not line:
                return
            parts = line.decode().split()
            if parts[0] == 'GEN':
                req = int(parts[1])
                drafter = int(parts[-2] if parts[-1] == 'PROGRESS' else parts[-1])
                if req == BASE + 1:
                    self.server.a = self
                    self.server.stage = 1
                    self.progress(req)
                elif req == BASE + 2:
                    self.server.stage = 2
                    if self.server.mode == 'missing-handoff':
                        self.server.stage = 0
                        self.server.a.complete(BASE + 1)
                        self.complete(req)
                    else:
                        self.progress(req)
                        timer = threading.Timer(.1, self.server.a.progress, [BASE + 1])
                        timer.daemon = True
                        timer.start()
                else:
                    self.complete(req, drafter)
            elif parts[0] == 'SNAP':
                stage = self.server.stage
                req = BASE + (2 if stage == 2 else 1)
                self.send(f'P 0 {1 if stage else 0} {req} 0 1 6')
                if stage == 2:
                    self.server.stage = 3
            elif parts[0] == 'X':
                self.complete(int(parts[1]), cancelled=True)
                self.server.a.complete(BASE + 1, cancelled=self.server.mode == 'wrong-peer')
            else:
                raise ValueError(parts[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = args.output / 'fixtures.json'
    fixtures.write_text(json.dumps(dict(a=[1, 2, 3, 4, 5, 6], b=[7, 8, 9, 10, 11, 12],
                                       recovery=[1, 2], stop_ids=[])))
    rows = []
    for observer in ['progress', 'snap']:
        for mode in ['correct', 'wrong-peer', 'missing-handoff']:
            with Server(('127.0.0.1', 0), Handler) as server:
                server.observer, server.mode, server.stage = observer, mode, 0
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                output = args.output / (observer + '-' + mode)
                try:
                    proc = subprocess.run([sys.executable, '-B',
                                           str(Path(__file__).with_name('prefill_cancel_live_test.py')),
                                           '--endpoint', f'127.0.0.1:{server.server_address[1]}',
                                           '--fixtures', str(fixtures), '--observer', observer,
                                           '--output', str(output)], capture_output=True, text=True, timeout=20)
                finally:
                    server.shutdown()
                    worker.join()
                (output / 'client.log').write_text(proc.stdout + proc.stderr)
                result = json.loads((output / 'results.json').read_text())
                expected = mode == 'correct'
                passed = result['passed'] == expected and (proc.returncode == 0) == expected
                if mode == 'wrong-peer':
                    passed &= any(item['case'] == 'uncancelled-peer-finish' and not item['passed']
                                  for item in result['checks'])
                if mode == 'missing-handoff':
                    passed &= any('required overlapping handoff' in item.get('error', '')
                                  for item in result['records'])
                row = dict(observer=observer, mode=mode, passed=bool(passed),
                           client_passed=result['passed'], exit_code=proc.returncode)
                rows.append(row)
                print(('PASS' if passed else 'FAIL') + f': {observer}/{mode}', flush=True)
    result = dict(passed=all(row['passed'] for row in rows), cases=rows,
                  evidence_scope='model-free test-client protocol fixtures')
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
