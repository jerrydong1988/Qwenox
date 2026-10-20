"""Black-box checks for the three selected upstream fixes on a running engine.

Uses only the supplied localhost endpoint. Does not launch or stop services.
Run checkpoint with one slot and RCKPT_MAX=2; parity/cancel with two slots.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import subprocess
import time


class Client:
    def __init__(self, port):
        self.sock = socket.create_connection(('127.0.0.1', port), timeout=10)
        self.sock.settimeout(240)
        self.reader = self.sock.makefile('rb')

    def send(self, line):
        self.sock.sendall((line + '\n').encode('ascii'))

    def read(self):
        raw = self.reader.readline()
        if not raw:
            raise RuntimeError('engine closed connection')
        return raw.decode().split()

    def start(self, req, ids, count=32, draft=1, suffix=''):
        self.send(f'GEN {req} {count} 0 {len(ids)} ' + ' '.join(map(str, ids)) +
                  f' {draft} ' + suffix)

    def finish(self, req):
        tokens, logprobs, progress = [], [], []
        while True:
            row = self.read()
            if row[0] == 'T':
                tokens.append(int(row[2]))
                if len(row) > 3:
                    logprobs.append(float(row[3]))
            elif row[0] == 'P':
                progress.append(row)
            elif row[0] == 'D':
                assert int(row[1]) == req, row
                assert row[2] != 'error', row
                return dict(tokens=tokens, logprobs=logprobs, done=row,
                            progress=progress)

    def gen(self, req, ids, count=32, draft=1, suffix=''):
        started = time.monotonic()
        self.start(req, ids, count, draft, suffix)
        result = self.finish(req)
        result['seconds'] = time.monotonic() - started
        return result

    def close(self):
        self.sock.shutdown(socket.SHUT_RDWR)
        self.reader.close()
        self.sock.close()


def identity(result):
    return result['tokens'], result['logprobs'], result['done'][2]


def run_request(port, req, ids, suffix=''):
    c = Client(port)
    try:
        return c.gen(req, ids, 64, 1, suffix)
    finally:
        c.close()


def parity(args):
    texts = ['Calculate 17 * 23. Answer with the integer only.',
             '\u8bf7\u7528\u4e2d\u6587\u89e3\u91ca\u4e3a\u4ec0\u4e48\u5929\u7a7a\u662f\u84dd\u8272\u3002',
             'Write a Python function that deduplicates a list preserving order.',
             ('Context: alpha beta gamma delta.\n' * 160) + 'Summarize in JSON.']
    commands = [dict(op='encode', text='<|im_start|>user\n' + text +
                     '<|im_end|>\n<|im_start|>assistant\n</think>\n\n') for text in texts]
    p = subprocess.run([args.tokenizer_cli, args.tokenizer],
                       input=''.join(json.dumps(x) + '\n' for x in commands),
                       capture_output=True, text=True, check=True,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    prompts = [json.loads(line)['ids'] for line in p.stdout.splitlines()]
    records = []
    for sampled in (False, True):
        suffix = 'SAMPLE 0.8 20 0.95 0 777 LOGPROBS' if sampled else ''
        serial = [run_request(args.port, 100 + i, ids, suffix)
                  for i, ids in enumerate(prompts)]
        for wave in range(2):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_request, args.port, 200 + i, ids, suffix)
                           for i, ids in enumerate(prompts)]
                concurrent = [f.result() for f in futures]
            for i, got in enumerate(concurrent):
                assert identity(got) == identity(serial[i]), (sampled, wave, i)
        records.append(dict(sampled=sampled, serial=serial))
    if args.reference:
        reference = json.loads(Path(args.reference).read_text())['records']
        for actual, expected in zip(records, reference):
            assert actual['sampled'] == expected['sampled']
            for a, b in zip(actual['serial'], expected['serial']):
                assert identity(a) == identity(b), 'baseline/candidate divergence'
    return dict(checks=24, baseline_compared=bool(args.reference), records=records)


def checkpoint(args):
    c = Client(args.port)
    prefix = [11 + i % 43 for i in range(4500)]
    a = prefix + [71] * 500
    b = prefix + [72] * 500
    target = b[:4300] + [73] * 700
    try:
        c.gen(1, a, 1)
        recompute = c.gen(2, b, 8, suffix='SNAPS 1 4096')
        restored = c.gen(3, target, 32)
        cached = int(restored['done'][10])
        assert int(recompute['done'][10]) == 0, recompute
        if args.expect_fixed:
            assert cached == 4096, restored
            # Evict the two-entry RAM tier with unrelated valid checkpoints,
            # then compare restore against the same fresh split arithmetic.
            for req, token in ((4, 81), (5, 82)):
                c.gen(req, [token] * 5000, 1, suffix='SNAPS 1 4096')
            fresh = c.gen(6, target, 32, suffix='SNAPS 1 4096')
            assert int(fresh['done'][10]) == 0, fresh
            assert identity(restored) == identity(fresh), 'restored/fresh mismatch'
        else:
            assert cached == 0, restored
            fresh = None
        return dict(checks=4 if args.expect_fixed else 2, cached=cached,
                    recompute=recompute, restored=restored, fresh=fresh)
    finally:
        c.close()


def started_prefill(c, req):
    while True:
        row = c.read()
        assert int(row[1]) == req, row
        assert row[0] != 'D', row
        if row[0] == 'P' and int(row[4]) > 0 and row[2] == '0':
            return row


def cancel(args):
    prompt = [17 + i % 37 for i in range(32768)]
    peer_ids = [21 + i % 29 for i in range(512)]
    peer_gold = run_request(args.port, 900, peer_ids)
    c = Client(args.port)
    try:
        c.start(901, prompt, 32, suffix='PROGRESS')
        initial = started_prefill(c, 901)
        began = time.monotonic()
        c.send('X 901')
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(run_request, args.port, 902, peer_ids)
            result = c.finish(901)
            peer = pending.result()
        latency = time.monotonic() - began
        assert result['done'][2] == 'cancel', result
        assert identity(peer) == identity(peer_gold), 'peer corrupted by cancellation'
        max_progress = max([int(initial[4])] + [int(p[4]) for p in result['progress']])
        if args.expect_fixed:
            assert not result['tokens'] and max_progress < len(prompt), result
        # Reuse the same connection after a cancelled prompt; no partial KV may
        # be mistaken for a valid cached prefix.
        recovery = c.gen(903, peer_ids, 64)
        assert identity(recovery) == identity(peer_gold), 'cancelled slot not reset'
    finally:
        c.close()
    gone = Client(args.port)
    gone.start(904, [90 + i % 31 for i in range(32768)], 32, suffix='PROGRESS')
    started_prefill(gone, 904)
    began = time.monotonic()
    gone.close()
    recovered = run_request(args.port, 905, peer_ids)
    assert identity(recovered) == identity(peer_gold), 'disconnect recovery failed'
    if args.expect_fixed and args.engine_log:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if 'req 904 prefill aborted: client gone' in Path(args.engine_log).read_text(errors='replace'):
                break
            time.sleep(.1)
        else:
            raise AssertionError('disconnected prefill did not reach its cancellation cleanup')
        assert identity(run_request(args.port, 906, peer_ids)) == identity(peer_gold)
    return dict(checks=5, cancel_seconds=latency, max_progress=max_progress,
                total=len(prompt), disconnect_peer_seconds=time.monotonic()-began,
                cancellation=result, peer=peer, recovery=recovery)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--stage', choices=('parity', 'checkpoint', 'cancel'), required=True)
    p.add_argument('--expect-fixed', action='store_true')
    p.add_argument('--reference')
    p.add_argument('--engine-log')
    p.add_argument('--tokenizer-cli')
    p.add_argument('--tokenizer')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = globals()[args.stage](args)
    result.update(stage=args.stage, passed=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ('records', 'recompute', 'restored', 'fresh', 'cancellation',
                       'peer', 'recovery')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
