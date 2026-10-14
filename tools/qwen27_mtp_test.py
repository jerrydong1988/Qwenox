"""Compare MTP with serial target execution, including rejected draft rollback."""
import argparse
import json
import subprocess
from pathlib import Path
from qwen27_live_test import Client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=18874)
    ap.add_argument('--baseline-port', type=int, default=18872)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--cli', type=Path, default=Path('build/eng-cli-win.exe'))
    ap.add_argument('--sampled', action='store_true', help='verify MTP target resampling instead of the earlier serial fallback')
    args = ap.parse_args()
    def encode(prompt):
        result = subprocess.run([str(args.cli.resolve()), '--dump-req', '--tokenizer', args.tokenizer,
                                 json.dumps({'prompt': prompt})], check=True, capture_output=True)
        parts = result.stdout.decode().split()
        start = 4+int(parts[3])
        return [int(t) for t in parts[start+1:start+1+int(parts[start])]]
    cases = [
        ('numeric', [1, 2, 3, 4], 64),
        ('python', encode('Write a Python function to return the first n Fibonacci numbers.\n```python\ndef fibonacci(n):'), 96),
        ('chinese', encode('<|im_start|>user\n用中文简要解释什么是二分查找。<|im_end|>\n<|im_start|>assistant\n'), 96),
        ('english', encode('<|im_start|>user\nExplain why the sky is blue in three sentences.<|im_end|>\n<|im_start|>assistant\n'), 96),
        ('one-token', [12, 14, 16], 1),
        ('two-tokens', [1, 2, 3, 4], 2),
        ('long', [1, 2, 3, 4, 5, 6, 7, 8]*257, 32),
    ]
    spec, base = Client(args.port), Client(args.baseline_port)
    results = []
    total_drafts = total_accepts = 0
    for i, (name, ids, count) in enumerate(cases):
        spec.gen(10000, [101], 1)
        base.gen(10000, [101], 1)
        a, sa = spec.gen(100+i, ids, count, '1')
        b, sb = base.gen(100+i, ids, count)
        record = {'case': name, 'pass': a == b, 'tokens': a, 'mtp': sa, 'serial': sb}
        print(json.dumps(record, ensure_ascii=False), flush=True)
        assert a == b, record
        total_drafts += int(sa[11])
        total_accepts += int(sa[9])-int(sa[8])
        # A speculative request may have consumed its last emitted token.
        continued, cached = spec.gen(500+i, ids+a, 8, '1')
        base.gen(10000, [102], 1)
        fresh, fresh_stats = base.gen(500+i, ids+a, 8)
        assert continued == fresh, (name, continued, fresh, cached, fresh_stats)
        results.append(record)
    # EOS can be the first token, an accepted draft, or the target correction.
    for eos in [5, 6, 24218]:
        spec.gen(10000, [103], 1)
        base.gen(10000, [103], 1)
        a, sa = spec.gen(eos, [1, 2, 3, 4], 16, '1', [eos])
        b, sb = base.gen(eos, [1, 2, 3, 4], 16, '', [eos])
        assert a == b and sa[2] == sb[2] == 'done', (eos, a, b, sa, sb)
    # Fixed-seed target sampling, with penalties and log probabilities.
    sampling = 'SAMPLE 0.7 20 0.9 0 12345 LOGPROBS PENALTY 0.2 0.1'
    spec.gen(10000, [104], 1)
    base.gen(10000, [104], 1)
    a, sa = spec.gen(700, [1, 2, 3, 4], 16, '1 '+sampling)
    b, sb = base.gen(700, [1, 2, 3, 4], 16, sampling)
    assert a == b and sa[7] == ('1' if args.sampled else '0'), (a, b, sa, sb)
    if args.sampled:
        for seed in [1, 7, 12345]:
            for name, ids, _ in cases[1:3]:
                suffix = f'SAMPLE 0.8 40 0.95 0.02 {seed} LOGPROBS PENALTY 0.1 0.1 BIAS 1 198 -0.25'
                spec.gen(10000, [105], 1);base.gen(10000, [105], 1)
                a, sa = spec.gen(800, ids, 48, '1 '+suffix)
                al = spec.last_logprobs
                b, sb = base.gen(800, ids, 48, suffix)
                delta = max(abs(x-y) for x, y in zip(al, base.last_logprobs))
                record = {'case': name, 'seed': seed, 'pass': a == b, 'logprob_max_abs': delta, 'mtp': sa, 'serial': sb}
                print(json.dumps(record), flush=True)
                assert a == b, (record, a, b)
                assert len(al) == len(a) and len(base.last_logprobs) == len(b) and delta < 1e-4, record
    spec.gen(10000, [106], 1)
    spec.send('GEN 900 128 0 4 1 2 3 4 1')
    while not spec.read().startswith('T 900 '):
        pass
    spec.send('X 900')
    while True:
        line = spec.read()
        if line.startswith('D 900 '):
            assert line.split()[2] == 'cancel', line
            break
    a, sa = spec.gen(901, [1, 2, 3, 4], 8, '1')
    assert a == [5, 6, 24218, 10, 4838, 1665, 25, 33898], (a, sa)
    assert 0 < total_accepts < total_drafts, (total_drafts, total_accepts)
    print(json.dumps({'pass': True, 'cases': len(results), 'drafted': total_drafts,
                      'accepted': total_accepts, 'acceptance': total_accepts/total_drafts}), flush=True)


if __name__ == '__main__':
    main()
