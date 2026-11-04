#!/usr/bin/env python3
"""Exercise actual Windows build functions with a fake compiler, without HIP.

Checks path quoting, default/empty overrides, successful atomic replacement,
and preservation of both existing outputs when a compiler fails. Does not
source runtime staging, download dependencies, or touch an installed service.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bash', default='bash')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'build_win.sh'
    text = source.read_text(encoding='utf-8')
    functions = '\n'.join(re.search(r'^' + name + r'\(\) \{\n.*?^\}', text,
                                   re.M | re.S)[0]
                          for name in ['build_engine', 'build_api'])
    temporary = tempfile.TemporaryDirectory(prefix='gdec build output ')
    work = Path(temporary.name)
    try:
        sdk = work / 'fake sdk'
        compiler = sdk / 'lib/llvm/bin/clang++.exe'
        compiler.parent.mkdir(parents=True)
        driver = work / 'compiler.py'
        driver.write_text('''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['COMPILE_TRACE'],'a',encoding='utf-8') as f:f.write(json.dumps(args)+'\\n')
out=Path(args[args.index('-o')+1]);out.write_bytes(b'partial' if os.environ['COMPILE_FAIL']=='1' else b'compiled')
raise SystemExit(37 if os.environ['COMPILE_FAIL']=='1' else 0)
''', encoding='utf-8')
        compiler.write_text('#!/usr/bin/env bash\nexec python "$FAKE_DRIVER" "$@"\n',
                            encoding='utf-8', newline='\n')
        compiler.chmod(0o755)
        harness = work / 'build-functions.sh'
        harness.write_text('''set -euo pipefail
HIPCC="$TR/lib/llvm/bin/clang++.exe"
FLAGS=(-O3 -std=c++17 --offload-arch=gfx1151)
build_webp() { WEBP_FLAGS=(-Iwebp-stub); }
''' + functions + '\n"build_$1"\n', encoding='utf-8', newline='\n')
        rows = []
        for target, variable, default in [('engine', 'QWENOX_ENGINE_OUT', 'qwenox-engine-win.exe'),
                                           ('api', 'QWENOX_API_OUT', 'qwenox-win.exe')]:
            for kind in ['default', 'empty', 'relative-space', 'absolute-space', 'legacy', 'precedence']:
                for fail in [False, True]:
                    case = work / f'{target}-{kind}-{int(fail)}'
                    (case / 'build').mkdir(parents=True)
                    defaults = [case / 'build/qwenox-engine-win.exe', case / 'build/qwenox-win.exe']
                    for path in defaults:
                        path.write_bytes(b'existing-default')
                    candidate = case / 'build/candidate with spaces.exe'
                    candidate.write_bytes(b'existing-candidate')
                    env = {k: v for k, v in os.environ.items()
                           if not k.startswith(('GDEC_', 'QWENOX_'))}
                    env.update(TR=sdk.as_posix(), FAKE_DRIVER=driver.as_posix(),
                               COMPILE_TRACE=str(case / 'args.jsonl'), COMPILE_FAIL=str(int(fail)))
                    legacy = variable.replace('QWENOX_', 'GDEC_')
                    if kind == 'legacy':
                        env[legacy] = candidate.as_posix()
                    elif kind == 'precedence':
                        env[variable] = candidate.as_posix()
                        env[legacy] = (case / 'build' / default).as_posix()
                    elif kind != 'default':
                        env[variable] = {'empty': '', 'relative-space': 'build/candidate with spaces.exe',
                                         'absolute-space': candidate.as_posix()}[kind]
                    expected = case / 'build' / default if kind in ['default', 'empty'] else candidate
                    before = {p: p.read_bytes() for p in [*defaults, candidate]}
                    proc = subprocess.run([args.bash, harness.as_posix(), target], cwd=case,
                                          env=env, capture_output=True, text=True, timeout=30)
                    argv = json.loads((case / 'args.jsonl').read_text().splitlines()[0])
                    actual_output = Path(argv[argv.index('-o') + 1])
                    if not actual_output.is_absolute():
                        actual_output = case / actual_output
                    passed = (proc.returncode != 0 if fail else proc.returncode == 0)
                    passed &= actual_output.resolve() == Path(str(expected) + '.tmp').resolve()
                    for path, content in before.items():
                        passed &= path.read_bytes() == (b'compiled' if path == expected and not fail else content)
                    if target == 'engine':
                        passed &= argv[:3] == ['-O3', '-std=c++17', '--offload-arch=gfx1151']
                    row = dict(target=target, override=kind, compiler_failed=fail,
                               passed=bool(passed), exit_code=proc.returncode, argv=argv,
                               stdout=proc.stdout, stderr=proc.stderr)
                    rows.append(row)
                    print(('PASS' if passed else 'FAIL') + f': {target}/{kind}/failure={fail}')
        result = dict(passed=all(row['passed'] for row in rows), checks=len(rows), rows=rows)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        return 0 if result['passed'] else 1
    finally:
        temporary.cleanup()


if __name__ == '__main__':
    raise SystemExit(main())
