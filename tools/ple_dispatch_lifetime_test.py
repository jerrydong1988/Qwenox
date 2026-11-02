"""Extract and run the production pump loop with deliberately deferred workers."""
import argparse,os,subprocess,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--cxx',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=False)
 io=(root/'src/gpu/parts/10_ple_io.inc').read_text(encoding='utf-8');model=(root/'src/gpu/parts/40_model.inc').read_text(encoding='utf-8')
 (a.output/'production_readers.inc').write_text(io[io.index('struct PleWin {'):io.index('struct PleWinPrefetch {')],encoding='utf-8')
 start=model.index('          ple_addrs.resize((size_t)P * 16);',model.index('if (ple_uring_ready()) {',model.index('  int prefill_chunk(')))
 end=model.index('        } else {',start);code=model[start:end]
 # A scheduler is permitted to defer every thread until after this block.
 if 'ple_pool.emplace_back([&, w, npump]' not in code:raise AssertionError('batch-local npump must have value lifetime')
 (a.output/'production_dispatch.inc').write_text(code,encoding='utf-8')
 exe=a.output/'dispatch-test.exe';command=[a.cxx,'-O2','-std=c++17','-D_CRT_SECURE_NO_WARNINGS','-I',str(a.output),str(root/'tools/ple_dispatch_lifetime_test.cpp'),'-o',str(exe)]
 built=subprocess.run(command,capture_output=True,text=True,timeout=120);(a.output/'compile.log').write_text(built.stdout+built.stderr,encoding='utf-8');built.check_returncode()
 env={k:v for k,v in os.environ.items() if not k.startswith('GDEC_')};env['GDEC_PLE_PAGE_IO']='0'
 result=subprocess.run([str(exe),str(a.output/'fixture.dat')],env=env,capture_output=True,text=True,timeout=120)
 (a.output/'results.json').write_text(json.dumps({'passed':result.returncode==0,'stdout':result.stdout,'stderr':result.stderr,'command':command},indent=2),encoding='utf-8');print(result.stdout);return result.returncode
if __name__=='__main__':raise SystemExit(main())
