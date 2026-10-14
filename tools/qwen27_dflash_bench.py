"""Serialize GPU A/B measurements and select a measured draft length.

Starts/stops only its own isolated candidate child, never the baseline service.
The child window is hidden on Windows. Baseline must be an MTP-enabled Qwen27.
"""
import argparse
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path
from qwen27_live_test import Client
from qwen27_quality import encode


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--engine',type=Path,required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--tokenizer',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--baseline-port',type=int,default=18870);ap.add_argument('--port',type=int,default=18874)
    ap.add_argument('--gammas',default='3,4,5,7');ap.add_argument('--repeats',type=int,default=2)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cli=Path('build/eng-cli-win.exe')
    cases=[('numeric',[1,2,3,4]),
      ('python',encode(cli,args.tokenizer,'Write a Python function to return the first n Fibonacci numbers.\n```python\ndef fibonacci(n):')),
      ('chinese',encode(cli,args.tokenizer,'<|im_start|>user\n用中文简要解释什么是二分查找。<|im_end|>\n<|im_start|>assistant\n')),
      ('english',encode(cli,args.tokenizer,'<|im_start|>user\nExplain why the sky is blue in three sentences.<|im_end|>\n<|im_start|>assistant\n'))]
    results=[];gold={};baseline=Client(args.baseline_port)
    def measure(client,name,ids,drafter,mode):
        client.gen(999,[100],1)
        ids_out,stats=client.gen(1,ids,96,str(drafter))
        if name in gold:assert ids_out==gold[name],(name,mode,ids_out,gold[name])
        else:gold[name]=ids_out
        item={'case':name,'mode':mode,'prefill_ms':float(stats[5]),'decode_ms':float(stats[6]),
              'tokens_per_second':95*1000/float(stats[6]),'drafted':int(stats[11]),
              'accepted':int(stats[9])-int(stats[8]),'rounds':int(stats[8]),
              'token_sha256':hashlib.sha256(json.dumps(ids_out).encode()).hexdigest()}
        results.append(item);print(json.dumps(item),flush=True)
        (args.output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    for name,ids in cases:
        measure(baseline,name,ids,0,'serial')
        for _ in range(args.repeats):measure(baseline,name,ids,1,'mtp3')
    baseline.close()
    for gamma in map(int,args.gammas.split(',')):
        # Fail before spawning when somebody else owns the candidate port.
        with socket.socket() as probe:probe.bind(('127.0.0.1',args.port))
        with (args.output/f'gamma{gamma}.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([str(args.engine.resolve()),args.model,'--serve','--port',str(args.port),
                 '--maxctx','8192','--qwen27-dflash','--qwen27-dflash-gamma',str(gamma)],stdout=log,stderr=log,
                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                deadline=time.monotonic()+120
                while True:
                    if child.poll() is not None:raise RuntimeError(f'candidate exited: {child.returncode}')
                    try:client=Client(args.port);break
                    except OSError:
                        if time.monotonic()>deadline:raise TimeoutError('candidate did not listen')
                        time.sleep(.25)
                try:
                    for name,ids in cases:
                        for _ in range(args.repeats):measure(client,name,ids,2,f'dflash{gamma}')
                    client.send('MEM');print(client.read(),flush=True)
                finally:client.close()
            finally:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
