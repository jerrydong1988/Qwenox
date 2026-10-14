"""Reject malformed new diagnostic/draft options before allocating GPU memory."""
import argparse
import json
import subprocess


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--engine',required=True);ap.add_argument('--model',required=True)
    args=ap.parse_args()
    cases=[['--qwen27-dflash-gamma','0'],['--qwen27-dflash-gamma','8'],
           ['--qwen27-dflash-probe','-1'],['--qwen27-dflash-probe','248320'],
           ['--qwen27-dflash','--qwen27-mtp-probe','5'],
           ['--qwen27-score','--gen','1'],['--qwen27-score','--qwen27-dflash'],
           ['--qwen27-score','--serve'],['--qwen27-score','--tokens','248320'],
           ['--qwen27-score','--tokens','-1']]
    for extra in cases:
        result=subprocess.run([args.engine,args.model,'--tokens','1,2,3,4','--gen','0']+extra,
            capture_output=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert result.returncode==1 and b'loaded layer' not in result.stderr and b'qwen27:' in result.stderr,(extra,result)
    print(json.dumps({'pass':True,'checks':len(cases),'cases':cases}))


if __name__=='__main__':main()
