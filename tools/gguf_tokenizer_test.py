"""Compare embedded GGUF tokenization to an existing tokenizer directory."""
import argparse,json,random,struct,subprocess,tempfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--tok-cli',required=True);p.add_argument('--gguf',required=True);p.add_argument('--reference',required=True);a=p.parse_args()
 rng=random.Random(7351);pieces=['中文','hello','مرحبا','😀','\n','\t',' C:/模型/文件.gguf','<|im_start|>','<|im_end|>','<tool_call>','</tool_call>','"path":','\\','é','１２３','def f(x): return x + 1']
 texts=['','你好，世界！','<|im_start|>user\n测试<|im_end|>\n']+[''.join(rng.choices(pieces,k=rng.randrange(1,25))) for _ in range(128)]
 queries=[{'op':'encode','text':t,'offsets':True} for t in texts]+[{'op':'specials'}]
 def run(source,queries):
  data=''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in queries).encode()
  r=subprocess.run([a.tok_cli,str(source)],input=data,capture_output=True,timeout=40)
  assert r.returncode==0,r.stderr.decode(errors='replace')
  return [json.loads(line) for line in r.stdout.splitlines()]
 actual=run(a.gguf,queries);ref=run(a.reference,queries);assert actual==ref,'token ids/offsets/specials differ'
 dec=[{'op':'decode','ids':x['ids'],'skip_special_tokens':False} for x in actual[:-1]]
 assert run(a.gguf,dec)==run(a.reference,dec),'decode differs'
 with tempfile.TemporaryDirectory(prefix='gguf-bad-meta-') as d:
  f=Path(d)/'bad.gguf';key=b'tokenizer.ggml.tokens'
  cases=[b'BAD!',b'GGUF'+struct.pack('<IQQ',3,0,1)+struct.pack('<Q',len(key))+key+struct.pack('<IIQ',9,8,2**64-1)]
  for data in cases:
   f.write_bytes(data);r=subprocess.run([a.tok_cli,str(f)],input=b'',capture_output=True,timeout=5)
   assert r.returncode==1 and b'load failed' in r.stderr,r
 print(f'PASS {len(texts)} encode/offset and {len(texts)} decode cases, special ids, 2 malformed metadata cases')
if __name__=='__main__':main()
