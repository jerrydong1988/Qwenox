"""Old transcript cache must be ignored; fresh v2 must reload and reuse safely."""
import argparse,json,os,socket,struct,subprocess,sys,time
from pathlib import Path
from api_disconnect_test import free_port,wait_for
from api_regression_test import get,request,check

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--api',required=True);parser.add_argument('--tokenizer',required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);cache=root/'legacy-cache.bin'
    tok=json.loads((Path(a.tokenizer)/'tokenizer.json').read_text(encoding='utf-8'))
    vocab=max([*tok['model']['vocab'].values(),*(x['id'] for x in tok['added_tokens'])])+1
    ids=[248045,12675,248046]
    cache.write_bytes(struct.pack('<8sIIQ',b'GDTC1\0\0\0',1,vocab,1)+struct.pack('<Q',len(ids))+struct.pack('<'+'i'*len(ids),*ids))
    env={k:v for k,v in os.environ.items() if not k.startswith(('GDEC_','QWENOX_'))}
    env.update(GDEC_API_TOKCACHE_FILE=str(cache),GDEC_API_TOKCACHE_SAVE_S='1',GDEC_REQSTAT='0',GDEC_API_ADMIN_KEY='',PYTHONDONTWRITEBYTECODE='1')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0;rows=[];counts=[]
    for turn in [1,2]:
        ep,ap=free_port(),free_port()
        while ep==ap:ap=free_port()
        children=[];logpath=root/f'process-{turn}.log'
        with logpath.open('w',encoding='utf-8') as log:
            try:
                children.append(subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('api_fake_engine.py')),'--port',str(ep)],cwd=root,env=env,stdout=log,stderr=log,creationflags=flags))
                def ready():
                    try:
                        with socket.create_connection(('127.0.0.1',ep),timeout=.1):return True
                    except OSError:return False
                wait_for(ready,'fake engine')
                children.append(subprocess.Popen([str(Path(a.api).resolve()),'--tokenizer',str(Path(a.tokenizer).resolve()),'--engine',f'127.0.0.1:{ep}','--host','127.0.0.1','--port',str(ap),'--overrides',''],cwd=root,env=env,stdout=log,stderr=log,creationflags=flags))
                base=f'http://127.0.0.1:{ap}'
                def api_ready():
                    try:return get(base,'/health')[0]==200
                    except OSError:return False
                wait_for(api_ready,'API',timeout=15)
                p={'model':'test','messages':[{'role':'user','content':'hi'}],'temperature':0,'enable_thinking':False,'max_tokens':16}
                status,_,raw=request(base,'/v1/chat/completions',p);j=json.loads(raw);rows.append({'turn':turn,'request':p,'status':status,'response':j})
                check(status==200 and j['choices'][0]['message']['content']=='Hi','healthy generation '+str(turn),j)
                counts.append(j['usage']['prompt_tokens'])
                wait_for(lambda:len(cache.read_bytes())>=12 and struct.unpack_from('<I',cache.read_bytes(),8)[0]==2,'v2 cache persisted',timeout=5)
            finally:
                for child in reversed(children):
                    if child.poll() is None:child.terminate()
                    child.wait(timeout=10)
        text=logpath.read_text(encoding='utf-8',errors='replace')
        if turn==1:check('truncated/mismatched' in text and 'loaded 1 entries' not in text,'legacy v1 is ignored',text[-1000:])
        else:check('loaded 1 entries' in text and 'tcache: reused' in text,'v2 reload and real prefix reuse',text[-1200:])
    check(counts[0]==counts[1],'cached and fresh prompt lengths agree',counts)
    (root/'responses.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('RESULT PASS')
if __name__=='__main__':main()
