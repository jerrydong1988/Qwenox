"""CPU integration for model-profile isolation and web configuration safety.
Usage: python tools/launcher_web_test.py --launcher PATH/launcher_settings.exe
Starts only the configuration service, using a temporary root and loopback port.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import socketserver
import threading
import time
import tempfile
import urllib.request
import urllib.error
from api_disconnect_test import free_port, wait_for

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launcher',required=True);args=parser.parse_args()
    exe=str(Path(args.launcher).resolve());port=free_port();ep=free_port();ap=free_port()
    assert len({port,ep,ap})==3
    base=f'http://127.0.0.1:{port}';count=0
    def request(path,body=None,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        req=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=5) as res:return res.status,json.load(res)
        except urllib.error.HTTPError as err:return err.code,json.load(err)
    def check(ok,name):
        nonlocal count
        assert ok,name;count+=1
    with tempfile.TemporaryDirectory(prefix='qwenox-web-') as temporary:
        root=Path(temporary)
        flash={'weights_ver':'v2','API_MODEL':'flash-test','MODEL_FILE':'C:/test/main.hgn','NGRAM_FILE':'C:/test/ngram.hgn',
               'MAX_CONTEXT':'8192','PARALLEL':'2','KV_PAGED':'1','KV_POOL_TOKENS':'16384',
               'GDEC_GR_COMBINE_SIG':'1','GDEC_AXPY_SG_OUT':'1','GDEC_LT_BUCKET':'1','GDEC_V2_HT_DEQ_EXACT':'1',
               'GDEC_SPEC_PRECISION':'aligned','GDEC_MTP_BRANCH_STRIDED':'1','V2_DENSE_BACKEND':'upstream'}
        q27={'weights_ver':'v1','API_MODEL':'q27-test','MODEL_FILE':'C:/test/q27.hgn','MAX_CONTEXT':'8192',
             'PARALLEL':'1','KV_PAGED':'0','KV_POOL_TOKENS':'0','GDEC_QWEN27_MTP':'1',
             'GDEC_QWEN27_MTP_GAMMA':'3','GDEC_QWEN27_PREFILL_MODE':'compensated'}
        gguf={'weights_ver':'gguf','API_MODEL':'gguf-test','GGUF_FILE':'C:/test/main.gguf','GGUF_MTP_FILE':'',
              'MAX_CONTEXT':'8192','PARALLEL':'2','KV_PAGED':'1','KV_POOL_TOKENS':'16384','QWENOX_GG_WMMA':'1'}
        conf={'ENGINE_HOST':'127.0.0.1','API_HOST':'127.0.0.1','ENGINE_PORT':str(ep),'API_PORT':str(ap),
              **{k:v for k,v in flash.items() if k!='weights_ver'}}
        (root/'service.conf').write_text('\n'.join(k+'="'+v+'"' for k,v in conf.items())+'\n# start_win: weights=v2\n')
        env={k:v for k,v in os.environ.items() if not k.startswith(('GDEC_','QWENOX_'))}
        with (root/'test.log').open('w+') as log:
            child=subprocess.Popen([exe,'--root',str(root),'--no-browser','--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            try:
                def ready():
                    try:return request('/api/config')[0]==200
                    except OSError:return False
                wait_for(ready,'web launcher',timeout=10)
                for name,data in [('Flash',flash),('27B',q27),('GGUF',gguf),('Flash',flash)]:
                    values={k:v for k,v in data.items() if k!='weights_ver'}
                    code,_=request('/api/config',{'values':values,'weights_ver':data['weights_ver'],'replace_profile':True})
                    check(code==200,'switch '+name)
                    c=request('/api/config')[1];v=c['values']
                    check(v['API_MODEL']==data['API_MODEL'],'API model '+name)
                    check(all(v[k]==x for k,x in values.items()),'all policy fields '+name)
                    if name=='27B':check(v.get('GDEC_SPEC_PRECISION','')=='' and v.get('V2_DENSE_BACKEND','')=='','stale Flash flags cleared')
                    if name=='GGUF':check(v.get('GDEC_QWEN27_MTP','')=='','stale 27B flag cleared')
                    code,j=request('/api/profiles',{'name':name+' 中文','data':data});check(code==200 and j['profiles'][name+' 中文']==data,'profile roundtrip '+name)
                    code,j=request('/api/profiles/delete',{'name':name+' 中文'});check(code==200 and name+' 中文' not in j['profiles'],'profile deletion '+name)
                for key,value in [('GDEC_QWEN27_MTP_GAMMA','99'),('GDEC_QWEN27_PREFILL_MODE','bad'),('GDEC_GR_COMBINE_SIG','bad'),('API_PORT',str(ep)),('API_MODEL','bad\nvalue'),('MODEL_FILE','$(bad)')]:
                    before=(root/'service.conf').read_bytes()
                    code,_=request('/api/config',{'values':{key:value}})
                    check(code==400,'reject invalid '+key);check((root/'service.conf').read_bytes()==before,'preserve invalid '+key)
                for path in ['/api/config','/api/start','/api/stop','/api/profiles/delete']:
                    code,_=request(path,{},origin='https://example.invalid');check(code==403,'reject cross-origin '+path)
                # Missing native supervisor produces an observable failure, not an orphan.
                code,_=request('/api/start',{});check(code==200,'start accepted asynchronously')
                wait_for(lambda:request('/api/status')[1]['state']=='failed','missing supervisor reported')
                check(not request('/api/status')[1]['api']['up'],'no API after startup failure')
                # Cancel while the native-start worker is still probing a local
                # occupied API port. A late failure must not overwrite idle.
                entered=threading.Event(); probe_ok=False
                class SlowProbe(socketserver.StreamRequestHandler):
                    def handle(self):
                        self.rfile.readline();entered.set();time.sleep(.7)
                        try:
                            body=b'{"engine_info_ready":true,"endpoints":[]}' if probe_ok else b''
                            status=b'200 OK' if probe_ok else b'503 Busy'
                            self.wfile.write(b"HTTP/1.1 "+status+b"\r\nContent-Length: "+str(len(body)).encode()+b"\r\nConnection: close\r\n\r\n"+body)
                        except OSError:pass
                server=socketserver.ThreadingTCPServer(('127.0.0.1',ap),SlowProbe)
                server.daemon_threads=True
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                try:
                    (root/'start_win.exe').write_bytes(b'not an executable; never reached')
                    for probe_ok in [False, True]:
                        entered.clear()
                        request('/api/start',{})
                        check(entered.wait(3),'start entered preflight probe')
                        code,_=request('/api/stop',{});check(code==200,'cancel pending start')
                        wait_for(lambda:request('/api/status')[1]['state']=='idle','pending start cancelled')
                        time.sleep(1)
                        state=request('/api/status')[1]
                        check(state['state']=='idle','late worker cannot overwrite cancelled state')
                        check(not state['api']['up'],'late successful probe cannot revive cancelled service')
                finally:server.shutdown();server.server_close();thread.join(timeout=3)
                code,_=request('/api/shutdown',{'force':True});check(code==200,'shutdown')
                child.wait(timeout=10);check(child.returncode==0,'clean web exit')
            except BaseException:
                log.flush();log.seek(0);print(log.read()[-5000:]);raise
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=10)
    print(f'PASS {count} launcher assertions')
if __name__=='__main__':main()
