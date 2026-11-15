"""CPU-only contract test for the launcher's read-only telemetry bridge."""
import argparse
import collections
import http.server
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import urllib.request
import urllib.error
from api_disconnect_test import free_port, wait_for

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launcher',required=True);args=parser.parse_args()
    counts=collections.Counter();mode={'value':'good'};checks=0
    class API(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            counts[self.path]+=1
            if self.path=='/health':
                value={'engine_info_ready':True,'endpoints':[],'model':'actual-running-model','slots':2,'queued':1}
            elif self.path=='/metrics/live':
                if mode['value']=='missing':self.send_error(404);return
                if mode['value']=='invalid':value=[]
                else:value={'session':'api-session','sample_ms':1000,'active':[],'output_tokens_total':12,'prefill_estimated_tokens_total':32}
            elif self.path=='/reqstat/summary':value={'requests':7,'decode_tok_per_s':44,'drafters':{}}
            elif self.path=='/reqstat/tail?n=5':value={'records':[]}
            else:self.send_error(404);return
            data=json.dumps(value).encode();self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    api=http.server.ThreadingHTTPServer(('127.0.0.1',0),API)
    thread=threading.Thread(target=api.serve_forever,daemon=True);thread.start()
    port=free_port();ep=free_port();next_port=free_port();base=f'http://127.0.0.1:{port}'
    def request(path,body=None,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        r=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
        try:
            with urllib.request.urlopen(r,timeout=6) as f:return f.status,json.load(f)
        except urllib.error.HTTPError as e:return e.code,json.load(e)
    def check(ok,label):
        nonlocal checks
        assert ok,label;checks+=1
    try:
        with tempfile.TemporaryDirectory(prefix='qwenox-metrics-') as temp:
            root=Path(temp)
            (root/'service.conf').write_text(f'API_PORT="{api.server_port}"\nENGINE_PORT="{ep}"\nAPI_HOST="127.0.0.1"\nENGINE_HOST="127.0.0.1"\nAPI_MODEL="next-start-model"\n')
            env={k:v for k,v in os.environ.items() if not k.startswith(('GDEC_','QWENOX_'))}
            with (root/'web.log').open('w') as log:
                child=subprocess.Popen([str(Path(args.launcher).resolve()),'--root',str(root),'--no-browser','--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                try:
                    def ready():
                        try:return request('/api/status')[1]['state']=='running'
                        except OSError:return False
                    wait_for(ready,'external API observed',timeout=10)
                    state=request('/api/status')[1]
                    check(state['health']['model']=='actual-running-model','actual model identity')
                    check(state['engine']['external'],'read-only external ownership')
                    live=request('/api/metrics')[1]
                    check(live['available'] and live['data']['output_tokens_total']==12,'live counters forwarded unchanged')
                    check(request('/api/metrics',origin='https://untrusted.invalid')[0]==403,'cross-origin metrics refused')
                    check(request('/api/history',origin='https://untrusted.invalid')[0]==403,'cross-origin history refused')
                    one=request('/api/history')[1];two=request('/api/history')[1]
                    check(one==two and one['summary']['requests']==7,'history returned consistently')
                    check(counts['/reqstat/summary']==1 and counts['/reqstat/tail?n=5']==1,'history cached across callers')
                    check(request('/api/config',{'values':{'API_PORT':str(next_port)}})[0]==200,'next-start endpoint changed')
                    check(request('/api/metrics')[1]['available'],'live target stays on running endpoint')
                    check(request('/api/status')[1]['console_url']==f'http://127.0.0.1:{api.server_port}/','console endpoint stays pinned')
                    mode['value']='missing'
                    check(request('/api/metrics')[1]=={'available':False,'reason':'unsupported'},'old API explicitly unsupported')
                    mode['value']='invalid'
                    check(not request('/api/metrics')[1]['available'],'malformed upstream rejected')
                    mode['value']='good'
                    check(request('/api/metrics')[1]['available'],'live polling recovers')
                    check(request('/api/stop',{})[0]==409,'external engine cannot be stopped')
                    for path,kind in [('/launcher.css','text/css'),('/telemetry.js','javascript')]:
                        with urllib.request.urlopen(base+path) as f:
                            check(kind in f.headers['Content-Type'] and len(f.read())>100,'embedded asset '+path)
                    request('/api/shutdown?force=1',{})
                    child.wait(timeout=10);check(child.returncode==0,'web clean exit')
                    with urllib.request.urlopen(f'http://127.0.0.1:{api.server_port}/health') as live_api:
                        check(live_api.status==200,'external API remains running')
                finally:
                    if child.poll() is None:child.terminate();child.wait(timeout=10)
    finally:api.shutdown();api.server_close();thread.join(timeout=3)
    print(f'PASS {checks} telemetry bridge assertions')
if __name__=='__main__':main()
