"""Deterministic Messages wire, strict-tool, stop, count and disconnect regression."""
import argparse,json,os,socket,subprocess,sys,time
from pathlib import Path
from api_disconnect_test import free_port,make_tokenizer,wait_for
from api_regression_test import get,request

def events(raw):
    rows=[]
    for block in raw.decode('utf-8').replace('\r\n','\n').split('\n\n'):
        fields={}
        for line in block.splitlines():
            if ': ' in line:
                k,v=line.split(': ',1);fields[k]=v
        if 'data' in fields:rows.append((fields.get('event'),json.loads(fields['data'])))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--api',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);make_tokenizer(root/'tokenizer')
    ep,port=free_port(),free_port()
    while ep==port:port=free_port()
    base=f'http://127.0.0.1:{port}';children=[];results=[];raw_rows=[]
    env={k:v for k,v in os.environ.items() if not k.startswith(('GDEC_','QWENOX_'))}
    env.update(GDEC_API_TOKCACHE_FILE='',GDEC_API_TOKCACHE='0',GDEC_REQSTAT='0',GDEC_API_ADMIN_KEY='',ROPE_FACTOR='1',PYTHONDONTWRITEBYTECODE='1')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    def check(name,value,detail=None):
        results.append({'name':name,'passed':bool(value),'detail':detail});print(('PASS ' if value else 'FAIL ')+name,flush=True)
    def call(path,body):
        status,ctype,raw=request(base,path,body);raw_rows.append({'path':path,'request':body,'status':status,'content_type':ctype,'body':raw.decode('utf-8')})
        return status,events(raw) if 'event-stream' in ctype else json.loads(raw)
    def prompt(**kw):return {'model':'test','max_tokens':512,'thinking':{'type':'disabled'},'messages':[{'role':'user','content':'hello'}],**kw}
    tools=[{'name':'read_file','input_schema':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}]
    def wire(command):
        with socket.create_connection(('127.0.0.1',ep),timeout=2) as c:
            c.sendall((command+'\n').encode('ascii'))
            with c.makefile('rb') as f:return json.loads(f.readline())
    def trace():return wire('TESTSTATE')
    try:
        with (root/'process.log').open('w',encoding='utf-8') as log:
            try:
                children.append(subprocess.Popen([sys.executable,str(Path(__file__).with_name('api_fake_engine.py')),'--port',str(ep)],cwd=root,env=env,stdout=log,stderr=log,creationflags=flags))
                def ready():
                    try:
                        with socket.create_connection(('127.0.0.1',ep),timeout=.1):return True
                    except OSError:return False
                wait_for(ready,'fake engine')
                children.append(subprocess.Popen([str(Path(a.api).resolve()),'--tokenizer',str(root/'tokenizer'),'--engine',f'127.0.0.1:{ep}','--host','127.0.0.1','--port',str(port),'--overrides','','--model','cpu-messages'],cwd=root,env=env,stdout=log,stderr=log,creationflags=flags))
                def api_ready():
                    try:return get(base,'/health')[0]==200
                    except OSError:return False
                wait_for(api_ready,'API',timeout=10)
                wire('IDSLOG');body=prompt();s,count=call('/v1/messages/count_tokens',body)
                check('count succeeds without GEN',s==200 and count['input_tokens']>0 and wire('IDSLOG')==[],count)
                s,text=call('/v1/messages',body);check('text content and stop',s==200 and text['content']==[{'type':'text','text':'Hi'}] and text['stop_reason']=='end_turn',text)
                check('count matches generation total prompt',count['input_tokens']==text['usage']['input_tokens']+text['usage']['cache_read_input_tokens'],text['usage'])
                s,stream=call('/v1/messages',prompt(stream=True));types=[x[0] for x in stream]
                check('named SSE lifecycle',s==200 and types[0]=='message_start' and types[-1]=='message_stop' and 'content_block_delta' in types,types)
                check('SSE event and payload type agree',all(t==d['type'] for t,d in stream))
                check('SSE text', ''.join(d['delta'].get('text','') for t,d in stream if t=='content_block_delta')=='Hi')
                for streaming in [False,True]:
                    p=prompt(temperature=1,seed=717100,tools=tools,stream=streaming)
                    s,value=call('/v1/messages',p)
                    if streaming:
                        blocks=[d['content_block'] for t,d in value if t=='content_block_start'];args=''.join(d['delta'].get('partial_json','') for t,d in value if t=='content_block_delta')
                        reason=[d['delta']['stop_reason'] for t,d in value if t=='message_delta'][-1]
                        valid=any(b.get('type')=='tool_use' and b['name']=='read_file' for b in blocks) and json.loads(args)=={'path':'a'} and reason=='tool_use'
                    else:valid=value['stop_reason']=='tool_use' and value['content'][0]['name']=='read_file' and value['content'][0]['input']=={'path':'a'}
                    check('tool use '+str(streaming),s==200 and valid,value)
                    p['seed']=717101;s,error=call('/v1/messages',p)
                    if streaming and isinstance(error,list):
                        errs=[d for t,d in error if t=='error'];valid=bool(errs) and errs[-1]['error']['type']=='api_error' and not any(t=='message_stop' for t,d in error)
                    else:valid=s==502 and error['type']=='error' and error['error']['type']=='api_error'
                    check('malformed tool is error, not token exhaustion '+str(streaming),valid,error)
                    p['seed']=717106;s,error=call('/v1/messages',p)
                    if streaming:
                        valid=s==200 and any(t=='error' and d['error']['type']=='api_error' for t,d in error) and not any(t=='message_stop' for t,d in error)
                    else:valid=s==502 and error['type']=='error'
                    check('error after a completed tool '+str(streaming),valid,error)
                    p['seed']=717104;s,value=call('/v1/messages',p)
                    reason=[d['delta']['stop_reason'] for t,d in value if t=='message_delta'][-1] if streaming else value['stop_reason']
                    check('actual truncated tool reports max_tokens '+str(streaming),s==200 and reason=='max_tokens',value)
                for streaming in [False,True]:
                    s,value=call('/v1/messages',prompt(stop_sequences=['i'],stream=streaming))
                    delta=[d['delta'] for t,d in value if t=='message_delta'][-1] if streaming else value
                    check('selected stop sequence '+str(streaming),s==200 and delta['stop_reason']=='stop_sequence' and delta['stop_sequence']=='i',value)
                for body in [{},{'messages':[]},prompt(messages='bad'),prompt(messages=[{'role':'user','content':[{'type':'bad'}]}]),prompt(tool_call_format='json')]:
                    s,value=call('/v1/messages',body);check('invalid request envelope '+str(len(results)),s==400 and value.get('type')=='error' and value['error']['type']=='invalid_request_error',value)
                # Preserve local strict template policy and do not execute unsupported extensions.
                s,value=call('/v1/messages',prompt(auto_disable_thinking_with_tools=True));check('unsupported extension explicitly rejected',s==400,value)
                s,value=call('/v1/messages',prompt(messages=[{'role':'user','content':'q'},{'role':'assistant','content':'prefix'}]));check('assistant prefill',s==200 and value['content'][0]['text']=='Hi',value)
                # Active Messages disconnect must reach the existing cancellation path.
                import http.client
                c=http.client.HTTPConnection('127.0.0.1',port,timeout=10);p=prompt(temperature=1,seed=616161,stream=True)
                c.request('POST','/v1/messages',json.dumps(p),{'Content-Type':'application/json'});resp=c.getresponse();check('disconnect probe starts SSE',resp.status==200)
                resp.close();c.close()
                def cancelled():
                    rows=trace();return bool(rows) and rows[-1]['cancel']>=1 and rows[-1]['done']
                wait_for(cancelled,'Messages cancellation',timeout=8)
                check('Messages disconnect cancels underlying request',cancelled(),trace())
                wait_for(lambda:json.loads(get(base,'/health')[2])['in_flight']==0,'drained after disconnect',timeout=8)
                s,value=call('/v1/messages',prompt());check('healthy generation after disconnect',s==200 and value['content'][0]['text']=='Hi',value)
            finally:
                for child in reversed(children):
                    if child.poll() is None:child.terminate()
                    child.wait(timeout=10)
    finally:
        (root/'responses.json').write_text(json.dumps(raw_rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (root/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    failed=[x['name'] for x in results if not x['passed']];print(json.dumps({'checks':len(results),'failed':failed}))
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
