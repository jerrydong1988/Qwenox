#!/usr/bin/env python3
"""Open-call repair across all API contracts; no model/GPU required."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from api_disconnect_test import free_port, make_tokenizer, wait_for
from api_regression_test import get, request

# This suite owns every process and ephemeral port.
FUNCTION = {'name':'read_file','parameters':{'type':'object','properties':{
    'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}
CALL = '<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>'
COUNT = 0

def check(ok, name, detail=''):
    global COUNT
    if not ok: raise AssertionError(f'{name}: {detail}')
    COUNT += 1

def wire(port, command):
    with socket.create_connection(('127.0.0.1', port), timeout=2) as conn:
        conn.sendall((command+'\n').encode())
        return json.loads(conn.makefile('rb').readline())

def body_for(api, case, stream=False, **extra):
    text = f'resume-case-{case}'
    body = dict(temperature=0.7, seed=7, max_tokens=1000, stream=stream, enable_thinking=case==2)
    if api == 'responses':
        body.update(input=text, tools=[{'type':'function',**FUNCTION}])
    elif api == 'messages':
        body.update(messages=[{'role':'user','content':text}],
                    tools=[{'name':FUNCTION['name'],'input_schema':FUNCTION['parameters']}],
                    thinking={'type':'enabled' if case==2 else 'disabled'})
    else:
        body.update(messages=[{'role':'user','content':text}],tools=[{'type':'function','function':FUNCTION}])
    body.update(extra)
    if api=='messages' and 'stop' in body:
        stop=body.pop('stop');body['stop_sequences']=[stop] if isinstance(stop,str) else stop
    return body

def path_for(api):
    return '/v1/'+('chat/completions' if api=='chat' else api)

def frames(raw, stream):
    if not stream or raw.lstrip().startswith(b'{'):
        return [json.loads(raw)]
    return [json.loads(line[6:]) for line in raw.decode().splitlines()
            if line.startswith('data: {')]

def outcome(api, vals):
    if api=='chat':
        finals=[c['finish_reason'] for v in vals for c in v.get('choices',[]) if c.get('finish_reason')]
        return finals[-1] if finals else None
    if api=='responses':
        last=vals[-1].get('response',vals[-1]);return last.get('status')
    final=[v.get('delta',v).get('stop_reason') for v in vals if v.get('delta',v).get('stop_reason')]
    return final[-1] if final else None

def checks(base, ep, ap):
    for api in ('chat','responses','messages'):
        for stream in (False,True):
            for case in (0,1,2,3):
                wire(ep,'RESUMELOG')
                status,_,raw=request(base,path_for(api),body_for(api,case,stream))
                logs=wire(ep,'RESUMELOG'); vals=frames(raw,stream)
                tag=f'{api}/{stream}/{case}'
                check(status==200,tag,raw)
                check(outcome(api,vals)=={'chat':'tool_calls','responses':'completed','messages':'tool_use'}[api],tag,raw)
                check(len(logs)==2 and not logs[0]['resumed'] and logs[1]['resumed'],tag,logs)
                check(logs[0]['connection']==logs[1]['connection'],tag+' same slot')
                first={1:CALL.split('\n</parameter>')[0],2:'</think>\n\n<tool_call>',3:CALL+'<tool_call>'}.get(case,'<tool_call>')
                check(logs[1]['ids']==logs[0]['ids']+[1024+b for b in first.encode()],tag+' exact ids')
                check(logs[1]['max']==min(128,1000-len(first)),tag+' remaining budget')
                check(dict(logs[1]['bias'])=={248046:-100,248044:-100,248043:-100},tag+' eos bias')
                # Reconstruct emitted arguments; every call occurs once, including
                # when an earlier complete call precedes the unfinished one.
                n=2 if case==3 else 1
                if api=='chat':
                    calls={}
                    for v in vals:
                        for c in v.get('choices',[]):
                            for tc in c.get('delta',c.get('message',{})).get('tool_calls',[]):
                                i=tc.get('index',len(calls) if not stream else 0)
                                calls[i]=calls.get(i,'')+tc.get('function',{}).get('arguments','')
                    check(len(calls)==n and all(json.loads(x)=={'path':'a'} for x in calls.values()),tag+' arguments',calls)
                elif api=='responses':
                    final=vals[-1].get('response',vals[-1]); calls=[x for x in final['output'] if x['type']=='function_call']
                    check(len(calls)==n and all(json.loads(x['arguments'])=={'path':'a'} for x in calls),tag+' arguments',calls)
                elif not stream:
                    calls=[x for x in vals[0]['content'] if x['type']=='tool_use']
                    check(len(calls)==n and all(x['input']=={'path':'a'} for x in calls),tag+' arguments',calls)
                    check(vals[0]['stop_sequence'] is None,tag+' internal stop hidden')
                else:
                    calls={}
                    for v in vals:
                        if v.get('type')=='content_block_start' and v['content_block']['type']=='tool_use':
                            calls[v['index']]=''
                        if v.get('type')=='content_block_delta' and v['delta']['type']=='input_json_delta':
                            calls[v['index']]+=v['delta']['partial_json']
                    check(len(calls)==n and all(json.loads(x)=={'path':'a'} for x in calls.values()),tag+' SSE arguments',calls)
                    ends=[v for v in vals if v.get('type')=='message_stop']
                    check(len(ends)==1,tag+' single terminal')
                    deltas=[v['delta'] for v in vals if v.get('type')=='message_delta']
                    check(deltas[-1]['stop_sequence'] is None,tag+' internal stop hidden')
                if not stream:
                    u=vals[0]['usage']; input_key='prompt_tokens' if api=='chat' else 'input_tokens'
                    output_key='completion_tokens' if api=='chat' else 'output_tokens'
                    check(u[input_key]==len(logs[0]['ids']),tag+' original prompt counted once',u)
                    check(u[output_key]==len(first)+len(CALL)-(len(first) if case==1 else len('<tool_call>')),tag+' repair tokens counted',u)
            for case,extra,expect_gens in (
                (4,{},1),(9,{},1),(7,{},1),(6,{},2),
                (5,{},2),(8,{'max_tokens':40},2),
                (0,{'max_tokens':11},1),
                (0,{'stop':'<tool_call>'},1),
                (0,{'stop':'</tool_call>'},1),
                (10,{'stop':'HALT'},2),
                (0,{'tools':[]},1),
                (0,{'temperature':0},2),
                (0,{'logit_bias':{'248046':5}},2)):
                wire(ep,'RESUMELOG')
                status,_,raw=request(base,path_for(api),body_for(api,case,stream,**extra))
                logs=wire(ep,'RESUMELOG'); vals=frames(raw,stream)
                tag=f'{api}/{stream}/edge{case}/{extra}'
                check(len(logs)==expect_gens,tag,[{k:v for k,v in x.items() if k!='ids'} for x in logs])
                if case in (4,9,6,5,10):
                    check('invalid_tool_call' in raw.decode() or 'model_output_error' in raw.decode()
                          or (api=='messages' and any(v.get('type')=='error' and
                              v.get('error',{}).get('type')=='api_error' for v in vals)),tag+' explicit error',raw)
                    check(outcome(api,vals) not in ('length','max_tokens','incomplete'),tag+' no false budget',raw)
                if case==8 or case==7:
                    check(outcome(api,vals)=={'chat':'length','responses':'incomplete','messages':'max_tokens'}[api],tag+' real budget',raw)
                if extra.get('temperature')==0:
                    check(logs[1]['bias']==[],tag+' greedy no bias')
                if 'logit_bias' in extra:
                    check(dict(logs[1]['bias'])[248046]==5,tag+' client bias preserved')
    # Force the real context clamp independently of the client's larger budget.
    body=body_for('messages',8,max_tokens=1000)
    _,_,raw=request(base,'/v1/messages/count_tokens',body)
    count=json.loads(raw)['input_tokens']
    body['messages'][0]['content']+='a'*(262144-40-count)
    wire(ep,'RESUMELOG')
    status,_,raw=request(base,'/v1/messages',body);logs=wire(ep,'RESUMELOG')
    check(status==200 and json.loads(raw)['stop_reason']=='max_tokens','context exhaustion stays max_tokens',raw)
    check([x['max'] for x in logs]==[40,29],'repair uses original context clamp')
    # A queued request must use the released slot after a disconnected repair.
    wire(ep,'RESUMELOG')
    body=body_for('chat',11,True); payload=json.dumps(body).encode()
    conn=socket.create_connection(('127.0.0.1',ap),timeout=2)
    conn.sendall((f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n').encode()+payload)
    wait_for(lambda: bool(wire(ep,'TESTSTATE')), 'repair entered')
    conn.close()
    wait_for(lambda: json.loads(get(base,'/health')[2])['in_flight']==0,'repair cancelled',timeout=15)
    check(any(x['cancel']==1 for x in wire(ep,'TESTSTATE')),'repair cancellation forwarded once')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(request,base,path_for('chat'),body_for('chat',0)) for _ in range(2)]
        check(all(j.result()[0]==200 for j in jobs),'concurrent requests after cancellation')
    check(json.loads(get(base,'/health')[2])['in_flight']==0,'all slots released')

def main():
    p=argparse.ArgumentParser();p.add_argument('--api',required=True);p.add_argument('--slots',type=int,default=1);a=p.parse_args()
    api=str(Path(a.api).resolve()); ep,ap=free_port(),free_port()
    while ep==ap:ap=free_port()
    children=[]
    with tempfile.TemporaryDirectory(prefix='qwenox-tool-resume-') as tmp:
        root=Path(tmp);make_tokenizer(root/'tokenizer')
        env=dict(os.environ,GDEC_API_TOKCACHE_FILE='',GDEC_API_TOKCACHE='0',GDEC_API_ADMIN_KEY='',GDEC_REQSTAT='0',ROPE_FACTOR='1',QWENOX_TOOL_CALL_RESUME_MAX='128')
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        with (root/'test.log').open('w+') as log:
            try:
                children.append(subprocess.Popen([sys.executable,str(Path(__file__).with_name('api_fake_engine.py')),'--port',str(ep),'--slots',str(a.slots)],stdout=log,stderr=log,env=env,creationflags=flags))
                def ready():
                    try:wire(ep,'TESTSTATE');return True
                    except OSError:return False
                wait_for(ready,'engine')
                children.append(subprocess.Popen([api,'--tokenizer',str(root/'tokenizer'),'--engine',f'127.0.0.1:{ep}','--port',str(ap)],stdout=log,stderr=log,env=env,creationflags=flags))
                base=f'http://127.0.0.1:{ap}'
                def health():
                    try:return get(base,'/health')[0]==200
                    except OSError:return False
                wait_for(health,'API');checks(base,ep,ap)
                print(f'PASS {COUNT} assertions')
            except BaseException:
                log.flush();log.seek(0);print(log.read()[-12000:],file=sys.stderr);raise
            finally:
                for child in reversed(children):
                    if child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)

if __name__=='__main__':main()
