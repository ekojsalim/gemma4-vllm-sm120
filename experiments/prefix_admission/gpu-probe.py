"""Bounded HTTP probe. Raw responses stay in the caller's private output directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import time
import urllib.request


def metrics(url):
    with urllib.request.urlopen(url+'/metrics',timeout=10) as r:
        return r.read().decode()


def counters(raw):
    result={}
    for line in raw.splitlines():
        if line.startswith('#') or not line.startswith('vllm:'):
            continue
        name=line.split('{')[0].split()[0]
        if any(x in name for x in ('spec_decode_num','prefix_cache','num_requests_running','num_requests_waiting','num_preemptions')):
            try:
                result[name]=result.get(name,0)+float(line.rsplit(' ',1)[-1])
            except ValueError:
                pass
    return result


def call(url, body, label):
    body=dict(body,model='prefix-trial',stream=True,stream_options={'include_usage':True})
    req=urllib.request.Request(url+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    start=time.perf_counter();first=None;output='';usage=None;finish=None;logprobs=[]
    with urllib.request.urlopen(req,timeout=180) as r:
        for raw in r:
            if not raw.startswith(b'data: '):
                continue
            data=raw[6:].strip()
            if data==b'[DONE]':
                break
            row=json.loads(data)
            if row.get('error'):
                raise RuntimeError(row['error'])
            usage=row.get('usage') or usage
            for choice in row.get('choices',[]):
                content=choice.get('delta',{}).get('content') or ''
                if content and first is None:
                    first=time.perf_counter()
                output+=content
                finish=choice.get('finish_reason') or finish
                if choice.get('logprobs'):
                    logprobs.extend(choice['logprobs'].get('content') or [])
    end=time.perf_counter()
    return dict(label=label,elapsed_ms=1000*(end-start),ttft_ms=1000*(first-start) if first else None,
                output=output,output_sha256=hashlib.sha256(output.encode()).hexdigest(),
                usage=usage,finish_reason=finish,logprobs=logprobs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:31839');p.add_argument('--root',required=True);p.add_argument('--variant',required=True)
    a=p.parse_args();root=Path(a.root);out=root/a.variant;out.mkdir(exist_ok=True)
    original=json.loads((root/'private-translation.json').read_text())
    chapter=(root/'public-chapter.txt').read_text()
    common=dict(temperature=0,seed=7,chat_template_kwargs={'enable_thinking':False})
    translation=dict(messages=original['messages'],temperature=0.5,seed=7,max_tokens=512)
    schema={'type':'object','properties':{'location':{'type':'string','enum':['market','river','library']}},'required':['location'],'additionalProperties':False}
    structured=dict(common,messages=[{'role':'system','content':'Answer the question about the chapter using the required JSON schema.'},{'role':'user','content':chapter+'\nWhere did Lila go at sunrise?'}],max_tokens=48,
        response_format={'type':'json_schema','json_schema':{'name':'location','strict':True,'schema':schema}})
    decision=dict(common,messages=[{'role':'system','content':'Answer with one letter only: A for market, B for river, C for library.'},{'role':'user','content':chapter+'\nWhere did Lila go at sunrise?'}],max_tokens=1,logprobs=True,top_logprobs=3)
    def span(i):
        return dict(common,messages=[{'role':'system','content':'Translate the assigned numbered Indonesian paragraph into English. Return only <span id="N">translation</span>, replacing N with the assigned paragraph number.'},{'role':'user','content':chapter+f'\nAssigned paragraph: {i}'}],max_tokens=160)
    # Unrelated short warm-up, not included in measurements.
    call(a.url,dict(common,messages=[{'role':'user','content':'Reply OK.'}],max_tokens=8),'warmup')
    rows=[];phases={}
    def phase(name,jobs,concurrency=1):
        before=metrics(a.url);(out/(name+'-before.metrics')).write_text(before)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results=list(executor.map(lambda pair:call(a.url,pair[1],pair[0]),jobs))
        after=metrics(a.url);(out/(name+'-after.metrics')).write_text(after)
        old,new=counters(before),counters(after)
        delta={k:new[k]-old.get(k,0) for k in new if k.endswith('_total')}
        rows.extend(results);(out/'responses.private.json').write_text(json.dumps(rows,indent=2))
        phases[name]={'delta':delta,'requests':[{'label':r['label'],'ttft_ms':r['ttft_ms'],'elapsed_ms':r['elapsed_ms'],'usage':r['usage'],'finish_reason':r['finish_reason'],'output_sha256':r['output_sha256']} for r in results]}
        print(json.dumps({'phase':name,'median_ttft_ms':median(r['ttft_ms'] for r in results if r['ttft_ms'] is not None),'delta':delta}),flush=True)
    for name,body in [('translation',translation),('structured',structured),('decision',decision),('span',span(1))]:
        for i in range(3):
            phase(f'{name}-{i}',[(f'{name}-{i}',body)])
    changed=dict(decision,messages=[decision['messages'][0],
        {'role':'user','content':chapter.replace('ke pasar ikan','ke tepi sungai')+
         '\nWhere did Lila go at sunrise?'}])
    phase('decision-changed',[('decision-changed',changed)])
    phase('decision-original-again',[('decision-original-again',decision)])
    for n in (4,8):
        phase(f'fanout-{n}',[(f'fanout-{n}-{i}',span(1+i%6)) for i in range(n)],n)
    valid={}
    for r in rows:
        name=r['label'];text=r['output']
        if name.startswith('structured'):
            valid[name]=json.loads(text)=={'location':'market'}
        elif name.startswith('decision'):
            valid[name]=text.strip()==('B' if name=='decision-changed' else 'A')
        elif name.startswith(('span','fanout')):
            expected=1 if name.startswith('span') else 1+int(name.rsplit('-',1)[1])%6
            valid[name]=bool(re.fullmatch(r'<span id="'+str(expected)+r'">[\s\S]+</span>',text.strip()))
    summary={'variant':a.variant,'phases':phases,'checks':valid,'all_checks_pass':all(valid.values()),'final_gauges':counters(metrics(a.url))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'all_checks_pass':all(valid.values()),'failed':[k for k,v in valid.items() if not v]}),flush=True)


if __name__=='__main__':
    main()
