"""Replay private span bodies unchanged except endpoint model, four in flight.
Raw responses remain in --out; only summary.json is suitable for publication.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
import re
import time
import urllib.request


def metrics(url):
    with urllib.request.urlopen(url+'/metrics',timeout=15) as r:
        raw=r.read().decode()
    result={}
    for line in raw.splitlines():
        if not line.startswith('vllm:'): continue
        name=line.split('{')[0].split()[0]
        if name.endswith(('_total','_sum','_count')) or name in ('vllm:num_requests_running','vllm:num_requests_waiting'):
            result[name]=result.get(name,0)+float(line.rsplit(' ',1)[-1])
    return result


def call(url,row,index):
    body=dict(row['body'],model='prefix-trial')
    req=urllib.request.Request(url+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    start=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=300) as r: response=json.load(r)
        elapsed=1000*(time.perf_counter()-start)
        choice=response['choices'][0];text=choice['message'].get('content') or ''
        if row['kind']=='generate':
            markers=[int(x) for x in re.findall(r'<<U\s*(\d+)>>',text)]
            valid=markers==row['units'] and not text.split('<<',1)[0].strip()
        else:
            valid=text.strip().lower() in ('ok','no') and bool(choice.get('logprobs',{}).get('content'))
        return dict(index=index,chapter=row['chapter'],kind=row['kind'],request=row.get('request'),unit=row.get('unit'),question=row.get('question'),elapsed_ms=elapsed,valid=valid,response=response)
    except Exception as exc:
        return dict(index=index,chapter=row['chapter'],kind=row['kind'],elapsed_ms=1000*(time.perf_counter()-start),valid=False,error=str(exc))


def quantile(xs,q):
    xs=sorted(xs);return xs[round((len(xs)-1)*q)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--url',default='http://127.0.0.1:31839');a=p.parse_args()
    rows=[json.loads(l) for l in Path(a.input).open()];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    phases=[]
    # Unrelated warm-up, retaining cold chapter prefixes.
    call(a.url,dict(kind='verify',chapter='warmup',body=dict(messages=[dict(role='user',content='Reply ok.')],max_tokens=1,temperature=0,logprobs=True,top_logprobs=20,chat_template_kwargs={'enable_thinking':False})),-1)
    with (out/'responses.private.jsonl').open('w') as f:
        for key,group in itertools.groupby(enumerate(rows),lambda x:(x[1]['chapter'],x[1]['kind'])):
            jobs=list(group);before=metrics(a.url);start=time.perf_counter();results=[]
            with ThreadPoolExecutor(max_workers=4) as pool:
                for r in pool.map(lambda pair:call(a.url,pair[1],pair[0]),jobs):
                    results.append(r);f.write(json.dumps(r)+'\n');f.flush()
            elapsed=time.perf_counter()-start;after=metrics(a.url)
            delta={k:after[k]-before.get(k,0) for k in after if k.endswith(('_total','_sum','_count'))}
            ms=[r['elapsed_ms'] for r in results]
            phase=dict(chapter=key[0],kind=key[1],requests=len(results),wall_seconds=elapsed,p50_ms=quantile(ms,.5),p95_ms=quantile(ms,.95),p99_ms=quantile(ms,.99),valid=sum(r['valid'] for r in results),errors=sum('error' in r for r in results),truncated=sum(r['kind']=='generate' and r.get('response',{}).get('choices',[{}])[0].get('finish_reason')=='length' for r in results),prompt_tokens=sum(r.get('response',{}).get('usage',{}).get('prompt_tokens',0) for r in results),output_tokens=sum(r.get('response',{}).get('usage',{}).get('completion_tokens',0) for r in results),metrics_delta=delta)
            phases.append(phase);(out/'summary.json').write_text(json.dumps(dict(phases=phases,final_running=after.get('vllm:num_requests_running'),final_waiting=after.get('vllm:num_requests_waiting')),indent=2)+'\n')
            print(json.dumps({k:v for k,v in phase.items() if k!='metrics_delta'}),flush=True)
            if phase['errors']: raise RuntimeError('Replay errors; inspect private responses before continuing')

if __name__=='__main__':main()
