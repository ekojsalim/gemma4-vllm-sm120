"""Diagnose verifier changes with unique cache salts in one running instance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import secrets

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--root',required=True);p.add_argument('--url',default='http://127.0.0.1:31839');a=p.parse_args()
spec=importlib.util.spec_from_file_location('replay',Path(__file__).with_name('span-replay.py'));replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
root=Path(a.root);rows=[json.loads(l) for l in Path(a.input).open()]
off=[json.loads(l) for l in (root/'spans-off/responses.private.jsonl').open()];on=[json.loads(l) for l in (root/'spans-on/responses.private.jsonl').open()]
def answer(r):return r['response']['choices'][0]['message']['content'].strip().lower()
flips=[x['index'] for x,y in zip(off,on) if x['kind']=='verify' and answer(x)!=answer(y)]
controls=[next(r['index'] for r in off if r['chapter']==ch and r['kind']=='verify' and r['index'] not in flips) for ch in ('zh1','zh2','ja1','ko1')]
indices=flips+controls
jobs=[(i,dict(rows[i],body=dict(rows[i]['body'],cache_salt=secrets.token_hex(32)))) for i in indices]
all_results={};deltas={}
for phase in ('cold','warm','warm-repeat'):
    before=replay.metrics(a.url)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda pair:replay.call(a.url,pair[1],pair[0]),jobs))
    after=replay.metrics(a.url)
    all_results[phase]=results
    deltas[phase]={k:after[k]-before.get(k,0) for k in after if k.startswith('vllm:prefix_cache_') and k.endswith('_total')}
    (root/'span-cold-warm.private.json').write_text(json.dumps(all_results,indent=2))
    print(phase,[(r['index'],answer(r)) for r in results],deltas[phase],flush=True)
summary=dict(selected_flips=flips,controls=controls,metrics=deltas,answers={phase:[dict(index=r['index'],answer=answer(r),elapsed_ms=r['elapsed_ms']) for r in results] for phase,results in all_results.items()})
(root/'span-cold-warm-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
