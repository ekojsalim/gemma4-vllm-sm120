"""Fill admission reservations, then disconnect clients and check queue drainage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import threading
import time
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('--root',required=True)
p.add_argument('--variant',required=True)
p.add_argument('--url',default='http://127.0.0.1:31839')
a=p.parse_args()
spec=importlib.util.spec_from_file_location('probe',Path(__file__).with_name('gpu-probe.py'))
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
root=Path(a.root)
body=dict(messages=json.loads((root/'private-translation.json').read_text())['messages'],
          model='prefix-trial',temperature=0.5,max_tokens=4096,ignore_eos=True,
          stream=True,stream_options={'include_usage':True})
release=threading.Event()
def client(i):
    req=urllib.request.Request(a.url+'/v1/chat/completions',json.dumps(body).encode(),
                               {'Content-Type':'application/json'})
    # Streaming headers arrive before queued inference completes. Close every
    # connection together, including waiting requests, without reading outputs.
    with urllib.request.urlopen(req,timeout=60) as r:
        release.wait(timeout=30)
    return i
samples=[]
with ThreadPoolExecutor(max_workers=64) as pool:
    futures=[pool.submit(client,i) for i in range(64)]
    try:
        for _ in range(40):
            samples.append(probe.counters(probe.metrics(a.url)))
            time.sleep(0.25)
    finally:
        release.set()
    completed=[f.result() for f in futures]
for _ in range(100):
    current=probe.counters(probe.metrics(a.url));samples.append(current)
    if current.get('vllm:num_requests_running',0)==0 and current.get('vllm:num_requests_waiting',0)==0:
        break
    time.sleep(0.2)
result=dict(clients_closed=len(completed),peak_running=max(x.get('vllm:num_requests_running',0) for x in samples),
            peak_waiting=max(x.get('vllm:num_requests_waiting',0) for x in samples),
            final_running=current.get('vllm:num_requests_running'),final_waiting=current.get('vllm:num_requests_waiting'))
# Ensure the server still produces a response after pressure/cancellation.
r=probe.call(a.url,dict(messages=[{'role':'user','content':'Reply OK.'}],max_tokens=8,
                       temperature=0,chat_template_kwargs={'enable_thinking':False}),'post-cancel')
result['followup_completed']=bool(r['finish_reason'])
(root/a.variant/'cancellation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
assert result['final_running']==0 and result['final_waiting']==0 and result['followup_completed']
