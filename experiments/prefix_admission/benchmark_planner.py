"""Synthetic CPU metadata timing only; no inference or production latency claims."""
import json
from statistics import median
from time import perf_counter_ns

from direct_planner import DirectTransactions, project
from lifecycle import EpochBlockPool, Lifecycle
from test_vllm_managers import fixture, publish
from transactions import Step, Transactions


def make(cls, capacity=4096, count=8):
    pool,ms=fixture(capacity=capacity,bs=16,window=1024,pool_class=EpochBlockPool)
    tx=DirectTransactions(Lifecycle(ms,6144))
    tx.commit(tx.prepare(1,[Step('parent',1024,512)]));tx.complete(1)
    for m in ms:
        publish(m,'parent',512)
    hits=tuple(tuple(b.block_id for b in m.req_to_blocks['parent']) for m in ms)
    tx.cancel('parent')
    controller=cls(tx.life)
    controller.seen=set(tx.seen)
    tx=controller
    return tx,tuple(Step(f'c{i}',1024,528,lookahead=4,computed=512,hits=hits) for i in range(count))


def main():
    reference,steps=make(Transactions)
    direct,_=make(DirectTransactions)
    times={k:[] for k in ('shadow_prepare_ms','direct_prepare_ms','projection_only_ms')}
    funcs={
        'shadow_prepare_ms':lambda:reference.prepare(2,steps),
        'direct_prepare_ms':lambda:direct.prepare(2,steps),
        'projection_only_ms':lambda:project(direct,2,steps),
    }
    unavailable={}
    for i in range(23):
        order=[k for k in (list(funcs) if i%2 else list(reversed(funcs))) if k not in unavailable]
        for key in order:
            start=perf_counter_ns()
            try:
                funcs[key]()
            except RecursionError:
                unavailable[key]='allocator deepcopy exceeds Python recursion limit'
                continue
            elapsed=(perf_counter_ns()-start)/1e6
            if i>=3:
                times[key].append(elapsed)
        reference._plans.clear();direct._plans.clear()
    print(json.dumps({'pool_blocks_including_null':4096,'requests':8,
        'samples_per_path':20,'unavailable':unavailable,'median_ms':{k:median(v) for k,v in times.items() if v}},indent=2))


if __name__=='__main__':
    main()
