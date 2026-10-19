"""Real pinned vLLM managers and BlockPool, metadata only, no GPU tensors.

Run explicitly in the isolated pinned image; host oracle discovery skips this
module when vLLM is absent. No replacement allocator or AST-extracted methods.
"""
import importlib.util
import unittest
from types import SimpleNamespace

if importlib.util.find_spec('vllm') is None:
    raise unittest.SkipTest('real-manager tests require the pinned vLLM image')

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_adapter import observe, request_budgets


def fixture(capacity=64, bs=4, window=8, inflight=0, extra=0, pool_class=BlockPool):
    pool = pool_class(capacity, True, bs)
    full = FullAttentionSpec(block_size=bs,num_kv_heads=1,head_size=32,dtype=torch.float16)
    swa = SlidingWindowSpec(block_size=bs,num_kv_heads=1,head_size=32,dtype=torch.float16,
                            sliding_window=window,extra_retained_tokens=extra)
    common = dict(block_pool=pool,enable_caching=True,scheduler_block_size=bs)
    managers = (FullAttentionManager(full,kv_cache_group_id=0,**common),
                SlidingWindowManager(swa,kv_cache_group_id=1,
                    max_admission_blocks_per_request=swa.max_admission_blocks_per_request(inflight,6144),**common))
    return pool,managers


def publish(m,rid,tokens):
    req=SimpleNamespace(request_id=rid,shared_prefix_boundary=0,
        block_hashes=[i.to_bytes(16,'big') for i in range(1,(tokens//m.block_size)+1)])
    m.cache_blocks(req,tokens,replay_boundaries=())


def allocate(m,rid,tokens,hit=(),computed=0):
    if rid not in m.num_cached_block:
        m.add_local_computed_blocks(rid,hit,computed,0)
    return m.allocate_new_blocks(rid,tokens,tokens)


def estimate(m,rid,tokens,hit=(),computed=0):
    return m.get_num_blocks_to_allocate(rid,tokens,hit,computed,computed,tokens)


def snapshot(pool,ms,extents,**kwargs):
    return observe(ms,extents,6144,{b.block_id:0 for b in pool.blocks},**kwargs)


class RealManagersTests(unittest.TestCase):
    def test_geometry_matches_pinned_aggregate(self):
        for bs in (4,16,32):
            for extra in (0,3):
                for inflight in (0,8,2048):
                    pool,ms=fixture(bs=bs,window=1024,inflight=inflight,extra=extra)
                    # Invoke the actual helper with the same managers; no engine.
                    parent=object.__new__(KVCacheManager)
                    parent.max_model_len=6144
                    parent.coordinator=SimpleNamespace(single_type_managers=ms)
                    for extent in (1,bs,bs+1,1024,2737,6144):
                        budgets=dict(request_budgets(ms,extent,6144))
                        self.assertEqual(sum(budgets.values()),parent.get_admission_blocks(extent))
                        self.assertEqual(budgets['full:0'],(extent+bs-1)//bs)
                        cap=(min(1023+extra+inflight,6144)+bs-1)//bs+1
                        self.assertEqual(budgets['swa:1'],min((extent+bs-1)//bs,cap))

    def test_cold_active_and_idle_hit_counts(self):
        pool,ms=fixture()
        m=ms[0]
        self.assertEqual(estimate(m,'a',8),2)
        allocate(m,'a',8);publish(m,'a',8)
        hit=list(m.req_to_blocks['a'])
        self.assertEqual(estimate(m,'b',12,hit,8),1)
        before=pool.get_num_free_blocks()
        allocate(m,'b',12,hit,8)
        self.assertEqual(before-pool.get_num_free_blocks(),1)
        self.assertTrue(all(b.ref_cnt==2 for b in hit))
        observation=snapshot(pool,ms,{'a':12,'b':12})
        self.assertEqual(observation.charge().pinned,3)
        m.free('a');m.free('b')
        self.assertTrue(all(b.ref_cnt==0 for b in hit))
        # Same two prefix blocks now leave the free queue when touched.
        self.assertEqual(estimate(m,'c',12,hit,8),3)
        allocate(m,'c',12,hit,8)
        self.assertEqual(snapshot(pool,ms,{'c':12}).charge().pinned,3)

    def test_swa_skips_nulls_and_reclaims_shared_refs(self):
        pool,ms=fixture()
        m=ms[1]
        allocate(m,'a',24);publish(m,'a',24)
        m.remove_skipped_blocks('a',24)
        self.assertEqual(sum(not b.is_null for b in m.req_to_blocks['a']),2)
        hit=list(m.req_to_blocks['a'])
        allocate(m,'b',24,hit,24)
        obs=snapshot(pool,ms,{'a':32,'b':32})
        self.assertEqual(obs.charge().pinned,2)
        self.assertFalse(any(b.lifetime_prefix for b in obs.snapshot.blocks))
        # SWA null placeholders do not consume real pool capacity or credit.
        m.remove_skipped_blocks('a',32)
        self.assertEqual(snapshot(pool,ms,{'a':32,'b':32}).charge().pinned,2)
        m.free('b');m.free('a')
        self.assertEqual(snapshot(pool,ms,{}).charge().total,0)

    def test_transient_prefill_exceeds_persistent_swa_cap(self):
        pool,ms=fixture()
        m=ms[1]
        self.assertEqual(m.admission_block_cap,3)
        self.assertEqual(estimate(m,'a',24),6)
        allocate(m,'a',24)
        obs=snapshot(pool,ms,{'a':32})
        self.assertEqual(obs.charge().pinned,6)
        self.assertEqual(obs.charge().future,8) # full-attention extent, no full blocks yet

    def test_deferred_free_requires_explicit_lease(self):
        pool,ms=fixture();m=ms[0]
        allocate(m,'a',8)
        retained=m.pop_blocks_for_free('a')
        with self.assertRaisesRegex(ValueError,'unexplained reference'):
            snapshot(pool,ms,{})
        obs=snapshot(pool,ms,{},leases={7:[(0,b) for b in retained]})
        self.assertEqual(obs.charge().total,2)
        pool.free_blocks(retained)
        self.assertEqual(snapshot(pool,ms,{}).charge().total,0)

    def test_hash_lookup_eviction_and_group_isolation(self):
        pool,ms=fixture(capacity=5);m=ms[0]
        allocate(m,'a',8);publish(m,'a',8)
        hashes=[i.to_bytes(16,'big') for i in (1,2)]
        def lookup(gid):
            return m.find_longest_cache_hit(hashes,8,[gid],pool,m.kv_cache_spec,False,4)
        hits,length=lookup(0)
        self.assertEqual(length,8)
        self.assertEqual(hits[0],m.req_to_blocks['a'])
        self.assertEqual(lookup(1)[1],0)
        m.free('a')
        self.assertEqual(lookup(0)[1],8) # idle cached entries still reusable
        allocate(m,'pressure',16) # force eviction of both idle cached entries
        self.assertEqual(lookup(0)[1],0)
        m.free('pressure')
        self.assertEqual(snapshot(pool,ms,{}).charge().total,0)

    def test_partial_hit_cow_cancel_and_copy_completion(self):
        pool,ms=fixture();m=ms[0]
        allocate(m,'a',8);publish(m,'a',8)
        hits=list(m.req_to_blocks['a'])
        # A cross-group convergence boundary may truncate a full hit to 6 tokens.
        self.assertEqual(estimate(m,'b',8,hits,6),1)
        allocate(m,'b',8,hits,6)
        copies=m.take_pending_cow_copies()
        self.assertEqual(len(copies),1)
        source,dest=copies[0]
        self.assertIs(source,hits[-1])
        self.assertIs(m.req_to_blocks['b'][-1],dest)
        self.assertEqual((source.ref_cnt,dest.ref_cnt),(2,2))
        with self.assertRaisesRegex(ValueError,'unexplained reference'):
            snapshot(pool,ms,{'a':12,'b':12})
        leases={1:[(0,source),(0,dest)]}
        self.assertEqual(snapshot(pool,ms,{'a':12,'b':12},leases=leases).charge().pinned,3)
        m.free('b');m.free('a') # cancelled before worker copy completes
        self.assertEqual(snapshot(pool,ms,{},leases=leases).charge().total,2)
        pool.free_blocks([source,dest]) # worker copy completion
        self.assertEqual(snapshot(pool,ms,{}).charge().total,0)

    def test_missing_epoch_fails_closed(self):
        pool,ms=fixture();allocate(ms[0],'a',4)
        with self.assertRaisesRegex(ValueError,'generation'):
            observe(ms,{'a':4},6144,{})


if __name__=='__main__':
    unittest.main()
