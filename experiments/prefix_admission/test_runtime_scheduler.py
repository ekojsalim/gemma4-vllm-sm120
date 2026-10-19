"""Actual Scheduler.schedule/update_from_output on CPU; worker outputs are synthetic."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec('vllm') is None:
    raise unittest.SkipTest('runtime scheduler tests require pinned vLLM')

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig, DeviceConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash, get_request_block_hasher
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput, DraftTokenIds
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


def make_scheduler(blocks=64, opt_in=True, speculative=False, async_mode=False, hash_size=16):
    with tempfile.TemporaryDirectory() as d:
        Path(d,'config.json').write_text(json.dumps(dict(
            architectures=['OPTForCausalLM'],model_type='opt',hidden_size=64,
            word_embed_proj_dim=64,ffn_dim=128,num_attention_heads=2,
            num_hidden_layers=2,max_position_embeddings=128,vocab_size=256,
            bos_token_id=1,eos_token_id=2,pad_token_id=0)))
        model=ModelConfig(model=d,dtype='float16',max_model_len=128,skip_tokenizer_init=True)
        sc=SchedulerConfig(max_num_seqs=8,max_num_batched_tokens=128,max_model_len=128,
                           enable_chunked_prefill=True,hybrid_kv_cache_admission=True,
                           async_scheduling=async_mode,watermark=0.0,is_encoder_decoder=False)
        cc=CacheConfig(block_size=16,enable_prefix_caching=True)
        cc.num_gpu_blocks=blocks
        config=VllmConfig(model_config=model,scheduler_config=sc,cache_config=cc,
                          device_config=DeviceConfig(device='cpu'))
        common=dict(block_size=16,num_kv_heads=1,head_size=16,dtype=torch.float16)
        kv=KVCacheConfig(num_blocks=blocks,kv_cache_tensors=[],kv_cache_groups=[
            KVCacheGroupSpec(['full'],FullAttentionSpec(**common)),
            KVCacheGroupSpec(['swa'],SlidingWindowSpec(**common,sliding_window=32))])
        if speculative:
            # Metadata-only Gemma MTP contract; no target or draft model runs.
            config.model_config.hf_text_config.model_type='gemma4_text'
            config.model_config.hf_config.architectures=['Gemma4ForConditionalGeneration']
            config.scheduler_config.hybrid_kv_speculative_admission=True
            config.speculative_config=SimpleNamespace(
                num_speculative_tokens=4,num_speculative_tokens_per_batch_size=None,
                method='mtp',enable_adaptive_verification=False,max_num_new_slots_for_drafting=0,parallel_drafting=False,use_eagle=lambda:True,use_eagle_block_drop=lambda:True,
                use_dflash=lambda:False,uses_draft_model=lambda:False,
                use_multi_module_mtp=lambda:False)
        register_all_kvcache_specs(config)
        with patch.dict(os.environ,{'GEMMA4_EXPERIMENTAL_PREFIX_ADMISSION':'1' if opt_in else '0'}):
            return (AsyncScheduler if async_mode else Scheduler)(
                config,kv,StructuredOutputManager(config),block_size=16,hash_block_size=hash_size)


def request(rid, max_tokens=4, suffix=99):
    init_none_hash(sha256)
    sampling=SamplingParams(max_tokens=max_tokens,ignore_eos=True)
    sampling.update_from_generation_config({},2)
    return Request(rid,list(range(1,49))+[suffix],sampling,pooling_params=None,
                   block_hasher=get_request_block_hasher(16,sha256))


def finish_step(scheduler, output):
    ids=list(output.num_scheduled_tokens)
    result=ModelRunnerOutput(req_ids=ids,req_id_to_index={rid:i for i,rid in enumerate(ids)},
                            sampled_token_ids=[[100] for _ in ids],logprobs=None,
                            prompt_logprobs_dict={},pooler_output=[])
    return scheduler.update_from_output(output,result)


class RuntimeSchedulerTests(unittest.TestCase):
    def test_opt_in_required(self):
        with self.assertRaisesRegex(ValueError,'prefix caching disabled'):
            make_scheduler(opt_in=False)

    def test_cold_then_warm_sibling_and_drain(self):
        s=make_scheduler()
        first=request('first',max_tokens=1)
        s.add_request(first)
        cold=s.schedule()
        self.assertEqual(cold.num_scheduled_tokens['first'],49)
        finish_step(s,cold)
        self.assertEqual(s.hybrid_kv_admission.charged_blocks,0)
        child=request('child',max_tokens=2,suffix=98)
        s.add_request(child)
        warm=s.schedule()
        self.assertLess(warm.num_scheduled_tokens['child'],49)
        self.assertLess(warm.num_scheduled_tokens['child'],49)
        finish_step(s,warm)
        finish_step(s,s.schedule())
        self.assertEqual(s.hybrid_kv_admission.charged_blocks,0)
        self.assertEqual(s.kv_cache_manager.block_pool.get_num_free_blocks(),63)

    def test_active_shared_prefix_cancel_before_completion(self):
        s=make_scheduler()
        parent=request('parent',max_tokens=8)
        s.add_request(parent);finish_step(s,s.schedule())
        child=request('child',max_tokens=8,suffix=98)
        s.add_request(child)
        batch=s.schedule()
        self.assertLess(batch.num_scheduled_tokens['child'],49)
        s.finish_requests(['child'],RequestStatus.FINISHED_ABORTED)
        finish_step(s,batch)
        s.finish_requests(['parent'],RequestStatus.FINISHED_ABORTED)
        self.assertEqual(s.hybrid_kv_admission.charged_blocks,0)
        self.assertEqual(s.kv_cache_manager.block_pool.get_num_free_blocks(),63)

    def test_fixed_k4_warm_hit_rejected_drafts_and_drain(self):
        s=make_scheduler(speculative=True)
        parent=request('parent',max_tokens=1)
        s.add_request(parent);finish_step(s,s.schedule())
        child=request('child',max_tokens=8,suffix=98)
        s.add_request(child)
        warm=s.schedule()
        self.assertLess(warm.num_scheduled_tokens['child'],49)
        finish_step(s,warm)
        for _ in range(2):
            s.update_draft_token_ids(DraftTokenIds(['child'],[[101,102,103,104]]))
            batch=s.schedule()
            self.assertEqual(len(batch.scheduled_spec_decode_tokens['child']),4)
            finish_step(s,batch) # one output: all four proposed drafts rejected
        s.finish_requests(['child'],RequestStatus.FINISHED_ABORTED)
        self.assertEqual(s.hybrid_kv_admission.charged_blocks,0)
        self.assertEqual(s.kv_cache_manager.block_pool.get_num_free_blocks(),63)

    def test_pressure_waits_then_finishes_all_requests(self):
        s=make_scheduler(blocks=16)
        for i in range(6):
            s.add_request(request(f'r{i}',max_tokens=3))
        waiting_seen=False
        for _ in range(30):
            if not s.requests:
                break
            batch=s.schedule()
            waiting_seen |= bool(s.waiting)
            self.assertGreater(batch.total_num_scheduled_tokens,0)
            self.assertLessEqual(s.hybrid_kv_admission.charged_blocks,15)
            finish_step(s,batch)
        self.assertTrue(waiting_seen)
        self.assertFalse(s.requests)
        self.assertEqual(s.hybrid_kv_admission.charged_blocks,0)
        self.assertEqual(s.kv_cache_manager.block_pool.get_num_free_blocks(),15)

    def test_async_two_batches_cancel_and_drain(self):
        s=make_scheduler(async_mode=True,speculative=True)
        parent=request('parent',max_tokens=8)
        s.add_request(parent);finish_step(s,s.schedule())
        child=request('child',max_tokens=8,suffix=98)
        s.add_request(child)
        first=s.schedule()
        second=s.schedule()
        self.assertEqual(len(s.hybrid_kv_admission.batches),2)
        s.finish_requests(['child','parent'],RequestStatus.FINISHED_ABORTED)
        self.assertGreater(s.hybrid_kv_admission.charged_blocks,0)
        finish_step(s,first)
        self.assertGreater(s.hybrid_kv_admission.charged_blocks,0)
        finish_step(s,second)
        self.assertEqual(s.hybrid_kv_admission.charged_blocks,0)
        self.assertEqual(s.kv_cache_manager.block_pool.get_num_free_blocks(),63)

    def test_fine_grained_hashing_rejected(self):
        with self.assertRaisesRegex(ValueError,'uniform full-block'):
            make_scheduler(hash_size=8)

    def test_partial_hit_rejected_before_allocation(self):
        s=make_scheduler()
        req=request('partial')
        before=s.kv_cache_manager.block_pool.get_num_free_blocks()
        with self.assertRaisesRegex(ValueError,'full-block hits'):
            s.kv_cache_manager.prepare_speculative_allocation(
                req,1,4,num_new_computed_tokens=15)
        self.assertEqual(s.kv_cache_manager.block_pool.get_num_free_blocks(),before)
