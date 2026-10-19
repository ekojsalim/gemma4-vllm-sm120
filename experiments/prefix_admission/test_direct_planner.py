"""Differential checks against shadow execution, plus shared transaction suite."""
import importlib.util
import random
import unittest

if importlib.util.find_spec('vllm') is None:
    raise unittest.SkipTest('direct planner tests require the pinned vLLM image')

import test_transactions
from direct_planner import DirectTransactions, project
from lifecycle import EpochBlockPool, Lifecycle
from test_vllm_managers import fixture, publish
from transactions import Step, Transactions


class DirectTransactionTests(test_transactions.TransactionIntegrationTests):
    def controller(self, capacity=64):
        pool, ms, life = test_transactions.setup(capacity)
        return pool, ms, DirectTransactions(life)


class DifferentialTests(unittest.TestCase):
    def test_seeded_projection_matches_real_shadow_mutations(self):
        accepted = rejected = 0
        for seed in range(200):
            rng = random.Random(seed)
            pool,ms = fixture(capacity=rng.choice((24,48,96)),
                              bs=rng.choice((4,8)), window=rng.choice((8,16)),
                              extra=rng.choice((0,3)),pool_class=EpochBlockPool)
            tx=Transactions(Lifecycle(ms,6144))
            tx.commit(tx.prepare(1,[Step('parent',32,24)]));tx.complete(1)
            for m in ms:
                publish(m,'parent',24)
            computed=rng.choice((0,8,15,16,23,24))
            hits=tuple(tuple(b.block_id for b in m.req_to_blocks['parent'][:
                             (computed+m.block_size-1)//m.block_size]) for m in ms)
            if rng.choice((False,True)):
                tx.cancel('parent')
            steps=tuple(Step(f'c{i}',rng.choice((32,48,64)),rng.choice((24,28,32)),
                             lookahead=rng.choice((0,4,8)),computed=computed,
                             hits=hits if computed else ()) for i in range(rng.randint(1,6)))
            extra=rng.choice((0,0,2,8,30))
            before=tx._fingerprint()
            try:
                expected=tx.prepare(2,steps,extra)
            except ValueError:
                rejected+=1
                with self.assertRaises(ValueError,msg=f'seed={seed}'):
                    project(tx,2,steps,extra)
            else:
                accepted+=1
                self.assertEqual(project(tx,2,steps,extra),expected.charge,f'seed={seed}')
                self.assertEqual(tx._fingerprint(),before)
                actual=tx.commit(expected)
                self.assertEqual(actual,expected.charge)
            if seed == 199:
                self.assertGreater(accepted,20)
                self.assertGreater(rejected,20)

    def test_direct_prepare_does_not_use_deepcopy(self):
        from unittest.mock import patch
        pool,ms=fixture(pool_class=EpochBlockPool)
        tx=DirectTransactions(Lifecycle(ms,6144))
        with patch('transactions.deepcopy',side_effect=AssertionError('copy forbidden')):
            plan=tx.prepare(1,[Step('a',32,16)])
            tx.commit(plan)

    def test_large_pool_prepare_and_commit_without_recursive_serialization(self):
        pool,ms=fixture(capacity=4096,pool_class=EpochBlockPool)
        tx=DirectTransactions(Lifecycle(ms,6144))
        charge=tx.commit(tx.prepare(1,[Step('a',64,32)]))
        self.assertEqual(charge.total,24)
        tx.complete(1);tx.cancel('a')
        self.assertEqual(pool.get_num_free_blocks(),4095)

    def test_new_swa_hit_with_skipped_null_placeholders(self):
        pool,ms=fixture(capacity=128,pool_class=EpochBlockPool)
        tx=Transactions(Lifecycle(ms,6144))
        tx.commit(tx.prepare(1,[Step('parent',32,24)]));tx.complete(1)
        for m in ms:
            publish(m,'parent',24)
        tx.reclaim('parent',24)
        hits=tuple(tuple(b.block_id for b in m.req_to_blocks['parent']) for m in ms)
        steps=(Step('child',48,28,computed=24,hits=hits),)
        expected=tx.prepare(2,steps)
        self.assertEqual(project(tx,2,steps),expected.charge)
        tx.commit(expected)
