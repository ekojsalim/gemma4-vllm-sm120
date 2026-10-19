"""Prepare/commit and worker lifetime tests against real allocator metadata."""
import importlib.util
import unittest

if importlib.util.find_spec('vllm') is None:
    raise unittest.SkipTest('transaction tests require the pinned vLLM image')

from model import CapacityError, StalePlan
from test_lifecycle import setup
from test_vllm_managers import publish
from transactions import Step, Transactions


class TransactionIntegrationTests(unittest.TestCase):
    def controller(self, capacity=64):
        pool, ms, life = setup(capacity)
        return pool, ms, Transactions(life)

    def submit(self, tx, bid, *steps, extra=0):
        return tx.commit(tx.prepare(bid, steps, extra))

    def test_prepare_has_no_live_allocator_effect_and_commit_once(self):
        pool, ms, tx = self.controller()
        before = tx._fingerprint()
        plan = tx.prepare(1, [Step('a',16,8)])
        self.assertEqual(tx._fingerprint(), before)
        self.assertEqual(tx.commit(plan).total,7)
        with self.assertRaises(StalePlan):
            tx.commit(plan)
        tx.complete(1); tx.cancel('a')
        self.assertEqual(tx.life.observation({}).charge().total,0)

    def test_prefill_transient_rejection_does_not_mutate_live_state(self):
        pool, ms, tx = self.controller(capacity=14)
        before = tx._fingerprint()
        # Persistent bound=11, but this prefill needs charge=14; usable pool=13.
        with self.assertRaises(CapacityError):
            tx.prepare(1,[Step('a',32,24)])
        self.assertEqual(tx._fingerprint(),before)
        self.assertEqual(self.submit(tx,1,Step('a',32,8)).total,11)

    def test_lookahead_slots_are_charged_beyond_hard_extent(self):
        pool, ms, tx = self.controller(capacity=8)
        before=tx._fingerprint()
        with self.assertRaises(ValueError):
            tx.prepare(1,[Step('a',8,8,lookahead=8)])
        self.assertEqual(tx._fingerprint(),before)
        self.assertEqual(self.submit(tx,1,Step('a',8,8)).total,4)

    def test_overlapping_extras_held_until_each_completion(self):
        pool,ms,tx=self.controller(capacity=15)
        self.assertEqual(self.submit(tx,1,Step('a',16,8),extra=3).total,10)
        self.assertEqual(self.submit(tx,2,Step('a',16,12),extra=4).total,14)
        before=tx._fingerprint()
        with self.assertRaises(ValueError):
            tx.complete(2)
        self.assertEqual(tx._fingerprint(),before)
        self.assertEqual(tx.complete(1).total,11)
        self.assertEqual(tx.complete(2).total,7)

    def test_cancel_waits_for_last_submitted_worker_use(self):
        pool,ms,tx=self.controller()
        self.submit(tx,1,Step('a',16,8))
        self.submit(tx,2,Step('a',16,12))
        self.assertEqual(tx.cancel('a').pinned,6)
        self.assertEqual(tx.complete(1).pinned,6)
        self.assertEqual(tx.complete(2).total,0)
        self.assertEqual(pool.get_num_free_blocks(),len(pool.blocks)-1)

    def test_sibling_plans_and_out_of_band_mutation_stale(self):
        pool,ms,tx=self.controller()
        p=tx.prepare(1,[Step('a',8,4)])
        q=tx.prepare(2,[Step('b',8,4)])
        tx.commit(p)
        with self.assertRaises(StalePlan):
            tx.commit(q)
        q=tx.prepare(2,[Step('b',8,4)])
        block=pool.get_new_blocks(1)[0]
        pool.free_blocks([block]) # same occupancy, different epoch/free ordering
        with self.assertRaises(StalePlan):
            tx.commit(q)

    def test_idle_hits_survive_other_group_allocations(self):
        pool,ms,tx=self.controller(capacity=20)
        self.submit(tx,1,Step('parent',8,8));tx.complete(1)
        for m in ms:
            publish(m,'parent',8)
        hits=tuple(tuple(b.block_id for b in m.req_to_blocks['parent']) for m in ms)
        tx.cancel('parent')
        charge=self.submit(tx,2,Step('a',16,12,computed=8,hits=hits),
                           Step('b',16,12,computed=8,hits=hits))
        self.assertEqual(charge.pinned,8) # four shared blocks, four private tails
        for m in ms:
            self.assertEqual(m.req_to_blocks['a'][:2],m.req_to_blocks['b'][:2])
        tx.cancel('a');tx.cancel('b');tx.complete(2)
        self.assertEqual(tx.life.observation({}).charge().total,0)

    def test_partial_hit_copies_and_cancel_complete_in_one_transaction(self):
        pool,ms,tx=self.controller()
        self.submit(tx,1,Step('parent',8,8));tx.complete(1)
        for m in ms:
            publish(m,'parent',8)
        hits=tuple(tuple(b.block_id for b in m.req_to_blocks['parent']) for m in ms)
        tx.cancel('parent')
        self.submit(tx,2,Step('child',16,8,computed=6,hits=hits))
        tx.cancel('child')
        self.assertGreater(tx.life.observation({}).charge().pinned,0)
        tx.complete(2)
        self.assertEqual(tx.life.observation({}).charge().total,0)

    def test_reclaim_waits_for_completion_and_allows_bounded_progress(self):
        pool,ms,tx=self.controller(capacity=20)
        self.submit(tx,1,Step('a',32,16))
        with self.assertRaises(ValueError):
            tx.reclaim('a',16)
        tx.complete(1)
        tx.reclaim('a',16)
        for bid,end in ((2,24),(3,32)):
            self.submit(tx,bid,Step('a',32,end))
            tx.complete(bid);tx.reclaim('a',end)
        tx.cancel('a')
        self.assertEqual(tx.life.observation({}).charge().total,0)

    def test_failed_multi_request_batch_is_atomic(self):
        pool,ms,tx=self.controller(capacity=10)
        before=tx._fingerprint()
        with self.assertRaises((ValueError,CapacityError)):
            tx.prepare(1,[Step('a',16,8),Step('b',16,8)])
        self.assertEqual(tx._fingerprint(),before)
        self.assertFalse(tx.extents)

    def test_hard_extent_cannot_change(self):
        pool,ms,tx=self.controller()
        self.submit(tx,1,Step('a',16,8));tx.complete(1)
        before=tx._fingerprint()
        with self.assertRaisesRegex(ValueError,'hard reservation'):
            tx.prepare(2,[Step('a',20,12)])
        self.assertEqual(tx._fingerprint(),before)

    def test_wrong_group_hit_rejected_before_live_mutation(self):
        pool,ms,tx=self.controller()
        self.submit(tx,1,Step('parent',8,8));tx.complete(1)
        for m in ms:
            publish(m,'parent',8)
        hits=tuple(tuple(b.block_id for b in m.req_to_blocks['parent']) for m in ms)
        before=tx._fingerprint()
        with self.assertRaisesRegex(ValueError,'another cache group'):
            tx.prepare(2,[Step('child',16,12,computed=8,hits=hits[::-1])])
        self.assertEqual(tx._fingerprint(),before)

    def test_transient_extra_rejection_preserves_other_reservations(self):
        pool,ms,tx=self.controller(capacity=12)
        self.submit(tx,1,Step('a',16,8),extra=2)
        before=tx._fingerprint()
        with self.assertRaises(CapacityError):
            tx.prepare(2,[Step('a',16,12)],unallocated_extra=3)
        self.assertEqual(tx._fingerprint(),before)
        tx.complete(1)
        self.assertEqual(self.submit(tx,2,Step('a',16,12),extra=3).total,10)
