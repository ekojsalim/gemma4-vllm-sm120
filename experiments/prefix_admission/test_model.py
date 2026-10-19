import random
import unittest
from dataclasses import replace

from model import Batch, Block, CapacityError, Handle, Ledger, Request, Snapshot, StalePlan, account


def block(slot, group='full', owner=None, durable=False, generation=0):
    return Block(Handle(slot, generation), group, owner, durable)


def request(name, blocks=(), full=4, swa=2):
    return Request(name, (('full', full), ('swa', swa)), frozenset(b.handle for b in blocks))


class AccountingTests(unittest.TestCase):
    def test_fanout_credits_full_prefix_once_but_keeps_private_growth(self):
        prefix = tuple(block(i, durable=True) for i in range(100))
        state = Snapshot(prefix, tuple(request(n, prefix, full=120, swa=0) for n in ('a','b')))
        charge = account(state, 140)
        self.assertEqual((charge.pinned, charge.future, charge.total), (100, 40, 140))
        with self.assertRaises(CapacityError):
            account(state, 139)

    def test_swa_prefix_does_not_discount_future_replacements(self):
        prefix = (block(0, 'swa'), block(1, 'swa'))
        state = Snapshot(prefix, (request('a', prefix, full=0), request('b', prefix, full=0)))
        self.assertEqual(account(state, 6).total, 6)
        with self.assertRaises(ValueError):
            account(replace(state, blocks=(replace(prefix[0], lifetime_prefix=True), prefix[1])), 6)

    def test_no_credit_moves_between_groups(self):
        prefix = tuple(block(i, durable=True) for i in range(4))
        state = Snapshot(prefix, (request('a', prefix, full=2, swa=2),))
        self.assertEqual(account(state, 6).total, 6)

    def test_idle_cached_blocks_are_evictable_not_charged(self):
        blocks = tuple(block(i, durable=True) for i in range(8))
        empty = Snapshot(blocks)
        self.assertEqual(account(empty, 8).total, 0)
        active = replace(empty, requests=(request('a', blocks[:2], full=4, swa=0),))
        self.assertEqual(account(active, 8).total, 4)

    def test_private_tail_cannot_be_shared(self):
        tail = block(0, owner='a')
        with self.assertRaises(ValueError):
            account(Snapshot((tail,), (request('a', (tail,)), request('b', (tail,)))), 20)

    def test_prefix_blocks_are_group_and_slot_validated(self):
        for blocks in [(block(0), block(0, generation=1)), (block(9),), (block(0, 'unknown'),)]:
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                account(Snapshot(blocks), 8)

    def test_transient_physical_excess_is_charged(self):
        held = tuple(block(i, owner='a') for i in range(6))
        state = Snapshot(held, (request('a', held, full=4, swa=0),))
        self.assertEqual(account(state, 8).total, 6)

    def test_overlapping_batches_pin_union_and_reserve_extra(self):
        b = block(0, owner='a')
        state = Snapshot((b,), (), (Batch(1, frozenset({b.handle}), 2), Batch(2, frozenset({b.handle}), 3)))
        charge = account(state, 6)
        self.assertEqual((charge.pinned, charge.future, charge.unallocated_extra), (1, 0, 5))

    def test_no_cache_baseline(self):
        self.assertEqual(account(Snapshot(requests=(request('a'),request('b'))), 12).total, 12)

    def test_unreferenced_private_storage_is_not_evictable_cache(self):
        with self.assertRaises(ValueError):
            account(Snapshot((block(0,owner='a'),)), 8)

    def test_partial_hit_copy_has_a_private_growth_charge(self):
        prefix = block(0,durable=True)
        tail = block(1,owner='a')
        copied = block(2,owner='b')
        state = Snapshot((prefix,tail,copied), (
            request('a',(prefix,tail),full=3,swa=0),
            request('b',(prefix,copied),full=3,swa=0)))
        self.assertEqual(account(state,5).total,5)


class TransactionTests(unittest.TestCase):
    def test_rejected_admission_rolls_back_and_stale_plan_rejected(self):
        ledger = Ledger(6)
        initial = Snapshot(requests=(request('a'),))
        ledger.replace(initial, expected_revision=0)
        with self.assertRaises(CapacityError):
            ledger.replace(replace(initial, requests=initial.requests+(request('b'),)), expected_revision=1)
        self.assertEqual(ledger.snapshot, initial)
        self.assertEqual(ledger.revision, 1)
        with self.assertRaises(StalePlan):
            ledger.replace(Snapshot(), expected_revision=0)

    def test_cancellation_retains_gpu_pins_until_both_batches_complete(self):
        ledger = Ledger(8)
        b = block(0, owner='a')
        batches = (Batch(1, frozenset({b.handle})), Batch(2, frozenset({b.handle})))
        ledger.replace(Snapshot((b,), (request('a',(b,)),), batches), expected_revision=0)
        ledger.replace(Snapshot((b,), (), batches), expected_revision=1)
        self.assertEqual(ledger.charge.total, 1)
        with self.assertRaises(ValueError):
            ledger.replace(Snapshot(), expected_revision=2)
        ledger.replace(Snapshot((b,), (), (batches[0],)), expected_revision=2, completed_batches=frozenset({2}))
        self.assertEqual(ledger.charge.total, 1)
        ledger.replace(Snapshot(), expected_revision=3, completed_batches=frozenset({1}))
        self.assertEqual(ledger.charge.total, 0)

    def test_lifetime_credit_cannot_be_silently_reclaimed(self):
        ledger = Ledger(10)
        p = block(0, durable=True)
        ledger.replace(Snapshot((p,), (request('a',(p,)),)), expected_revision=0)
        with self.assertRaises(ValueError):
            ledger.replace(Snapshot((p,), (request('a'),)), expected_revision=1)
        with self.assertRaises(ValueError):
            ledger.replace(Snapshot((p,), (request('a',(p,),full=3),)), expected_revision=1)

    def test_reused_slot_rejects_old_generation_and_stale_handles(self):
        ledger = Ledger(10)
        p = block(0, durable=True)
        ledger.replace(Snapshot((p,)), expected_revision=0)
        ledger.replace(Snapshot(), expected_revision=1)
        with self.assertRaises(ValueError):
            ledger.replace(Snapshot((p,)), expected_revision=2)
        fresh = block(0, durable=True, generation=1)
        ledger.replace(Snapshot((fresh,)), expected_revision=2)
        with self.assertRaises(ValueError):
            ledger.replace(Snapshot((fresh,), (request('a',(p,)),)), expected_revision=3)

    def test_live_block_metadata_cannot_be_reinterpreted(self):
        ledger = Ledger(10)
        p = block(0)
        ledger.replace(Snapshot((p,)), expected_revision=0)
        with self.assertRaises(ValueError):
            ledger.replace(Snapshot((replace(p,lifetime_prefix=True),)), expected_revision=1)

    def test_cold_prefill_publishes_only_after_completion_then_fans_out(self):
        ledger=Ledger(6)
        private=tuple(block(i,owner='warm') for i in range(2))
        handles=frozenset(b.handle for b in private)
        batch=Batch(1,handles)
        ledger.replace(Snapshot(private,(request('warm',private,full=2,swa=0),),(batch,)),
                       expected_revision=0)
        cached=tuple(replace(b,private_owner=None,lifetime_prefix=True) for b in private)
        with self.assertRaises(ValueError):
            ledger.replace(Snapshot(cached,(),(batch,)),expected_revision=1,published_blocks=handles)
        ledger.replace(Snapshot(cached),expected_revision=1,
                       completed_batches=frozenset({1}),published_blocks=handles)
        self.assertEqual(ledger.charge.total,0)
        ledger.replace(Snapshot(cached,tuple(request(n,cached,full=4,swa=0) for n in ('a','b'))),
                       expected_revision=2)
        self.assertEqual(ledger.charge.total,6)

    def test_duplicate_batch_completion_and_id_reuse(self):
        ledger = Ledger(10)
        ledger.replace(Snapshot(batches=(Batch(1),)), expected_revision=0)
        ledger.replace(Snapshot(), expected_revision=1, completed_batches=frozenset({1}))
        with self.assertRaises(ValueError):
            ledger.replace(Snapshot(), expected_revision=2, completed_batches=frozenset({1}))
        with self.assertRaises(ValueError):
            ledger.replace(Snapshot(batches=(Batch(1),)), expected_revision=2)

    def test_swa_replacement_and_reclamation_preserve_capacity(self):
        ledger=Ledger(4)
        prefix=(block(0,'swa'),block(1,'swa'))
        ledger.replace(Snapshot(prefix,(request('a',prefix,full=0),)),expected_revision=0)
        private=(block(2,'swa','a'),block(3,'swa','a'))
        ledger.replace(Snapshot(prefix+private,(request('a',private,full=0),)),expected_revision=1)
        self.assertEqual(ledger.charge.total,2)
        # Old shared SWA entries remain idle/evictable; new private pages consume
        # the two reserved slots. Replacement does not lose a lifetime discount.

    def test_new_chapter_waits_for_canceled_batch_to_drain(self):
        ledger=Ledger(6)
        held=tuple(block(i,owner='old') for i in range(4))
        batch=Batch(1,frozenset(b.handle for b in held))
        ledger.replace(Snapshot(held,(),(batch,)),expected_revision=0)
        with self.assertRaises(CapacityError):
            ledger.replace(Snapshot(held,(request('new'),),(batch,)),expected_revision=1)
        ledger.replace(Snapshot(requests=(request('new'),)),expected_revision=1,
                       completed_batches=frozenset({1}))
        self.assertEqual(ledger.charge.total,6)

    def test_randomized_fanout_can_reach_every_hard_limit_then_drain(self):
        # Constructively realize all independent future blocks, in random order.
        # If the admission discount was unsafe, these transitions would exceed
        # the initial exact capacity. This is not a vLLM allocator integration test.
        for seed in range(100):
            rng = random.Random(seed)
            n = rng.randint(1, 8)
            prefix = (block(0,durable=True), block(1,durable=True), block(2,'swa'), block(3,'swa'))
            ledger = Ledger(4+4*n)
            requests = [request(str(i),prefix) for i in range(n)]
            blocks = list(prefix)
            ledger.replace(Snapshot(tuple(blocks),tuple(requests)),expected_revision=0)
            self.assertEqual(ledger.charge.total, ledger.capacity)
            growth = [(i,g) for i in range(n) for g in ('full','full','swa','swa')]
            rng.shuffle(growth)
            for i,g in growth:
                b = block(len(blocks),g,str(i))
                blocks.append(b)
                requests[i] = replace(requests[i],held=requests[i].held|{b.handle})
                ledger.replace(Snapshot(tuple(blocks),tuple(requests)),expected_revision=ledger.revision)
                self.assertLessEqual(ledger.charge.total,ledger.capacity)
            self.assertEqual(ledger.charge.future,0)
            rng.shuffle(requests)
            while requests:
                retired=requests.pop().request_id
                blocks=[b for b in blocks if b.private_owner!=retired]
                ledger.replace(Snapshot(tuple(blocks),tuple(requests)),expected_revision=ledger.revision)
            self.assertEqual(ledger.charge.total,0)


if __name__ == '__main__':
    unittest.main()
