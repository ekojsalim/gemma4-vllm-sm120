"""CPU lifecycle integration tests using actual pinned allocator methods."""
import importlib.util
import random
import unittest

if importlib.util.find_spec('vllm') is None:
    raise unittest.SkipTest('lifecycle tests require the pinned vLLM image')

from lifecycle import EpochBlockPool, Lifecycle
from test_vllm_managers import fixture, allocate, publish


def setup(capacity=64):
    pool, managers = fixture(capacity=capacity, pool_class=EpochBlockPool)
    return pool, managers, Lifecycle(managers, 6144)


class LifecycleTests(unittest.TestCase):
    def test_idle_touch_preserves_epoch_reallocation_advances(self):
        pool, ms, life = setup(capacity=5)
        allocate(ms[0], 'a', 8); publish(ms[0], 'a', 8)
        hits = list(ms[0].req_to_blocks['a'])
        original = {b.block_id: pool.epochs[b.block_id] for b in hits}
        ms[0].free('a')
        allocate(ms[0], 'b', 8, hits, 8)
        self.assertEqual(original, {b.block_id: pool.epochs[b.block_id] for b in hits})
        life.observation({'b': 8}).charge()
        ms[0].free('b')
        allocate(ms[0], 'pressure', 16)
        for bid, epoch in original.items():
            self.assertEqual(pool.epochs[bid], epoch + 1)
        ms[0].free('pressure')
        self.assertEqual(life.observation({}).charge().total, 0)

    def test_failed_allocation_does_not_change_epochs_or_free_queue(self):
        pool, ms, life = setup(capacity=5)
        before = dict(pool.epochs), pool.get_num_free_blocks()
        with self.assertRaises(ValueError):
            pool.get_new_blocks(5)
        self.assertEqual(before, (pool.epochs, pool.get_num_free_blocks()))
        self.assertEqual(life.observation({}).charge().total, 0)

    def test_copy_then_cancel_same_fence_preserves_multiplicity(self):
        pool, ms, life = setup(); m = ms[0]
        allocate(m, 'a', 8); publish(m, 'a', 8)
        allocate(m, 'b', 8, list(m.req_to_blocks['a']), 6)
        copies = life.collect_copies(1)
        self.assertEqual(len(copies), 1)
        life.defer_request(1, 'b')
        life.defer_request(1, 'a')
        # The copy source/destination and request refs overlap physically.
        self.assertEqual(life.observation({}).charge().pinned, 3)
        life.complete(1, {})
        self.assertEqual(life.observation({}).charge().total, 0)
        self.assertEqual(pool.get_num_free_blocks(), len(pool.blocks) - 1)
        before = pool.get_num_free_blocks()
        with self.assertRaises(ValueError):
            life.complete(1, {})
        with self.assertRaises(ValueError):
            life.collect_copies(1)
        self.assertEqual(pool.get_num_free_blocks(), before)

    def test_cross_group_deferred_release_and_out_of_order_completion(self):
        pool, ms, life = setup()
        for m in ms:
            allocate(m, 'a', 8); publish(m, 'a', 8)
            allocate(m, 'b', 8, list(m.req_to_blocks['a']), 8)
        life.defer_request(10, 'a')
        life.defer_request(11, 'b')
        self.assertEqual(life.observation({}).charge().pinned, 4)
        life.complete(11, {})
        self.assertEqual(life.observation({}).charge().pinned, 4)
        life.complete(10, {})
        self.assertEqual(life.observation({}).charge().total, 0)

    def test_rejected_third_fence_leaves_request_attached(self):
        pool, ms, life = setup()
        for rid in ('a', 'b', 'c'):
            allocate(ms[0], rid, 4)
        life.defer_request(1, 'a'); life.defer_request(2, 'b')
        before = life.observation({'c': 4})
        with self.assertRaisesRegex(ValueError, 'too many'):
            life.defer_request(3, 'c')
        self.assertEqual(life.observation({'c': 4}), before)
        life.complete(1, {'c': 4})
        life.defer_request(3, 'c')
        life.complete(2, {}); life.complete(3, {})
        self.assertEqual(life.observation({}).charge().total, 0)

    def test_rejected_copy_fence_does_not_drain_pending_copies(self):
        pool, ms, life = setup(); m = ms[0]
        life.collect_copies(1); life.collect_copies(2)
        allocate(m, 'a', 8); publish(m, 'a', 8)
        allocate(m, 'b', 8, list(m.req_to_blocks['a']), 6)
        before = [b.ref_cnt for b in pool.blocks]
        with self.assertRaisesRegex(ValueError, 'too many'):
            life.collect_copies(3)
        self.assertEqual(before, [b.ref_cnt for b in pool.blocks])
        # Existing fence can accept the pending copies after the rejected transfer.
        self.assertEqual(len(life.collect_copies(2)), 1)
        life.defer_request(2, 'a'); life.defer_request(2, 'b')
        with self.assertRaisesRegex(ValueError, 'already deferred'):
            life.defer_request(2, 'a')
        life.complete(1, {}); life.complete(2, {})
        self.assertEqual(life.observation({}).charge().total, 0)

    def test_external_release_and_reuse_detected_before_completion(self):
        pool, ms, life = setup(capacity=3)
        allocate(ms[0], 'a', 4)
        block = ms[0].req_to_blocks['a'][0]
        life.defer_request(1, 'a')
        pool.free_blocks([block]) # deliberately simulate incorrect lifecycle integration
        with self.assertRaisesRegex(ValueError, 'unexplained reference'):
            life.complete(1, {})
        pool.get_new_blocks(2)
        with self.assertRaisesRegex(ValueError, 'reused before completion'):
            life.complete(1, {})

    def test_randomized_real_manager_fanout_and_drain(self):
        for seed in range(30):
            rng = random.Random(seed)
            pool, ms, life = setup(capacity=128)
            for m in ms:
                allocate(m, 'parent', 8); publish(m, 'parent', 8)
            extents = {'parent': 16}
            for i in range(4):
                rid = f'child-{i}'
                for m in ms:
                    allocate(m, rid, 12, list(m.req_to_blocks['parent']), 8)
                extents[rid] = 16
                life.observation(extents).charge()
            order = list(extents); rng.shuffle(order)
            for i, rid in enumerate(order):
                for m in ms:
                    allocate(m, rid, 16)
                    if m is ms[1]:
                        m.remove_skipped_blocks(rid, 16)
                life.observation(extents).charge()
                life.defer_request(i, rid); del extents[rid]
                life.observation(extents).charge()
                life.complete(i, extents)
            self.assertEqual(life.observation({}).charge().total, 0)
            self.assertEqual(pool.get_num_free_blocks(), len(pool.blocks) - 1)
