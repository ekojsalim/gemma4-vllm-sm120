"""CPU lifecycle prototype around real allocators, not installed runtime hooks.

Single scheduler-thread ownership is required. A batch token identifies a worker
completion fence, not a request. Transfer existing references; never synthesize
an additional refcount model or touch blocks merely to record ownership.
"""
from collections import Counter
from dataclasses import dataclass

from vllm.v1.core.block_pool import BlockPool
from vllm_adapter import observe


class EpochBlockPool(BlockPool):
    """Prototype allocation hook: only storage reassignment advances identity.

    Cache touches, release to the idle cache and hash publication preserve the
    epoch. Install at construction, before managers allocate any blocks.
    """
    def __init__(self, *args, **kwargs):
        self.epochs = {}
        super().__init__(*args, **kwargs)

    def get_new_blocks(self, num_blocks):
        blocks = super().get_new_blocks(num_blocks)
        for block in blocks:
            self.epochs[block.block_id] = self.epochs.get(block.block_id, -1) + 1
        return blocks


@dataclass(frozen=True)
class Retained:
    group_id: int
    block: object
    generation: int


class Lifecycle:
    def __init__(self, managers, max_model_len, max_batches=2):
        if not managers or max_batches not in (1, 2):
            raise ValueError('invalid lifecycle configuration')
        self.managers = tuple(managers)
        self.pool = managers[0].block_pool
        if not isinstance(self.pool, EpochBlockPool):
            raise ValueError('allocation epochs must be tracked from construction')
        self.max_model_len = max_model_len
        self.max_batches = max_batches
        self._leases = {}
        self._completed = set()
        self._deferred_requests = set()
        # Validate all supported geometry before any lifecycle mutation.
        self.observation({})

    def _check_batch(self, batch_id):
        if not isinstance(batch_id, int) or batch_id < 0:
            raise ValueError('invalid batch ID')
        if batch_id in self._completed:
            raise ValueError('batch already completed')
        if batch_id not in self._leases and len(self._leases) >= self.max_batches:
            raise ValueError('too many outstanding batches')

    def _retain(self, batch_id, refs):
        # Called only after validating batch and allocator-owned endpoints.
        self._leases.setdefault(batch_id, []).extend(
            Retained(gid, b, self.pool.epochs[b.block_id])
            for gid, b in refs if not b.is_null)

    def collect_copies(self, batch_id):
        """Transfer existing CoW refs to their worker-completion fence."""
        self._check_batch(batch_id)
        copies = []
        for manager in self.managers:
            pending = manager.take_pending_cow_copies()
            self._retain(batch_id, [(manager.kv_cache_group_id, b)
                                    for pair in pending for b in pair])
            copies.extend(pending)
        return copies

    def defer_request(self, batch_id, request_id):
        """Detach request tables while preserving refs until the fence completes.

        Caller must choose the last worker step that can still access the request.
        This CPU prototype cannot infer device completion or ordering.
        """
        self._check_batch(batch_id)
        if request_id in self._deferred_requests:
            raise ValueError('request already deferred')
        if not any(request_id in m.req_to_blocks for m in self.managers):
            raise ValueError('unknown request')
        for manager in self.managers:
            if request_id in manager.req_to_blocks:
                blocks = manager.pop_blocks_for_free(request_id)
                self._retain(batch_id, [(manager.kv_cache_group_id, b) for b in blocks])
        self._deferred_requests.add(request_id)

    def observation(self, extents):
        leases = {}
        for bid, records in self._leases.items():
            leases[bid] = []
            for r in records:
                if self.pool.epochs.get(r.block.block_id) != r.generation:
                    raise ValueError('retained block was reused before completion')
                leases[bid].append((r.group_id, r.block))
        return observe(self.managers, extents, self.max_model_len,
                       self.pool.epochs, leases=leases)

    def complete(self, batch_id, extents):
        """Validate all ownership before releasing exactly this fence's refs."""
        if batch_id not in self._leases:
            raise ValueError('unknown or already completed batch')
        self.observation(extents) # strict ref conservation before mutation
        records = self._leases[batch_id]
        counts = Counter(r.block.block_id for r in records)
        if any(self.pool.blocks[i].ref_cnt < n for i, n in counts.items()):
            raise ValueError('retained references were released outside lifecycle')
        self.pool.free_blocks([r.block for r in records])
        del self._leases[batch_id]
        self._completed.add(batch_id)
