"""CPU scheduler transaction reference. Metadata copies are NOT a serving hot path.

Uses real manager mutations on a shadow state, then replays a validated batch on
unchanged live state. Single-threaded ownership and ordered worker completions
are required. Unexpected commit-time allocator exceptions are fatal, not rolled
back. Do not install this harness in a production scheduler.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import pickle

from model import StalePlan
from vllm.v1.core.kv_cache_utils import get_group_id


@dataclass(frozen=True)
class Step:
    request_id: str
    extent: int
    main_tokens: int
    lookahead: int = 0
    computed: int = 0
    # Per manager, actual upstream lookup block IDs, including null placeholders.
    hits: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class Prepared:
    revision: int
    baseline: bytes
    batch_id: int
    steps: tuple[Step, ...]
    extra: int
    charge: object


class Transactions:
    def __init__(self, lifecycle):
        self.life = lifecycle
        self.extents = {}
        self.seen = set()
        self.inflight = {} # insertion order is worker submission order
        self.main_tokens = {}
        self.revision = 0
        self._plans = {}

    def _fingerprint(self):
        # Full metadata graph detects out-of-band publication, free-queue
        # changes, touches and ABA reuse, not just transaction revision changes.
        state = (self.life, self.extents, self.seen, self.inflight, self.main_tokens)
        return sha256(pickle.dumps(state, protocol=5)).digest()

    def _apply(self, batch_id, steps, extra):
        if batch_id in self.inflight or batch_id in self.life._leases:
            raise ValueError('batch already submitted')
        self.life._check_batch(batch_id)
        if not steps or len({s.request_id for s in steps}) != len(steps):
            raise ValueError('batch needs unique requests')
        if not isinstance(extra, int) or extra < 0:
            raise ValueError('invalid transient reservation')
        ms = self.life.managers
        new = []
        for s in steps:
            if not s.request_id or not 0 < s.main_tokens <= s.extent <= self.life.max_model_len:
                raise ValueError('invalid hard extent or scheduled target extent')
            if s.lookahead < 0 or s.main_tokens + s.lookahead > self.life.max_model_len:
                raise ValueError('invalid lookahead extent')
            if s.request_id in self.extents:
                if s.extent != self.extents[s.request_id]:
                    raise ValueError('hard reservation cannot change')
                if s.hits or s.computed or s.main_tokens < self.main_tokens[s.request_id]:
                    raise ValueError('running request cannot adopt hits or regress')
            else:
                if s.request_id in self.seen:
                    raise ValueError('request ID reused')
                if not 0 <= s.computed <= s.main_tokens:
                    raise ValueError('invalid computed boundary')
                hits = s.hits or tuple(() for _ in ms)
                if len(hits) != len(ms):
                    raise ValueError('one hit list per group required')
                for m, ids in zip(ms, hits):
                    if len(ids) != (s.computed + m.block_size - 1) // m.block_size:
                        raise ValueError('hit list does not cover computed boundary')
                    for bid in ids:
                        if not 0 <= bid < len(self.life.pool.blocks):
                            raise ValueError('invalid hit block')
                        b = self.life.pool.blocks[bid]
                        if not b.is_null:
                            if b.block_hash is None:
                                raise ValueError('hit is not cached')
                            if get_group_id(b.block_hash) != m.kv_cache_group_id:
                                raise ValueError('hit belongs to another cache group')
                new.append((s, hits))
                self.extents[s.request_id] = s.extent
                self.seen.add(s.request_id)
        # Protect EVERY hit before allocating, including other requests/groups.
        for s, hits in new:
            for m, ids in zip(ms, hits):
                blocks = [self.life.pool.blocks[i] for i in ids]
                m.add_local_computed_blocks(s.request_id, blocks, s.computed, 0)
        for s in steps:
            for m in ms:
                m.allocate_new_blocks(s.request_id, s.main_tokens + s.lookahead, s.main_tokens)
            self.main_tokens[s.request_id] = s.main_tokens
        self.life.collect_copies(batch_id)
        self.life.reserve_extra(batch_id, extra)
        self.inflight[batch_id] = tuple(s.request_id for s in steps)
        return self.life.observation(self.extents).charge()

    def prepare(self, batch_id, steps, unallocated_extra=0):
        steps = tuple(steps)
        baseline = self._fingerprint()
        # Do not clone prepared plans; they are capabilities of this controller.
        shadow = Transactions(deepcopy(self.life))
        shadow.extents = dict(self.extents)
        shadow.seen = set(self.seen)
        shadow.inflight = dict(self.inflight)
        shadow.main_tokens = dict(self.main_tokens)
        charge = shadow._apply(batch_id, steps, unallocated_extra)
        plan = Prepared(self.revision, baseline, batch_id, steps, unallocated_extra, charge)
        self._plans[id(plan)] = plan
        return plan

    def commit(self, plan):
        if self._plans.get(id(plan)) is not plan:
            raise StalePlan('unknown or consumed plan')
        if plan.revision != self.revision or plan.baseline != self._fingerprint():
            raise StalePlan('allocator changed since prepare')
        charge = self._apply(plan.batch_id, plan.steps, plan.extra)
        if charge != plan.charge:
            raise RuntimeError('shadow/live allocator mismatch; stop scheduling')
        self.revision += 1
        self._plans.clear()
        return charge

    def cancel(self, request_id):
        if request_id not in self.extents:
            raise ValueError('unknown request')
        batches = [bid for bid, ids in self.inflight.items() if request_id in ids]
        if batches:
            self.life.defer_request(batches[-1], request_id)
        else:
            for m in self.life.managers:
                m.free(request_id)
        del self.extents[request_id]
        del self.main_tokens[request_id]
        self.revision += 1
        self._plans.clear()
        return self.life.observation(self.extents).charge()

    def complete(self, batch_id):
        if not self.inflight or next(iter(self.inflight)) != batch_id:
            raise ValueError('completion must follow worker submission order')
        self.life.complete(batch_id, self.extents)
        del self.inflight[batch_id]
        self.revision += 1
        self._plans.clear()
        return self.life.observation(self.extents).charge()

    def reclaim(self, request_id, processed_tokens):
        if request_id not in self.extents or any(request_id in ids for ids in self.inflight.values()):
            raise ValueError('reclaim requires a live request with no outstanding worker reads')
        if not 0 <= processed_tokens <= self.main_tokens[request_id]:
            raise ValueError('invalid processed boundary')
        for m in self.life.managers:
            m.remove_skipped_blocks(request_id, processed_tokens)
        self.revision += 1
        self._plans.clear()
        return self.life.observation(self.extents).charge()
