"""CPU specification, deliberately not imported by the runtime overlay.

Inputs describe authoritative allocator ownership at a transaction boundary.
This model neither hashes prefixes nor calculates vLLM's allocation bounds.
"""
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Handle:
    slot: int
    generation: int


@dataclass(frozen=True)
class Block:
    handle: Handle
    group: str
    private_owner: str | None = None
    lifetime_prefix: bool = False


@dataclass(frozen=True)
class Request:
    request_id: str
    budgets: tuple[tuple[str, int], ...]
    held: frozenset[Handle] = frozenset()


@dataclass(frozen=True)
class Batch:
    batch_id: int
    pinned: frozenset[Handle] = frozenset()
    # Additional capacity NOT covered by the request lifetime budgets.
    unallocated_extra: int = 0


@dataclass(frozen=True)
class Snapshot:
    blocks: tuple[Block, ...] = ()
    requests: tuple[Request, ...] = ()
    batches: tuple[Batch, ...] = ()


@dataclass(frozen=True)
class Charge:
    pinned: int
    future: int
    unallocated_extra: int

    @property
    def total(self):
        return self.pinned + self.future + self.unallocated_extra


class CapacityError(ValueError):
    pass


class StalePlan(ValueError):
    pass


def account(snapshot, capacity, groups=frozenset({'full', 'swa'}),
            lifetime_groups=frozenset({'full'}), max_batches=2):
    """Count unique occupancy plus remaining independent obligations.

    A lifetime-prefix credit is an adapter assertion of retention, not something
    inferred from a cache hit. Shared SWA gets no such credit in this first model.
    """
    if capacity <= 0 or not lifetime_groups <= groups:
        raise ValueError('invalid pool policy')
    blocks = {b.handle: b for b in snapshot.blocks}
    if len(blocks) != len(snapshot.blocks):
        raise ValueError('duplicate block identity')
    if len({h.slot for h in blocks}) != len(blocks):
        raise ValueError('multiple generations occupy one physical slot')
    if any(h.slot < 0 or h.slot >= capacity or h.generation < 0 for h in blocks):
        raise ValueError('invalid physical handle')
    for b in blocks.values():
        if b.group not in groups:
            raise ValueError('unknown cache group')
        if b.lifetime_prefix and (b.group not in lifetime_groups or b.private_owner is not None):
            raise ValueError('invalid lifetime-prefix credit')
    ids = [r.request_id for r in snapshot.requests]
    if len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError('invalid request identity')
    batch_ids = [b.batch_id for b in snapshot.batches]
    if len(set(batch_ids)) != len(batch_ids) or len(batch_ids) > max_batches:
        raise ValueError('invalid in-flight batch set')
    pinned = set()
    future = 0
    for r in snapshot.requests:
        budgets = dict(r.budgets)
        if len(budgets) != len(r.budgets) or set(budgets) != groups:
            raise ValueError('one explicit budget per group required')
        if any(not isinstance(n, int) or n < 0 for n in budgets.values()):
            raise ValueError('invalid budget')
        credit = dict.fromkeys(groups, 0)
        for h in r.held:
            if h not in blocks:
                raise ValueError('request references missing/stale block')
            b = blocks[h]
            if b.private_owner is not None and b.private_owner != r.request_id:
                raise ValueError('writable private block shared by another request')
            if b.private_owner == r.request_id or b.lifetime_prefix:
                credit[b.group] += 1
        future += sum(max(0, n - credit[g]) for g, n in budgets.items())
        pinned.update(r.held)
    extras = 0
    for b in snapshot.batches:
        if not isinstance(b.unallocated_extra, int) or b.unallocated_extra < 0:
            raise ValueError('invalid pending allocation bound')
        if not b.pinned <= blocks.keys():
            raise ValueError('batch references missing/stale block')
        pinned.update(b.pinned)
        extras += b.unallocated_extra
    if any(b.private_owner is not None and h not in pinned for h, b in blocks.items()):
        raise ValueError('unreferenced private storage must be freed, not treated as idle cache')
    result = Charge(len(pinned), future, extras)
    if result.total > capacity:
        raise CapacityError(f'charge {result.total} exceeds capacity {capacity}')
    return result


class Ledger:
    """Atomic reference transitions; failed proposals leave all state untouched.

    GPU completion is explicit. Existing batches can only disappear when named
    in completed_batches; callers cannot silently drop their pins. Credits for
    retained prefixes may not disappear from a still-live request. More general
    request suspension/re-prefill is intentionally outside this model.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.snapshot = Snapshot()
        self.revision = 0
        self._generations = {}
        self._seen_batches = set()
        self._seen_requests = set()

    @property
    def charge(self):
        return account(self.snapshot, self.capacity)

    def replace(self, candidate, *, expected_revision, completed_batches=frozenset(),
                published_blocks=frozenset()):
        if expected_revision != self.revision:
            raise StalePlan('snapshot changed since planning')
        old_blocks = {b.handle: b for b in self.snapshot.blocks}
        new_blocks = {b.handle: b for b in candidate.blocks}
        old_requests = {r.request_id: r for r in self.snapshot.requests}
        new_requests = {r.request_id: r for r in candidate.requests}
        old_batches = {b.batch_id: b for b in self.snapshot.batches}
        new_batches = {b.batch_id: b for b in candidate.batches}
        if not completed_batches <= old_batches.keys():
            raise ValueError('completion of an unknown batch')
        if completed_batches & new_batches.keys():
            raise ValueError('completed batch is still present')
        if not published_blocks <= old_blocks.keys() & new_blocks.keys():
            raise ValueError('publishing a missing block')
        for bid, batch in old_batches.items():
            if bid not in completed_batches and new_batches.get(bid) != batch:
                raise ValueError('outstanding batch contract changed without completion')
        if (new_batches.keys() - old_batches.keys()) & self._seen_batches:
            raise ValueError('batch ID reused')
        if (new_requests.keys() - old_requests.keys()) & self._seen_requests:
            raise ValueError('request ID reused')
        for h, b in new_blocks.items():
            if h in old_blocks:
                if b != old_blocks[h]:
                    old = old_blocks[h]
                    if (h not in published_blocks or old.private_owner is None
                            or b.private_owner is not None or old.group != b.group):
                        raise ValueError('block identity metadata changed without publication')
                    if any(h in batch.pinned for bid, batch in old_batches.items()
                           if bid not in completed_batches):
                        raise ValueError('publishing before GPU users settle')
                elif h in published_blocks:
                    raise ValueError('publication must transition private to immutable')
            elif h.generation <= self._generations.get(h.slot, -1):
                raise ValueError('recycled slot needs a fresh generation')
        for rid in old_requests.keys() & new_requests.keys():
            old, new = old_requests[rid], new_requests[rid]
            if old.budgets != new.budgets:
                raise ValueError('live hard reservation changed')
            retained = {h for h in old.held if old_blocks[h].lifetime_prefix}
            if not retained <= new.held:
                raise ValueError('lifetime prefix lost while request remains active')
        result = account(candidate, self.capacity)
        self.snapshot = candidate
        self.revision += 1
        self._seen_batches.update(new_batches)
        self._seen_requests.update(new_requests)
        for h in new_blocks:
            self._generations[h.slot] = h.generation
        return result
