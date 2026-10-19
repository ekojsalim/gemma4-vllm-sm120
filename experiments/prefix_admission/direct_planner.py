"""Direct capacity projection for supported full/SWA transaction operations.

No allocator cloning, allocation or hypothetical block IDs. Current ownership
still comes from a full metadata observation; fingerprinting remains the CPU
controller's stale-plan fence. Incremental runtime hooks are not installed.
"""
from collections import Counter

from model import CapacityError, Charge, Handle
from transactions import Prepared, Transactions
from vllm_adapter import group_name, request_budgets


def project(controller, batch_id, steps, extra=0):
    new = controller._validate(batch_id, steps, extra)
    life = controller.life
    observation = life.observation(controller.extents)
    current = observation.charge()
    snapshot = observation.snapshot
    blocks = {b.handle: b for b in snapshot.blocks}
    requests = {r.request_id: r for r in snapshot.requests}
    # These sets express occupied storage, including batch-only references.
    pinned = set(blocks)
    slots = {b.block_id: i for i, b in enumerate(b for b in life.pool.blocks if not b.is_null)}
    new_by_id = {s.request_id: hits for s, hits in new}
    future = current.future
    fresh = 0
    touched_idle = set()
    for s in steps:
        budgets = dict(request_budgets(life.managers, s.extent, life.max_model_len))
        credit = Counter()
        old = requests.get(s.request_id)
        if old is not None:
            for h in old.held:
                b = blocks[h]
                if b.private_owner == s.request_id or b.lifetime_prefix:
                    credit[b.group] += 1
            future -= sum(max(0, n-credit[g]) for g,n in old.budgets)
        for i, manager in enumerate(life.managers):
            group = group_name(manager)
            size = manager.block_size
            required = (s.main_tokens + s.lookahead + size - 1) // size
            if s.request_id in new_by_id:
                ids = new_by_id[s.request_id][i]
                skipped = manager.get_num_skipped_tokens(s.computed) // size
                effective = ids[skipped:]
                table_length = skipped + len(effective)
                cow = bool(effective) and s.computed % size != 0
                for j, bid in enumerate(effective):
                    b = life.pool.blocks[bid]
                    if b.is_null:
                        continue
                    # CoW tail stays occupied as a copy source even though the
                    # request receives a new private block instead.
                    if b.ref_cnt == 0:
                        touched_idle.add(bid)
                    pinned.add(Handle(slots[bid], life.pool.epochs[bid]))
                    boundary = b.block_hash_num_tokens
                    if (not (cow and j == len(effective)-1)
                            and group in observation.lifetime_groups
                            and boundary is not None and boundary % size == 0):
                        credit[group] += 1
            else:
                table_length = len(manager.req_to_blocks[s.request_id])
                cow = False
            allocated = max(0, required-table_length) + int(cow)
            fresh += allocated
            credit[group] += allocated # new private storage consumes owner's budget
        future += sum(max(0, n-credit[g]) for g,n in budgets.items())
    # Idle cache entries are free, but all adopted hits must be touched before
    # any allocation. Count shared idle hits once across the entire batch.
    if fresh + len(touched_idle) > life.pool.get_num_free_blocks():
        raise CapacityError('physical allocations and idle hits exceed free capacity')
    result = Charge(len(pinned)+fresh, future, current.unallocated_extra+extra)
    if result.total > observation.capacity:
        raise CapacityError(f'charge {result.total} exceeds capacity {observation.capacity}')
    return result


class DirectTransactions(Transactions):
    def prepare(self, batch_id, steps, unallocated_extra=0):
        steps = tuple(steps)
        baseline = self._fingerprint()
        charge = project(self, batch_id, steps, unallocated_extra)
        plan = Prepared(self.revision, baseline, batch_id, steps, unallocated_extra, charge)
        self._plans[id(plan)] = plan
        return plan
