"""CPU adapter for real pinned vLLM manager metadata; not a runtime hook.

Epochs and retained-reference leases must come from allocation/lifecycle hooks.
A periodic snapshot cannot discover slot reuse or asynchronous ownership alone.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

from model import Batch, Block, Handle, Request, Snapshot, account


def manager_kind(manager):
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    if manager.dcp_world_size != 1 or manager.pcp_world_size != 1:
        raise ValueError('only single-device cache geometry is supported')
    if type(manager) is FullAttentionManager and type(manager.kv_cache_spec) is FullAttentionSpec:
        return 'full'
    if type(manager) is SlidingWindowManager and type(manager.kv_cache_spec) is SlidingWindowSpec:
        return 'swa'
    raise ValueError('unsupported manager/spec; do not infer retention from inheritance')


def group_name(manager):
    return f'{manager_kind(manager)}:{manager.kv_cache_group_id}'


def request_budgets(managers, extent, max_model_len):
    """Match KVCacheManager.get_admission_blocks, preserving per-group budgets.

    The manager's configured cap is authoritative. A zero-in-flight persistent
    cap is not a bound on a complete transient prefill allocation.
    """
    if not 0 < extent <= max_model_len:
        raise ValueError('invalid declared request extent')
    result = []
    for manager in managers:
        name = group_name(manager)
        size = manager.block_size
        required = (extent + size - 1) // size
        cap = manager.admission_block_cap
        if cap is not None:
            required = min(required, cap)
        result.append((name, required))
    if len(dict(result)) != len(result):
        raise ValueError('duplicate cache group')
    return tuple(result)


@dataclass(frozen=True)
class Observation:
    snapshot: Snapshot
    capacity: int
    groups: frozenset[str]
    lifetime_groups: frozenset[str]

    def charge(self):
        return account(self.snapshot, self.capacity, self.groups, self.lifetime_groups)


def observe(managers, extents, max_model_len, epochs, leases=None, extras=None):
    """Observe requests plus explicit extra/deferred BlockPool references.

    leases: batch_id -> [(group_id, KVCacheBlock), ...], including multiplicity.
    These are real retained refs, not merely a list of all GPU reads. Refcount
    conservation fails closed if a CoW endpoint/deferred free is not supplied.
    Idle cache entries are omitted; the pool's null block is excluded/remapped.
    """
    leases = {} if leases is None else leases
    extras = {} if extras is None else extras
    if set(extras) - set(leases):
        raise ValueError('extra reservation needs an explicit batch')
    if not managers:
        raise ValueError('at least one cache manager is required')
    pool = managers[0].block_pool
    if any(m.block_pool is not pool for m in managers):
        raise ValueError('multiple pools are outside this adapter')
    names = {m.kv_cache_group_id: group_name(m) for m in managers}
    if len(names) != len(managers):
        raise ValueError('duplicate group IDs')
    groups = frozenset(names.values())
    lifetime = frozenset(group_name(m) for m in managers if manager_kind(m) == 'full')
    block_sizes = {m.kv_cache_group_id: m.block_size for m in managers}
    slots = {b.block_id: i for i,b in enumerate(b for b in pool.blocks if not b.is_null)}
    refs = Counter()
    owners = defaultdict(set)
    block_group = {}
    held = {rid: set() for rid in extents}

    def visit(gid, b, owner=None):
        if b.is_null:
            return None
        if b.pool is not pool or pool.blocks[b.block_id] is not b:
            raise ValueError('foreign pool block')
        if gid not in names or (b.block_id in block_group and block_group[b.block_id] != gid):
            raise ValueError('physical block assigned to incompatible groups')
        if b.block_id not in epochs:
            raise ValueError('missing allocator generation')
        block_group[b.block_id] = gid
        refs[b.block_id] += 1
        if owner is not None:
            owners[b.block_id].add(owner)
        return Handle(slots[b.block_id], epochs[b.block_id])

    for m in managers:
        for rid, blocks in m.req_to_blocks.items():
            if rid not in extents:
                raise ValueError('allocator request has no hard reservation')
            for b in blocks:
                h = visit(m.kv_cache_group_id, b, rid)
                if h is not None:
                    held[rid].add(h)
    batches = []
    for bid, retained in leases.items():
        pins = frozenset(h for gid,b in retained if (h := visit(gid,b)) is not None)
        batches.append(Batch(bid, pins, extras.get(bid,0)))
    for b in pool.blocks:
        if not b.is_null and b.ref_cnt != refs[b.block_id]:
            raise ValueError(f'unexplained reference count for block {b.block_id}')
    blocks = []
    for slot,gid in block_group.items():
        b = pool.blocks[slot]
        cached = b.block_hash is not None
        boundary = b.block_hash_num_tokens
        complete = cached and boundary is not None and boundary % block_sizes[gid] == 0
        durable = complete and names[gid] in lifetime
        who = owners[slot]
        if not cached and len(who) > 1:
            raise ValueError('unhashed writable block shared across requests')
        owner = next(iter(who)) if not cached and len(who) == 1 else None
        blocks.append(Block(Handle(slots[slot],epochs[slot]),names[gid],owner,durable))
    requests = tuple(Request(rid,request_budgets(managers,n,max_model_len),frozenset(held[rid]))
                     for rid,n in extents.items())
    return Observation(Snapshot(tuple(blocks),requests,tuple(batches)),len(slots),groups,lifetime)
