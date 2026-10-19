# Pinned vLLM cache-manager bounds and CPU adapter

This extends the [implementation scope](prefix-cache-first-implementation.md).
The adapter is experimental metadata code, not an installed runtime hook.
Tests use the actual vLLM 0.31 BlockPool, FullAttentionManager and
SlidingWindowManager with this repository's overlay. No GPU tensors, model
weights, network or GPU device mounts are needed.

## Bounds from the managers

For declared hard extent E (prompt plus hard output allowance), group block size
B, and the manager's configured admission cap C:

- Full attention: ceil(E / B).
- Sliding window: min(ceil(E / B), C).
- C = ceil(min(window - 1 + extra_retained_tokens + max_in_flight_tokens,
  max_model_len) / B) + 1. The last block covers alignment slack.

The scheduler constructs the hybrid-admission cache manager with
max_in_flight_tokens=0. For window=1024, B=16 and no extra retention, C=65;
with three extra retained tokens it is 66. These are illustrative geometry
values, not a report of current deployed cache-group sizes. Read actual manager
specs at startup; do not multiply group block counts by layer count.

The tests sweep block sizes 4/16/32, extra retention 0/3, in-flight tokens
0/8/2048 and six extents, and compare per-group totals to the real
KVCacheManager.get_admission_blocks helper. Its aggregate helper is invoked
without constructing an engine. Allocator lifecycle tests use real managers and
pool instances.

A persistent SWA cap is **not** a transient allocation bound: one prefill chunk
can materialize more blocks before skipped blocks are reclaimed. Runtime work
must reserve those transient peaks and speculative/async overlap explicitly.

## Ownership observed from real managers

`experiments/prefix_admission/vllm_adapter.py` converts manager request tables
and explicit retained-reference leases into the existing accounting oracle:

- Exclude the permanently allocated null block; ignore null SWA placeholders.
- Count unique physical occupied blocks, not summed request-table lengths.
- Credit only complete hashed full-attention blocks as lifetime-shared prefixes.
  This requires the runtime to preserve their reference until request release.
- Keep shared SWA replacement budgets reserved. Hashing alone does not make
  sliding-window storage a lifetime credit.
- Omit idle cached entries: they are reclaimable, but touching one removes it
  from the free queue. The allocation estimator counts idle hits plus new blocks;
  active hits add ownership without consuming another physical block.
- Require exact refcount conservation against request tables and supplied extra
  leases. CoW retains a source ref and an extra destination ref until worker copy
  completion; cancellation of both requests does not release these copy leases.
- Require caller-supplied allocation epochs. Snapshots cannot detect ABA reuse.
  Fixture epochs are zero only within observations that do not track transitions;
  stale/recycled-handle transitions are covered separately by the oracle tests.

Only exact supported full/SWA spec-manager pairs, one pool and single-device
geometry are accepted. Unsupported manager subclasses fail closed. Hash lookup
and group isolation are tested through the actual full-attention lookup; the
adapter trusts upstream matching rather than implementing its own hash cache.

## Reproduce

```sh
python3 -m unittest discover -s experiments/prefix_admission -v
experiments/prefix_admission/run-vllm-cpu.sh
```

The host run skips real-manager tests when vLLM is absent. The second command
uses the digest-pinned image, copies the overlay into a disposable container,
checks vLLM's version and runs both suites. Podman/image availability is required;
the image may need downloading first. Torch may print a no-CUDA-runtime warning.
The runner removes its container on exit.

## Remaining runtime work

1. Hook allocation/reuse epochs and deferred/CoW lease transfer and completion;
   scan-only reconciliation is insufficient. Preserve all existing hidden refs.
2. Prepare candidate admission and transient bounds before mutation, protect all
   groups' prefix hits before allocations can evict them, then commit atomically.
   The current adapter observes state; it does not implement admission rollback.
3. Wire admission/release/publication to the scheduler, then run CPU scheduler
   pressure, starvation, full-growth, cancellation and completion-order tests.
4. Qualify real cache contents, warm/cold logits and latency, speculative decode,
   pressure and forward progress on GPU before enabling production caching.

No runtime guards, release manifest hashes or production arguments are changed
by this stage. Passing these tests does not qualify prefix-enabled deployment.

Validation on 2026-10-19: 29 tests passed in the disposable pinned-image runner
(21 oracle tests and 8 real-manager tests), including the 100 randomized oracle
schedules. Host run passed all 21 oracle tests and skipped the real-manager
module as intended. `git diff --check` passed. No GPU or production service was
used or changed.

## Allocation and completion lifecycle prototype

`experiments/prefix_admission/lifecycle.py` now supplies epochs from a real
BlockPool subclass instead of fixture constants. It advances an epoch only when
`get_new_blocks` assigns storage to new content. Freeing a cached block and
subsequently touching that same prefix preserve its epoch. Failed pool
allocations leave epochs and free capacity unchanged.

The lifecycle helper transfers the managers' existing CoW and detached-request
references into worker-completion fences. Transfers do not increment refcounts.
Each fence stores all references, including multiplicity when a copy endpoint
is also part of a cancelled request. Physical occupancy remains a set; release
remains a multiset. Completion checks allocator epochs and full refcount
conservation before releasing references. Duplicate completions, reuse of
completed fence IDs, duplicate deferred requests, and a third outstanding fence
are rejected. These checks run before draining pending-copy lists or detaching
request tables where applicable.

Tests exercise actual manager allocation, publication, cache touches, CoW,
cancellation and completion. Thirty seeded fan-out schedules grow every request
to its hard extent, reclaim SWA storage, and drain all active references. Fault
injection verifies that premature external release and storage reuse are detected
before completion can release another owner's blocks. Synthetic out-of-order
completion tests validate reference accounting only; the runtime must still
supply a fence whose completion proves that all relevant worker accesses ended.

This prototype requires a fresh pool and single scheduler-thread ownership.
It is not installed in the overlay and does not intercept scheduler methods.
It transfers references but does not decide whether a new request can be
admitted. The next CPU stage remains pre-mutation admission/transient planning
and scheduler integration, including adapting the oracle's transition contract
for idle cached identities returning without a new allocation epoch and for
references transferred to an existing worker fence. Snapshot accounting is
already checked; the immutable Ledger transition API is not wired to this
lifecycle helper. Pending copies must be collected before reconciliation.

Lifecycle validation on 2026-10-19: 37 pinned-image CPU tests passed, including
8 lifecycle tests and 30 real-manager randomized schedules in addition to the
prior oracle schedules. Host oracle tests and `git diff --check` passed. No GPU
or production service was used or changed.

## CPU scheduler transaction reference

`experiments/prefix_admission/transactions.py` connects the real manager mutations
and lifecycle helper through prepare/commit, ordered completion, cancellation and
safe SWA reclamation. It remains a CPU reference harness, not vLLM Scheduler
integration. Preparation copies allocator metadata and executes the proposed
batch on that shadow. A successful prepare returns the resulting occupancy plus
future hard obligations and explicit unallocated transient extras. Failed
preparation leaves live tables, pool refs, epochs, reservations and free-queue
order unchanged.

Commit checks a complete metadata fingerprint and transaction revision before
replaying the prepared operations on the live managers. Plans are single-use and
belong to one controller. Changes outside the controller, even a free/reallocate
cycle that leaves occupancy unchanged, invalidate a prepared plan. Commit-time
allocator exceptions or shadow/live disagreement are fatal programming errors;
there is no rollback of arbitrary partial live mutations. This is appropriate
for a serialized CPU reference, not sufficient production failure handling.

The transaction sequence is:

1. Validate fixed hard extents, scheduled target extent, lookahead, batch identity
   and cache-group ownership of supplied hits. Actual prefix matching remains
   upstream's responsibility; block IDs are internal lookup results, not an API.
2. Adopt all requests' hits across all groups before any new allocation can evict
   an idle hit needed later in the batch.
3. Allocate target and speculative slots through the real managers. Transfer
   pending CoW refs to the submitted batch's completion fence.
4. Check unique occupancy plus future per-group obligations plus explicitly
   supplied unallocated transient reservations. Already allocated lookahead and
   prefill storage are charged physically; do not add them again as extras.
5. Commit only if the live state still matches the prepared baseline.

Two batches may overlap. Cancellation detaches tables but transfers their refs
to the last submitted batch using that request. The controller requires ordered
worker completions, so that fence implies earlier accesses have also finished.
SWA reclamation is allowed only when the request has no outstanding worker use
and only through a processed-token boundary no later than the scheduled target
extent. This is deliberately conservative; async rejected-token rollback and
reclamation while older steps are in flight remain future runtime work.

Tests include large-prefill rejection followed by smaller-chunk progress,
lookahead exceeding hard extent, overlapping transient extras, multi-request
atomic rejection, stale sibling plans, out-of-band allocation changes,
wrong-group hits, idle multi-group fan-out, partial-hit CoW, cancellation across
two batches, and growth/reclamation to the full declared extent.

### What this does not establish

Copying/pickling the pool is too expensive for the production scheduler. This
path supplies an executable reference for a bounded delta planner; it is not a
candidate serving implementation. Explicit unallocated extras are caller inputs,
not an automatically derived bound on every asynchronous scheduler operation.
The harness does not use the original immutable `Ledger.replace` transition API:
real-manager observations supply accounting and fingerprints supply stale-plan
validation, so idle cached identities can return without inventing new epochs.

Next runtime integration must compare a bounded planner against this reference,
wire the vLLM waiting/running loops and completion sequence, and handle prefix
publication, speculative rollback, preemption and external KV paths explicitly.
Unsupported paths should fail closed. Prefix-cache rejection guards and release
manifests remain unchanged. GPU availability is useful only after that wiring
passes CPU tests; these commits alone are not ready for a serving trial.

Transaction validation on 2026-10-19: all 50 pinned-image CPU tests passed,
including 13 transaction tests. Host oracle tests passed with the three
vLLM-dependent modules skipped as intended. No GPU or production service was
used or changed.
