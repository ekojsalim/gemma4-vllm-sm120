# CPU shared-prefix admission specification

Run from repository root, no third-party packages, model files or GPU:

```sh
python3 -m unittest discover -s experiments/prefix_admission -v
```

See [scope and integration plan](../../docs/prefix-cache-first-implementation.md).
This is an immutable snapshot oracle, not a vLLM replacement allocator. The
runtime's prefix-cache rejection guards and release manifests are unchanged.

Inputs must come from a validated upstream allocator snapshot. Block handles
include reuse generations; no hash matching is implemented here. Explicit
publication models completed private blocks becoming immutable cached blocks.
It conservatively waits for every old GPU batch touching those blocks to settle.
The adapter must establish that published blocks are complete and immutable;
this model has no token tensors from which to infer that fact.

Per-group hard budgets are inputs, not estimated from max_tokens in this code.
Full-attention lifetime-prefix credits require retention until the request ends.
Shared sliding-window blocks receive no lifetime discount; their future private
replacement capacity stays reserved. Private holdings consume their owner's
reservation, and surplus physical blocks are always charged. Idle cached blocks
are reclaimable; batch-only pins are not. Unallocated batch extras are explicit
and conservatively summed, not derived from the existing scalar upper bounds.

Tests cover cold publication, fan-out, cancellation/late completion, partial-tail
private copies, idle hits, cross-group credit isolation, SWA replacement, capacity
rejection/rollback, stale snapshots, stale/reused physical handles and duplicate
completion. The randomized test realizes every admitted child's entire private
budget in 100 shuffled schedules before draining all requests. It does not prove
real scheduler liveness, sliding-window geometry bounds or vLLM integration.

The real-manager adapter and derived bounds are now documented in
[cache-manager adapter](../../docs/prefix-cache-manager-adapter.md). Run its
isolated pinned-image suite with `experiments/prefix_admission/run-vllm-cpu.sh`.
It tests real allocator metadata without GPU tensors. Runtime hooks, transient
reservation planning and scheduler integration remain to be implemented.
Do not enable prefix caching based solely on these tests.

The lifecycle prototype now obtains allocation epochs directly from a real pool
subclass and transfers CoW/deferred-free references to completion fences. The
same pinned-image runner exercises these paths, including randomized fan-out and
full-growth drains. See the lifecycle section of the adapter document for its
remaining scheduler and transaction-boundary limitations.

`transactions.py` adds a CPU scheduler transaction reference: shadow prepare,
stale-checked commit, hard reservations, allocated lookahead, explicit pending
extras, ordered completion and cancellation. It is not the production scheduler
or a performant admission planner. Its tests run in the same pinned-image suite.

`direct_planner.py` replaces shadow allocation with direct capacity projection.
The shared transaction suite and 200 seeded differential cases compare it with
real shadow execution. Pool scans and a flat metadata fingerprint remain; this
is not yet wired into vLLM's serving scheduler. The adapter document includes
CPU timing and the remaining runtime work.

The first runtime candidate now uses the original per-request ledger, not this
shared-credit framework. `test_runtime_scheduler.py` exercises the actual
Scheduler/AsyncScheduler with that conservative opt-in. See
[GPU candidate](../../docs/prefix-cache-gpu-candidate.md) for the exact scope.
