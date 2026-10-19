# Chapter-prefix reuse: first implementation

Date: 2026-10-19. Status: design plus CPU reference model; not deployable.

## Scope and workload

Support ordinary marked-span generation and per-unit verification requests that
reuse chapter context. Evidence: omni-translation-spans at 6c60774. Generation
uses roughly 180-source-token requests containing ~60-token units. Verification
uses separate support/coverage calls, including one-token Core decisions with
logprobs. These are starting workload shapes, not permanently fixed policies.

Generation and verification normally have different instruction prefixes.
Request-specific glossary, assignment, candidate and question content follows
stable marked source. Long-chapter windows are separate prefix groups. Group IDs
coordinate work; actual tokenized prefixes and compatible model state establish
cache identity. Never merge KV merely because text or chapter IDs match.

Preserve hard max_tokens reservations. No output overbooking, persistent context
API, generated-state forks, attention changes, draft quantization or adaptive
verification in this milestone. Keep the pinned 0.31/fixed-k4 baseline.

## Components

1. Client prompt audit: verify rendered token prefixes; no separate prefix/suffix
   tokenization assumption. Keep current per-request glossary after stable source.
2. Client warm-up: complete one real request per prefix group before releasing
   siblings; bounded groups/concurrency, failure/cancellation cleanup. Measure the
   cold latency penalty of waiting for a long first generation. No promise of
   permanent cache residency; later eviction means a normal cache miss.
3. Shared admission: unique physical occupancy plus independent remaining growth
   obligations, outstanding allocations and GPU lifetime protection.
4. Cache-manager adapter: upstream physical ownership remains authoritative;
   snapshot reference acquisition, eviction eligibility, allocation and release
   consistently. Do not implement a competing runtime refcount system.
5. Metrics: cached tokens, physical occupancy, future reservation, pending batch
   protection, admission wait, TTFT, group completion and preemptions.
6. Packaging: updated pinned overlay/manifests/tests/launch recipe and rollback.

## Accounting contract and an important limitation

An immutable full-attention prefix retained for a request's lifetime can count
against each child's full-attention extent while its physical blocks are charged
only once globally. A reusable sliding-window prefix cannot automatically get
that same lifetime credit: it will slide out, and each child may require its own
replacement blocks. Reserve independent sliding-window growth until a tighter
peak-overlap proof exists. This may overcharge transient shared-window occupancy.

The CPU reference model counts:

    unique blocks held by requests or outstanding batches
    + remaining per-request, per-cache-group reservation
    + explicitly reserved unallocated extra batch blocks

Private held blocks consume their owner's reservation. Shared prefix blocks earn
per-request credit only for groups declared to retain them throughout the request
(the first candidate is full attention). No cross-group credit. Idle cached blocks
are evictable and consume no admission credit, though they still occupy physical
slots until reused/evicted. Batch-only pins remain charged after cancellation.
Speculative extra allocations not covered by a request's reservation must be
explicit; the integration must derive these, not infer them from draft length
alone. The model conservatively sums unallocated extras across batches.

Partial writable tails must be private; immutable complete prefixes alone may be
shared. Model block identities include a slot generation to detect stale handles.
A revisioned atomic snapshot replacement models reserve/commit/rollback without
mutating upstream state. This is an executable oracle, not a runtime hook or a
proof that existing vLLM allocator bounds satisfy the contract.

## Source integration points inspected

- core/kv_admission.py: scalar per-request max(persistence, physical, batch-bound)
  ledger; validator rejects prefix caching.
- core/kv_cache_manager.py: prepare_speculative_allocation rejects caching and
  assumes no new computed blocks. Must handle prefix acquisition and final
  ownership, not just allocator delta.
- core/single_type_kv_cache_manager.py: get_num_blocks_to_allocate counts newly
  allocated blocks plus idle hit blocks removed from the free queue, and partial
  hit CoW allocation. This is NOT a count of newly owned unique blocks.
- core/sched/scheduler.py: request ownership counting, reservation, rollback,
  post-allocation binding and delayed batch settlement must move together.

The runtime guards remain enabled until integration is qualified. Avoid a patch
that only removes the two rejection checks.

## CPU stages

A. Dependency-free reference accounting and transactional snapshot tests (started).
B. Derive cache-group growth bounds from request extent and hybrid geometry;
   prove retained-prefix credits cannot disappear while the promise remains.
C. Adapter tests using actual pinned vLLM cache-manager objects and tiny pools:
   cold/active/idle/partial hits, CoW, skipped blocks, speculative rollback,
   allocator failure, cancellation, two in-flight batches and eviction.
D. Scheduler integration plus randomized event sequences checked against the
   independent oracle; all admitted requests can finish to their declared limits.
E. Client group coordination/token-prefix tests with mocked requests/tokenization.

Stage A tests are synthetic snapshots, not substitutes for C/D. CPU tests must
cover identical-prefix/different-suffix fan-out, multiple chapters, full-length
outputs, partial-block boundaries, shared blocks surviving sibling completion,
last-reference release, stale plans, cache-slot reuse, batch completion order and
no leaked credits. Include liveness tests, not only capacity arithmetic.

## GPU qualification (after CPU stages)

1. Cache off/cold/warm generation and one-token verification: actual hits, marker
   completeness, decisions/logprobs, contamination checks. Preserve same-shape
   implementation gates; use quality checks for changed cache/schedule shapes.
2. Pressure with resident companion services, multiple chapters, full-length
   outputs, cancellation and eviction. Check allocator/ledger agreement, forward
   progress, unexpected preemption and steady memory. Idle cached blocks can
   remain after active reservations drain.
3. Compare cache off, caching alone, and caching plus client coordination. Measure
   group completion, cold/warm TTFT, tail latency, cached tokens and concurrency.
   Include coordination's cold-start cost and fairness between chapter groups.

Do not enable prefix caching in production until these pass and a complete
rollback image/overlay/arguments combination is recorded.

## Current delivery

experiments/prefix_admission/model.py implements the initial immutable snapshot
oracle. Its tests run on the CPU with Python's standard library. Runtime overlay,
manifest hashes, services, GPU state and deployment arguments are unchanged.

Initial validation: 21 CPU tests passed, including 100 randomized full-growth
and drain schedules; git diff --check passed. Explicit cold-prefix publication
waits for outstanding GPU references to settle before immutable reuse. These
are specification tests, not actual vLLM cache-manager or GPU qualification.

Worktree: feature/prefix-admission-cpu, based on the local integrated 0.31 bundle
at 69213ca. This development branch does not change a deployed release.

Follow-up CPU stage: [derived bounds and real-manager adapter](prefix-cache-manager-adapter.md)
now exercise the pinned allocator's active/idle hits, eviction, SWA skipping,
transient prefill occupancy and deferred/CoW reference lifetimes. This remains
experimental code outside the runtime overlay.
