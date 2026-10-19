# Prefix-cache GPU qualification candidate

Candidate: **0.2.1-prefix.2**, pinned vLLM 0.31 image from release.env.
Status: bounded GPU checks and the representative 666-request span replay passed
structural checks; full replay time improved 2.18×. Application-level quality
validation remains open. See [span replay results](results/span-prefix-replay-2026-10-19/README.md)
and the [initial GPU results](results/prefix-cache-gpu-2026-10-19/README.md).
The [historical c8 regression check](results/prefix-c8-regression-2026-10-19/README.md)
also passed: median 925 tok/s baseline versus 932 cold / 940 warm candidate,
with a first-use compilation outlier documented separately.
Production baseline remains 0.2.0, fixed-k4, prefix caching disabled.

## Deliberately small implementation

Enable ordinary full-block chapter-prefix reuse while retaining the existing
hard per-request completion reservations. Every request is charged for its own
full declared extent and physical upper bound, including shared blocks in its
block tables. Summing these holdings overcounts shared storage, so sharing does
not create extra admission credit. This candidate targets prefill compute savings,
not a global concurrency increase. It does not install the experimental direct
planner, snapshot oracle, lifecycle tracker or a second scheduler.

The waiting path now passes upstream's local hit and hit length into the existing
allocation-bound helper. The helper includes active shared hits in per-request
ownership; vLLM's free-queue estimator otherwise counts only new allocations and
idle hits. Running decode, reservations, rollback, deferred free and completion
settlement stay on their existing paths. A conservative physical-growth check can
reject an active hit even when sharing would fit; this is accepted for the first
candidate and is a measurement target, not a promised capacity improvement.

## Launch delta for a private trial

Use a fresh pinned-image container and install this candidate archive before
importing vLLM, following the existing deployment recipe. Keep target/draft
weights, memory budget, companion services and fixed-k4 settings unchanged.
Change only:

```sh
export GEMMA4_EXPERIMENTAL_PREFIX_ADMISSION=1
# Replace --no-enable-prefix-caching with --enable-prefix-caching.
# Keep --hybrid-kv-cache-admission --hybrid-kv-speculative-admission.
# Keep enable_adaptive_verification=false and num_speculative_tokens=4.
```

Use a private port and isolated compilation/cache directories. Check startup's
actual group geometry before sending requests. The opt-in accepts only exact
full-attention/SWA specs whose block sizes are multiples of the smallest
group/hash block size (including Gemma SWA16/full32),
one device, no connector, no bounded-prefix replay, no dynamic/adaptive depth,
and fixed-k4 (or no speculation for diagnosis). Startup rejects unsupported
geometry; do not bypass it to make the candidate launch. Partial-hit admission is
also rejected before allocation. Lookup falls back to cold prefill if a returned
hit is not aligned to every group; it never truncates SWA tables to an earlier
window boundary. Standard aligned hits and EAGLE's upstream block
drop remain enabled. Without the environment opt-in, the old prefix-cache guard
still rejects caching with hybrid admission.

## CPU evidence and its limits

The actual Scheduler and AsyncScheduler are instantiated with an offline tiny
model configuration, real requests/hash lookup, real cache coordinator/managers
and the real reservation ledger. Tests call schedule and update_from_output;
worker outputs are synthetic, and fixed-k4 uses a metadata-only Gemma MTP contract.
No target/draft forward pass or grammar backend is exercised on CPU.

Covered: cold versus warm sibling scheduling, idle and active shared hits,
normal finish, cancellation before completion, fixed-k4 all-draft rejection,
two in-flight asynchronous batches, bounded-pool waiting and complete drain,
opt-in guard, fine-grained hashing rejection and partial-hit rejection.
These checks demonstrate scheduler accounting and progress in the tested cases;
they do not demonstrate valid cached GPU tensors, model acceptance or latency.

## First bounded GPU session

1. Startup/geometry and one cold request. Verify installed overlay hashes and
   fixed-k4 configuration; preserve the working baseline image/overlay/args.
2. Compare cache-off and candidate cold/warm marked-span generations with identical
   settings. Keep stable chapter context before variable assignment/glossary.
   Record actual cached tokens, TTFT, total latency, generated tokens and speculative
   acceptance. Diagnose changed outputs with logprobs/quality checks rather than
   demanding cross-shape bitwise equality.
3. Repeat with strict-schema verification and one-token decisions/logprobs,
   including the actual production chat-template thinking flags and schema.
   Check schema validity, marker coverage and decisions, not only throughput.
4. Fan out 4 then 8 siblings, include a different chapter, and cancel some requests
   during decode. Exercise bounded pressure/full output limits. Watch queue
   progress, assertion failures, preemptions and allocator/ledger drain. Idle cached
   blocks may remain hashed after active references are released.
5. Stop on wrong-prefix reuse, invalid outputs, accounting failures or stalled
   progress. Do not promote the candidate based on a warm-TTFT win alone.

Rollback is the complete 0.2.0 image/overlay/arguments combination. No existing
service or GPU is changed by building this archive. GPU execution and production
promotion are separate steps.

## Validation record (2026-10-19)

The candidate archive installed successfully into a fresh digest-pinned image
with no network or GPU devices. From that installed package, 75 unittest cases
passed (including 8 actual Scheduler/AsyncScheduler cases), followed by all 47
existing admission pytest regressions. Bundle manifest verification and
`git diff --check` passed. This records the initial CPU-only .1 archive. Candidate .2 additionally passed
77 prefix-admission tests and the same 47 regressions, followed by the bounded
GPU session linked above. It is not a production promotion.
