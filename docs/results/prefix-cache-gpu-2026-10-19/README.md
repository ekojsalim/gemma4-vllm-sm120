# Conservative prefix admission: first GPU qualification

The candidate works for bounded chapter-prefix reuse on the real GPU. It saves
prefill time without granting extra admission capacity. Basic output checks and
pressure/cancellation passed. **Do not promote yet:** translation output and
speculative acceptance vary across launches/configurations, and this small sample
does not establish translation quality or an overall latency win.

A subsequent [saved-output quality spot-check](quality-spot-check.md) found no
obvious cache-related semantic regression. It supports treating the observed
variation as acceptable in this sample, while leaving representative workload
validation open. An overall speedup on the old translation fixture is not a
requirement for this prefix-reuse feature.

## Configuration and startup correction

RTX 5090, pinned vLLM 0.31 image from release.env, production target and assistant
checkpoints, FP8 KV, FlashInfer CUTLASS linear backend with autotuning, Triton
attention, asynchronous fixed-k4, max length 6144, batch-token budget 1024,
32 sequences, KV budget 9,506,652,160 bytes. The production chat template was
copied unchanged. Candidate runs used an isolated loopback port and cache dirs.
Private model inputs and outputs are not included here.

Initial startup failed closed: the original guard assumed uniform 16-token
blocks. Actual Gemma geometry has five SWA groups with 16-token blocks and one
full-attention group with 32-token blocks, hash size 16. Candidate .2 permits
this divisible geometry, still requiring full-group-aligned hits. An unaligned
lookup falls back to cold prefill, without truncating a SWA table (which could
lose the earlier window's required blocks). The allocation helper independently
rejects partial hits. No partial-block sharing or capacity discount was added.

77 CPU prefix-admission tests passed, including 10 real Scheduler/AsyncScheduler
cases and new heterogeneous-geometry/alignment coverage. All 47 existing admission
regressions passed. Both GPU variants installed the archive in a fresh pinned
image; the installed scheduler/manager hashes matched the final runtime files.
The review patch also passed application checking against the preceding overlay.

## Bounded workloads and results

One private production translation prompt (1,637 tokens), a public six-paragraph
Indonesian chapter, strict-schema location verification, one-token A/B/C decisions
with logprobs, and marked-span translation. Public prompts are only 248–264 tokens;
these fan-outs do not represent long-chapter scaling. Translation uses temperature
0.5, seed 7 and its normal thinking default; public checks use temperature 0 and
thinking disabled. Translation max output is 512, and all translations ended
normally rather than at that limit.

Order: cache-on initial, cache-off control, cache-on repeat. Each task has one
initial request and two repeats. The table uses the two repeats' mean, except
fan-outs, which use a single batch's median. Initial requests are excluded because
first-use compilation/grammar setup inflated them. These are exploratory samples,
not interleaved confidence intervals or p99 measurements.

| Workload | Cache-off TTFT | Cache-on repeat TTFT |
| --- | ---: | ---: |
| Private translation | 138.0 ms | 31.5 ms |
| Strict JSON verification | 55.2 ms | 27.1 ms |
| One-token decision | 52.0 ms | 20.9 ms |
| Marked span | 72.6 ms | 43.9 ms |
| Four sibling spans | 127.4 ms | 80.2 ms |
| Eight sibling spans | 183.8 ms | 120.0 ms |

Warm private requests reused 1,600/1,637 prompt tokens. The 4/8-sibling batches
reused 800/1,056 and 1,728/2,112 prompt tokens respectively. All 23 checked public
outputs in each final variant passed JSON/location, decision or marker/id checks.
Changing the chapter's location changed the decision from A to B, and restoring
the original chapter restored A. Marker validity is not a translation-quality
score; the subsequent qualitative spot-check is linked above. No labeled
translation-quality evaluation was performed.

Translation is the material caveat:

| Run (warm repeats) | Output tokens | Speculative acceptance | Total latency |
| --- | ---: | ---: | ---: |
| Cache-off | 346 | 57.1% | 2,188 ms |
| Cache-on initial | 348 | 47.7% | 2,374 ms |
| Cache-on repeat | 338 | 49.8% | 2,237 ms |

The two warm responses within each run matched exactly. Initial cache-on cold and
warm outputs also matched; on the repeated launch, the cold output differed from
warm outputs (acceptance 50.0% versus 49.8%). Outputs differed between cache-on
launches despite the same request settings. Autotuning/kernel and execution-shape
numerics are possible confounders, not an established root cause. Lower TTFT did
not yield a total translation-latency win here. Cache correctness cannot be
inferred solely from exact-match or mismatch under these changing shapes.

## Pressure and cancellation

With 64 simultaneous clients reserving up to 4,096 output tokens, cache-on reached
14 running and 50 waiting requests; cache-off reached 14 running and 61 waiting
(peaks at different instants). After closing clients, both variants returned to
zero running/waiting and completed a follow-up request. The GPU log reported 0%
active KV usage after drainage. The initial shorter 32-client run also drained
but did not demonstrate waiting pressure. No preemptions occurred in the bounded
functional phases. CPU tests check ledger/free-block drainage directly; GPU
HTTP metrics do not expose a direct ledger assertion.

## Scope and next gate

This is enough to proceed with application integration testing, not production
promotion. Before deployment, hold or record autotune tactics, compare more private
translation requests and actual marked-span/verification payloads with quality
checks, and run longer-context fan-outs under bounded pressure. Keep hard
max_tokens reservations and fixed-k4; shared-capacity credit and partial-block
ownership remain out of scope.

Scripts: `experiments/prefix_admission/launch-gpu.sh`, `gpu-probe.py`, and
`gpu-cancel-probe.py`. Set TARGET, ASSISTANT, TRIAL_ROOT and CHAT_TEMPLATE for the
launcher. The probes expect private-translation.json and public-chapter.txt in
TRIAL_ROOT. Raw SSE outputs/logs stay there, outside this repository. Sanitized
aggregate values are in summary.json. Trial servers were stopped after testing;
the unchanged production container was restarted without changing routing.
