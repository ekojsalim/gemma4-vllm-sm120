# Historical c8 throughput regression check

**No material steady c8 throughput regression was detected.** The prefix
candidate matches the freshly measured released baseline on the historical quick
fixture, including runs that start with an empty prefix cache. This clears the
normal-throughput gate before promotion; it is not a long-running soak or a
claim that performance is numerically identical.

## Method

Reused the original `benchmarks.gemma4.replay` client and private quick subset
from the October 15 fixed-k4 PR comparisons: seed 42, 32 measured requests,
concurrency 8, observed streaming, unbounded arrival rate, four-request preliminary
warmup and two short internal warmup requests per run. Prompt and output text
are not retained by this client. Request sampling settings are unchanged; the
seed selects the workload, not necessarily deterministic model outputs.

Control: rebuilt **0.2.0**, prefix caching disabled. Its scheduler and cache-manager
file hashes match the installed production baseline. Candidate: **0.2.1-prefix.2**,
prefix caching enabled with the conservative admission opt-in. Both use the same
pinned vLLM 0.31 image, target/draft weights, production chat template, async
fixed-k4, FP8 KV, Triton attention and FlashInfer CUTLASS/autotuning. KV budget is
9,506,652,160 bytes; length limit 6144; batch tokens 1024; maximum sequences 32.
Other inference workloads did not run during this comparison.

Six baseline repetitions were followed by three candidate cold/warm pairs and
one additional candidate cold run. Before each candidate cold run, the upstream
reset endpoint confirmed that the prefix cache was reset. Warm runs immediately
reuse the preceding workload without a reset. The reset endpoint was enabled
only on the isolated loopback trial servers; it is not a production requirement.
Runs are sequential within each startup, not interleaved across configurations.

The old published ~830 tok/s result used a different memory-budget/service-
residency setup. The fresh matched baseline is the relevant comparator here;
this test does not attribute the difference from that historical number.

## Results

Values below are medians across runs; E2E is the median of per-run means.

| Configuration | Runs | Output tok/s | Mean E2E | Mean TTFT | Spec acceptance, pooled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Released 0.2.0, caching off | 6 | 925.2 | 2,495 ms | 154 ms | 49.13% |
| Prefix candidate, cache reset | 4 | 931.7 | 2,538 ms | 150 ms | 49.29% |
| Prefix candidate, warm cache | 3 | 940.5 | 2,454 ms | 134 ms | 49.53% |

The candidate cold median is +0.7% and warm median +1.7% versus baseline.
These small differences should be treated as comparable performance, not a
proven general speedup. E2E p99 medians were 4,084 / 4,184 / 4,198 ms respectively;
the short, stochastic runs do not establish tail-latency equivalence.

All **416 measured requests** succeeded, with no recorded preemptions. Cold
candidate runs still obtained an 8.9% token hit rate from natural sharing within
the run; warm runs obtained 13.6%. This is the actual historical workload,
not an artificial per-request-salted zero-reuse stress test. The warm results do
not establish that all 32 prompts remain fully resident between passes.

### First-use outlier

Individual throughput values (tok/s):

- Baseline: 927.5, 922.9, 933.2, 931.9, 913.1, 906.4.
- Candidate cold: **843.9**, 938.3, 932.0, 931.3.
- Candidate warm: 964.2, 937.0, 940.5.

The first candidate cold pass was about 9% below baseline. The server's JIT
monitor recorded first-use `_topk_topp_kernel` compilation during that pass;
other sampler kernels also compiled during initial warmup. Later reset-cache
passes recovered consistently. Compilation is a plausible contributor, not a
quantified attribution of the entire outlier. The initial result remains in
summary.json and the all-cold aggregate above. Do not describe this as zero
startup overhead.

Before returning traffic after promotion, warm the representative c8 shapes,
not just a single greeting or four requests. Clearing prefix entries after that
warmup, if desired in an isolated test, does not undo kernel compilation. There
is no reason to enable development/reset endpoints on the production service.

## Reproduction and scope

`experiments/prefix_admission/throughput-replay.sh LABEL OUT` uses the existing
benchmark client and dataset mounts. `RESET_CACHE=1` resets before cold passes;
`BENCH_PASSES=r4-cold` runs an additional cold pass. The private launcher accepts
an `OVERLAY_ARCHIVE` override to compare the released baseline with the candidate.
Its development-mode flag defaults off.

This check complements the representative span replay; it does not replace the
span quality observations or the end-to-end generate→verify integration test.
No runtime algorithm was changed during this check. Trial servers were stopped,
and the unchanged production baseline was restored; routing was not modified.
