# Local validation — October 15, 2026

These are quick experiments on one RTX 5090 using a private mixed-length
64-request corpus, not a statistical benchmark or broad quality evaluation.
Some requests use n>1, yielding 68 generated sequences. Timings exclude warmup.

## Selected configuration

Use local patches, fixed-k4 verification, Triton attention and FlashInfer
CUTLASS. The latency follow-up below supersedes the original adaptive selection.

## Production latency follow-up

**Request-fidelity correction:** the exporter subsequently confirmed that
`sampling_parameters` omits the `json_schema` response format added to structured
calls. The proxy replays below did not reproduce those grammar constraints.
Healthy fresh-adaptive results do not rule out a live request-body/mix effect,
and these tests alone do not establish recovered live production latency.

A subsequent after-tuning proxy replay restored the exact application schema
builders, `enable_thinking:false` for structured tasks, and `n:2` for the second
non-streamed translation candidate. Across two 120-second windows (241 calls,
56 structured calls, 25 n=2 calls), all calls succeeded with no preemptions.
Overall speculative acceptance was 51.0%, versus approximately 20% reported
from the degraded live instance. Per-task acceptance was 48.8% translation,
50.5% self-selection, 80.1% terms, 89.7% page-title metadata and 94.3% opening
metadata. Thus these samples did not reproduce a structured-task collapse.
Per-request metrics attributed n=1 calls; the global counter residual was
assigned to translation because all missing-metric calls were translation
(25 n=2 calls and one streamed n=1 call). Prompt substitution remains a limit.

The original saturated throughput test missed a production latency regression.
We compared the actual degraded adaptive instance with 0.31 fixed-k4 and the
old 0.26 fixed-k4 bundle, retaining the model checkpoints, explicit KV budget,
and resident OCR/term services. The 0.31 variants retain the same 17 patches.

| Configuration | Normal mean E2E | Busy mean E2E | Normal decode ms/output token | Busy decode ms/output token |
|---|---:|---:|---:|---:|
| 0.31 degraded adaptive instance | 2.646 s | 3.300 s | 13.42 | 13.63 |
| 0.31 fixed-k4 | 1.797 s | 2.697 s | 8.55 | 10.42 |
| 0.26 fixed-k4 | 1.847 s | 2.715 s | 8.56 | 10.52 |
| 0.31 fresh adaptive instance | 1.717 s | 2.113 s | 8.10 | 8.25 |

Each window is 120 seconds, containing 123 or 220 requests sent at recorded
arrival offsets. All requests succeeded with zero preemptions. Payloads are
task-matched proxies using a translation corpus and 45 private metadata/term
samples; they are not the exact production conversations. Some trace token
counts are missing. Outputs are not forced; aggregate output-token totals
between degraded adaptive and 0.31 fixed differed by less than 0.5%.

Separate 32-request screens at concurrency 1/2/4/8 showed comparable fixed-k4
performance between 0.26 and 0.31. A fresh adaptive instance also recovered
healthy low-concurrency performance and beat fixed-k4 on the busy replay.
Fixed-k4 is selected as the conservative fallback, not the fastest observed
short-run configuration. The cause of the degraded instance is unresolved.
Startup calibration and request-history effects require further
isolation; a healthy short restart test does not establish sustained behavior.
A second adaptive startup using a copy of the production cache also recovered
c4 performance (556 output tokens/s, 7.08 ms median TPOT). This short check
does not reproduce the degradation from the cache alone.

Adaptive verification may skip scheduled draft tokens that still enter the
draft counter, so its reported acceptance is not a direct draft-quality metric.
Server inter-token latency measures output-event gaps and can improve while
time per delivered token worsens. Use client E2E, TTFT and TPOT alongside server
decode time per output token. Short metadata tasks in the busy replay showed
little improvement; most of the benefit was on translation.

## Original saturated-throughput selection (superseded)

Local patches + adaptive verification up to four tokens + Triton attention +
FlashInfer CUTLASS gave approximately 1,293 output tokens/s at concurrency 24
in the original comparison, against 864–881 for the matched production 0.26
controls. Fixed k 4 on 0.31 gave 860; fixed k 2 gave 1,190. Concurrency 8 did not show
a consistent gain. Subsequent deployed-endpoint validation gave 1,285 with
64/64 successes and zero preemptions.

## Admission

| 0.31 adaptive configuration | c 24 output tok/s | Preemption events |
|---|---:|---:|
| Local hybrid/speculative admission | 1308.6 | 0 |
| Upstream scheduler/cache admission | 1404.4 | 11 |
| Upstream, max active sequences 15 | 1315.9 | 0 |

The upstream comparison retains local attention/scale/graph fixes; it is not
an entirely unmodified vLLM baseline. A16-request forced-growth workload
(1,024 input /4,096 output tokens each) produced zero preemptions with local
admission and two with upstream capped at 15 active sequences. Local throughput
was 1,724 versus 1,610 tokens/s. A4,096 input /1,024 output workload was essentially
tied. The patch helps prevent preemption under mixed/growing outputs; it is
not a universal throughput improvement.

## FlashInfer alternatives

All linear comparisons kept adaptive verification and Triton attention.

| Linear backend | c 24 output tok/s | c 8 output tok/s | c 24 mean prefill / decode |
|---|---:|---:|---:|
| CUTLASS | 1261.3 | 873.4 | 128.7 ms /2.780 s |
| B12 X | 1208.3 | 826.3 | 127.4 ms /2.989 s |
| cuDNN | 1155.7 | 798.5 | 148.4 ms /3.154 s |

All six runs had zero failures/preemptions. Sampled total GPU peaks including
OCR and term were 31,730/31,726/31,758 MiB respectively. Samples were one second
apart and include startup; they are not exact allocation peaks. Phase means
are measured-only request durations (68 observations), not exclusive GPU time.
Output token counts differ slightly across backends. Three saved simple output
checks were sensible; this does not establish model-quality equivalence.

FlashInfer attention with CUTLASS and fixed k 2 failed warmup at the same KV
budget: a 42 MiB allocation was requested with 31.81 MiB free. It also downgraded
full/piecewise graphs to piecewise for speculative decoding. No attention
throughput result was obtained. Lower KV budgets and mixed backend variants
were not tested. FlashInfer adaptive verification requires its SM100-family
path and is unsupported on the 5090 in this version.

## Fixed-k4 Gemma PR re-verification

The October 15 follow-up retained the selected fixed-k4 configuration and
compared the local baseline with PR60317, PR60689's head changes, and a
PR60689 head plus query-only backport. No candidate was promoted.

| Candidate | c1 / c4 / c8 output tok/s | Evening mean E2E | Quiet mean E2E |
|---|---:|---:|---:|
| Selected baseline | 164.5 / 499.1 / 829.7 | 2.194 s | 2.965 s |
| PR60317 | 162.5 / 495.0 / 827.7 | 2.230 s | 2.960 s |
| PR60689 head only | 163.5 / 495.2 / 803.9 | 2.197 s | 3.061 s |
| PR60689 head + query only | 166.8 / 500.7 / 842.9 | 2.139 s | 2.810 s |

Each candidate ran 32 measured requests at concurrency 1, 4 and 8 after
warmup, followed by 241 calls across two 120-second arrival-paced windows.
The replay included strict per-task JSON schemas, thinking overrides, Nova
`n:2` calls and streaming options. Translation and self-selection prompts
remained proxies. All 384 quick requests and 964 paced requests succeeded,
with no recorded preemptions. Paced acceptance was 51.4%, 51.5%, 52.0% and
51.4% respectively. These are short single-pass comparisons, not a soak or
quality evaluation.

PR60317 offered no useful latency gain, although sampled total GPU use fell
by 142 MiB. Head-only changes were flat. Query-only changes improved quiet
mean latency by about 5.2%, with only about 0–2% quick throughput gains.
Evening output lengths were also shorter (313.3 versus 319.4 tokens on
average), so its 2.5% E2E reduction is not an equivalent-work speedup.
The evidence does not establish a broad production improvement.

Pinned trial revisions and scope:

- [PR60317](https://github.com/vllm-project/vllm/pull/60317):
  `b6e9de5607e167e3cd4e4688414ed4ce0e843e0b`, using the existing 0.31 port
  and its prior 24-case GPU kernel checks. It replaces the local small-query
  attention optimization while retaining admission and KV patches.
- [PR60689](https://github.com/vllm-project/vllm/pull/60689):
  `972b5416882e9be2e1c34ae337dcdfbc487e6ce0`. The head subset enables dense
  greedy top-token selection and sparse gather changes; the sparse path is
  inactive for this assistant. The query-only trial additionally ports input
  compaction and all-draft-step decode into 0.31's static-k graph machinery.
  It excludes the later dynamic-k refactor and all relaxed-verification
  changes. Target sampling and strict verification remain unchanged.

Final focused GPU checks passed nine cases, including query-only mapping at
the deployed hidden width, sparse maximum-score parity and dense-head checks.
Three earlier sparse exact-token-ID failures were investigated as equal-score
BF16 ties, not score mismatches. Basic output checks are not a quality gate.
Only aggregate measurements are published; private replay records are excluded.

## Adaptive verification investigation

The issue search on October 15 found related upstream reports, but no confirmed
explanation for our single-GPU Gemma regression:

- [#54011](https://github.com/vllm-project/vllm/issues/54011): acceptance
  collapse and hanging with tensor-parallel rank budget divergence on eight
  GPUs; this is not the same deployment topology.
- [#52057](https://github.com/vllm-project/vllm/issues/52057): startup cost
  mismatch, context dependence and online profiling design concerns.
- [#54046](https://github.com/vllm-project/vllm/issues/54046): impossible
  tail profiling shapes; proposed fixes #54065/#54077 were unmerged when checked.

These reports are investigation leads, not a root-cause diagnosis. Fixed-k4
remains selected; healthy fresh adaptive runs do not establish sustained
production behavior.

## Limits

The original 0.31 port passed 70 focused CPU tests. Runtime validation includes
startup, replay, simple outputs, memory samples, and production vLLM/OCR/term
smoke checks. It does not include a fresh remote Vast boot, broad model-quality
evaluation, every KV/batch-size configuration, or long-duration soak testing.
The benchmark corpus and private request/response records are not distributed.

## Package checks

The packaged overlay passed all 70 focused CPU tests inside the pinned image
without GPU access. Reproduce after installation with `bash scripts/check-cpu.sh`
from a source checkout. Fresh installation, byte-identical reinstallation, and
rejection of a modified runtime file were checked in a disposable container.
Applying the review patch to a clean source index reproduced all 17 overlay
files byte-for-byte. Release archive generation was checked for determinism.
