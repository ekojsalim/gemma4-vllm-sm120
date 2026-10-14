# Local validation — October 15, 2026

These are quick experiments on one RTX 5090 using a private mixed-length
64-request corpus, not a statistical benchmark or broad quality evaluation.
Some requests use n>1, yielding 68 generated sequences. Timings exclude warmup.

## Selected configuration

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

PR60689 MTP fusion and PR60317 attention experiments showed no clear benefit
in the earlier comparison and are excluded from this release.

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
