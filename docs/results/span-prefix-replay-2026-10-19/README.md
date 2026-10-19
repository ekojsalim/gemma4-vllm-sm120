# Representative span generation and verification replay

**Prefix reuse helps this workload.** At four in-flight requests, the supplied
666-request replay took 248.5 seconds with caching disabled and 114.2 seconds
with caching enabled: **2.18× faster, 54% less elapsed time**. All format checks
passed in both variants. Generation quality looked broadly comparable in a
source-based spot-check, with specific imperfections noted below. These results
support a controlled application integration trial; no production rollout was
performed.

## Setup

Candidate 0.2.1-prefix.2 on the pinned vLLM 0.31 image, production target/assistant
weights and chat template, asynchronous fixed-k4, FP8 KV, Triton attention,
FlashInfer CUTLASS/autotuning, 9,506,652,160-byte KV budget, length limit 6144,
1024 batch tokens, 32 maximum sequences. GPU: RTX 5090, observed memory clock
15,801 MHz. The conservative per-request hard output reservations remain enabled;
sharing does not grant additional capacity credit.

The private fixture contains 92 generation calls and 574 verification calls.
Generation requests cover 289 marked units, 1–4 per call. The five fixture labels
are zh1, zh2, zh3, ja1 and ko1. **ko1 is predominantly Chinese with embedded
Hangul characters**, so this is not evidence of clean Korean-language coverage.
The first generation source for ko1 contains 1,737 CJK ideographs and 90 Hangul
syllables. No private prompts or generated prose are committed here.

The exact supplied request bodies were used except for the local served-model
alias. In particular: non-streamed calls, generation temperature 0.5 without an
added seed, verifier temperature 0/max_tokens 1/top_logprobs 20, thinking disabled.
No JSON grammar was introduced. All prompt+declared-output budgets fit 6144.

The replay preserves file order in a four-worker queue, draining between chapter
and task groups for metric attribution. Each server receives one unrelated warmup.
Order was cache-off then cache-on, with separate startups and autotuning; this is
one exploratory pass per variant, not an interleaved statistical experiment.
Initial chapter misses and first-use costs are included. Startup time is excluded.
Timing is complete non-streamed request latency, not client-measured TTFT.

## Timing and cache reuse

| Phase totals | Cache-off | Cache-on | Reduction |
| --- | ---: | ---: | ---: |
| 92 generation calls | 57.2 s | 47.5 s | 17.0% |
| 574 verification calls | 191.3 s | 66.8 s | 65.1% |
| All 666 calls | 248.5 s | 114.2 s | 54.0% |

Generation produced 17,969 versus 18,030 tokens (+0.34%). Speculative acceptance
was 54.94% versus 54.78%; no acceptance collapse appeared in this replay.

| Verification group | Off p50 | On p50 | On p95 | On prompt-token hit rate |
| --- | ---: | ---: | ---: | ---: |
| zh1 | 612 ms | 106 ms | 335 ms | 85.1% |
| zh2 | 884 ms | 110 ms | 238 ms | 91.4% |
| zh3 | 1,850 ms | 120 ms | 1,782 ms | 59.3% |
| ja1 | 795 ms | 107 ms | 252 ms | 89.1% |
| ko1 | 893 ms | 112 ms | 241 ms | 91.2% |

The long zh3 chapter uses 25 distinct generation excerpts and 54 verification
windows, rather than one common chapter prefix. Its verification phase drops
from 135.1 to 57.5 seconds, but p99 remains 1,847 ms versus 1,858 ms off. A low
warm-hit median does not remove the expensive window-miss tail. No requests were
reordered to manufacture more reuse. Generation and verification have different
system prompts and cannot directly share their chapter prefix with each other.

One exception to the generation gain: the first zh1 group took 5.3 seconds on
versus 4.1 seconds off. Its small group includes cold/first-use costs; this test
does not isolate their contribution. All other generation groups improved.

## Structural checks and verifier stability

Both variants completed all 666 requests without HTTP errors. All 92 generation
responses had exactly the requested markers in order, with no leading prose,
and none exhausted its output limit. All 574 verifier responses were valid
ok/no answers with logprobs. No preemptions occurred in either full replay;
final running and waiting gauges were zero.

Verifier decisions agreed on **566/574 (98.61%)**, with eight changes: six no→ok
and two ok→no. Some changes were confident, not merely near-ties. There is no
gold answer set, so agreement is not accuracy and neither variant is the oracle.
Across all cases, the mean absolute difference in normalized top-20 ok/no mass
was 0.0170; the 95th percentile was 0.0290. These are probabilities normalized
over the observed ok/no token variants, not all vocabulary probability mass.

A focused follow-up tested the eight disagreements and four unchanged controls
inside the same cache-on instance. Each request received its own random cache
salt, then the same bodies/salts were repeated twice. The first round recorded
0/39,180 prefix-hit tokens; subsequent rounds recorded 38,560/39,180. One of the
12 answers changed between cold and warm, and both warm rounds agreed on all 12.
Several answers also differed from their earlier concurrent replay results.
This demonstrates execution-shape sensitivity; it neither proves an accounting
bug nor establishes that all differences are caused only by startup autotuning.
The selected set is intentionally biased toward disagreements, so 1/12 is not a
general error-rate estimate.

## Source-based generation spot-check

Reviewed the first and last generation request for each chapter: ten paired
requests covering 28 source units, including late requests after preceding
siblings. Main events, speaker exchanges, quantities and requested unit identities
were broadly comparable. There was no obvious sibling-context substitution,
truncation or marker leakage in these samples. This was an assistant qualitative
review, not a blinded or labeled benchmark.

Specific limitations matter:

- In ko1's first request, cache-on used a different romanization for one name
  instead of the supplied glossary target; cache-off followed the target. Across
  that chapter's generation outputs, cache-on used the alternate name once and
  the glossary form six times, versus seven glossary-form occurrences off.
  That is an actual consistency defect in the sample, not just harmless wording.
  With temperature 0.5, unseeded generation and separate startups, this does not
  establish a systematic cache-induced defect.
- Both modes retain some website notice/navigation boilerplate. Cache-on omits
  one inline advertisement that the control translates. The application's desired
  cleanup policy should be scored explicitly.
- Both render a source school subject-exam reference as an entrance exam, losing
  specificity/changing its characterization. Other modest tense/hedge differences
  and an added physical gesture also occur. Neither mode is uniformly better.

The 574 verifier fixtures contain pre-existing translations. They do **not**
verify the new 92 generations produced in this experiment. An end-to-end trial
must substitute the newly generated units into the verification requests.

## Artifacts and next step

`summary.json` contains sanitized phase measurements, disagreement metadata and
the cold/warm diagnostic results. Raw source/response/log artifacts remain local
under the private trial directory. Reproduction scripts:

- `experiments/prefix_admission/span-replay.py --input REQUESTS --out OUT`
- `experiments/prefix_admission/span-cold-warm.py --input REQUESTS --root TRIAL_ROOT`

The second script expects spans-off and spans-on response directories from the
first. Both default to the private trial endpoint on loopback port 31839.

Next: integrate the candidate into the actual generate→verify workflow and judge
its output with the application's policy, particularly name consistency and
boilerplate handling. There is no reason to require an overall speedup on the old
long-output translation fixture as a prerequisite for this feature. Shared-credit
admission and partial-block ownership remain out of scope.

After testing, trial servers were stopped and the unchanged production baseline
was restored and health-checked. Routing was not changed.
