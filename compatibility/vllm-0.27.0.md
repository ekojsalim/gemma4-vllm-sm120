# vLLM 0.27.0 upgrade assessment

Status: audit only; do not deploy the v0.26 overlay on this image.

## Upstream input

- Release date: 2026-08-10.
- Tag/build commit: `4bdc8a788d2e2ce9165d552b3d4d8b72604626bf`.
- Official image digest:
  `sha256:07ea4e292adf3a26b05ac97114b28849cf4551a26beb1fbe7decd3842d752ed7`.
- Major environment changes: Torch 2.13.0, Triton 3.7.1, Transformers 5.14.1,
  and FlashInfer 0.6.16.post3.

The new warmup infrastructure, scheduler/KV changes, Model Runner V2 changes,
and compiled dependency updates require full runtime validation even where a
patch applies without conflict.

## Patch disposition

| v0.26 patch | Direct context check | v0.27 assessment |
| --- | --- | --- |
| 0001 language-only attention | applies | Still explicit; port and retest selection. |
| 0002 small-M segmented Triton | applies | Still explicit; Triton/kernel changes require exact correctness and performance reruns. |
| 0003 transactional hybrid admission | conflicts | Scheduler, KV manager, and tests changed. Upstream admission caps/watermarks are not proof of the physical-accounting/liveness contract. Rebase by intent. |
| 0004 shared FP8 scale ownership | partial conflict | PR #48666 merged after the 0.27.0 tag. This tag still needs assessment; later releases may supersede part of our patch. Retain backend-general/fail-closed requirements unless proven upstream. |
| 0005 speculative overlay module | applies | Module alone is inert; reconsider together with admission integration. |
| 0006 graph depth/admission split | conflicts | Dynamic speculative decoding and full-CUDA-graph support are upstream. Drop duplicated graph-policy code and retain only independently required admission/liveness integration. |
| 0007 optional FP8 assistant config | applies | Recheck against the new multi-layer MTP speculator and loader contracts. |
| 0008 node-local immutable tuner | conflicts | Warmup and FlashInfer changed. Port tune-once/coverage/freeze intent onto the new public tuner lifecycle. |
| 0009 Q/probability-scale warning | applies | Reconfirm actual kernel consumers before retaining. |

“Applies” only means `git apply --check` accepted the patch context against the
tag. It is not a numerical, ABI, graph, allocation, or capacity result.

## Relevant upstream changes

- Dynamic speculative decoding landed in PR #32374.
- Full CUDA graph compatibility for dynamic speculative decoding landed in PR
  #45953, making a substantial part of patch 0006 redundant.
- Gemma 4 FA4 FP8/shared-scale work in PR #48666 merged on 2026-08-14, after the
  v0.27.0 tag; it is therefore not assumed present in this release.
- v0.27 includes broader hybrid-cache admission and in-flight-window changes,
  but they do not directly establish our global transient ledger, rejected-tail
  charging, or c15 progress guarantee.

## Recommended next step

Keep production on v0.26. Create a new v0.27 port from the tag source, starting
with target-only language selection, optional-scale behavior, and the small-M
attention test suite. Then reconcile shared-scale ownership, node-local tuning,
and hybrid/speculative admission separately. Only after those gates pass should
the exact startup pool, corpus, graphs, and OCR-resident physical capacity be
measured.
