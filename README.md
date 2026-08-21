# Gemma 4 / RTX 5090 vLLM overlays

This repository carries small, version-pinned runtime overlays for the Gemma 4
ModelOpt NVFP4 target and Gemma 4 MTP assistant on RTX 5090/SM120. It does not
publish a custom container image.

Every supported vLLM base has its own immutable compatibility bundle under
`versions/`. Full-file overlays are never reused across vLLM versions. A bundle
contains its exact base-image contract, review patches, runtime snapshot, and
focused tests.

## Support matrix

| vLLM | Status | Runtime image |
| --- | --- | --- |
| 0.26.0 | Production control | `sha256:ffb2d59b...abf52` |
| 0.27.0 | Audit only; not deployable | `sha256:07ea4e2...52ed7` |

The deployed Vast startup remains pinned to release `v0.1.0`. Repository
cleanup does not mutate that tag, release asset, or startup gist. The cleaned
`v0.2.0` packaging candidate contains the same validated v0.26 runtime bytes and
will require a separate release before deployment.

## Layout

- `versions/vllm-0.26.0/`: complete validated compatibility bundle;
- `compatibility/`: assessments for newer upstream releases;
- `scripts/`: version-agnostic verification, build, install, and audit tools;
- `docs/upgrading.md`: the required port and validation sequence; and
- `Makefile`: short local entry points.

## Verify and build

```bash
make verify
make release
```

The release builder packages only the selected runtime overlay, manifests,
installer, license, and attribution. It does not package tests, mail patches,
weights, prompts, corpus data, node secrets, or FlashInfer tactic profiles.

The installer validates the installed vLLM version and every base file it will
replace before the first `import vllm`. A mismatched image or partially patched
runtime fails closed.

## Production shape

- fixed default `num_speculative_tokens=4`;
- BF16 Gemma 4 assistant;
- ModelOpt NVFP4 target and FP8 KV cache;
- Triton SWA/full attention;
- CUDA graph coverage through effective M=64;
- node-local tune-once/freeze FlashInfer NVFP4 profiles;
- `9,506,652,160` KV-cache bytes under measured OCR pressure;
- `max_model_len=6144`, `max_num_seqs=32`, and
  `max_num_batched_tokens=1024`; and
- separate ordinary hybrid-KV and speculative admission contracts.

The v0.26 control demonstrated 14.03 simultaneous full-length-equivalent slots
with OCR resident. A newer vLLM base must reproduce correctness, performance,
graph behavior, allocation stability, and physical capacity before promotion.

