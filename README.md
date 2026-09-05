# Gemma 4 / RTX 5090 vLLM runtime overlay

This repository packages the reviewed Python runtime changes used for a Gemma
4 ModelOpt NVFP4 target with the Gemma 4 MTP assistant on one RTX 5090. It is a
small overlay for the official vLLM image, not a custom container image.

The deployment base is pinned to:

```text
docker.io/vllm/vllm-openai@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
```

The image reports vLLM `0.26.0` and build revision
`ffd46bfab2128bb84146050e98b51a617c6575ab`. The review patch base is
`568afb3a13806beb53bb2e6bd518269357b237c0`.

## Deployment contract

The release asset contains only the overlay, compatibility manifests, license,
and installer. A node startup script downloads the immutable asset, verifies
its SHA-256, and runs:

```bash
bash scripts/install-overlay.sh
```

The installer verifies vLLM `0.26.0`, every source byte in the asset, and every
base file it will replace. It accepts an already-installed byte-identical
overlay, but fails closed on any other base. Installation finishes before any
vLLM process is started.

FlashInfer NVFP4 tactic profiles are deliberately absent. Each SM120 node tunes
the complete bounded target domain once before readiness and reuses its frozen,
node-local profile on later starts.

## Production shape

- fixed default `num_speculative_tokens=4`;
- BF16 Gemma 4 assistant;
- ModelOpt NVFP4 target and FP8 KV cache;
- Triton SWA/full attention;
- CUDA graph captures through effective M=64;
- `9,506,652,160` KV-cache bytes under the measured OCR memory pressure;
- `max_model_len=6144`, `max_num_seqs=32`, and
  `max_num_batched_tokens=1024`;
- prefix caching disabled and async scheduling enabled; and
- ordinary hybrid-KV admission plus the separate speculative overlay.

The measured local configuration retained 14.03 simultaneous full-length
equivalent slots with OCR resident and safely served the representative
mixed-length corpus at concurrency 32. A new node must revalidate physical
capacity because its driver and OCR runtime may reserve different memory.

## Repository contents

- `overlay/`: exact runtime files installed into the official image;
- `patches/`: nine-part review stack;
- `tests/`: focused source-level regression tests;
- `manifests/`: base compatibility and overlay integrity contracts;
- `scripts/install-overlay.sh`: fail-closed runtime installer;
- `scripts/build-release.sh`: deterministic release-asset builder; and
- `docs/`: provenance and deployment notes.

No checkpoint, Hugging Face cache, prompts, corpus data, generated tactic map,
or node secret is published here.


## OCR deployment update

The September 6, 2026 OCR bundle adds shared GPU workspace, cropped-line
recognition, short-line profiles, and a fix for small/wide/tall full-page
inputs. It is delivered through the existing Hugging Face OCR repository; the
vLLM overlay release and startup gist remain compatible. See
[deployment and validation details](docs/deployment.md#ocr-bundle-update-2026-09-06).
