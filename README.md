# Gemma 4 / RTX 5090 vLLM runtime overlay

This repository packages the local runtime changes validated for a Gemma 4
ModelOpt NVFP4 target with a BF16 Gemma 4 MTP assistant on one RTX 5090 (SM120).
Overlay **0.2.0** targets **vLLM 0.31.0** and uses fixed verification of
four draft tokens. It is an overlay for the official image, not a custom image.

```text
docker.io/vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
```

The review patch base is `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
The old 0.1.0 release remains paired with vLLM 0.26; do not install the new
overlay over an already-patched 0.26 environment.

## Deployment contract

Build with `bash scripts/build-release.sh`. Install the asset in a fresh pinned
image **before importing or starting vLLM**:

```bash
bash scripts/install-overlay.sh
```

The installer checks the vLLM version, overlay hashes, and the exact original
bytes of every replaced file. It accepts a byte-identical installed overlay
and rejects a different base. Python commands use `uv` and the image's Python
interpreter (`VLLM_PYTHON` can override `/usr/bin/python3`).

See [deployment instructions and serving arguments](docs/deployment.md).
An existing startup script must update the image, asset/checksum, serving
arguments, and cache handling together. The old 0.26 startup gist is not a
0.31 deployment recipe.

## Selected serving configuration

- Fixed MTP verification, `num_speculative_tokens=4` and
  `enable_adaptive_verification=false`.
- ModelOpt NVFP4 target, BF16 assistant, FP8 KV cache.
- Triton sliding-window/full attention and FlashInfer CUTLASS linear kernels.
- Upstream 0.31 FlashInfer autotuning and a persistent, version-isolated cache.
- Local hybrid-KV and speculative admission; CUDA graphs through 64.
- Context 6144, max sequences 32, batch tokens 1024, async scheduling,
  prefix caching disabled.
- Validated local KV allocation: **8,432,910,336 bytes**, with OCR and term
  services resident. Revalidate capacity for each node's resident services.

The previous custom tune-once/freeze overlay and optional FP8-assistant loader
are not included in this port. The assistant evaluated here is BF16.

## Validation

Production latency follow-up supersedes the original adaptive recommendation.
In short arrival-paced proxy replays, 0.31 fixed-k4 reduced mean request latency
from **2.65 to 1.80 seconds** in a normal window and **3.30 to 2.70 seconds**
in a busy window, compared with the degraded adaptive production instance.
Matched 0.26 fixed-k4 controls were **1.85 and 2.71 seconds**. A fresh adaptive
instance achieved **1.72 and 2.11 seconds**; the underlying instance-specific
cause is not established. Fixed-k4 is the conservative fallback while that
instability is investigated. These are workload-specific performance tests, not a
broad quality or statistical claim.

FlashInfer B12 X and cuDNN linear alternatives were slower than CUTLASS in the
latest paired tests. FlashInfer attention failed warmup at the current memory
budget; adaptive verification is also unsupported by that backend on SM120.
The local admission patch remains useful for avoiding preemption under mixed
output lengths. See [results and limitations](docs/validation-0.31.md).

## Repository contents

- `overlay/`:17 exact runtime files installed into the pinned image.
- `patches/`: reviewable runtime/test patch against the pinned source commit.
- `tests/`: corresponding source regression tests.
- `manifests/`: base compatibility and overlay integrity contracts.
- `scripts/`: installer, source verifier, patch application and release builder.
- `docs/`: deployment, provenance, measurements and historical OCR notes.

No checkpoint, Hugging Face cache, prompts, corpus data, generated tactic map,
or node secret is distributed. Historical OCR deployment details remain in
[the deployment notes](docs/deployment.md#ocr-bundle-update-2026-09-06).
