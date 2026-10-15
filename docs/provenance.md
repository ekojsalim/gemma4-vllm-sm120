# Provenance

- Upstream: https://github.com/vllm-project/vllm.
- License: Apache License 2.0; see `LICENSE` and `NOTICE`.
- Overlay version: 0.2.0.
- Source patch base: `db9527a46873454610df6dbedf79a36d6bf1a7f6` (v0.31.0).
- Runtime image: `docker.io/vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`.
- Locally inspected image ID: `c76d0e2225a4b1cb1e2109ace39639f55e714abd1a7a427acc8b0bbd7f6a83b3`.
- Image packages: vLLM 0.31.0, Torch 2.13.0+cu130, FlashInfer 0.7.0.post1,
  Transformers 5.17.0, Triton 3.7.1.
- GPU validation: NVIDIA GeForce RTX 5090 / SM120, October 15, 2026.

Base-file hashes were extracted from a fresh unmodified pinned image, not
from the source checkout. Overlay bytes match the deployed local 0.31 bundle.
The review patch includes both runtime modifications and corresponding tests.
No generated autotune cache is distributed; upstream 0.31 creates node-local
cache entries. The historical 0.26 release and its profile lifecycle remain
available in the repository history and existing 0.1.0 release.
