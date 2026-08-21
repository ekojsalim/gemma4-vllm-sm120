# Porting to a newer vLLM release

An upgrade is a semantic port, not an overlay copy.

1. Add `versions/vllm-X.Y.Z/` with its own `release.env`.
2. Pin the upstream tag commit and official image manifest digest.
3. Inventory dependency changes, especially Torch, CUDA, Triton, FlashInfer,
   Transformers, compiled attention/GEMM modules, and Python ABI.
4. Run `scripts/audit-patches.sh` for a quick context report.
5. Classify every old patch as upstream, obsolete, still required, or requiring
   redesign. Port intent against the new source; never copy old full files.
6. Generate the new overlay from the ported checkout and measure base-image
   hashes directly from the exact official image.
7. Re-run focused CPU tests, content-free target/MTP correctness, the production
   distributional gate, corpus acceptance/performance, graph coverage, stable
   allocation, reclamation, and physical OCR-resident capacity.
8. Publish a new release asset only after all gates pass. Update the startup
   gist in a separate immutable revision.

Patch application success is not a correctness result. A cleanly applying
attention or scheduler patch still requires all numerical, liveness, and
capacity gates.
