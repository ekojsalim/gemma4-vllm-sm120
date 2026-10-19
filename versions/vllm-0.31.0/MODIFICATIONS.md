# Modified vLLM files

`manifests/runtime-files.txt` identifies the 17 modified or added runtime files
in `overlay/vllm/`. `patches/0001-Port-SM120-runtime-to-vllm-0.31.patch` contains
the corresponding changes and regression tests against upstream commit
`db9527a46873454610df6dbedf79a36d6bf1a7f6`.

The port preserves language-only attention backend validation, bounded
small-query segmented Triton attention, transactional hybrid-KV admission,
separate speculative admission, shared FP8 KV-scale ownership, the optional
Q/probability-scale contract, and the draft single-query graph guard.

FlashInfer autotuning uses upstream 0.31 behavior. Adaptive verification is
unmodified upstream code, but is disabled in the selected serving configuration
following the production-latency regression.
The old optional FP8 assistant loader and custom tune-once/freeze autotune
implementation are excluded. PR60689/60317 experiments are not included.

Only the pinned image and per-file contract in `manifests/base-files.sha256`
are supported. No model weights, tokenizer files, prompts, benchmark corpora,
tactic profiles, or private deployment configuration are included.

## 0.2.1 promotion

`patches/0002-Experimental-conservative-prefix-admission.patch` adds opt-in
full-block prefix reuse through the real scheduler, retaining per-request hard
reservations without shared-capacity discounts. See the candidate document for
supported geometry, CPU evidence and bounded GPU results. Translation quality
and performance qualification remain open. The promoted runtime is byte-identical to the qualified prefix.2 candidate. The allocator's block/reference ownership is unchanged.
