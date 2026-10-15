# vLLM 0.31.0 compatibility bundle

Selected production configuration: fixed-k4 verification, adaptive disabled,
BF16 assistant, FP8 KV, Triton attention and FlashInfer CUTLASS.
All 17 runtime files match the selected production bundle.

See ../../docs/deployment.md and ../../docs/validation-0.31.md for serving
arguments, measurements and limitations. The single review patch is a plain
Git diff, including focused regression tests; it is not a mail patch.
