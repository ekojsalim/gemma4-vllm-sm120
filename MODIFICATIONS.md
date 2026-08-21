# Modified vLLM files

Every file under `overlay/vllm/` is a modified or newly added runtime file for
the Gemma 4 SM120 deployment. The exact file list is
`manifests/runtime-files.txt`; the corresponding review series is under
`patches/`.

The changes provide:

- language-only Gemma 4 attention selection;
- bounded small-M segmented Triton attention;
- transactional hybrid-KV admission and a separate speculative overlay;
- shared FP8 KV-scale ownership;
- bounded graph-aware speculative depth support;
- optional assistant FP8 configuration;
- node-local tune-once/freeze FlashInfer NVFP4 profiles; and
- corrected optional Q/probability-scale handling and warnings.

The runtime overlay is compatible only with the base-image and per-file
contracts recorded in `manifests/base-files.sha256`. It does not contain model
weights, tokenizer files, prompts, benchmark corpora, tactic profiles, or
private deployment configuration.

