# Modified vLLM files

Each directory under `versions/` is an independently validated derivative of a
specific vLLM base. Its `manifests/runtime-files.txt` lists every modified or
new runtime file, and its `patches/` directory records the reviewable changes.

The vLLM 0.26 bundle provides:

- language-only Gemma 4 attention selection;
- bounded small-M segmented Triton attention;
- transactional hybrid-KV admission and a separate speculative overlay;
- shared FP8 KV-scale ownership;
- bounded graph-aware speculative-depth integration;
- optional assistant FP8 configuration;
- node-local tune-once/freeze FlashInfer NVFP4 profiles; and
- corrected optional Q/probability-scale handling and warnings.

These modifications are not assumed to apply to another vLLM release. See the
corresponding compatibility assessment before porting them.

No bundle contains model weights, tokenizer files, prompts, benchmark corpora,
generated tactic profiles, private deployment paths, or credentials.

