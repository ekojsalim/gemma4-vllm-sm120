# Startup-script deployment

The active Vast deployment remains pinned to the immutable `v0.1.0` release
asset and its published SHA-256. Repository main is not a deployment channel.

For any release, the node startup script must:

1. use the exact official image digest recorded by the compatibility bundle;
2. download one immutable overlay release asset;
3. verify the complete asset SHA-256;
4. extract into a fresh temporary directory;
5. run `scripts/install-overlay.sh` before the first `import vllm`;
6. persist Hugging Face, compile, and FlashInfer profile caches locally;
7. make OCR resident before allocating the exact vLLM KV pool; and
8. require API readiness plus a complete frozen FlashInfer profile record.

The profile directory is disposable node-local state and must not be copied
between nodes. Target, assistant, and OCR inputs use immutable checkpoint
revisions. Authentication tokens stay in process environment and must never be
written to profile scripts, release assets, logs, or this repository.

Promoting a new compatibility bundle requires a new release tag, asset checksum,
and immutable gist revision. Existing tags and gist revisions are controls and
are never rewritten.

