# Startup-script deployment

Use the official image by immutable digest. Before launching vLLM, the node
startup script must:

1. download one versioned release asset;
2. verify the asset SHA-256;
3. extract it into a fresh temporary directory;
4. run `scripts/install-overlay.sh` before the first `import vllm`;
5. persist Hugging Face, vLLM compile, and FlashInfer profile caches on the
   node-local workspace;
6. make OCR resident and healthy before allocating the exact vLLM KV pool; and
7. wait for both API readiness and the frozen FlashInfer profile log record.

The profile directory is disposable node-local state. Do not copy profiles
between nodes. A cold node may take several additional minutes for framework
compilation; the measured FlashInfer target-domain tuning itself took about ten
seconds. Runtime missing shapes and tuner mutation fail closed.

The startup script must receive target and OCR checkpoint revisions as immutable
commit hashes. Authentication tokens stay in process environment and must not
be copied into `/etc/profile.d`, logs, release assets, or this repository.

