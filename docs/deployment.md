# Deploying overlay 0.2.0 / vLLM 0.31

Use the image digest in the root README. Build/download the matching overlay
asset, verify its SHA256, extract it, and run `scripts/install-overlay.sh`
before any vLLM process starts. Use a fresh image; replacing a running server's
files or upgrading an old patched 0.26 container in place is unsupported.

Mount the target at `/models/gemma4` and BF16 assistant at `/models/assistant`.
Use a persistent cache directory dedicated to 0.31, for example:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_CACHE_ROOT=/cache/vllm031
export TRITON_CACHE_DIR=/cache/triton031
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve /models/gemma4 \
  --served-model-name gemma4 --host 0.0.0.0 --port 8000 \
  --quantization modelopt --dtype auto --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 --kv-cache-memory-bytes 8432910336 \
  --max-model-len 6144 --max-num-seqs 32 --max-num-batched-tokens 1024 \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --language-model-only --linear-backend flashinfer_cutlass \
  --enable-flashinfer-autotune --async-scheduling \
  --no-enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --hybrid-kv-cache-admission --hybrid-kv-speculative-admission \
  --speculative-config '{"method":"mtp","model":"/models/assistant","num_speculative_tokens":4,"enable_adaptive_verification":true}' \
  --cudagraph-capture-sizes 1 2 4 8 16 24 32 40 48 56 64 \
  --attention-config '{"backend_per_kind":{"sliding_window":"TRITON_ATTN","full_attention":"TRITON_ATTN"}}' \
  --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 \
  --chat-template /vllm-workspace/examples/tool_chat_template_gemma4.jinja
```

This is the evaluated local shape, with a deployment-neutral model alias.
The exact KV budget fits our OCR+term resident configuration; other nodes must
recheck available memory. Keep companion services resident during validation.
Wait for `/health`, then test text generation and companion services. Cold
compilation/autotuning can take several minutes.

0.31 uses its upstream FlashInfer autotune cache. Remove old startup checks
that wait for a custom frozen-profile marker; the old
`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` overlay contract does not apply here.
Do not copy old 0.26 tactic files into the new cache. Keep model credentials in
the process environment, not assets or logs.

Rollback requires the complete previous image+overlay+arguments combination,
not just the image tag. Overlay 0.1.0 remains paired with 0.26 and fixed k 4.
The historical startup gist below remains a 0.26 recipe until separately updated.
No fresh remote Vast boot was performed for this 0.31 update.

## Historical OCR deployment notes

The following describes the September 0.26 deployment. Its statements about
unchanged vLLM assets apply to that OCR-only update, not this 0.31 migration.

## OCR bundle update: 2026-09-06

OCR is distributed independently of the vLLM overlay, through
[`ekojs/ppocrv6-tensorrt-ocr-bundle`](https://huggingface.co/ekojs/ppocrv6-tensorrt-ocr-bundle).
The September update supplies:

- shared TensorRT enqueue workspace reserved at OCR startup;
- cropped multi-character line recognition at `/v1/lines/recognize`;
- parallel line preprocessing and a three-profile recognition engine; and
- white detector padding for small or extreme-aspect pages, with padding
  excluded from detections and original-image coordinates preserved.

The existing [startup gist revision 54b267c](https://gist.github.com/ekojsalim/5cd2ac2ff3d0b43fb5e488e5ee1ea8da/54b267cb7372f1e0d837605da99b1017a2c53d6f)
uses the same file paths and TensorRT `10.14.1.48-1+cuda13.0` runtime. Its script
and the vLLM `v0.1.0` release asset/checksum do not need changing for this update.
The release remains pinned to official vLLM 0.26.0 at image digest
`sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52`.

With `OCR_REVISION` unset, the next startup downloads the OCR repository's
default branch. Existing processes do not hot-reload the bundle: restart or
redeploy the worker using its normal startup lifecycle. If `OCR_REVISION` is
set, advance it to the accepted bundle commit (see `ocr-bundle.json`). An
operator who explicitly uses an older gist revision should update to the
linked compatible revision; no newly published gist is necessary.

The target follows `MODEL_NAME` by default, or an existing `MODEL_PATH`;
`MODEL_REVISION` is an optional exact-commit override. The assistant retains
its existing default pinned revision. This matches the repository-based model
release contract used by the current startup script.

### Validation and limits

The promoted shared-workspace/three-profile stack completed 96 synthetic vLLM
requests overlapping 360 OCR requests without OOM or preemption. Separate runs
reached 32 active sequences and 90.5% KV usage at the unchanged 9,506,652,160-byte
KV budget. Lowest sampled free VRAM was 236 MiB. The subsequent detector-padding
fix passed 18 input-shape checks, five text/coordinate controls, and the full
page/glyph/line regression suites with vLLM resident at unchanged warm memory.
These are local RTX 5090 results; a fresh remote Vast boot was not performed.

The recognition rebuild flips four synthetic glyph predictions between close
candidates. All four intended characters are outside the model vocabulary;
accuracy on supported filename labels remains 559/751. This investigated change
was accepted, but is not a production accuracy equivalence claim.

Exact bundle hashes and the published/previous Hugging Face revisions are
recorded in [`ocr-bundle.json`](ocr-bundle.json). To roll back OCR, set
`OCR_REVISION` to that previous revision and redeploy; this also restores the
older memory behavior and detector minimum-shape bug. Keep the vLLM overlay
release unchanged.
