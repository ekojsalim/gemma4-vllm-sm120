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

Target and OCR repositories follow their default branches unless the operator
sets optional exact-commit `MODEL_REVISION` / `OCR_REVISION` overrides.
Authentication tokens stay in process environment and must not be copied into
`/etc/profile.d`, logs, release assets, or this repository.


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
