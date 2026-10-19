#!/usr/bin/env bash
# Private candidate only; does not stop/restart any other service.
set -euo pipefail
variant=${1:?usage: launch-gpu.sh on|off}
[[ "$variant" == on || "$variant" == off ]]
: "${TARGET:?}" "${ASSISTANT:?}" "${TRIAL_ROOT:?}" "${CHAT_TEMPLATE:?}"
root=$(cd "$(dirname "$0")/../.." && pwd)
source "$root/versions/vllm-0.31.0/release.env"
archive="$root/dist/gemma4-vllm-sm120-overlay-v${BUNDLE_VERSION}.tar.gz"
[[ -f "$archive" ]]
container="gemma4-prefix-gpu-$variant"
if podman container exists "$container"; then
  echo "Container $container already exists; inspect it before continuing." >&2
  exit 1
fi
mkdir -p "$TRIAL_ROOT/cache-$variant"
flag=--no-enable-prefix-caching
[[ "$variant" == off ]] || flag=--enable-prefix-caching
podman run -d --name "$container" --device nvidia.com/gpu=all \
  --security-opt label=disable --network host --ipc host \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e GEMMA4_EXPERIMENTAL_PREFIX_ADMISSION=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_CACHE_ROOT=/eval-cache/vllm -e TRITON_CACHE_DIR=/eval-cache/triton \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$archive:/candidate.tar.gz:ro" \
  -v "$TARGET:/models/gemma4:ro" -v "$ASSISTANT:/models/assistant:ro" \
  -v "$TRIAL_ROOT/cache-$variant:/eval-cache:rw" \
  -v "$CHAT_TEMPLATE:/trial-chat.jinja:ro" \
  --entrypoint /bin/bash "$VLLM_IMAGE_REF" -c '
set -euo pipefail
tar -xzf /candidate.tar.gz -C /tmp
bash /tmp/gemma4-vllm-sm120-overlay-*/scripts/install-overlay.sh
exec vllm serve "$@"
' _ /models/gemma4 --served-model-name prefix-trial --host 127.0.0.1 --port "${PORT:-31839}" \
  --quantization modelopt --dtype auto --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 --kv-cache-memory-bytes "${KV_BYTES:-9506652160}" \
  --max-model-len 6144 --max-num-seqs 32 --max-num-batched-tokens 1024 \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' --language-model-only \
  --linear-backend flashinfer_cutlass --enable-flashinfer-autotune \
  --async-scheduling "$flag" --no-disable-hybrid-kv-cache-manager \
  --hybrid-kv-cache-admission --hybrid-kv-speculative-admission \
  --speculative-config '{"method":"mtp","model":"/models/assistant","num_speculative_tokens":4,"enable_adaptive_verification":false}' \
  --cudagraph-capture-sizes 1 2 4 8 16 24 32 40 48 56 64 \
  --attention-config '{"backend_per_kind":{"sliding_window":"TRITON_ATTN","full_attention":"TRITON_ATTN"}}' \
  --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 \
  --chat-template /trial-chat.jinja
