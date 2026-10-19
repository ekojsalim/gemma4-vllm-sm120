#!/usr/bin/env bash
# Real allocator metadata tests only: no GPU devices, model mounts or network.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
image=docker.io/vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
container=$(podman run -d --network none --security-opt label=disable \
  -e CUDA_VISIBLE_DEVICES= -v "$root:/workspace:ro" \
  --entrypoint sleep "$image" infinity)
trap 'podman rm -f -t 0 "$container" >/dev/null' EXIT
podman exec "$container" cp -a /workspace/versions/vllm-0.31.0/overlay/vllm/. /usr/local/lib/python3.12/dist-packages/vllm/
podman exec "$container" /usr/bin/python3 -c 'import vllm; assert vllm.__version__ == "0.31.0", vllm.__version__'
if (( $# )); then
  podman exec "$container" /usr/bin/python3 "$@"
else
  podman exec "$container" /usr/bin/python3 -m unittest discover -s /workspace/experiments/prefix_admission -v
fi
