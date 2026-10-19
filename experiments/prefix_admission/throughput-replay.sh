#!/usr/bin/env bash
# Existing October 15 quick replay, repeated at c8; no prompt/output text saved.
set -euo pipefail
label=${1:?usage: throughput-replay.sh LABEL OUT}
out=${2:?};mkdir -p "$out"
: "${BENCH_SOURCE:=/home/ekojs/Dev/gemma4-vllm-server}"
: "${BENCH_DATA:=/home/ekojs/Dev/wu-1.6-core-preview-vllm-sample}"
: "${TELEMETRY:=/home/ekojs/Dev/gemma4-rtx5090-runner/experiments/vllm031/telemetry-prometheus.py}"
read -ra passes <<< "${BENCH_PASSES:-warmup r1-cold r1-warm r2-cold r2-warm r3-cold r3-warm}"
for pass in "${passes[@]}"; do
  requests=32
  [[ "$pass" != warmup ]] || requests=4
  reset=()
  if [[ "${RESET_CACHE:-0}" == 1 && "$pass" == *-cold ]]; then reset=(--reset-prefix-cache); fi
  podman run --rm --network host --security-opt label=disable \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$BENCH_SOURCE:/src:ro" -v "$TELEMETRY:/src/benchmarks/gemma4/prometheus.py:ro" \
    -v "$BENCH_DATA:/data:ro" -v "$out:/results:rw" -w /src --entrypoint uv \
    localhost/gemma4-vllm-production:v0.26-pinned \
    run --no-project --python /usr/bin/python3 python -m benchmarks.gemma4.replay \
    --data-root /data --api-base http://127.0.0.1:31839 \
    --metrics-url http://127.0.0.1:31839/metrics --model prefix-trial \
    --variant "$label-$pass-c8" --mode quick --max-requests "$requests" \
    --max-concurrency 8 --warmup-requests 2 --seed 42 --stream-mode observed \
    --progress-every 16 --output-root "/results/$pass" "${reset[@]}" \
    > "$out/$pass.log" 2>&1
  echo "$label $pass completed"
done
