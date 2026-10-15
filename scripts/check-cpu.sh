#!/usr/bin/env bash
# Run inside the pinned image after install-overlay.sh; no GPU is required.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv=$(mktemp -d)
trap 'rm -rf -- "$venv"' EXIT
uv venv --system-site-packages "$venv"
uv pip install --python "$venv/bin/python" pytest
cd "$root"
uv run --no-project --python "$venv/bin/python" python -m pytest --noconftest -q \
  tests/model_executor/layers/test_attention.py \
  tests/quantization/test_kv_cache_scale_contract.py \
  tests/v1/core/test_kv_admission.py \
  tests/v1/worker/test_shared_kv_scale_binding.py \
  tests/v1/spec_decode/test_dynamic_sd_cug.py \
  tests/v1/core/test_kv_cache_utils.py::test_global_hybrid_admission_replaces_per_group_transient_reserve
