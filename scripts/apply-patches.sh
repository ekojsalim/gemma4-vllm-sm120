#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s CLEAN_VLLM_CHECKOUT\n' "$0" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkout=$(realpath "$1")
expected=568afb3a13806beb53bb2e6bd518269357b237c0

[[ "$(git -C "${checkout}" rev-parse HEAD)" == "${expected}" ]] || {
  printf 'checkout is not at %s\n' "${expected}" >&2
  exit 1
}
[[ -z "$(git -C "${checkout}" status --porcelain)" ]] || {
  printf 'checkout must be clean\n' >&2
  exit 1
}

git -C "${checkout}" am \
  "${root}/patches/0001-Fix-attention-selection-for-language-only-Gemma.patch" \
  "${root}/patches/0002-Support-bounded-small-M-segmented-Triton-attention.patch" \
  "${root}/patches/0003-Add-transactional-hybrid-KV-cache-admission.patch"

git -C "${checkout}" apply \
  "${root}/patches/0004-Share-FP8-KV-scale-ownership.patch" \
  "${root}/patches/0005-Add-speculative-admission-module.patch" \
  "${root}/patches/0006-Split-speculative-admission-and-graph-depth.patch" \
  "${root}/patches/0007-Allow-optional-FP8-assistant-config.patch" \
  "${root}/patches/0008-Add-node-local-immutable-NVFP4-autotune-profiles.patch" \
  "${root}/patches/0009-Clarify-FP8-Q-and-probability-scale-contract.patch"

printf 'patch stack applied; review and commit patches 0004-0009\n'

