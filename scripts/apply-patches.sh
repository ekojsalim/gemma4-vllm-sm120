#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s CLEAN_VLLM_CHECKOUT\n' "$0" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkout=$(realpath "$1")
expected=db9527a46873454610df6dbedf79a36d6bf1a7f6

[[ "$(git -C "${checkout}" rev-parse HEAD)" == "${expected}" ]] || {
  printf 'checkout is not at %s\n' "${expected}" >&2
  exit 1
}
[[ -z "$(git -C "${checkout}" status --porcelain)" ]] || {
  printf 'checkout must be clean\n' >&2
  exit 1
}

git -C "${checkout}" apply --check "${root}/patches/0001-Port-SM120-runtime-to-vllm-0.31.patch"
git -C "${checkout}" apply "${root}/patches/0001-Port-SM120-runtime-to-vllm-0.31.patch"
printf 'v0.31 runtime and regression-test patch applied; review before committing\n'
