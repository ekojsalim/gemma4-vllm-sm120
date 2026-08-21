#!/usr/bin/env bash
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${root}"

sha256sum --check manifests/overlay-files.sha256
bash -n scripts/install-overlay.sh
bash -n scripts/build-release.sh
bash -n scripts/apply-patches.sh

while IFS= read -r path; do
  [[ -f "overlay/${path}" ]] || {
    printf 'missing overlay file: %s\n' "${path}" >&2
    exit 1
  }
done <manifests/runtime-files.txt

[[ "$(find overlay -type f | wc -l)" == "$(wc -l <manifests/runtime-files.txt)" ]]
[[ ! -e config/flashinfer-autotune ]]

printf 'source and overlay verification: PASS\n'
