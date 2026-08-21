#!/usr/bin/env bash
set -Eeuo pipefail

release_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
package_root=${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}
expected_version=0.26.0
lock_file=${VLLM_OVERLAY_LOCK_FILE:-/tmp/gemma4-vllm-sm120-overlay.lock}

log() {
  printf '[gemma4-overlay] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

[[ -d "${package_root}/vllm" ]] || die "vLLM package not found under ${package_root}"
[[ -f "${release_root}/VERSION" ]] || die "release VERSION is missing"
[[ "$(<"${release_root}/VERSION")" == "0.1.0" ]] || die "unexpected overlay release version"

installed_version=$(
  python3 -c 'from importlib.metadata import version; print(version("vllm"))'
)
[[ "${installed_version}" == "${expected_version}" ]] || \
  die "expected vLLM ${expected_version}, found ${installed_version}"

command -v flock >/dev/null 2>&1 || die "flock is required for serialized installation"
exec 9>"${lock_file}"
flock -w 120 9 || die "timed out acquiring ${lock_file}"

(
  cd "${release_root}"
  sha256sum --check manifests/overlay-files.sha256
) >/dev/null || die "release overlay checksum validation failed"

overlay_hash_for() {
  local path=$1
  awk -v wanted="overlay/${path}" '$2 == wanted {print $1}' \
    "${release_root}/manifests/overlay-files.sha256"
}

needs_install=0
while read -r expected path; do
  [[ -n "${expected}" && -n "${path}" ]] || continue
  target="${package_root}/${path}"
  overlay_expected=$(overlay_hash_for "${path}")
  [[ -n "${overlay_expected}" ]] || die "overlay manifest has no entry for ${path}"

  if [[ -f "${target}" ]]; then
    actual=$(sha256sum "${target}" | awk '{print $1}')
    if [[ "${actual}" == "${overlay_expected}" ]]; then
      continue
    fi
    [[ "${expected}" != MISSING && "${actual}" == "${expected}" ]] || \
      die "incompatible base file ${path}: ${actual}"
    needs_install=1
  else
    [[ "${expected}" == MISSING ]] || die "required base file is missing: ${path}"
    needs_install=1
  fi
done <"${release_root}/manifests/base-files.sha256"

if [[ "${needs_install}" == 0 ]]; then
  log "overlay 0.1.0 is already installed and byte-identical"
  exit 0
fi

while IFS= read -r path; do
  [[ -n "${path}" ]] || continue
  source_file="${release_root}/overlay/${path}"
  target="${package_root}/${path}"
  [[ -f "${source_file}" ]] || die "release file is missing: overlay/${path}"
  install -D -m 0644 "${source_file}" "${target}"
done <"${release_root}/manifests/runtime-files.txt"

while read -r expected source_path; do
  [[ -n "${expected}" && -n "${source_path}" ]] || continue
  path=${source_path#overlay/}
  target="${package_root}/${path}"
  [[ -f "${target}" ]] || die "installed file is missing: ${path}"
  actual=$(sha256sum "${target}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || die "installed checksum mismatch: ${path}"
done <"${release_root}/manifests/overlay-files.sha256"

log "installed and verified overlay 0.1.0 for vLLM ${installed_version}"

