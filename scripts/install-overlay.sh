#!/usr/bin/env bash
set -Eeuo pipefail

release_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f "${release_root}/release.env" ]] || {
  printf '[gemma4-overlay] ERROR: release.env is missing\n' >&2
  exit 1
}
# shellcheck disable=SC1090
source "${release_root}/release.env"

: "${BUNDLE_VERSION:?}"
: "${VLLM_VERSION:?}"
: "${VLLM_IMAGE_DIGEST:?}"
: "${PYTHON_SITE_PACKAGES:?}"

package_root=${VLLM_SITE_PACKAGES:-${PYTHON_SITE_PACKAGES}}
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
[[ "$(<"${release_root}/VERSION")" == "${BUNDLE_VERSION}" ]] || \
  die "VERSION and release.env disagree"

installed_version=$(
  python3 -c 'from importlib.metadata import version; print(version("vllm"))'
)
[[ "${installed_version}" == "${VLLM_VERSION}" ]] || \
  die "expected vLLM ${VLLM_VERSION}, found ${installed_version}"

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
  log "overlay ${BUNDLE_VERSION} is already installed and byte-identical"
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

log "installed overlay ${BUNDLE_VERSION} for vLLM ${installed_version}"
log "validated base image contract ${VLLM_IMAGE_DIGEST}"

