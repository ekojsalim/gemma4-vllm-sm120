#!/usr/bin/env bash
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
version_arg=${1:-versions/vllm-0.26.0}
version_dir=$(realpath "${root}/${version_arg}")

case "${version_dir}" in
  "${root}"/versions/*) ;;
  *)
    printf 'version directory must be under %s/versions\n' "${root}" >&2
    exit 2
    ;;
esac

[[ -f "${version_dir}/release.env" ]] || {
  printf 'missing release.env: %s\n' "${version_dir}" >&2
  exit 1
}
# shellcheck disable=SC1090
source "${version_dir}/release.env"

: "${BUNDLE_VERSION:?}"
: "${VLLM_VERSION:?}"
: "${VLLM_IMAGE_DIGEST:?}"
: "${PYTHON_SITE_PACKAGES:?}"

[[ "$(<"${version_dir}/VERSION")" == "${BUNDLE_VERSION}" ]] || {
  printf 'VERSION and BUNDLE_VERSION disagree\n' >&2
  exit 1
}

name="gemma4-vllm-sm120-overlay-v${BUNDLE_VERSION}"
dist="${root}/dist"
stage=$(mktemp -d)
trap 'rm -rf -- "${stage}"' EXIT

mkdir -p "${dist}" "${stage}/${name}/manifests" "${stage}/${name}/scripts"
cp -a "${version_dir}/overlay" "${stage}/${name}/"
cp "${root}/LICENSE" "${root}/NOTICE" "${root}/MODIFICATIONS.md" \
  "${version_dir}/VERSION" "${version_dir}/release.env" "${stage}/${name}/"
cp "${version_dir}/manifests/base-files.sha256" \
  "${version_dir}/manifests/overlay-files.sha256" \
  "${version_dir}/manifests/runtime-files.txt" "${stage}/${name}/manifests/"
cp "${root}/scripts/install-overlay.sh" "${stage}/${name}/scripts/"

archive="${dist}/${name}.tar.gz"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -C "${stage}" -czf "${archive}" "${name}"
sha256sum "${archive}"

