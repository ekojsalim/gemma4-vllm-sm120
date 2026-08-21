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

[[ -f "${version_dir}/release.env" ]]
# shellcheck disable=SC1090
source "${version_dir}/release.env"

: "${BUNDLE_VERSION:?}"
: "${VLLM_VERSION:?}"
: "${VLLM_IMAGE_REF:?}"
: "${VLLM_IMAGE_DIGEST:?}"
: "${VLLM_BUILD_COMMIT:?}"
: "${PATCH_BASE_COMMIT:?}"
: "${PATCH_MAIL_COUNT:?}"
: "${PYTHON_SITE_PACKAGES:?}"

[[ "$(<"${version_dir}/VERSION")" == "${BUNDLE_VERSION}" ]]

(
  cd "${version_dir}"
  sha256sum --check manifests/overlay-files.sha256
)

bash -n "${root}/scripts/install-overlay.sh"
bash -n "${root}/scripts/build-release.sh"
bash -n "${root}/scripts/apply-patches.sh"
bash -n "${root}/scripts/audit-patches.sh"

while IFS= read -r path; do
  [[ -f "${version_dir}/overlay/${path}" ]] || {
    printf 'missing overlay file: %s\n' "${path}" >&2
    exit 1
  }
done <"${version_dir}/manifests/runtime-files.txt"

overlay_count=$(find "${version_dir}/overlay" -type f | wc -l)
manifest_count=$(wc -l <"${version_dir}/manifests/runtime-files.txt")
[[ "${overlay_count}" == "${manifest_count}" ]]
[[ ! -e "${version_dir}/config/flashinfer-autotune" ]]

printf 'version bundle verification: PASS (%s, vLLM %s)\n' \
  "${BUNDLE_VERSION}" "${VLLM_VERSION}"
