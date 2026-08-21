#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'usage: %s CLEAN_VLLM_CHECKOUT [VERSION_DIR]\n' "$0" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkout=$(realpath "$1")
version_arg=${2:-versions/vllm-0.26.0}
version_dir=$(realpath "${root}/${version_arg}")

case "${version_dir}" in
  "${root}"/versions/*) ;;
  *)
    printf 'version directory must be under %s/versions\n' "${root}" >&2
    exit 2
    ;;
esac

# shellcheck disable=SC1090
source "${version_dir}/release.env"

[[ "$(git -C "${checkout}" rev-parse HEAD)" == "${PATCH_BASE_COMMIT}" ]] || {
  printf 'checkout is not at %s\n' "${PATCH_BASE_COMMIT}" >&2
  exit 1
}
[[ -z "$(git -C "${checkout}" status --porcelain)" ]] || {
  printf 'checkout must be clean\n' >&2
  exit 1
}

mapfile -t patch_files < <(
  find "${version_dir}/patches" -maxdepth 1 -type f -name '*.patch' | sort
)
[[ "${#patch_files[@]}" -gt "${PATCH_MAIL_COUNT}" ]]

git -C "${checkout}" am "${patch_files[@]:0:PATCH_MAIL_COUNT}"
git -C "${checkout}" apply "${patch_files[@]:PATCH_MAIL_COUNT}"

printf 'patch stack applied; review and commit non-mail patches\n'

