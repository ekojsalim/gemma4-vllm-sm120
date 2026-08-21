#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'usage: %s UPSTREAM_CHECKOUT [VERSION_DIR]\n' "$0" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkout=$(realpath "$1")
version_arg=${2:-versions/vllm-0.26.0}
version_dir=$(realpath "${root}/${version_arg}")

git -C "${checkout}" rev-parse --verify HEAD >/dev/null
printf 'upstream_commit\t%s\n' "$(git -C "${checkout}" rev-parse HEAD)"
printf 'patch\tdirect_context_check\n'

while IFS= read -r patch; do
  if git -C "${checkout}" apply --check "${patch}" >/dev/null 2>&1; then
    status=applies
  else
    status=conflicts
  fi
  printf '%s\t%s\n' "$(basename "${patch}")" "${status}"
done < <(find "${version_dir}/patches" -maxdepth 1 -type f -name '*.patch' | sort)

printf '\nContext checks are independent and do not account for patch dependencies.\n'
printf 'They are an upgrade triage aid, not a correctness or compatibility gate.\n'
