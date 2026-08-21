#!/usr/bin/env bash
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
version=$(<"${root}/VERSION")
name="gemma4-vllm-sm120-overlay-v${version}"
dist="${root}/dist"
stage=$(mktemp -d)
trap 'rm -rf -- "${stage}"' EXIT

mkdir -p "${dist}" "${stage}/${name}/manifests" "${stage}/${name}/scripts"
cp -a "${root}/overlay" "${stage}/${name}/"
cp "${root}/LICENSE" "${root}/NOTICE" "${root}/MODIFICATIONS.md" \
  "${root}/VERSION" "${stage}/${name}/"
cp "${root}/manifests/base-files.sha256" \
  "${root}/manifests/overlay-files.sha256" \
  "${root}/manifests/runtime-files.txt" "${stage}/${name}/manifests/"
cp "${root}/scripts/install-overlay.sh" "${stage}/${name}/scripts/"

archive="${dist}/${name}.tar.gz"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -C "${stage}" -czf "${archive}" "${name}"
sha256sum "${archive}"

