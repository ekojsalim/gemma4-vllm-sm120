# vLLM 0.26.0 compatibility bundle

Status: historical validated control; not the default configuration.

The overlay files are byte-identical to release `v0.1.0` and the preserved
production deployment. Bundle version `0.1.1` changes only repository/release
organization and the generic installer metadata; it does not change runtime
Python bytes.

The exact runtime, build, patch-base, and site-packages contracts are in
`release.env`. `manifests/base-files.sha256` was measured from the immutable
official image. Two runtime modules are new files and are marked `MISSING` in
the base manifest.

The accepted historical FlashInfer tactic table is intentionally absent. A node
generates and freezes a complete local profile before readiness.

The 0.1.1 archive is a rebuilt packaging candidate, not the published 0.1.0
asset. Existing tags and assets are unchanged.
