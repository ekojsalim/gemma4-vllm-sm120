VERSION_DIR ?= versions/vllm-0.31.0
UPSTREAM_CHECKOUT ?=

.PHONY: verify release audit audit-027

verify:
	bash scripts/verify-version.sh "$(VERSION_DIR)"

release: verify
	bash scripts/build-release.sh "$(VERSION_DIR)"

audit:
	@test -n "$(UPSTREAM_CHECKOUT)" || \
		(echo 'set UPSTREAM_CHECKOUT to a clean upstream vLLM checkout' >&2; exit 2)
	bash scripts/audit-patches.sh "$(UPSTREAM_CHECKOUT)" "$(VERSION_DIR)"

# Historical audit uses the historical patch stack explicitly.
audit-027:
	$(MAKE) audit VERSION_DIR=versions/vllm-0.26.0 UPSTREAM_CHECKOUT="$(UPSTREAM_CHECKOUT)"
