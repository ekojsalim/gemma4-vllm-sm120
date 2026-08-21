VERSION_DIR ?= versions/vllm-0.26.0
UPSTREAM_CHECKOUT ?=

.PHONY: verify release audit-027

verify:
	bash scripts/verify-version.sh "$(VERSION_DIR)"

release: verify
	bash scripts/build-release.sh "$(VERSION_DIR)"

audit-027:
	@test -n "$(UPSTREAM_CHECKOUT)" || \
		(echo 'set UPSTREAM_CHECKOUT to a clean vLLM 0.27.0 checkout' >&2; exit 2)
	bash scripts/audit-patches.sh "$(UPSTREAM_CHECKOUT)" "$(VERSION_DIR)"
