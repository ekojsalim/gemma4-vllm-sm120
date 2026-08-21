# Provenance

## Validated vLLM 0.26 control

- Upstream: `https://github.com/vllm-project/vllm`.
- Review patch base: `568afb3a13806beb53bb2e6bd518269357b237c0`.
- Official image tag: `vllm/vllm-openai:v0.26.0`.
- Immutable image digest:
  `sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52`.
- Image build revision: `ffd46bfab2128bb84146050e98b51a617c6575ab`.
- Image creation time: `2026-07-25T02:53:04.076259045Z`.
- GPU validation target: NVIDIA GeForce RTX 5090 / SM120.
- Historical tactic-table oracle SHA-256:
  `eac5588baf4584fb186662316d7f7943e67dbcffa96f495fa2812994f2acef7f`.

The tactic oracle is not distributed or required at runtime. The production
executor generates and freezes a compatible node-local mapping before serving.

## vLLM 0.27 audit input

- Upstream tag/build revision: `4bdc8a788d2e2ce9165d552b3d4d8b72604626bf`.
- Official image digest:
  `sha256:07ea4e292adf3a26b05ac97114b28849cf4551a26beb1fbe7decd3842d752ed7`.
- Image creation time: `2026-08-10T18:43:57.128093607Z`.

No v0.27 overlay or deployment asset exists yet.

