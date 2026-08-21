# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backend import (
    resolve_kv_scale_owner,
    sync_kv_scale_consumers,
)
from vllm.v1.worker import utils as worker_utils
from vllm.v1.worker.utils import bind_kv_cache


class _AttentionStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_q_scale", torch.ones(1))
        self.register_buffer("_k_scale", torch.ones(1))
        self.register_buffer("_v_scale", torch.ones(1))
        self.register_buffer("_k_scale_cpu", torch.ones(1))
        self.register_buffer("_v_scale_cpu", torch.ones(1))
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        self.kv_cache_dtype = "auto"
        self.kv_sharing_target_layer_name = None
        self.calculate_kv_scales = False


@pytest.fixture(autouse=True)
def _cpu_platform(monkeypatch):
    monkeypatch.setattr(worker_utils.current_platform, "is_cpu", lambda: True)


def test_bind_kv_cache():
    ctx = {
        "layers.0.self_attn": _AttentionStub(),
        "layers.1.self_attn": _AttentionStub(),
        "layers.2.self_attn": _AttentionStub(),
        "layers.3.self_attn": _AttentionStub(),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.zeros((1,)),
        "layers.2.self_attn": torch.zeros((1,)),
        "layers.3.self_attn": torch.zeros((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx["layers.0.self_attn"].kv_cache is kv_cache["layers.0.self_attn"]
    assert ctx["layers.1.self_attn"].kv_cache is kv_cache["layers.1.self_attn"]
    assert ctx["layers.2.self_attn"].kv_cache is kv_cache["layers.2.self_attn"]
    assert ctx["layers.3.self_attn"].kv_cache is kv_cache["layers.3.self_attn"]

    assert runner_kv_caches[0] is kv_cache["layers.0.self_attn"]
    assert runner_kv_caches[1] is kv_cache["layers.1.self_attn"]
    assert runner_kv_caches[2] is kv_cache["layers.2.self_attn"]
    assert runner_kv_caches[3] is kv_cache["layers.3.self_attn"]


def test_bind_kv_cache_non_attention():
    # example from Jamba PP=2
    ctx = {
        "model.layers.20.attn": _AttentionStub(),
        "model.layers.28.attn": _AttentionStub(),
    }
    kv_cache = {
        "model.layers.20.attn": torch.zeros((1,)),
        "model.layers.28.attn": torch.zeros((1,)),
    }

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.20.attn"].kv_cache is kv_cache["model.layers.20.attn"]
    assert ctx["model.layers.28.attn"].kv_cache is kv_cache["model.layers.28.attn"]

    assert runner_kv_caches[0] is kv_cache["model.layers.20.attn"]
    assert runner_kv_caches[1] is kv_cache["model.layers.28.attn"]


def test_bind_kv_cache_draft_model():
    layer_names = [
        "model.layers.0.attn",
        "model.layers.1.attn",
        "draft_model.layers.0.attn",
        "draft_model.layers.1.attn",
    ]
    ctx = {layer_name: _AttentionStub() for layer_name in layer_names}
    kv_cache = {layer_name: torch.zeros((1,)) for layer_name in layer_names}
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.0.attn"].kv_cache is kv_cache["model.layers.0.attn"]
    assert ctx["model.layers.1.attn"].kv_cache is kv_cache["model.layers.1.attn"]
    assert (
        ctx["draft_model.layers.0.attn"].kv_cache
        is kv_cache["draft_model.layers.0.attn"]
    )
    assert (
        ctx["draft_model.layers.1.attn"].kv_cache
        is kv_cache["draft_model.layers.1.attn"]
    )

    # caches are ordered by layer_index, interleaving target and draft model
    assert runner_kv_caches[0] is kv_cache["model.layers.0.attn"]
    assert runner_kv_caches[1] is kv_cache["draft_model.layers.0.attn"]
    assert runner_kv_caches[2] is kv_cache["model.layers.1.attn"]
    assert runner_kv_caches[3] is kv_cache["draft_model.layers.1.attn"]


def _configure_fp8_scale_owner(owner, consumer, owner_name):
    owner.kv_cache_dtype = "fp8"
    consumer.kv_cache_dtype = "fp8"
    consumer.kv_sharing_target_layer_name = owner_name
    owner._k_scale.fill_(0.125)
    owner._v_scale.fill_(0.375)
    owner._k_scale_float = 0.125
    owner._v_scale_float = 0.375
    owner._k_scale_cpu.fill_(0.125)
    owner._v_scale_cpu.fill_(0.375)


@pytest.mark.parametrize("attention_kind", ["sliding", "full"])
def test_bind_shared_fp8_kv_scale_owner(attention_kind):
    from vllm.v1.attention.backends import flashinfer, triton_attn

    owner_name = f"model.layers.58.{attention_kind}_attn"
    consumer_name = f"draft_model.layers.0.{attention_kind}_attn"
    owner = _AttentionStub()
    consumer = _AttentionStub()
    _configure_fp8_scale_owner(owner, consumer, owner_name)
    consumer._q_scale.fill_(0.75)
    cache = torch.arange(16, dtype=torch.uint8)
    cache_before = cache.clone()
    kv_cache = {owner_name: cache, consumer_name: cache}

    bind_kv_cache(kv_cache, {owner_name: owner, consumer_name: consumer}, [])

    assert resolve_kv_scale_owner(consumer) is owner
    assert triton_attn.resolve_kv_scale_owner(consumer) is owner
    assert flashinfer.resolve_kv_scale_owner(consumer) is owner
    assert consumer._k_scale.item() == pytest.approx(0.125)
    assert consumer._v_scale.item() == pytest.approx(0.375)
    assert consumer._k_scale_float == pytest.approx(0.125)
    assert consumer._v_scale_float == pytest.approx(0.375)
    assert consumer._k_scale_cpu.item() == pytest.approx(0.125)
    assert consumer._v_scale_cpu.item() == pytest.approx(0.375)
    assert consumer._q_scale.item() == pytest.approx(0.75)
    assert torch.equal(cache, cache_before)


def test_bind_shared_bf16_kv_has_no_scale_owner():
    owner_name = "model.layers.59.self_attn"
    consumer_name = "draft_model.layers.3.self_attn"
    owner = _AttentionStub()
    consumer = _AttentionStub()
    owner.kv_cache_dtype = "auto"
    consumer.kv_cache_dtype = "auto"
    consumer.kv_sharing_target_layer_name = owner_name
    cache = torch.zeros(16, dtype=torch.bfloat16)

    bind_kv_cache(
        {owner_name: cache, consumer_name: cache},
        {owner_name: owner, consumer_name: consumer},
        [],
    )

    assert resolve_kv_scale_owner(consumer) is consumer
    assert consumer._k_scale.item() == 1.0
    assert consumer._v_scale.item() == 1.0


def test_shared_fp8_kv_scale_owner_refreshes_after_loading():
    owner_name = "model.layers.58.self_attn"
    consumer_name = "draft_model.layers.0.self_attn"
    owner = _AttentionStub()
    consumer = _AttentionStub()
    owner.kv_cache_dtype = consumer.kv_cache_dtype = "fp8"
    consumer.kv_sharing_target_layer_name = owner_name
    cache = torch.zeros(16, dtype=torch.uint8)
    bind_kv_cache(
        {owner_name: cache, consumer_name: cache},
        {owner_name: owner, consumer_name: consumer},
        [],
    )

    owner._k_scale.fill_(0.0625)
    owner._v_scale.fill_(0.5)
    owner._k_scale_float = 0.0625
    owner._v_scale_float = 0.5
    owner._k_scale_cpu.fill_(0.0625)
    owner._v_scale_cpu.fill_(0.5)
    sync_kv_scale_consumers(owner)

    assert consumer._k_scale.item() == pytest.approx(0.0625)
    assert consumer._v_scale.item() == pytest.approx(0.5)
    assert consumer._k_scale_float == pytest.approx(0.0625)
    assert consumer._v_scale_float == pytest.approx(0.5)


def test_bind_shared_kv_rejects_incompatible_quantization():
    owner_name = "model.layers.58.self_attn"
    consumer_name = "draft_model.layers.0.self_attn"
    owner = _AttentionStub()
    consumer = _AttentionStub()
    owner.kv_cache_dtype = "fp8"
    consumer.kv_cache_dtype = "auto"
    consumer.kv_sharing_target_layer_name = owner_name
    cache = torch.zeros(16, dtype=torch.uint8)

    with pytest.raises(ValueError, match="quantization mismatch"):
        bind_kv_cache(
            {owner_name: cache, consumer_name: cache},
            {owner_name: owner, consumer_name: consumer},
            [],
        )


def test_bind_shared_fp8_kv_rejects_scale_granularity():
    owner_name = "model.layers.58.self_attn"
    consumer_name = "draft_model.layers.0.self_attn"
    owner = _AttentionStub()
    consumer = _AttentionStub()
    owner.kv_cache_dtype = consumer.kv_cache_dtype = "fp8"
    consumer.kv_sharing_target_layer_name = owner_name
    owner._buffers["_k_scale"] = torch.ones(16, dtype=torch.float32)
    cache = torch.zeros(16, dtype=torch.uint8)

    with pytest.raises(ValueError, match="scale shape mismatch"):
        bind_kv_cache(
            {owner_name: cache, consumer_name: cache},
            {owner_name: owner, consumer_name: consumer},
            [],
        )
