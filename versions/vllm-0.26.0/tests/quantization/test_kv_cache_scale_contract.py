# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod


def make_layer(*, quantized_query: bool) -> torch.nn.Module:
    layer = torch.nn.Module()
    set_default_quant_scales(layer, register_buffer=True)
    layer.kv_cache_dtype = "fp8"
    layer.calculate_kv_scales = False
    layer.query_quant = object() if quantized_query else None
    layer.impl = SimpleNamespace(supports_quant_query_input=quantized_query)
    method = BaseKVCacheMethod(quant_config=None)
    method.create_weights(layer)
    layer.k_scale.weight_loader(layer.k_scale, torch.tensor(0.0035))
    layer.v_scale.weight_loader(layer.v_scale, torch.tensor(0.0356))
    return layer


@pytest.mark.parametrize("quantized_query", [False, True])
def test_missing_q_and_prob_scales_do_not_inherit_kv_scales(
    quantized_query: bool,
) -> None:
    layer = make_layer(quantized_query=quantized_query)

    BaseKVCacheMethod(quant_config=None).process_weights_after_loading(layer)

    assert layer._k_scale_float == pytest.approx(0.0035)
    assert layer._v_scale_float == pytest.approx(0.0356)
    assert layer._q_scale_float == 1.0
    assert layer._prob_scale_float == 1.0
    assert layer._q_scale.item() == 1.0
    assert layer._prob_scale.item() == 1.0


def test_checkpoint_q_and_prob_scales_remain_independent() -> None:
    layer = make_layer(quantized_query=True)
    layer.q_scale.weight_loader(layer.q_scale, torch.tensor(0.125))
    layer.prob_scale.weight_loader(layer.prob_scale, torch.tensor(0.25))

    BaseKVCacheMethod(quant_config=None).process_weights_after_loading(layer)

    assert layer._q_scale_float == pytest.approx(0.125)
    assert layer._prob_scale_float == pytest.approx(0.25)
    assert layer._q_scale.item() == pytest.approx(0.125)
    assert layer._prob_scale.item() == pytest.approx(0.25)
