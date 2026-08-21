# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.attention.attention import _should_use_mm_prefix


@pytest.mark.parametrize(
    ("is_mm_prefix_lm", "language_model_only", "expected"),
    [(False, False, False), (True, False, True), (True, True, False)],
)
def test_language_only_model_does_not_require_mm_prefix_backend(
    is_mm_prefix_lm: bool, language_model_only: bool, expected: bool
) -> None:
    model_config = SimpleNamespace(
        is_mm_prefix_lm=is_mm_prefix_lm,
        multimodal_config=SimpleNamespace(language_model_only=language_model_only),
    )

    assert _should_use_mm_prefix(model_config) is expected
