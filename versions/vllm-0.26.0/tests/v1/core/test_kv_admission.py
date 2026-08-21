# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.kv_admission import (
    HybridKVAdmissionLedger,
    get_request_admission_length,
    validate_hybrid_kv_admission,
)
from vllm.v1.core.speculative_admission import (
    SpeculativeAdmissionOverlay,
    select_graph_aware_speculative_depth,
    speculative_depth_fallbacks,
    validate_hybrid_mtp_admission,
)


def reserve(
    ledger: HybridKVAdmissionLedger,
    batch_id: int,
    request_id: str,
    persistent: int,
    physical: int,
    upper: int,
):
    reservation = ledger.try_reserve(
        batch_id=batch_id,
        request_id=request_id,
        persistent_blocks=persistent,
        physical_blocks=physical,
        physical_upper_bound=upper,
    )
    assert reservation is not None
    return reservation


def test_admission_reserves_declared_request_extent():
    assert (
        get_request_admission_length(
            prompt_tokens=669, max_tokens=2048, max_model_len=6144
        )
        == 2717
    )
    assert (
        get_request_admission_length(
            prompt_tokens=4096, max_tokens=4096, max_model_len=6144
        )
        == 6144
    )


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_q3_settlement_retains_rejected_physical_tail(accepted: int):
    ledger = HybridKVAdmissionLedger(32, 2)
    reservation = reserve(ledger, 1, "r", 10, 10, 12)
    ledger.commit_allocation(reservation, 12)

    # Rejection changes logical persistence, not the already allocated page.
    persistent_after_acceptance = 10 + int(accepted == 2)
    ledger.settle_batch(
        1,
        physical_blocks={"r": 12},
        persistent_blocks={"r": persistent_after_acceptance},
    )

    assert ledger.physical["r"] == 12
    assert ledger.charged_blocks == 12
    assert ledger.stranded_blocks == 12 - persistent_after_acceptance


def test_retained_tail_is_reused_and_then_promoted():
    ledger = HybridKVAdmissionLedger(32, 2)
    first = reserve(ledger, 1, "r", 10, 10, 12)
    ledger.commit_allocation(first, 12)
    ledger.settle_batch(1, {"r": 12}, {"r": 10})

    second = reserve(ledger, 2, "r", 10, 12, 12)
    ledger.commit_allocation(second, 12)
    ledger.settle_batch(2, {"r": 12}, {"r": 12})
    assert ledger.charged_blocks == 12
    assert ledger.stranded_blocks == 0


def test_full_and_swa_page_crossings_are_one_atomic_bound():
    ledger = HybridKVAdmissionLedger(30, 2)
    # The upper bound represents one full-attention and five SWA crossings.
    reservation = reserve(ledger, 1, "r", 12, 12, 18)
    assert ledger.provisional_blocks == 6
    ledger.commit_allocation(reservation, 18)
    assert ledger.physical_blocks == 18
    assert ledger.charged_blocks == 18


def test_two_in_flight_batches_charge_union_physical_peak_once():
    ledger = HybridKVAdmissionLedger(30, 2)
    first = reserve(ledger, 1, "r", 10, 10, 12)
    ledger.commit_allocation(first, 12)
    second = reserve(ledger, 2, "r", 10, 12, 13)
    ledger.commit_allocation(second, 13)

    assert ledger.charged_blocks == 13
    assert ledger.physical_blocks == 13
    ledger.settle_batch(1, {"r": 13}, {"r": 10})
    assert ledger.charged_blocks == 13
    ledger.settle_batch(2, {"r": 13}, {"r": 10})
    assert ledger.stranded_blocks == 3


def test_two_in_flight_batches_record_independent_depth_contracts():
    ledger = HybridKVAdmissionLedger(30, 2)
    overlay = SpeculativeAdmissionOverlay(ledger)
    first = overlay.try_reserve(
        batch_id=1,
        request_id="r1",
        persistent_blocks=10,
        physical_blocks=10,
        physical_upper_bound=12,
        selected_k=4,
        lookahead_tokens=4,
        completion_epoch=1,
    )
    second = overlay.try_reserve(
        batch_id=2,
        request_id="r2",
        persistent_blocks=10,
        physical_blocks=10,
        physical_upper_bound=12,
        selected_k=2,
        lookahead_tokens=2,
        completion_epoch=2,
    )
    assert first is not None and second is not None
    assert overlay.batch_contract(1).selected_k == 4
    assert overlay.batch_contract(2).selected_k == 2
    assert overlay.batch_contract(1).request_ids == ("r1",)
    assert overlay.batch_contract(2).completion_epoch == 2


def test_one_batch_rejects_mixed_depth_or_epoch():
    ledger = HybridKVAdmissionLedger(30, 2)
    overlay = SpeculativeAdmissionOverlay(ledger)
    first = overlay.try_reserve(
        batch_id=1,
        request_id="r1",
        persistent_blocks=10,
        physical_blocks=10,
        physical_upper_bound=10,
        selected_k=3,
        lookahead_tokens=3,
        completion_epoch=1,
    )
    assert first is not None
    with pytest.raises(ValueError, match="inconsistent contract"):
        overlay.try_reserve(
            batch_id=1,
            request_id="r2",
            persistent_blocks=10,
            physical_blocks=10,
            physical_upper_bound=10,
            selected_k=2,
            lookahead_tokens=2,
            completion_epoch=1,
        )


def test_atomic_reservation_failure_and_rollback():
    ledger = HybridKVAdmissionLedger(12, 2)
    assert (
        ledger.try_reserve(
            batch_id=1,
            request_id="r",
            persistent_blocks=10,
            physical_blocks=10,
            physical_upper_bound=13,
        )
        is None
    )
    assert ledger.charged_blocks == 0

    reservation = reserve(ledger, 1, "r", 10, 10, 12)
    ledger.rollback(reservation)
    assert ledger.charged_blocks == 0
    assert not ledger.batches


def test_allocation_must_not_exceed_reserved_bound():
    ledger = HybridKVAdmissionLedger(20, 2)
    reservation = reserve(ledger, 1, "r", 10, 10, 12)
    with pytest.raises(RuntimeError, match="exceeded bound"):
        ledger.commit_allocation(reservation, 13)


@pytest.mark.parametrize("terminal_reason", ["cancellation", "abort", "failure"])
def test_terminal_request_retires_until_every_gpu_output_epoch(terminal_reason: str):
    del terminal_reason  # The ledger intentionally treats every terminal path alike.
    ledger = HybridKVAdmissionLedger(30, 2)
    first = reserve(ledger, 1, "r", 10, 10, 12)
    ledger.commit_allocation(first, 12)
    second = reserve(ledger, 2, "r", 10, 12, 13)
    ledger.commit_allocation(second, 13)

    ledger.finish_request("r", physical_blocks=13)
    assert ledger.retired_blocks == 13
    assert ledger.free_blocks == 17
    ledger.settle_batch(1, {}, {})
    assert ledger.retired_blocks == 13
    ledger.settle_batch(2, {}, {})
    assert ledger.retired_blocks == 0
    assert ledger.free_blocks == 30


def test_normal_finish_releases_without_active_batch():
    ledger = HybridKVAdmissionLedger(20, 1)
    reservation = reserve(ledger, 1, "r", 10, 10, 10)
    ledger.commit_allocation(reservation, 10)
    ledger.settle_batch(1, {"r": 10}, {"r": 10})
    ledger.finish_request("r", physical_blocks=10)
    assert ledger.free_blocks == 20


def test_duplicate_or_stale_batch_settlement_fails_closed():
    ledger = HybridKVAdmissionLedger(20, 1)
    reservation = reserve(ledger, 1, "r", 10, 10, 10)
    ledger.commit_allocation(reservation, 10)
    ledger.settle_batch(1, {"r": 10}, {"r": 10})
    with pytest.raises(KeyError):
        ledger.settle_batch(1, {"r": 10}, {"r": 10})


def test_stale_completion_epoch_fails_before_settlement():
    ledger = HybridKVAdmissionLedger(20, 1)
    overlay = SpeculativeAdmissionOverlay(ledger)
    reservation = overlay.try_reserve(
        batch_id=4,
        request_id="r",
        persistent_blocks=10,
        physical_blocks=10,
        physical_upper_bound=10,
        selected_k=3,
        lookahead_tokens=3,
        completion_epoch=9,
    )
    assert reservation is not None
    with pytest.raises(ValueError, match="stale completion epoch"):
        overlay.settle_batch(
            4,
            {"r": 10},
            {"r": 10},
            completion_epoch=8,
        )
    assert overlay.batch_contract(4).completion_epoch == 9


def test_batch_settlement_validation_is_atomic():
    ledger = HybridKVAdmissionLedger(30, 1)
    first = reserve(ledger, 1, "r1", 10, 0, 10)
    second = reserve(ledger, 1, "r2", 10, 0, 10)
    ledger.commit_allocation(first, 10)
    ledger.commit_allocation(second, 10)

    before = (dict(ledger.persistent), dict(ledger.physical), dict(ledger.batches))
    with pytest.raises(ValueError, match="r2"):
        ledger.settle_batch(1, {"r1": 9}, {"r1": 10})

    assert dict(ledger.persistent) == before[0]
    assert dict(ledger.physical) == before[1]
    assert dict(ledger.batches) == before[2]


def test_batch_limit_fails_closed():
    ledger = HybridKVAdmissionLedger(20, 1)
    reserve(ledger, 1, "r1", 5, 0, 5)
    assert (
        ledger.try_reserve(
            batch_id=2,
            request_id="r2",
            persistent_blocks=5,
            physical_blocks=0,
            physical_upper_bound=5,
        )
        is None
    )


def valid_mtp_config() -> dict[str, object]:
    return {
        "speculative_method": "mtp",
        "num_spec_tokens": 2,
        "use_eagle": True,
        "num_lookahead_tokens": 2,
        "dynamic_speculation": False,
        "model_type": "gemma4_text",
        "architectures": ("Gemma4ForConditionalGeneration",),
    }


def test_target_only_hybrid_cache_admission_is_supported():
    validate_hybrid_kv_admission(
        has_connector=False,
        is_encoder_decoder=False,
        has_mamba_layers=False,
        prefix_caching=False,
        cache_groups_supported=True,
        max_concurrent_batches=2,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("has_connector", True, "connectors"),
        ("is_encoder_decoder", True, "decoder-only"),
        ("has_mamba_layers", True, "decoder-only"),
        ("prefix_caching", True, "prefix caching"),
        ("cache_groups_supported", False, "SWA/full-attention"),
        ("max_concurrent_batches", 3, "at most two"),
    ],
)
def test_unsupported_hybrid_cache_configurations_fail_closed(
    field: str, value: object, message: str
):
    config = {
        "has_connector": False,
        "is_encoder_decoder": False,
        "has_mamba_layers": False,
        "prefix_caching": False,
        "cache_groups_supported": True,
        "max_concurrent_batches": 2,
    }
    config[field] = value
    with pytest.raises(ValueError, match=message):
        validate_hybrid_kv_admission(**config)


@pytest.mark.parametrize("num_spec_tokens", [1, 2, 4, 8])
def test_fixed_mtp_depth_is_supported(num_spec_tokens: int):
    config = valid_mtp_config()
    config["num_spec_tokens"] = num_spec_tokens
    config["num_lookahead_tokens"] = num_spec_tokens
    validate_hybrid_mtp_admission(**config)


def test_bounded_dynamic_mtp_depths_are_supported():
    config = valid_mtp_config()
    config.update(
        num_spec_tokens=4,
        num_lookahead_tokens=4,
        dynamic_speculation=True,
        dynamic_speculative_depths=(4, 3, 2),
    )
    validate_hybrid_mtp_admission(**config)


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [(1, (4, 1)), (12, (4, 12)), (16, (3, 16)), (24, (2, 21))],
)
def test_graph_aware_depth_and_membership_bound(
    batch_size: int, expected: tuple[int, int]
):
    lookup = [0] + [4] * 12 + [3] * 4 + [2] * 16
    assert (
        select_graph_aware_speculative_depth(
            candidate_count=batch_size,
            depth_lookup=lookup,
            max_graph_tokens=64,
        )
        == expected
    )


def test_depth_fallback_order_ends_target_only():
    assert speculative_depth_fallbacks(4) == (4, 3, 2, 0)
    assert speculative_depth_fallbacks(3) == (3, 2, 0)
    assert speculative_depth_fallbacks(2) == (2, 0)


@pytest.mark.parametrize(
    "depths", [(1, 2, 3), (2, 4, 8), (), (0, 2, 4)]
)
def test_unbounded_dynamic_mtp_depths_fail_closed(depths: tuple[int, ...]):
    config = valid_mtp_config()
    config.update(
        num_spec_tokens=4,
        num_lookahead_tokens=4,
        dynamic_speculation=True,
        dynamic_speculative_depths=depths,
    )
    with pytest.raises(ValueError, match="bounded k=2/3/4"):
        validate_hybrid_mtp_admission(**config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("speculative_method", "draft_model", "built-in MTP"),
        ("num_spec_tokens", 0, "bounded speculation"),
        ("use_eagle", False, "built-in MTP"),
        ("num_lookahead_tokens", 3, "matching maximum lookahead"),
        ("model_type", "llama", "Gemma 4"),
        ("architectures", ("Gemma4TextModel",), "Gemma 4"),
    ],
)
def test_unsupported_speculative_configurations_fail_closed(
    field: str, value: object, message: str
):
    config = valid_mtp_config()
    config[field] = value
    with pytest.raises(ValueError, match=message):
        validate_hybrid_mtp_admission(**config)
