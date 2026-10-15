# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

import vllm.envs as envs
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    FlashInferProfileError,
    FlashInferProfileStore,
    build_nvfp4_profile_contract,
    collect_target_nvfp4_signatures,
    freeze_flashinfer_autotuner,
    nvfp4_profile_identity,
    nvfp4_tuning_buckets,
    validate_nvfp4_table,
)


SIGNATURES = (
    (2688, 16384),
    (2688, 20480),
    (2688, 43008),
    (4096, 5376),
    (8192, 5376),
    (10752, 5376),
)
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024)
COMPATIBILITY = {
    "flashinfer_version": "0.6.14",
    "flashinfer_nvfp4_source_sha256": "source",
    "flashinfer_nvfp4_module_sha256": "module",
    "vllm_nvfp4_binding_sha256": "binding",
    "torch_version": "2.10",
    "torch_cuda": "13.0",
    "torch_cxx11_abi": True,
    "python_abi": "cpython-312",
    "platform": "x86_64",
    "gpu_name": "NVIDIA GeForce RTX 5090",
    "gpu_capability": [12, 0],
}


@dataclass
class _ModelConfig:
    dtype: str = "torch.bfloat16"
    quantization: str = "modelopt"


@dataclass
class _ParallelConfig:
    tensor_parallel_size: int = 1


@dataclass
class _SchedulerConfig:
    max_num_batched_tokens: int = 1024


class _VllmConfig:
    def __init__(self, *, speculative_config=None, tp: int = 1) -> None:
        self.model_config = _ModelConfig()
        self.parallel_config = _ParallelConfig(tp)
        self.speculative_config = speculative_config


class _Runner:
    def __init__(self, *, speculative_config=None, tp: int = 1) -> None:
        self.vllm_config = _VllmConfig(
            speculative_config=speculative_config, tp=tp
        )
        self.scheduler_config = _SchedulerConfig()


def _contract(runner: _Runner | None = None, **kwargs):
    return build_nvfp4_profile_contract(
        runner or _Runner(),
        signatures=SIGNATURES,
        compatibility=kwargs.pop("compatibility", COMPATIBILITY),
        max_m=kwargs.pop("max_m", 1024),
        **kwargs,
    )


def _table(
    *,
    signatures=SIGNATURES,
    buckets=BUCKETS,
    tactic: int = 4,
    extra: tuple[int, int, int] | None = None,
) -> bytes:
    payload = {
        "_metadata": {
            "flashinfer_version": "0.6.14",
            "gpu": "NVIDIA GeForce RTX 5090",
        }
    }
    triples = [(m, k, n) for k, n in signatures for m in buckets]
    if extra is not None:
        triples.append(extra)
    for m, k, n in triples:
        profile = (
            (m, k),
            (k, n),
            (-1, max(k // 8, 1)),
            (max(k // 8, 1), n),
            (),
            (0,),
            (-1, n),
            (0,),
            (0,),
            (-1,),
        )
        key = str(("fp4_gemm", "CutlassFp4GemmRunner", profile, ()))
        payload[key] = ["CutlassFp4GemmRunner", tactic]
    return json.dumps(payload, sort_keys=True, indent=2).encode()


def _publish(store: FlashInferProfileStore, contents: bytes):
    candidate = store.temporary_candidate()
    candidate.write_bytes(contents)
    return store.publish(
        candidate,
        signatures=SIGNATURES,
        buckets=BUCKETS,
        tuning_duration_s=1.0,
    )


def test_hybrid_buckets_cover_production_domain() -> None:
    assert nvfp4_tuning_buckets(1024) == BUCKETS
    with pytest.raises(FlashInferProfileError, match="M=1..64"):
        nvfp4_tuning_buckets(32)


def test_signature_collection_uses_packed_padded_k() -> None:
    kernel_type = type("FlashInferCutlassNvFp4LinearKernel", (), {})
    holder_type = type("Holder", (), {})
    layer_type = type("Layer", (), {})
    layer = layer_type()
    layer.input_size_per_partition = 5376
    layer.output_size_per_partition = 16384
    layer.weights_padding_cols = 16
    layer.quant_method = holder_type()
    layer.quant_method.kernel = kernel_type()
    model = type("Model", (), {"modules": lambda self: [layer]})()
    runner = type("Runner", (), {"get_model": lambda self: model})()
    assert collect_target_nvfp4_signatures(runner) == ((2704, 16384),)


def test_target_only_and_mtp_share_profile_identity() -> None:
    target = _contract(_Runner())
    mtp = _contract(_Runner(speculative_config={"method": "mtp", "k": 4}))
    assert target == mtp
    assert nvfp4_profile_identity(target) == nvfp4_profile_identity(mtp)


@pytest.mark.parametrize(
    "field,value",
    [
        ("torch_cuda", "13.1"),
        ("torch_cxx11_abi", False),
        ("gpu_capability", [12, 1]),
        ("flashinfer_nvfp4_source_sha256", "new-source"),
        ("flashinfer_nvfp4_module_sha256", "new-module"),
        ("vllm_nvfp4_binding_sha256", "new-binding"),
    ],
)
def test_runtime_compatibility_changes_identity(field: str, value) -> None:
    changed = dict(COMPATIBILITY)
    changed[field] = value
    assert nvfp4_profile_identity(_contract()) != nvfp4_profile_identity(
        _contract(compatibility=changed)
    )


def test_tp_signature_and_m_bound_change_identity() -> None:
    base = nvfp4_profile_identity(_contract())
    assert base != nvfp4_profile_identity(_contract(_Runner(tp=2)))
    assert base != nvfp4_profile_identity(
        build_nvfp4_profile_contract(
            _Runner(),
            signatures=SIGNATURES[:-1],
            compatibility=COMPATIBILITY,
            max_m=1024,
        )
    )
    assert base != nvfp4_profile_identity(_contract(max_m=768))


def test_workspace_contract_changes_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    base = nvfp4_profile_identity(_contract())
    monkeypatch.setattr(
        envs,
        "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
        envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE + 4096,
    )
    assert base != nvfp4_profile_identity(_contract())


def test_complete_profile_load_does_not_rewrite(tmp_path: Path) -> None:
    store = FlashInferProfileStore(tmp_path, _contract())
    generation = _publish(store, _table())
    before = generation.config_path.stat().st_mtime_ns
    loaded = store.load_current(signatures=SIGNATURES, buckets=BUCKETS)
    assert loaded == generation
    assert generation.config_path.stat().st_mtime_ns == before


def test_concurrent_startup_publishes_once(tmp_path: Path) -> None:
    store = FlashInferProfileStore(tmp_path, _contract())
    tune_count = 0
    tune_count_lock = threading.Lock()

    def start():
        nonlocal tune_count
        with store.lock():
            current = store.load_current(signatures=SIGNATURES, buckets=BUCKETS)
            if current is not None:
                return current
            with tune_count_lock:
                tune_count += 1
            return _publish(store, _table())

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(lambda _: start(), range(2)))
    assert tune_count == 1
    assert first.generation == second.generation


@pytest.mark.parametrize("kind", ["partial", "corrupt", "stale", "checksum"])
def test_invalid_profile_is_rejected(tmp_path: Path, kind: str) -> None:
    store = FlashInferProfileStore(tmp_path, _contract())
    generation = _publish(store, _table())
    if kind == "partial":
        generation.manifest_path.unlink()
    elif kind == "corrupt":
        generation.config_path.write_text("not-json")
    elif kind == "stale":
        manifest = json.loads(generation.manifest_path.read_text())
        manifest["contract"]["kernel_contract"]["tensor_parallel_size"] = 2
        generation.manifest_path.write_text(json.dumps(manifest))
    else:
        manifest = json.loads(generation.manifest_path.read_text())
        manifest["config_sha256"] = "0" * 64
        generation.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FlashInferProfileError):
        store.load_current(signatures=SIGNATURES, buckets=BUCKETS)


def test_missing_key_unexpected_shape_and_fallback_rejected() -> None:
    missing = json.loads(_table())
    missing.pop(next(key for key in missing if key != "_metadata"))
    with pytest.raises(FlashInferProfileError, match="incomplete"):
        validate_nvfp4_table(
            json.dumps(missing).encode(), signatures=SIGNATURES, buckets=BUCKETS
        )

    with pytest.raises(FlashInferProfileError, match="unexpected"):
        validate_nvfp4_table(
            _table(extra=(3, *SIGNATURES[0])),
            signatures=SIGNATURES,
            buckets=BUCKETS,
        )

    fallback = _table(tactic=-1)
    with pytest.raises(FlashInferProfileError, match="invalid tactic"):
        validate_nvfp4_table(fallback, signatures=SIGNATURES, buckets=BUCKETS)


class _FakeTuner:
    def __init__(self, tactic: int = 4) -> None:
        self.is_tuning_mode = False
        self.tactic = tactic

    def choose_one(self, *args, **kwargs):
        return "runner", self.tactic

    def load_configs(self, *args, **kwargs):
        return True

    def save_configs(self, *args, **kwargs):
        return None

    def clear_cache(self, *args, **kwargs):
        return None


def test_frozen_profile_rejects_live_mutation_and_fallback() -> None:
    tuner = _FakeTuner()
    freeze_flashinfer_autotuner(tuner, profile_id="profile", generation="generation")
    assert tuner.choose_one("fp4_gemm", [], None, []) == ("runner", 4)
    for method in (tuner.load_configs, tuner.save_configs, tuner.clear_cache):
        with pytest.raises(FlashInferProfileError, match="mutation"):
            method("ignored")

    fallback = _FakeTuner(tactic=-1)
    freeze_flashinfer_autotuner(
        fallback, profile_id="profile", generation="generation"
    )
    with pytest.raises(FlashInferProfileError, match="no tactic"):
        fallback.choose_one("fp4_gemm", [], None, [])

    tuning = _FakeTuner()
    freeze_flashinfer_autotuner(
        tuning, profile_id="profile", generation="generation"
    )
    tuning.is_tuning_mode = True
    with pytest.raises(FlashInferProfileError, match="live"):
        tuning.choose_one("fp4_gemm", [], None, [])
