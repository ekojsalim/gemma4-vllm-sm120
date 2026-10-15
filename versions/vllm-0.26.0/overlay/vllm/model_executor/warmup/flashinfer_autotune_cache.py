# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Node-local, immutable FlashInfer autotune profiles.

The FlashInfer cache keys already contain the concrete GEMM signature.  This
module supplies the missing outer lifecycle: a kernel-contract identity, exact
coverage validation, single-writer publication, and immutable generations.
"""

import ast
import fcntl
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import types
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import vllm.envs as envs
from vllm import __version__

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


PROFILE_SCHEMA_VERSION = 1
PROFILE_FILE_NAME = "autotune_configs.json"
PROFILE_MANIFEST_NAME = "profile.json"
CURRENT_FILE_NAME = "CURRENT"
NVFP4_OP = "fp4_gemm"
NVFP4_RUNNER = "CutlassFp4GemmRunner"
NVFP4_KERNEL = "FlashInferCutlassNvFp4LinearKernel"
MIN_PRODUCTION_M = 64
MAX_SERIALIZED_TACTIC = 4095


class FlashInferProfileError(RuntimeError):
    """A node-local profile is incomplete, incompatible, or mutable."""


@dataclass(frozen=True)
class ProfileGeneration:
    profile_id: str
    generation: str
    config_path: Path
    manifest_path: Path
    config_sha256: str
    entries: int


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=f".{path.name}."
    )
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    """Compatibility wrapper retained for non-profile FlashInfer warmups."""
    _atomic_write(cache_path, contents)


def nvfp4_tuning_buckets(max_m: int) -> tuple[int, ...]:
    """Mirror FlashInfer 0.6.14's hybrid token buckets exactly."""
    if max_m < MIN_PRODUCTION_M:
        raise FlashInferProfileError(
            f"NVFP4 profile max M {max_m} does not cover required M=1..64"
        )

    buckets: list[int] = []
    value = 1
    while value <= min(max_m, 256):
        buckets.append(value)
        value *= 2
    value = 512
    while value <= min(max_m, 2048):
        buckets.append(value)
        value += 256
    value = 2560
    while value <= min(max_m, 4096):
        buckets.append(value)
        value += 512
    value = 8192
    while value <= max_m:
        buckets.append(value)
        value *= 2
    if not buckets or buckets[-1] != max_m:
        buckets.append(max_m)
    return tuple(sorted(set(buckets)))


def collect_target_nvfp4_signatures(
    runner: "GPUModelRunner",
) -> tuple[tuple[int, int], ...]:
    """Return target CUTLASS NVFP4 ``(K, N)`` signatures.

    ``runner.get_model()`` is the target model.  The speculative assistant is
    owned separately by the drafter and is deliberately outside this walk.
    """
    signatures: set[tuple[int, int]] = set()
    for layer in runner.get_model().modules():
        for holder_name in ("quant_method", "scheme"):
            holder = getattr(layer, holder_name, None)
            kernel = getattr(holder, "kernel", None)
            if kernel is None or kernel.__class__.__name__ != NVFP4_KERNEL:
                continue
            logical_k = getattr(layer, "input_size_per_partition", None)
            n = getattr(layer, "output_size_per_partition", None)
            if (
                not isinstance(logical_k, int)
                or not isinstance(n, int)
                or logical_k <= 0
                or n <= 0
            ):
                raise FlashInferProfileError(
                    f"cannot derive NVFP4 signature from {layer.__class__.__name__}"
                )
            # FlashInfer keys the packed uint8 FP4 A/B tensors, so K is half
            # the logical BF16 width plus any CUTLASS byte-column padding.
            padding_bytes = getattr(layer, "weights_padding_cols", 0)
            if not isinstance(padding_bytes, int) or padding_bytes < 0:
                raise FlashInferProfileError("invalid NVFP4 CUTLASS K padding")
            if logical_k % 2:
                raise FlashInferProfileError("NVFP4 logical K must be even")
            packed_k = logical_k // 2 + padding_bytes
            signatures.add((packed_k, n))
    return tuple(sorted(signatures))


def _kernel_source_digest() -> str:
    """Hash the installed Python dispatch and SM120 source specialization."""
    import flashinfer

    root = Path(flashinfer.__file__).resolve().parent
    paths = (
        root / "gemm" / "gemm_base.py",
        root / "data" / "csrc" / "fp4_gemm_cutlass_sm120.cu",
        root
        / "data"
        / "include"
        / "flashinfer"
        / "gemm"
        / "fp4_gemm_cutlass_template_sm120.h",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FlashInferProfileError(
                f"FlashInfer NVFP4 compatibility source is missing: {path}"
            )
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _installed_module_digest() -> str:
    """Hash the exact precompiled SM120 module selected by this runner."""
    import flashinfer_jit_cache

    root = Path(flashinfer_jit_cache.__file__).resolve().parent
    module = (
        root
        / "jit_cache"
        / "fp4_gemm_cutlass_sm120"
        / "fp4_gemm_cutlass_sm120.so"
    )
    if not module.is_file():
        raise FlashInferProfileError(
            f"FlashInfer SM120 NVFP4 module is missing: {module}"
        )
    return _sha256_file(module)


def _vllm_binding_digest() -> str:
    """Hash the runner-side packing, workspace, and epilogue binding."""
    binding = (
        Path(__file__).resolve().parents[1]
        / "kernels"
        / "linear"
        / "nvfp4"
        / "flashinfer.py"
    )
    if not binding.is_file():
        raise FlashInferProfileError(f"vLLM NVFP4 binding is missing: {binding}")
    return _sha256_file(binding)


def runtime_compatibility() -> dict[str, Any]:
    import flashinfer
    import torch

    if not torch.cuda.is_available():
        raise FlashInferProfileError("CUDA is required for an NVFP4 profile")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "flashinfer_version": flashinfer.__version__,
        "flashinfer_nvfp4_source_sha256": _kernel_source_digest(),
        "flashinfer_nvfp4_module_sha256": _installed_module_digest(),
        "vllm_nvfp4_binding_sha256": _vllm_binding_digest(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch.compiled_with_cxx11_abi()),
        "python_abi": sys.implementation.cache_tag,
        "platform": platform.machine(),
        "gpu_name": props.name,
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
    }


def build_nvfp4_profile_contract(
    runner: "GPUModelRunner",
    *,
    signatures: Sequence[tuple[int, int]] | None = None,
    compatibility: Mapping[str, Any] | None = None,
    max_m: int | None = None,
) -> dict[str, Any]:
    """Build an identity containing only target-kernel compatibility facts."""
    config = runner.vllm_config
    if signatures is None:
        signatures = collect_target_nvfp4_signatures(runner)
    normalized_signatures = tuple(sorted(set(signatures)))
    if not normalized_signatures:
        raise FlashInferProfileError("target has no FlashInfer CUTLASS NVFP4 signatures")
    if max_m is None:
        max_m = runner.scheduler_config.max_num_batched_tokens
    buckets = nvfp4_tuning_buckets(max_m)
    if compatibility is None:
        compatibility = runtime_compatibility()

    model_config = config.model_config
    parallel_config = config.parallel_config
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "runtime": dict(sorted(compatibility.items())),
        "vllm_version": __version__,
        "kernel_contract": {
            "backend": NVFP4_KERNEL,
            "format": "ModelOpt-NVFP4-W4A4",
            "activation_scale_layout": "128x4-swizzled",
            "weight_scale_layout": "128x4-swizzled",
            "output_dtype": str(model_config.dtype),
            "quantization": str(model_config.quantization),
            "tensor_parallel_size": parallel_config.tensor_parallel_size,
            "workspace_bytes": envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            "signatures_kn": [list(signature) for signature in normalized_signatures],
            "m_buckets": list(buckets),
            "max_m": max_m,
        },
    }


def nvfp4_profile_identity(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    """Return the canonical target NVFP4 profile identity.

    This intentionally ignores speculative decoding, served-model aliases,
    scheduler policy, graph capture sizes, and unrelated request limits.
    """
    return nvfp4_profile_identity(build_nvfp4_profile_contract(runner))


def _profile_root() -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        return Path(override_dir).expanduser()
    return Path(envs.VLLM_CACHE_ROOT) / "flashinfer_nvfp4_profiles"


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    """Compatibility path for callers that still expect a single cache file."""
    return _profile_root() / flashinfer_autotune_cache_hash(runner) / PROFILE_FILE_NAME


def _decode_profile_key(raw_key: str) -> tuple[Any, ...]:
    try:
        key = ast.literal_eval(raw_key)
    except (SyntaxError, ValueError) as exc:
        raise FlashInferProfileError(f"invalid FlashInfer profile key: {raw_key}") from exc
    if not isinstance(key, tuple) or len(key) != 4:
        raise FlashInferProfileError(f"unexpected FlashInfer profile key: {raw_key}")
    return key


def validate_nvfp4_table(
    contents: bytes,
    *,
    signatures: Sequence[tuple[int, int]],
    buckets: Sequence[int],
) -> tuple[str, int]:
    """Validate exact target signature/M coverage and absence of fallbacks."""
    try:
        payload = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FlashInferProfileError("profile is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise FlashInferProfileError("profile root must be a JSON object")

    expected = {
        (int(m), int(k), int(n))
        for k, n in signatures
        for m in buckets
    }
    actual: set[tuple[int, int, int]] = set()
    for raw_key, value in payload.items():
        if raw_key == "_metadata":
            continue
        op, runner_name, profile, extras = _decode_profile_key(raw_key)
        if op != NVFP4_OP or runner_name != NVFP4_RUNNER or extras != ():
            raise FlashInferProfileError(
                f"unexpected autotune entry {op!r}/{runner_name!r}/{extras!r}"
            )
        if not isinstance(profile, tuple) or len(profile) < 2:
            raise FlashInferProfileError("NVFP4 profile entry has invalid shapes")
        a_shape, b_shape = profile[0], profile[1]
        if (
            not isinstance(a_shape, tuple)
            or len(a_shape) != 2
            or not isinstance(b_shape, tuple)
            or len(b_shape) != 2
        ):
            raise FlashInferProfileError("NVFP4 profile entry has invalid A/B shapes")
        m, k = a_shape
        b_k, n = b_shape
        if k != b_k:
            raise FlashInferProfileError("NVFP4 profile entry has inconsistent K")
        if (
            not isinstance(value, list)
            or len(value) != 2
            or value[0] != NVFP4_RUNNER
            or not isinstance(value[1], int)
            or not 0 <= value[1] <= MAX_SERIALIZED_TACTIC
        ):
            raise FlashInferProfileError("NVFP4 profile entry has invalid tactic")
        actual.add((m, k, n))

    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise FlashInferProfileError(
            "incomplete NVFP4 profile: "
            f"missing={sorted(missing)[:8]} ({len(missing)} total), "
            f"unexpected={sorted(unexpected)[:8]} ({len(unexpected)} total)"
        )
    return _sha256_bytes(contents), len(actual)


class FlashInferProfileStore:
    """Immutable generations plus an atomically replaced current pointer."""

    def __init__(self, root: Path, contract: Mapping[str, Any]) -> None:
        self.root = root
        self.contract = dict(contract)
        self.profile_id = nvfp4_profile_identity(contract)
        self.profile_dir = root / self.profile_id
        self.generations_dir = self.profile_dir / "generations"
        self.current_path = self.profile_dir / CURRENT_FILE_NAME
        self.lock_path = root / f".{self.profile_id}.lock"

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def temporary_candidate(self) -> Path:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(
            dir=self.profile_dir, prefix=".candidate.", suffix=".json"
        )
        os.close(fd)
        path = Path(raw_path)
        path.unlink()
        return path

    def load_current(
        self,
        *,
        signatures: Sequence[tuple[int, int]],
        buckets: Sequence[int],
    ) -> ProfileGeneration | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = json.loads(self.current_path.read_text())
            generation = pointer["generation"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise FlashInferProfileError("invalid current profile pointer") from exc
        if (
            not isinstance(generation, str)
            or len(generation) != 64
            or any(c not in "0123456789abcdef" for c in generation)
        ):
            raise FlashInferProfileError("invalid current profile generation")

        generation_dir = self.generations_dir / generation
        config_path = generation_dir / PROFILE_FILE_NAME
        manifest_path = generation_dir / PROFILE_MANIFEST_NAME
        try:
            manifest = json.loads(manifest_path.read_text())
            contents = config_path.read_bytes()
        except (OSError, json.JSONDecodeError) as exc:
            raise FlashInferProfileError("current profile generation is incomplete") from exc
        if manifest.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise FlashInferProfileError("profile schema is incompatible")
        if manifest.get("profile_id") != self.profile_id:
            raise FlashInferProfileError("profile identity is incompatible")
        if manifest.get("contract") != self.contract:
            raise FlashInferProfileError("profile contract is stale or incompatible")
        checksum, entries = validate_nvfp4_table(
            contents, signatures=signatures, buckets=buckets
        )
        if checksum != generation or manifest.get("config_sha256") != checksum:
            raise FlashInferProfileError("profile checksum mismatch")
        if manifest.get("entries") != entries:
            raise FlashInferProfileError("profile entry count mismatch")
        return ProfileGeneration(
            profile_id=self.profile_id,
            generation=generation,
            config_path=config_path,
            manifest_path=manifest_path,
            config_sha256=checksum,
            entries=entries,
        )

    def publish(
        self,
        candidate: Path,
        *,
        signatures: Sequence[tuple[int, int]],
        buckets: Sequence[int],
        tuning_duration_s: float,
    ) -> ProfileGeneration:
        contents = candidate.read_bytes()
        checksum, entries = validate_nvfp4_table(
            contents, signatures=signatures, buckets=buckets
        )
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        generation_dir = self.generations_dir / checksum
        if generation_dir.exists():
            existing = generation_dir / PROFILE_FILE_NAME
            if not existing.is_file() or _sha256_file(existing) != checksum:
                raise FlashInferProfileError(
                    "immutable profile generation already exists with different content"
                )
        else:
            staging = Path(
                tempfile.mkdtemp(dir=self.generations_dir, prefix=".generation.")
            )
            try:
                config_path = staging / PROFILE_FILE_NAME
                shutil.copyfile(candidate, config_path)
                manifest = {
                    "schema_version": PROFILE_SCHEMA_VERSION,
                    "profile_id": self.profile_id,
                    "config_sha256": checksum,
                    "entries": entries,
                    "contract": self.contract,
                    "created_unix_ns": time.time_ns(),
                    "tuning_duration_s": tuning_duration_s,
                }
                _atomic_write(
                    staging / PROFILE_MANIFEST_NAME,
                    json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n",
                )
                os.replace(staging, generation_dir)
            except BaseException:
                with suppress(OSError):
                    shutil.rmtree(staging)
                raise

        pointer = json.dumps({"generation": checksum}, sort_keys=True).encode() + b"\n"
        _atomic_write(self.current_path, pointer)
        candidate.unlink(missing_ok=True)
        loaded = self.load_current(signatures=signatures, buckets=buckets)
        assert loaded is not None
        return loaded


def freeze_flashinfer_autotuner(
    tuner: Any, *, profile_id: str, generation: str
) -> None:
    """Fail closed on post-readiness tuning, mutation, or uncovered shapes.

    FlashInfer's documented ``choose_one`` entry point returns tactic ``-1``
    for an uncovered shape.  Intercept that result before the caller launches
    the fallback kernel.  Mutation APIs are disabled after the validated map
    has been loaded.
    """
    frozen = getattr(tuner, "_vllm_frozen_nvfp4_profile", None)
    identity = (profile_id, generation)
    if frozen is not None:
        if frozen != identity:
            raise FlashInferProfileError(
                f"FlashInfer tuner is already frozen to {frozen}, not {identity}"
            )
        return

    original_choose_one = tuner.choose_one

    def frozen_choose_one(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self.is_tuning_mode:
            raise FlashInferProfileError(
                "live FlashInfer autotuning is prohibited after profile freeze"
            )
        runner, tactic = original_choose_one(*args, **kwargs)
        if tactic == -1:
            custom_op = args[0] if args else kwargs.get("custom_op", "unknown")
            raise FlashInferProfileError(
                f"frozen FlashInfer profile has no tactic for {custom_op}"
            )
        return runner, tactic

    def frozen_mutation(self: Any, *args: Any, **kwargs: Any) -> Any:
        del self, args, kwargs
        raise FlashInferProfileError(
            "FlashInfer profile mutation is prohibited after profile freeze"
        )

    tuner.choose_one = types.MethodType(frozen_choose_one, tuner)
    for method_name in ("load_configs", "save_configs", "clear_cache"):
        setattr(tuner, method_name, types.MethodType(frozen_mutation, tuner))
    tuner._vllm_frozen_nvfp4_profile = identity
