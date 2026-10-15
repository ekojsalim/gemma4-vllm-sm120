# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_warmup,
)
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    FlashInferProfileError,
    FlashInferProfileStore,
    build_nvfp4_profile_contract,
    collect_target_nvfp4_signatures,
    freeze_flashinfer_autotuner,
    nvfp4_tuning_buckets,
    resolve_flashinfer_autotune_file,
    validate_nvfp4_table,
    write_flashinfer_autotune_cache,
)
from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    deepseek_v4_sparse_mla_attention_warmup,
    flashinfer_sparse_mla_decode_autotune_warmup,
)
from vllm.model_executor.warmup.qwen_triton_warmup import qwen_triton_warmup
from vllm.model_executor.warmup.sparse_mla_triton_warmup import (
    sparse_mla_triton_warmup_if_needed,
)
from vllm.model_executor.warmup.v1_block_table_warmup import (
    warm_v1_block_table_kernels,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_LL_BF16_WARMUP_MODEL_SHAPES: tuple[tuple[int, int], ...] = (
    (6144, 264),  # Inkling
    (7168, 256),  # DSV3
    (7168, 384),  # DSV4-Pro
    (14400, 256),  # DSV4-Flash
)
_LL_BF16_WARMUP_M_RANGE = range(1, 17)


def _warmup_ll_bf16_router_gemm() -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        is_available as is_ll_bf16_gemm_available,
    )
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        ll_bf16_gemm_kernel,
    )

    if not is_ll_bf16_gemm_available():
        return

    logger.info("Warming up ll_bf16 router GEMM kernels.")
    ll_bf16_gemm_kernel.warmup(
        shapes=_LL_BF16_WARMUP_MODEL_SHAPES,
        m_values=_LL_BF16_WARMUP_M_RANGE,
    )


def kernel_warmup(worker: "Worker"):
    from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
        minimax_m3_msa_warmup,
    )

    # Pooling models do not use the generation slot-mapping path.
    if not worker.use_v2_model_runner and not worker.model_runner.is_pooling_model:
        warm_v1_block_table_kernels(
            getattr(worker.model_runner, "device", torch.device("cuda")),
            worker.scheduler_config.max_num_batched_tokens,
        )
    qwen_triton_warmup(worker.model_runner, worker.vllm_config.model_config)

    # DSv4 mHC TileLang kernels (hc_pre/hc_post/hc_head_op) run every decoder
    # layer per token; warm them across token sizes first so the first real
    # request doesn't pay JIT cost. No-op for non-DSv4 models (gated inside).
    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        ),
    )

    # Run next so input-prep kernels JIT against pristine runner state.
    sparse_mla_triton_warmup_if_needed(worker)
    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    deepseek_v4_sparse_mla_attention_warmup(worker)

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    minimax_m3_msa_warmup(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    if current_platform.has_device_capability(90):
        _warmup_ll_bf16_router_gemm()

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )

    if worker.vllm_config.kernel_config.enable_cutedsl_warmup:
        cutedsl_warmup()


def _flashinfer_autotune_skip_ops(runner: "GPUModelRunner") -> set[str] | None:
    if envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS is not None:
        return set(envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS) or None

    from vllm.model_executor.kernels.linear import (
        FlashInferCuteDslNvFp4LinearKernel,
    )

    for module in runner.get_model().modules():
        for holder_name in ("quant_method", "scheme"):
            kernel = getattr(getattr(module, holder_name, None), "kernel", None)
            # CuTe-DSL mm_fp4 tuning JIT-compiles every tactic and its
            # fallback is already the heuristic; all mm_fp4 backends share
            # the "fp4_gemm" op name, so skip only when cute-dsl is selected.
            if isinstance(kernel, FlashInferCuteDslNvFp4LinearKernel):
                return {"fp4_gemm"}
    return None


def _file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_loaded_nvfp4_profile(
    runner: "GPUModelRunner",
    *,
    config_path,
    signatures: tuple[tuple[int, int], ...],
    buckets: tuple[int, ...],
) -> None:
    """Exercise every covered M with tuning disabled and no file writes."""
    import vllm.utils.flashinfer as fi_utils
    from flashinfer.autotuner import AutoTuner

    checksum_before = _file_sha256(config_path)
    validate_nvfp4_table(
        config_path.read_bytes(), signatures=signatures, buckets=buckets
    )
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.reset_statistics()
    with torch.inference_mode(), fi_utils.autotune(
        tune_mode=False,
        cache=str(config_path),
        tuning_buckets=buckets,
    ):
        for num_tokens in buckets:
            runner._dummy_run(
                num_tokens=num_tokens,
                skip_eplb=True,
                is_profile=True,
            )
    checksum_after = _file_sha256(config_path)
    if checksum_after != checksum_before:
        raise FlashInferProfileError(
            "prevalidated FlashInfer profile changed during load-only validation"
        )
    if tuner.stats.cache_misses:
        raise FlashInferProfileError(
            "prevalidated FlashInfer profile used "
            f"{tuner.stats.cache_misses} fallback tactic(s): {tuner.stats}"
        )
    if tuner.is_tuning_mode:
        raise FlashInferProfileError("FlashInfer tuner remained in tuning mode")


def _flashinfer_nvfp4_profile(runner: "GPUModelRunner") -> bool:
    """Load or generate the target's node-local NVFP4 profile.

    Returns ``False`` when this runner has no FlashInfer CUTLASS NVFP4 target
    layers, allowing the upstream generic warmup to handle other models.
    """
    import vllm.utils.flashinfer as fi_utils
    from flashinfer.autotuner import AutoTuner
    from vllm.distributed.parallel_state import get_world_group

    signatures = collect_target_nvfp4_signatures(runner)
    if not signatures:
        return False
    max_m = runner.scheduler_config.max_num_batched_tokens
    buckets = nvfp4_tuning_buckets(max_m)
    contract = build_nvfp4_profile_contract(
        runner, signatures=signatures, max_m=max_m
    )
    store = FlashInferProfileStore(
        root=resolve_flashinfer_autotune_file(runner).parent.parent,
        contract=contract,
    )
    world = get_world_group()
    is_leader = world.rank_in_group == 0
    lock_context = store.lock() if is_leader else None
    if lock_context is not None:
        lock_context.__enter__()

    try:
        descriptor = None
        if is_leader:
            try:
                current = store.load_current(
                    signatures=signatures, buckets=buckets
                )
            except FlashInferProfileError as exc:
                logger.warning(
                    "Rejecting node-local FlashInfer profile %s: %s. "
                    "A new immutable generation will be tuned.",
                    store.profile_id,
                    exc,
                )
                current = None
            descriptor = {
                "profile_id": store.profile_id,
                "mode": "load" if current is not None else "tune",
                "config_path": str(current.config_path) if current else None,
                "generation": current.generation if current else None,
            }
        descriptor = world.broadcast_object(descriptor, src=0)
        if descriptor["profile_id"] != store.profile_id:
            raise FlashInferProfileError(
                "ranks derived different FlashInfer NVFP4 profile identities"
            )

        generated = descriptor["mode"] == "tune"
        tuning_duration_s = 0.0
        if generated:
            candidate = store.temporary_candidate() if is_leader else None
            candidate_path = world.broadcast_object(
                str(candidate) if candidate is not None else None, src=0
            )
            candidate = Path(candidate_path)
            tuner = AutoTuner.get()
            tuner.clear_cache()
            # FlashInfer uses strict '<' selection, so exact ties retain the
            # first (canonical lowest-id) tactic.  Increase samples over the
            # package default to reduce near-tie timing churn across nodes.
            tuner.warmup = max(tuner.warmup, 5)
            tuner.repeat = max(tuner.repeat, 20)
            start = time.perf_counter()
            with torch.inference_mode():
                if is_leader:
                    with fi_utils.autotune(
                        tune_mode=True,
                        cache=str(candidate),
                        tuning_buckets=buckets,
                    ):
                        runner._dummy_run(
                            num_tokens=max_m,
                            skip_eplb=True,
                            is_profile=True,
                        )
                else:
                    runner._dummy_run(
                        num_tokens=max_m,
                        skip_eplb=True,
                        is_profile=True,
                    )
            torch.cuda.synchronize()
            tuning_duration_s = time.perf_counter() - start
            world.barrier()
            if is_leader:
                validate_nvfp4_table(
                    candidate.read_bytes(),
                    signatures=signatures,
                    buckets=buckets,
                )

            # Validate every bucket through FlashInfer's public load-only path
            # before publishing or accepting readiness.
            _validate_loaded_nvfp4_profile(
                runner,
                config_path=candidate,
                signatures=signatures,
                buckets=buckets,
            )
            world.barrier()
            published = None
            if is_leader:
                published = store.publish(
                    candidate,
                    signatures=signatures,
                    buckets=buckets,
                    tuning_duration_s=tuning_duration_s,
                )
            descriptor = world.broadcast_object(
                (
                    {
                        "config_path": str(published.config_path),
                        "generation": published.generation,
                        "entries": published.entries,
                    }
                    if published is not None
                    else None
                ),
                src=0,
            )
        else:
            descriptor = {
                "config_path": descriptor["config_path"],
                "generation": descriptor["generation"],
                "entries": len(signatures) * len(buckets),
            }

        config_path = Path(descriptor["config_path"])
        # Published profiles and restart hits both take the same load-only
        # route.  A fresh generation gets a second check at its immutable path.
        _validate_loaded_nvfp4_profile(
            runner,
            config_path=config_path,
            signatures=signatures,
            buckets=buckets,
        )
        world.barrier()

        checksum = _file_sha256(config_path)
        if checksum != descriptor["generation"]:
            raise FlashInferProfileError("loaded FlashInfer generation checksum changed")
        tuner = AutoTuner.get()
        freeze_flashinfer_autotuner(
            tuner,
            profile_id=store.profile_id,
            generation=descriptor["generation"],
        )
        logger.info(
            "FlashInfer NVFP4 profile %s: status=%s generation=%s "
            "entries=%d signatures=%s buckets=%s tuning_duration_s=%.3f "
            "fallbacks=0 frozen=true",
            store.profile_id,
            "generated" if generated else "loaded",
            descriptor["generation"],
            descriptor["entries"],
            signatures,
            buckets,
            tuning_duration_s,
        )
        return True
    finally:
        if lock_context is not None:
            lock_context.__exit__(None, None, None)


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Tuning is performed only on rank 0. The resulting cache is broadcast
    to every rank so all ranks dispatch the same kernel tactic.
    """
    if _flashinfer_nvfp4_profile(runner):
        return

    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import get_world_group

    autotune_kwargs: dict = {}
    skip_ops = _flashinfer_autotune_skip_ops(runner)
    if skip_ops:
        logger.info(
            "Skipping FlashInfer autotuning for ops %s",
            sorted(skip_ops),
        )
        autotune_kwargs["skip_ops"] = skip_ops

    use_persistent_cache = True

    # When distributed, tune on every rank so the collectives stay synchronized.
    if get_world_group().world_size > 1:
        use_persistent_cache = False

    if not use_persistent_cache:
        with torch.inference_mode(), fi_utils.autotune(**autotune_kwargs):
            runner._dummy_run(
                num_tokens=runner.scheduler_config.max_num_batched_tokens,
                skip_eplb=True,
                is_profile=True,
            )
        get_world_group().barrier()
        return

    world = get_world_group()
    is_leader = world.rank_in_group == 0

    cache_path = resolve_flashinfer_autotune_file(runner)
    if is_leader:
        logger.info("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # When autotuning with number of tokens m, flashinfer will autotune
    # operations for all number of tokens up to m, so we only need to
    # run with the max number of tokens.
    dummy_run_kwargs = dict(
        num_tokens=runner.scheduler_config.max_num_batched_tokens,
        skip_eplb=True,
        is_profile=True,
    )

    with torch.inference_mode():
        if is_leader:
            with fi_utils.autotune(
                tune_mode=True, cache=str(cache_path), **autotune_kwargs
            ):
                runner._dummy_run(**dummy_run_kwargs)
        else:
            runner._dummy_run(**dummy_run_kwargs)

    # Broadcast autotune cache from rank 0 to all other ranks so every
    # rank loads the same set of chosen tactics.
    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)

    if tune_results is None:
        logger.warning(
            "No FlashInfer autotune cache entries found."
            "Falling back to default tactics."
        )
    else:
        write_flashinfer_autotune_cache(cache_path, tune_results)
        world.barrier()
        from flashinfer.autotuner import AutoTuner

        AutoTuner.get().load_configs(str(cache_path))
        logger.info(
            "FlashInfer autotune cache loaded on rank %d from %s.",
            world.rank_in_group,
            cache_path,
        )
