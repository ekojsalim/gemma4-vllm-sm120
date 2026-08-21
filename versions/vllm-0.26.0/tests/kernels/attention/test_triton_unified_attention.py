# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

DEVICE_TYPE = current_platform.device_type

NUM_HEADS = [(4, 4), (8, 2), (5, 1)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16]

DTYPES = [torch.bfloat16]
QDTYPES = [None, current_platform.fp8_dtype()]
FP8_DTYPE = current_platform.fp8_dtype()

# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]

# 0: use 2D kernel for decode
# 8: use 3D kernel for decode
SEQ_THRESHOLD_3D_VALUES = [0, 8]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 64, 128, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("q_dtype", QDTYPES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    q_dtype: torch.dtype | None,
    seq_threshold_3D: int,
) -> None:
    torch.set_default_device(DEVICE_TYPE)

    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    kv_quant_mode = KVQuantMode.NONE
    if q_dtype is not None:
        # Use non-1 scales so FP8 Q/K/V descale handling is tested explicitly.
        q_scale = torch.tensor(0.75, dtype=torch.float32)
        k_scale = torch.tensor(0.5, dtype=torch.float32)
        v_scale = torch.tensor(0.25, dtype=torch.float32)
        q_descale = q_scale
        scale_shape = (num_seqs, num_kv_heads)
        k_descale = torch.full(scale_shape, k_scale.item(), dtype=torch.float32)
        v_descale = torch.full(scale_shape, v_scale.item(), dtype=torch.float32)
        maybe_quantized_query = (query / q_scale).to(q_dtype)
        maybe_quantized_key_cache = (key_cache / k_scale).to(q_dtype)
        maybe_quantized_value_cache = (value_cache / v_scale).to(q_dtype)
        kv_quant_mode = KVQuantMode.FP8_PER_TENSOR

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=kv_quant_mode,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    atol, rtol = 1.5e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_bf16_query_fp8_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Test bf16 Q with FP8 per-tensor KV cache (dequant via _cast_kv_tile)."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (-1, -1)
    scale = head_size**-0.5

    dtype = torch.bfloat16
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.25, dtype=torch.float32)
    fp8_key_cache = (key_cache / k_scale).to(FP8_DTYPE)
    fp8_value_cache = (value_cache / v_scale).to(FP8_DTYPE)

    scale_shape = (num_seqs, num_kv_heads)
    k_descale = torch.full(scale_shape, k_scale.item(), dtype=torch.float32)
    v_descale = torch.full(scale_shape, v_scale.item(), dtype=torch.float32)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=fp8_key_cache,
        v=fp8_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )

    atol, rtol = 1.5e-1, 1.5e-1
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize("query_len", [2, 3])
@torch.inference_mode()
def test_triton_swa_single_block_matches_sequential_rows(query_len: int) -> None:
    """A small SWA query block must preserve sequential decode reductions."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    block_size = 16
    sliding_window = 64
    context_len = 128 - query_len
    seq_len = context_len + query_len
    num_query_heads = 4
    num_kv_heads = 2
    head_size = 256
    scale = head_size**-0.5

    query = torch.ones(
        query_len, num_query_heads, head_size, dtype=torch.bfloat16
    )
    key = -torch.rand(seq_len, num_kv_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.25, dtype=torch.float32)
    key_cache = (key / k_scale).to(FP8_DTYPE).view(
        -1, block_size, num_kv_heads, head_size
    )
    value_cache = (value / v_scale).to(FP8_DTYPE).view_as(key_cache)
    block_table = torch.arange(
        key_cache.shape[0], dtype=torch.int32
    ).unsqueeze(0)
    scale_shape = (1, num_kv_heads)
    k_descale = torch.full(scale_shape, k_scale.item(), dtype=torch.float32)
    v_descale = torch.full(scale_shape, v_scale.item(), dtype=torch.float32)

    def run(q: torch.Tensor, length: int) -> torch.Tensor:
        output = torch.empty_like(q)
        unified_attention(
            q=q,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=torch.tensor([0, len(q)], dtype=torch.int32),
            max_seqlen_q=len(q),
            seqused_k=torch.tensor([length], dtype=torch.int32),
            max_seqlen_k=length,
            softmax_scale=scale,
            causal=True,
            window_size=(sliding_window - 1, 0),
            block_table=block_table,
            softcap=0,
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=0,
            kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
        )
        return output

    block_output = run(query, seq_len)
    sequential_output = torch.cat(
        [
            run(query[row : row + 1], context_len + row + 1)
            for row in range(query_len)
        ]
    )

    torch.testing.assert_close(block_output, sequential_output, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("head_size", "sliding_window", "context_len"),
    [
        (256, 1024, 254),
        (256, 1024, 1022),
        (512, None, 254),
    ],
)
@pytest.mark.parametrize("query_len", [2, 3])
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@torch.inference_mode()
def test_triton_small_m_3d_matches_sequential_decode_rows(
    head_size: int,
    sliding_window: int | None,
    context_len: int,
    query_len: int,
    batch_size: int,
) -> None:
    """Bounded 3D verification must reproduce each q=1 reduction."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    block_size = 16
    num_query_heads = 32
    num_kv_heads = 16
    seq_threshold_3d = 8
    num_segments = 16
    seq_len = context_len + query_len
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_seq
    scale = head_size**-0.5

    query = torch.randn(
        batch_size * query_len,
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.25, dtype=torch.float32)
    key_cache = (key / k_scale).to(FP8_DTYPE)
    value_cache = (value / v_scale).to(FP8_DTYPE)
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(
        batch_size, blocks_per_seq
    )
    query_start = torch.arange(
        0,
        (batch_size + 1) * query_len,
        query_len,
        dtype=torch.int32,
    )
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)
    window_size = (
        (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    )

    scratch_tokens = seq_threshold_3d * 3
    scratch_output = torch.empty(
        scratch_tokens,
        num_query_heads,
        num_segments,
        head_size,
        dtype=torch.float32,
    )
    scratch_max = torch.empty(
        scratch_tokens,
        num_query_heads,
        num_segments,
        dtype=torch.float32,
    )
    scratch_expsum = torch.empty_like(scratch_max)

    def run(
        run_query: torch.Tensor,
        run_query_start: torch.Tensor,
        run_seq_lens: torch.Tensor,
        run_block_table: torch.Tensor,
        run_max_query_len: int,
    ) -> torch.Tensor:
        output = torch.empty_like(run_query)
        num_run_seqs = len(run_seq_lens)
        descale_shape = (num_run_seqs, num_kv_heads)
        unified_attention(
            q=run_query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=run_query_start,
            max_seqlen_q=run_max_query_len,
            seqused_k=run_seq_lens,
            max_seqlen_k=int(run_seq_lens.max()),
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=run_block_table,
            softcap=0,
            q_descale=None,
            k_descale=torch.full(
                descale_shape, k_scale.item(), dtype=torch.float32
            ),
            v_descale=torch.full(
                descale_shape, v_scale.item(), dtype=torch.float32
            ),
            seq_threshold_3D=seq_threshold_3d,
            num_par_softmax_segments=num_segments,
            softmax_segm_output=scratch_output,
            softmax_segm_max=scratch_max,
            softmax_segm_expsum=scratch_expsum,
            kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
        )
        return output

    block_output = run(
        query,
        query_start,
        seq_lens,
        block_table,
        query_len,
    )
    sequential_rows = []
    if batch_size > seq_threshold_3d:
        for row in range(query_len):
            sequential_rows.append(
                run(
                    query.reshape(
                        batch_size,
                        query_len,
                        num_query_heads,
                        head_size,
                    )[:, row],
                    torch.arange(batch_size + 1, dtype=torch.int32),
                    torch.full(
                        (batch_size,),
                        context_len + row + 1,
                        dtype=torch.int32,
                    ),
                    block_table,
                    1,
                )
            )
        sequential_output = torch.stack(sequential_rows, dim=1).flatten(0, 1)
    else:
        for sequence in range(batch_size):
            sequence_start = sequence * query_len
            for row in range(query_len):
                sequential_rows.append(
                    run(
                        query[sequence_start + row : sequence_start + row + 1],
                        torch.tensor([0, 1], dtype=torch.int32),
                        torch.tensor([context_len + row + 1], dtype=torch.int32),
                        block_table[sequence : sequence + 1],
                        1,
                    )
                )
        sequential_output = torch.cat(sequential_rows)

    torch.testing.assert_close(block_output, sequential_output, atol=0, rtol=0)


@torch.inference_mode()
def test_triton_small_m_uses_2d_above_sequence_threshold() -> None:
    """Batch 9 must retain the established 2D q=1/q=3 geometry."""
    test_triton_small_m_3d_matches_sequential_decode_rows(
        head_size=256,
        sliding_window=1024,
        context_len=254,
        query_len=3,
        batch_size=9,
    )


@torch.inference_mode()
def test_triton_small_m_3d_mixed_rows_and_segment_boundaries() -> None:
    """Mixed q=1/2/3 rows keep their own segment and SWA tile geometry."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    block_size = 16
    sliding_window = 1024
    head_size = 256
    num_query_heads = 32
    num_kv_heads = 16
    num_segments = 16
    seq_threshold_3d = 8
    query_lens = [1, 2, 3, 3, 2, 1]
    # Includes an empty final segment, partial tiles, a tiles-per-segment
    # transition, the 1,024-token boundary, and a later logical wrap.
    context_lens = [14, 15, 254, 255, 1022, 2046]
    seq_lens_list = [
        context + query
        for context, query in zip(context_lens, query_lens, strict=True)
    ]
    batch_size = len(query_lens)
    max_blocks_per_seq = (
        max(seq_lens_list) + block_size - 1
    ) // block_size
    num_blocks = batch_size * max_blocks_per_seq

    query_start_list = [0]
    for query_len in query_lens:
        query_start_list.append(query_start_list[-1] + query_len)
    query_start = torch.tensor(query_start_list, dtype=torch.int32)
    query = torch.randn(
        query_start_list[-1],
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.25, dtype=torch.float32)
    key_cache = (key / k_scale).to(FP8_DTYPE)
    value_cache = (value / v_scale).to(FP8_DTYPE)
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(
        batch_size, max_blocks_per_seq
    )
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    scratch_tokens = seq_threshold_3d * 3
    scratch_output = torch.empty(
        scratch_tokens,
        num_query_heads,
        num_segments,
        head_size,
        dtype=torch.float32,
    )
    scratch_max = torch.empty(
        scratch_tokens,
        num_query_heads,
        num_segments,
        dtype=torch.float32,
    )
    scratch_expsum = torch.empty_like(scratch_max)

    def run(
        run_query: torch.Tensor,
        run_query_start: torch.Tensor,
        run_seq_lens: torch.Tensor,
        run_block_table: torch.Tensor,
        run_max_query_len: int,
    ) -> torch.Tensor:
        output = torch.empty_like(run_query)
        descale_shape = (len(run_seq_lens), num_kv_heads)
        unified_attention(
            q=run_query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=run_query_start,
            max_seqlen_q=run_max_query_len,
            seqused_k=run_seq_lens,
            max_seqlen_k=int(run_seq_lens.max()),
            softmax_scale=head_size**-0.5,
            causal=True,
            window_size=(sliding_window - 1, 0),
            block_table=run_block_table,
            softcap=0,
            q_descale=None,
            k_descale=torch.full(
                descale_shape, k_scale.item(), dtype=torch.float32
            ),
            v_descale=torch.full(
                descale_shape, v_scale.item(), dtype=torch.float32
            ),
            seq_threshold_3D=seq_threshold_3d,
            num_par_softmax_segments=num_segments,
            softmax_segm_output=scratch_output,
            softmax_segm_max=scratch_max,
            softmax_segm_expsum=scratch_expsum,
            kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
        )
        return output

    block_output = run(query, query_start, seq_lens, block_table, 3)
    sequential_rows = []
    for sequence, (context_len, query_len) in enumerate(
        zip(context_lens, query_lens, strict=True)
    ):
        sequence_start = query_start_list[sequence]
        for row in range(query_len):
            sequential_rows.append(
                run(
                    query[sequence_start + row : sequence_start + row + 1],
                    torch.tensor([0, 1], dtype=torch.int32),
                    torch.tensor([context_len + row + 1], dtype=torch.int32),
                    block_table[sequence : sequence + 1],
                    1,
                )
            )
    sequential_output = torch.cat(sequential_rows)

    torch.testing.assert_close(block_output, sequential_output, atol=0, rtol=0)


@torch.inference_mode()
def test_triton_small_m_3d_cuda_graph_matches_eager() -> None:
    """The bounded q=3 segment buffers and launch geometry are graph-stable."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    query_len = 3
    context_len = 254
    seq_len = context_len + query_len
    block_size = 16
    head_size = 256
    num_query_heads = 32
    num_kv_heads = 16
    num_segments = 16
    blocks = (seq_len + block_size - 1) // block_size
    query = torch.randn(
        query_len, num_query_heads, head_size, dtype=torch.bfloat16
    )
    key = torch.randn(
        blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.25, dtype=torch.float32)
    key_cache = (key / k_scale).to(FP8_DTYPE)
    value_cache = (value / v_scale).to(FP8_DTYPE)
    block_table = torch.arange(blocks, dtype=torch.int32).unsqueeze(0)
    query_start = torch.tensor([0, query_len], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    k_descale = torch.full((1, num_kv_heads), k_scale.item())
    v_descale = torch.full((1, num_kv_heads), v_scale.item())
    scratch_output = torch.empty(
        8 * 3,
        num_query_heads,
        num_segments,
        head_size,
        dtype=torch.float32,
    )
    scratch_max = torch.empty(
        8 * 3, num_query_heads, num_segments, dtype=torch.float32
    )
    scratch_expsum = torch.empty_like(scratch_max)

    def launch(output: torch.Tensor) -> None:
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=query_start,
            max_seqlen_q=query_len,
            seqused_k=seq_lens,
            max_seqlen_k=seq_len,
            softmax_scale=head_size**-0.5,
            causal=True,
            window_size=(1023, 0),
            block_table=block_table,
            softcap=0,
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=8,
            num_par_softmax_segments=num_segments,
            softmax_segm_output=scratch_output,
            softmax_segm_max=scratch_max,
            softmax_segm_expsum=scratch_expsum,
            kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
        )

    eager_output = torch.empty_like(query)
    launch(eager_output)
    torch.cuda.synchronize()
    graph_output = torch.empty_like(query)
    launch(graph_output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(graph_output)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_output, eager_output, atol=0, rtol=0)



@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (5, 18), (129, 463)],
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1)] * 533,
        [(533, 533)] * 533,
    ],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 64, 128, 256])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_fp16_input_fp8_output(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Test with fp16 input and fp8 output using output_scale."""
    torch.set_default_device(DEVICE_TYPE)

    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    dtype = torch.float16
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty(sum(query_lens), num_query_heads, head_size, dtype=FP8_DTYPE)

    output_scale = torch.tensor(0.5, dtype=torch.float32)

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        output_scale=output_scale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    output_fp16 = output.to(torch.float32) * output_scale.item()
    output_fp16 = output_fp16.to(torch.float16)

    atol, rtol = 2e-1, 2e-1
    (
        torch.testing.assert_close(output_fp16, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output_fp16 - ref_output))}",
    )


# USE_TD path covers two head-size regimes:
# - pow2 (HEAD_SIZE == HEAD_SIZE_PADDED): full TD path including Q/O.
# - non-pow2 (96, HEAD_SIZE_PADDED=128): gates USE_TD_QO off — Q load
#   and output store fall back to pointer path, KV tile TD load remains.
# The non-pow2 case mirrors real models like Phi-3-mini (head_size=96).
HEAD_SIZES_USE_TD = [128, 256, 96]


def _run_use_td_case(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    sliding_window: int | None,
    soft_cap: float | None,
    seq_threshold_3D: int,
    dtype: torch.dtype = torch.bfloat16,
    num_blocks: int = 2048,
) -> None:
    """Shared driver for the USE_TD test cases.

    Runs ``unified_attention(..., use_td=True)`` and compares against the
    reference paged-attention implementation that the sibling non-TD
    tests use.
    """
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        use_td=True,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    torch.testing.assert_close(output, ref_output, atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES_USE_TD)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_use_td(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Exercise the USE_TD (tensor-descriptor) Q/K/V load/store path.

    Covers both 2D and 3D kernels via ``seq_threshold_3D``. Two routes
    to the USE_TD_QO=False fallback (pointer path for Q/O with TD still
    active for KV tile loads):

    - non-pow2 ``num_queries_per_kv`` via ``NUM_HEADS`` entry ``(5, 1)``,
    - non-pow2 ``head_size`` via ``HEAD_SIZES_USE_TD`` entry ``96``.
    """
    _run_use_td_case(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        seq_threshold_3D=seq_threshold_3D,
        num_blocks=num_blocks,
    )


# Prefill-heavy shape: long query drives the prefill kernel path where
# ``_get_tile_size`` returns 32, which exceeds block_size=16 and must be
# clamped by the fix in 'clamp TILE_SIZE to block_size when USE_TD'.
# Only the prefill launch exercises the clamp, so parameterize only over
# the (num_heads, seq_threshold_3D=0) combinations needed to cover it.
@pytest.mark.parametrize("num_heads", [(4, 4), (5, 1)])
@torch.inference_mode()
def test_triton_unified_attn_use_td_tile_clamp(
    num_heads: tuple[int, int],
) -> None:
    """Regression guard: ``USE_TD`` needs ``BLOCK_SIZE % TILE_SIZE == 0``.

    With ``block_size=16`` and ``head_size=128`` (non-Gemma3),
    ``_get_tile_size`` returns 32 for prefill, which violates the
    ``USE_TD`` constraint unless clamped to ``block_size``.  Without
    the clamp the triton kernel ``static_assert`` fires at compile time.
    """
    _run_use_td_case(
        seq_lens=[(256, 256), (128, 128)],
        num_heads=num_heads,
        head_size=128,
        block_size=16,
        sliding_window=None,
        soft_cap=None,
        seq_threshold_3D=0,
    )
