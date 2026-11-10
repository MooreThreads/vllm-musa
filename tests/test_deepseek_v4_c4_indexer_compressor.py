# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import struct
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / "csrc/musa/attention/deepseek_v4_c4_indexer_compressor.mu"
WRAPPER = ROOT / "vllm_musa/kernels/deepseek_v4_c4_indexer_compressor.py"
SERIES_PATCH = (
    ROOT
    / "vllm_musa/patches/series/0133-MUSA-dispatch-DeepSeek-V4-sparse-compressor-to-nativ.patch"
)


def _bf16_roundtrip(value: float) -> float:
    """Round one Python float through IEEE fp32 then bf16 (round-to-nearest-even)."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", rounded & 0xFFFF0000))[0]


def _state_value(page: int, offset: int, dim: int) -> float:
    # Deterministic, non-symmetric values make page/head-offset mistakes visible.
    return math.sin((page * 4099 + offset * 521 + dim * 17 + 1) * 0.001)


def _cpu_reference_row(
    req: int,
    position: int,
    block_table: list[list[int]],
    eps: float,
) -> tuple[list[float], float]:
    values_by_dim: list[float] = []
    for dim in range(128):
        row_values = []
        row_scores = []
        for row, source_position in enumerate(range(position - 7, position + 1)):
            if source_position < 0:
                row_values.append(0.0)
                row_scores.append(float("-inf"))
                continue
            page = block_table[req][source_position // 4]
            page_offset = source_position % 4
            head_offset = 128 if row >= 4 else 0
            row_values.append(_state_value(page, page_offset, head_offset + dim))
            row_scores.append(_state_value(page, page_offset, 256 + head_offset + dim))
        max_score = max(row_scores)
        softmax = [
            0.0 if score == float("-inf") else math.exp(score - max_score)
            for score in row_scores
        ]
        denominator = sum(softmax)
        values_by_dim.append(
            sum(value * score for value, score in zip(row_values, softmax, strict=True))
            / denominator
        )

    variance = sum(value * value for value in values_by_dim) / 128.0
    inv_rms = 1.0 / math.sqrt(variance + eps)
    output = [
        value * inv_rms * (1.0 + (dim % 11 - 5) * 0.002)
        for dim, value in enumerate(values_by_dim)
    ]
    compressed_position = (position // 4) * 4
    for pair in range(32):
        dim = 64 + pair * 2
        angle = compressed_position * 0.003 + pair * 0.007
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)
        even, odd = output[dim], output[dim + 1]
        output[dim] = even * cos_value - odd * sin_value
        output[dim + 1] = odd * cos_value + even * sin_value
    output = [_bf16_roundtrip(value) for value in output]
    absmax = max(1e-4, max(abs(value) for value in output))
    scale = 2.0 ** math.ceil(math.log2(absmax / 448.0))
    return output, scale


def _cpu_reference_store(
    token_to_req: list[int],
    positions: list[int],
    state_slots: list[int],
    block_table: list[list[int]],
    kv_slots: list[int],
    kv_block_size: int,
    eps: float,
) -> dict[tuple[int, int], tuple[list[float], float]]:
    output = {}
    for token, position in enumerate(positions):
        if state_slots[token] < 0 or (position + 1) % 4 != 0:
            continue
        kv_slot = kv_slots[token]
        if kv_slot < 0:
            continue
        page, offset = divmod(kv_slot, kv_block_size)
        output[(page, offset)] = _cpu_reference_row(
            token_to_req[token], position, block_table, eps
        )
    return output


def test_native_op_is_registered_and_built() -> None:
    kernel = KERNEL.read_text()
    setup = (ROOT / "setup.py").read_text()
    header = (ROOT / "csrc/musa/musa_ops.h").read_text()
    bindings = (ROOT / "csrc/musa/torch_bindings.cpp").read_text()
    custom_ops = (ROOT / "vllm_musa/_custom_ops.py").read_text()

    assert "deepseek_v4_c4_indexer_compressor.mu" in setup
    assert "void deepseek_v4_c4_indexer_compress_cache(" in header
    assert "deepseek_v4_c4_indexer_compress_cache(Tensor state_cache" in bindings
    assert "def deepseek_v4_c4_indexer_compress_cache(" in custom_ops
    assert "deepseek_v4_c4_indexer_compressor_kernel" in kernel


def test_kernel_uses_warp_per_token_register_pipeline() -> None:
    source = KERNEL.read_text()

    assert "constexpr int64_t kCompressRows = 2 * kCompressRatio;" in source
    assert "constexpr int64_t kWarpsPerBlock = 4;" in source
    assert "const int warp = threadIdx.x >> 5;" in source
    assert "const int lane = threadIdx.x & 31;" in source
    assert "float kv[kCompressRows][4];" in source
    assert "float score[kCompressRows][4];" in source
    assert "const float exp0 = exp2f" in source
    assert "const float exp3 = exp2f" in source
    assert "warp_reduce_max_u32" in source
    assert "const __mt_fp8x4_e4m3 packed" in source
    assert "__shared__" not in source
    assert "__syncthreads" not in source


def test_kernel_preserves_vllm_pad_boundary_and_page_abi() -> None:
    source = KERNEL.read_text()

    assert "state_slot < 0" in source
    assert "(position + 1) % kCompressRatio != 0" in source
    assert "req_idx * block_table_stride + logical_block" in source
    assert "head_offset = row >= kCompressRatio ? kHeadDim : 0" in source
    assert "kv_slot < 0" in source
    assert "kv_block_idx * kv_cache_stride0" in source
    assert "kKvBlockSize * kTokenValueBytes" in source
    assert "__bfloat162float(__float2bfloat16(output[elem]))" in source
    assert "Do not import SGLang's subsequent Hadamard transform" in source


def test_source_patch_keeps_triton_fallback() -> None:
    patch = SERIES_PATCH.read_text()
    native_call = patch.index("try_musa_deepseek_v4_c4_indexer_compressor(")
    triton_dispatch = patch.index("if head_dim == 512:")

    assert native_call < triton_dispatch
    assert "if handled:" in patch
    assert "+            return" in patch


def test_wrapper_is_default_on_and_shape_bounded() -> None:
    source = WRAPPER.read_text()

    assert "_SUPPORTED_KV_BLOCK_SIZES = (64, 256)" in source
    assert "0 < num_rows <= _MAX_DECODE_ROWS" in source
    assert "return False, reason" in source
    assert "VLLM_MUSA_DEEPSEEK_V4_C4_INDEXER_COMPRESS_IMPL" not in source
    assert "os.environ" not in source


def test_cpu_oracle_keeps_nonboundary_and_pad_slots_untouched() -> None:
    block_table = [[3, 1, 6, 0], [2, 5, 7, 4]]
    cache = _cpu_reference_store(
        token_to_req=[0, 0, 1, 1],
        positions=[3, 4, 7, 11],
        state_slots=[0, 1, 2, -1],
        block_table=block_table,
        kv_slots=[0, 1, 257, 258],
        kv_block_size=256,
        eps=1e-6,
    )

    assert set(cache) == {(0, 0), (1, 1)}
    first_values, first_scale = cache[(0, 0)]
    second_values, second_scale = cache[(1, 1)]
    assert len(first_values) == len(second_values) == 128
    assert all(math.isfinite(value) for value in first_values + second_values)
    assert math.isfinite(first_scale) and first_scale > 0
    assert math.isfinite(second_scale) and second_scale > 0
    assert first_values != second_values


def test_bf16_cpu_oracle_uses_round_to_nearest_even() -> None:
    # Exact bf16 values remain unchanged; an in-between fp32 value rounds to the
    # nearest bf16 representation instead of being blindly truncated.
    assert _bf16_roundtrip(1.0) == 1.0
    assert _bf16_roundtrip(-2.0) == -2.0
    value = struct.unpack("<f", struct.pack("<I", 0x3F80C000))[0]
    assert _bf16_roundtrip(value) == 1.0078125
