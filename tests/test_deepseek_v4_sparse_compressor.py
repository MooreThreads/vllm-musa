# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import struct
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / "csrc/musa/attention/deepseek_v4_sparse_compressor.mu"
WRAPPER = ROOT / "vllm_musa/kernels/deepseek_v4_sparse_compressor.py"
def _series_patch(number: str) -> Path:
    # Entries are addressed by number: `musa_sync regen` owns the slug part of
    # the file name, so a rename must not break a source assertion.
    matches = sorted((ROOT / "vllm_musa/patches/series").glob(f"{number}-*.patch"))
    assert len(matches) == 1, f"series entry {number} resolves to {matches}"
    return matches[0]


SERIES_PATCH = _series_patch("0164")
FUSED_SAVE_PATCH = _series_patch("0165")


def _bf16_roundtrip(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", rounded & 0xFFFF0000))[0]


def test_native_op_is_registered_and_built() -> None:
    kernel = KERNEL.read_text()
    setup = (ROOT / "setup.py").read_text()
    header = (ROOT / "csrc/musa/musa_ops.h").read_text()
    bindings = (ROOT / "csrc/musa/torch_bindings.cpp").read_text()
    custom_ops = (ROOT / "vllm_musa/_custom_ops.py").read_text()

    assert "deepseek_v4_sparse_compressor.mu" in setup
    assert "void deepseek_v4_sparse_compress_cache(" in header
    assert "deepseek_v4_sparse_compress_cache(Tensor! state_cache" in bindings
    assert "Tensor? kv_states=None, Tensor? score_states=None, Tensor? ape=None" in bindings
    assert "def deepseek_v4_sparse_compress_cache(" in custom_ops
    assert "kv_states: Optional[torch.Tensor] = None" in custom_ops
    assert "deepseek_v4_sparse_compressor_kernel" in kernel
    assert "deepseek_v4_sparse_save_partial_kernel" in kernel


def test_kernel_uses_block_per_token_512d_geometry() -> None:
    source = KERNEL.read_text()

    assert "constexpr int64_t kHeadDim = 512;" in source
    assert "constexpr int64_t kNopeDim = kHeadDim - kRopeDim;" in source
    assert "constexpr int64_t kTokenStride = 576;" in source
    assert "constexpr int64_t kScaleDim = 8;" in source
    assert "constexpr int64_t kThreadsPerBlock = 128;" in source
    assert "constexpr int kCompressRows = (1 + static_cast<int>(kOverlap)) * kCompressRatio;" in source
    assert "const int64_t token = static_cast<int64_t>(blockIdx.x);" in source
    assert "half_warp_reduce_max_u32" in source
    assert "const __mt_fp8x4_e4m3 packed" in source
    assert "Hadamard" not in source


def test_kernel_preserves_vllm_pad_boundary_and_page_abi() -> None:
    source = KERNEL.read_text()

    assert "state_slot < 0" in source
    assert "deepseek_v4_sparse_save_partial_kernel" in source
    assert "launch_sparse_save_partial<4, true, 4>" in source
    assert "(position + 1) % kCompressRatio != 0" in source
    assert "req_idx * block_table_stride + logical_block" in source
    assert "head_offset = row >= kCompressRatio ? kHeadDim : 0" in source
    assert "kv_slot < 0" in source
    assert "kv_block_idx * kv_cache_stride0" in source
    assert "kv_block_size * kTokenValueBytes" in source
    assert "__bfloat162float(__float2bfloat16" in source
    assert "scale_ptr[kNopeBlocks] = 0" in source
    assert "reinterpret_cast<__mt_bfloat16*>(value_ptr + kNopeDim)" in source


def test_source_patch_keeps_triton_fallback() -> None:
    patch = SERIES_PATCH.read_text()
    native_call = patch.index("try_musa_deepseek_v4_sparse_compressor(")
    triton_dispatch = patch.index("if head_dim == 512:")

    assert native_call < triton_dispatch
    assert "if handled:" in patch
    assert "+            return" in patch
    assert "compress_ratio in (4, 128)" in patch


def test_fused_save_patch_skips_triton_save_on_native_hit() -> None:
    patch = FUSED_SAVE_PATCH.read_text()
    wrapper = WRAPPER.read_text()

    native_call = patch.index("try_musa_deepseek_v4_sparse_compressor(")
    skip_save = patch.index("if handled:")
    skip_return = patch.index("+                return")
    assert native_call < skip_save < skip_return
    assert "save_partial_states(" not in patch
    assert "kv_states=kv" in patch
    assert "score_states=score" in patch
    assert "ape=self.ape" in patch
    assert "kv_states: torch.Tensor | None = None" in wrapper
    assert "kv_states, score_states, and ape must be supplied together" in wrapper


def test_wrapper_is_default_on_and_shape_bounded() -> None:
    source = WRAPPER.read_text()

    assert "4: (4, 1024, 2048)" in source
    assert "128: (8, 512, 1024)" in source
    assert "0 < num_rows <= _MAX_DECODE_ROWS" in source
    assert "return False, reason" in source
    assert "VLLM_MUSA_DEEPSEEK_V4_SPARSE_COMPRESS_IMPL" not in source
    assert "os.environ" not in source


def test_cpu_oracle_c4_overlap_keeps_nonboundary_and_pad_slots_untouched() -> None:
    values = _cpu_reference_store_c4(
        token_to_req=[0, 0, 1],
        positions=[3, 4, 7],
        state_slots=[0, 1, -1],
        block_table=[[0, 1, 2, 3], [4, 5, 6, 7]],
        kv_slots=[0, 1, 2],
        kv_block_size=64,
        eps=1e-6,
    )
    assert set(values) == {(0, 0)}
    nope, rope, scales = values[(0, 0)]
    assert len(nope) == 448 and len(rope) == 64 and len(scales) == 8
    assert scales[7] == 0
    assert all(math.isfinite(v) for v in nope + rope)


def _state_value(page: int, offset: int, dim: int) -> float:
    return math.sin((page * 4099 + offset * 521 + dim * 17 + 1) * 0.001)


def _cpu_reference_store_c4(
    token_to_req: list[int],
    positions: list[int],
    state_slots: list[int],
    block_table: list[list[int]],
    kv_slots: list[int],
    kv_block_size: int,
    eps: float,
) -> dict[tuple[int, int], tuple[list[float], list[float], list[int]]]:
    output: dict[tuple[int, int], tuple[list[float], list[float], list[int]]] = {}
    for token, position in enumerate(positions):
        if state_slots[token] < 0 or (position + 1) % 4 != 0:
            continue
        kv_slot = kv_slots[token]
        if kv_slot < 0:
            continue
        values_by_dim: list[float] = []
        for dim in range(512):
            row_values = []
            row_scores = []
            for row, source_position in enumerate(range(position - 7, position + 1)):
                if source_position < 0:
                    row_values.append(0.0)
                    row_scores.append(float("-inf"))
                    continue
                page = block_table[token_to_req[token]][source_position // 4]
                page_offset = source_position % 4
                head_offset = 512 if row >= 4 else 0
                row_values.append(_state_value(page, page_offset, head_offset + dim))
                row_scores.append(
                    _state_value(page, page_offset, 1024 + head_offset + dim)
                )
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
        variance = sum(value * value for value in values_by_dim) / 512.0
        inv_rms = 1.0 / math.sqrt(variance + eps)
        output_vec = [
            value * inv_rms * (1.0 + (dim % 11 - 5) * 0.002)
            for dim, value in enumerate(values_by_dim)
        ]
        compressed_position = (position // 4) * 4
        rope = []
        for pair in range(32):
            dim = 448 + pair * 2
            angle = compressed_position * 0.003 + pair * 0.007
            even, odd = output_vec[dim], output_vec[dim + 1]
            rope.append(even * math.cos(angle) - odd * math.sin(angle))
            rope.append(odd * math.cos(angle) + even * math.sin(angle))
        nope = [_bf16_roundtrip(value) for value in output_vec[:448]]
        scales = []
        for block in range(7):
            chunk = nope[block * 64 : (block + 1) * 64]
            absmax = max(1e-4, max(abs(value) for value in chunk))
            exponent = math.ceil(math.log2(absmax / 448.0))
            scales.append(max(0, min(255, int(exponent) + 127)))
        scales.append(0)
        page, offset = divmod(kv_slot, kv_block_size)
        output[(page, offset)] = (nope, rope, scales)
    return output
