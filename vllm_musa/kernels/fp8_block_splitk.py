# SPDX-License-Identifier: Apache-2.0
"""Split-K FP8 GEMM for small M and 32x32 block-scaled weights.

Each program multiplies a 16- or 32-row tile by a BLOCK_N x (K / SPLIT) weight
slice, one 32-wide scale group per FP8 dot, so there are enough programs to keep
the GPU busy at decode M; row tiles that share a weight slice run back to back.
A BF16 activation is quantized per 32-wide group inside the kernel with the
arithmetic of ``per_token_group_quant_fp8``; each group lies inside one K slice.
A second kernel sums the SPLIT partials in a fixed order and rounds once, so a
row's result depends on neither M nor the tile height.
"""

from __future__ import annotations

import functools
import os

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from vllm_musa import tuning

_FP8_MAX = 448.0
_QUANT_EPS = 1e-10
BLOCK_M = 16

# Measured rows live in vllm_musa/tuning/fp8_block_splitk.json, keyed by (N, K) and
# optionally by multiprocessor count; a config is (SPLIT, BLOCK_N, num_warps, UNROLL,
# MIN_ROWS, MAX_ROWS, TILE_M). The reduction is a second launch, so below MIN_ROWS the
# single-kernel GEMV wins; past MAX_ROWS the BF16 GEMM of fp8_block_bf16 or the block
# GEMM does. TILE_M rows per program where that pads no more rows than BLOCK_M-row
# tiles. MAX_ROWS holds while a CUDA graph is captured (decode); eager steps, prefill
# included, use the eager limit, past which the block GEMM runs up to
# fp8_block_bf16.MIN_ROWS.
_ENTRY_KEYS = frozenset(
    "n k split block_n num_warps unroll tile_m min_rows max_rows".split()
)
_MODES = frozenset({"capture", "eager"})


def _from_table(entries: list[dict]) -> dict:
    """``(N, K, multiprocessor count or None) -> (config, eager max rows)``."""
    rows = {}
    for index, entry in enumerate(entries):
        where = f"fp8_block_splitk.json entry {index}"
        if set(entry) - {"multiprocessor_count"} != _ENTRY_KEYS:
            raise ValueError(
                f"{where}: keys must be {sorted(_ENTRY_KEYS)}, "
                "optionally with multiprocessor_count"
            )
        max_rows = entry["max_rows"]
        if not isinstance(max_rows, dict) or set(max_rows) != _MODES:
            raise ValueError(f"{where}: max_rows must have exactly {sorted(_MODES)}")
        values = [v for key, v in entry.items() if key != "max_rows"]
        if any(isinstance(v, dict) for v in values + list(max_rows.values())):
            raise ValueError(f"{where}: values other than max_rows must be integers")
        if entry["min_rows"] > min(max_rows.values()):
            raise ValueError(f"{where}: min_rows exceeds max_rows")
        key = (entry["n"], entry["k"], entry.get("multiprocessor_count"))
        if key in rows:
            raise ValueError(f"{where}: repeats {key}")
        config = (
            entry["split"],
            entry["block_n"],
            entry["num_warps"],
            entry["unroll"],
            entry["min_rows"],
            max_rows["capture"],
            entry["tile_m"],
        )
        rows[key] = (config, max_rows["eager"])
    return rows


_ROWS = _from_table(tuning.load("fp8_block_splitk"))
# The default rows serve every multiprocessor count without rows of its own.
_CONFIGS = {(n, k): row[0] for (n, k, mp), row in _ROWS.items() if mp is None}
_EAGER_MAX_ROWS = {(n, k): row[1] for (n, k, mp), row in _ROWS.items() if mp is None}
_BINNED_SHAPES = frozenset((n, k) for n, k, mp in _ROWS if mp is not None)
_ENABLED = os.environ.get("VLLM_MUSA_FP8_SPLITK_MM", "1") != "0"


@triton.jit
def _fp8_block_splitk_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    part_ptr,
    M,
    N,
    stride_am,
    stride_asm,
    stride_bn,
    stride_bsn,
    QUANT_A: tl.constexpr,
    A_GROUP: tl.constexpr,
    GROUP_N: tl.constexpr,
    K_PER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    EVEN_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_EPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 32)
    m_mask = offs_m < M
    n_mask = offs_n < N
    a_rows = a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :]
    b_cols = b_ptr + offs_n[None, :].to(tl.int64) * stride_bn + offs_k[:, None]
    bs_cols = b_scale_ptr + (offs_n // GROUP_N).to(tl.int64) * stride_bsn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_start = pid_k * K_PER
    for k0 in range(k_start, k_start + K_PER, 32 * UNROLL):
        for j in tl.static_range(UNROLL):
            kk = k0 + j * 32
            if EVEN_N:
                b = tl.load(b_cols + kk)
                b_s = tl.load(bs_cols + kk // 32)
            else:
                b = tl.load(b_cols + kk, mask=n_mask[None, :], other=0.0)
                b_s = tl.load(bs_cols + kk // 32, mask=n_mask, other=0.0)
            if QUANT_A:
                x = tl.load(a_rows + kk, mask=m_mask[:, None], other=0.0).to(tl.float32)
                a_s = tl.maximum(tl.max(tl.abs(x), axis=1), QUANT_EPS) / FP8_MAX
                a = tl.clamp(x / a_s[:, None], -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            else:
                a = tl.load(a_rows + kk, mask=m_mask[:, None], other=0.0)
                a_s = tl.load(
                    a_scale_ptr + offs_m * stride_asm + kk // A_GROUP,
                    mask=m_mask,
                    other=0.0,
                )
            acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
    parts = part_ptr + (pid_k * M + offs_m[:, None]).to(tl.int64) * N + offs_n[None, :]
    tl.store(parts, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _fp8_block_splitk_reduce_kernel(
    part_ptr,
    c_ptr,
    M,
    N,
    stride_cm,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + s * M * N + offs, mask=mask, other=0.0)
    rows = offs // N
    cols = offs - rows * N
    tl.store(c_ptr + rows * stride_cm + cols, acc.to(c_ptr.dtype.element_ty), mask=mask)


@functools.cache
def _multiprocessor_count(device_index: int | None) -> int:
    return current_platform.num_compute_units(device_index)


def _rows(weight: torch.Tensor) -> tuple[tuple | None, int | None]:
    """The (config, eager max rows) for ``weight`` on its device's multiprocessor count."""
    shape = tuple(weight.shape)
    if shape in _BINNED_SHAPES:
        mp = _multiprocessor_count(weight.device.index)
        binned = _ROWS.get((*shape, mp))
        if binned is not None:
            return binned
    return _CONFIGS.get(shape), _EAGER_MAX_ROWS.get(shape)


def fp8_block_splitk_config(
    weight: torch.Tensor, weight_scale: torch.Tensor, rows: int
) -> tuple[int, int, int, int, int, int, int] | None:
    """Measured config for ``rows`` activation rows, or None when the GEMM does not apply."""
    if not _ENABLED or weight.dim() != 2:
        return None
    config, eager_max_rows = _rows(weight)
    if config is None:
        return None
    max_rows = config[5]
    if not torch.cuda.is_current_stream_capturing():
        max_rows = eager_max_rows
    if not config[4] <= rows <= max_rows:
        return None
    n, k = weight.shape
    if (
        weight.dtype != torch.float8_e4m3fn
        or weight_scale.dtype != torch.float32
        or weight.stride(-1) != 1
        or weight_scale.stride(-1) != 1
        or tuple(weight_scale.shape) != (triton.cdiv(n, 32), k // 32)
        or k % (config[0] * 32 * config[3])
    ):
        return None
    return config


def _tile_rows(rows: int, tile_m: int) -> int:
    """``tile_m`` unless it pads more rows than ``BLOCK_M``-row tiles."""
    if triton.cdiv(rows, tile_m) * tile_m == triton.cdiv(rows, BLOCK_M) * BLOCK_M:
        return tile_m
    return BLOCK_M


def fp8_block_splitk_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    config: tuple[int, int, int, int, int, int, int],
) -> torch.Tensor:
    """``output = dequant(a) @ dequant(weight).T`` for the rows ``config`` admits.

    Without ``a_scale``, ``a`` is BF16 and quantized like
    ``per_token_group_quant_fp8(a, 32)``. With it, ``a`` is FP8 with 32/64/128-wide
    group scales.
    """
    split, block_n, num_warps, unroll, _, _, tile_m = config
    m = a.shape[0]
    n, k = weight.shape
    block_m = _tile_rows(m, tile_m)
    parts = torch.empty((split, m, n), dtype=torch.float32, device=a.device)
    quant = a_scale is None
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split)
    _fp8_block_splitk_kernel[grid](
        a,
        a if quant else a_scale,
        weight,
        weight_scale,
        parts,
        m,
        n,
        a.stride(0),
        0 if quant else a_scale.stride(0),
        weight.stride(0),
        weight_scale.stride(0),
        QUANT_A=quant,
        A_GROUP=32 if quant else k // a_scale.shape[-1],
        GROUP_N=32,
        K_PER=k // split,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        UNROLL=unroll,
        EVEN_N=n % block_n == 0,
        FP8_MAX=_FP8_MAX,
        QUANT_EPS=_QUANT_EPS,
        num_warps=num_warps,
        num_stages=1,
    )
    # Narrow blocks keep more reduction programs in flight; measured on MTT S5000.
    _fp8_block_splitk_reduce_kernel[(triton.cdiv(m * n, 128),)](
        parts,
        output,
        m,
        n,
        output.stride(0),
        SPLIT=split,
        BLOCK=128,
        num_warps=2 if split >= 16 else 4,
    )
    return output
