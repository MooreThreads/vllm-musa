# SPDX-License-Identifier: Apache-2.0
"""Small-M FP8 GEMV for block-scaled weights.

DeepGEMM and the native MUSA GEMV take 64/128-wide scale blocks only. Weights
quantized with finer blocks (e.g. 32x32) otherwise run the generic Triton block
GEMM, whose K tile is limited to one scale block. These kernels read a wide K
tile per step, apply the activation and weight scales per scale group, and
optionally gather the weight of a routed expert per row. A BF16 activation can
be quantized per group inside the kernel, with the arithmetic of
``per_token_group_quant_fp8``.
"""

from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton

from vllm_musa.kernels.fp8_block_splitk import (
    fp8_block_splitk_config,
    fp8_block_splitk_mm,
)

_FP8_MAX = 448.0
_QUANT_EPS = 1e-10

# Native GEMV configs (``vllm_musa/jit_kernel/csrc/gemv``) measured on MTT S5000
# with a cold L2 for the DeepSeek-V4.1 TP8 projections, keyed by (N, K): the
# config id for 1-8 rows, or None where the Triton GEMV is faster. The BF16 table
# includes the separate activation quantization. One row keeps the Triton GEMV: its
# in-kernel quantization saves a launch, which outweighs the faster loop in a graph.
_NATIVE_BF16_CFG = {
    (1792, 5120): (None, 9, 9, 10, 10, 10, 10, 8),
    (1024, 4096): (None, 9, 3, 3, 2, 3, 2, 0),
    (576, 5120): (None, 9, 9, 3, 9, 9, 9, 2),
}
_NATIVE_FP8_CFG = {
    (1792, 5120): (None, 9, 9, 10, 10, 10, 10, 8),
    (1024, 4096): (None, 9, 3, 3, 2, 3, 2, 0),
    (576, 5120): (None, 9, 9, 3, 9, 9, 9, 2),
}
_NATIVE_GEMV = os.environ.get("VLLM_MUSA_FP8_NATIVE_GEMV", "1") != "0"


# (BLOCK_N, BLOCK_K, num_warps) measured on MTT S5000, keyed by (N, K, routed
# experts), for the DeepSeek-V4.1 TP8 projections: dense entries at M=1, routed
# entries at 1-64 tokens.
_MEASURED_TILES = {
    (576, 5120, True): (4, 512, 4),
    (576, 5120, False): (4, 4096, 16),
    (1792, 5120, False): (8, 2048, 16),
    (1024, 4096, False): (4, 4096, 8),
    (5120, 288, True): (32, 512, 16),
    (5120, 288, False): (32, 512, 16),
    (25600, 6144, False): (32, 256, 4),
}


def _tile(n: int, k: int, routed: bool = False, rows: int = 1) -> tuple[int, int, int]:
    """(BLOCK_N, BLOCK_K, num_warps) for an N x K weight at decode M."""
    measured = _MEASURED_TILES.get((n, k, routed))
    if measured is not None and (routed or rows == 1):
        return measured
    if k >= 4096 and n <= 1024:
        return 4, 2048, 8
    if k >= 4096 and n <= 2048:
        return 8, 1024, 8
    if n >= 16384:
        return 16, 512, 4
    return 8, 512, 4


def _rows_tile(n: int, k: int, rows: int) -> tuple[int, int, int, int] | None:
    """(ROWS, BLOCK_N, BLOCK_K, num_warps) for the multi-row dense GEMV, or None for one row each.

    Measured on MTT S5000: sharing a weight tile across rows wins for the K=5120
    DeepSeek-V4.1 TP8 projections from 4 rows on; elsewhere one row per program wins.
    """
    if (n, k) == (576, 5120) and rows >= 4:
        return (
            (2, 8, 1024, 8)
            if rows <= 8
            else (2, 8, 512, 4)
            if rows <= 16
            else (4, 8, 512, 4)
        )
    if (n, k) == (1792, 5120) and rows >= 4 and rows != 5:
        return (4, 8, 512, 4) if 6 <= rows <= 8 else (2, 8, 512, 4)
    return None


@triton.jit
def _fp8_block_gemv_scaled_activation(
    a_ptr,
    a_scale_ptr,
    bs_row,
    a_row,
    k0,
    K,
    stride_am,
    stride_asm,
    QUANT_A: tl.constexpr,
    SWIGLU: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    A_GROUP_K: tl.constexpr,
    B_GROUP_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_EPS: tl.constexpr,
):
    """One BLOCK_K slice of an activation row, dequantized and scaled by the weight scales."""
    k = k0 + tl.arange(0, BLOCK_K)
    k_mask = k < K
    a = tl.load(a_ptr + a_row * stride_am + k, mask=k_mask, other=0.0).to(tl.float32)
    if SWIGLU:
        # [gate | up] row: bf16(bf16(silu(min(gate, L))) * clamp(up, -L, L)), as
        # silu_and_mul_with_clamp rounds it.
        up = tl.load(a_ptr + a_row * stride_am + K + k, mask=k_mask, other=0.0).to(
            tl.float32
        )
        up = tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
        gate = tl.minimum(a, SWIGLU_LIMIT)
        silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        a = (silu * up).to(tl.bfloat16).to(tl.float32)
    a = tl.reshape(a, (BLOCK_K // GROUP_K, GROUP_K))
    g_start = k0 + tl.arange(0, BLOCK_K // GROUP_K) * GROUP_K
    if QUANT_A:
        # One activation group per GROUP_K slice: absmax from eps, scale = absmax / max.
        a_s = tl.maximum(tl.max(tl.abs(a), axis=1), QUANT_EPS) / FP8_MAX
        a = (
            tl.clamp(a / a_s[:, None], -FP8_MAX, FP8_MAX)
            .to(tl.float8e4nv)
            .to(tl.float32)
        )
    else:
        ga = g_start // A_GROUP_K
        a_s = tl.load(
            a_scale_ptr + a_row * stride_asm + ga,
            mask=ga < tl.cdiv(K, A_GROUP_K),
            other=0.0,
        )
    gb = g_start // B_GROUP_K
    b_s = tl.load(bs_row + gb, mask=gb < tl.cdiv(K, B_GROUP_K), other=0.0)
    return tl.reshape(a * (a_s * b_s)[:, None], (BLOCK_K,)), k, k_mask


@triton.jit
def _fp8_block_gemv_dot(
    a_ptr,
    a_scale_ptr,
    b_rows,
    bs_row,
    a_row,
    n_mask,
    K,
    stride_am,
    stride_asm,
    QUANT_A: tl.constexpr,
    SWIGLU: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    A_GROUP_K: tl.constexpr,
    B_GROUP_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_EPS: tl.constexpr,
):
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a, k, k_mask = _fp8_block_gemv_scaled_activation(
            a_ptr,
            a_scale_ptr,
            bs_row,
            a_row,
            k0,
            K,
            stride_am,
            stride_asm,
            QUANT_A,
            SWIGLU,
            SWIGLU_LIMIT,
            A_GROUP_K,
            B_GROUP_K,
            GROUP_K,
            BLOCK_K,
            FP8_MAX,
            QUANT_EPS,
        )
        b = tl.load(
            b_rows + k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0
        )
        acc += tl.sum(b.to(tl.float32) * a[None, :], axis=1)
    return acc


@triton.jit
def _fp8_block_gemv_rows_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_asm,
    stride_bn,
    stride_bsn,
    stride_cm,
    QUANT_A: tl.constexpr,
    SWIGLU: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    GROUP_N: tl.constexpr,
    A_GROUP_K: tl.constexpr,
    B_GROUP_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_EPS: tl.constexpr,
):
    """``ROWS`` activation rows per program against one weight tile, read once per K slice."""
    pid_n = tl.program_id(0)
    row0 = tl.program_id(1).to(tl.int64) * ROWS
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    # BLOCK_N divides GROUP_N, so a program's rows share one weight-scale row.
    bs_row = b_scale_ptr + (pid_n * BLOCK_N // GROUP_N).to(tl.int64) * stride_bsn
    b_rows = b_ptr + offs_n[:, None].to(tl.int64) * stride_bn
    offs_r = tl.arange(0, ROWS)
    acc = tl.zeros((ROWS, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        b = tl.load(
            b_rows + k[None, :], mask=n_mask[:, None] & (k[None, :] < K), other=0.0
        )
        b = b.to(tl.float32)
        for r in tl.static_range(ROWS):
            # Rows past M recompute the last row; their results are not stored.
            a, _, _ = _fp8_block_gemv_scaled_activation(
                a_ptr,
                a_scale_ptr,
                bs_row,
                tl.minimum(row0 + r, M - 1),
                k0,
                K,
                stride_am,
                stride_asm,
                QUANT_A,
                SWIGLU,
                SWIGLU_LIMIT,
                A_GROUP_K,
                B_GROUP_K,
                GROUP_K,
                BLOCK_K,
                FP8_MAX,
                QUANT_EPS,
            )
            part = tl.sum(b * a[None, :], axis=1)
            acc += tl.where(offs_r[:, None] == r, part[None, :], 0.0)
    rows = row0 + offs_r
    tl.store(
        c_ptr + rows[:, None] * stride_cm + offs_n[None, :],
        acc.to(c_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & n_mask[None, :],
    )


@triton.jit
def _fp8_block_gemv_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    expert_ids_ptr,
    routed_weights_ptr,
    N,
    K,
    num_experts,
    stride_am,
    stride_asm,
    stride_be,
    stride_bn,
    stride_bse,
    stride_bsn,
    stride_cm,
    A_ROWS_PER_TOKEN: tl.constexpr,
    QUANT_A: tl.constexpr,
    SWIGLU: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    HAS_EXPERTS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    SUM_TOPK: tl.constexpr,
    SLOTS: tl.constexpr,
    GROUP_N: tl.constexpr,
    A_GROUP_K: tl.constexpr,
    B_GROUP_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_EPS: tl.constexpr,
):
    """One output row per program row; with ``SUM_TOPK`` a token row sums its ``SLOTS`` routed rows."""
    pid_n = tl.program_id(0)
    out_row = tl.program_id(1).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = offs_n < N
    # BLOCK_N divides GROUP_N, so a program's rows share one weight-scale row.
    scale_row = (pid_n * BLOCK_N // GROUP_N).to(tl.int64)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for slot in tl.static_range(SLOTS):
        row = out_row * SLOTS + slot if SUM_TOPK else out_row
        n_mask = out_mask
        b_base = b_ptr
        bs_base = b_scale_ptr
        if HAS_EXPERTS:
            expert = tl.load(expert_ids_ptr + row).to(tl.int64)
            valid = (expert >= 0) & (expert < num_experts)
            # Rows routed outside the local experts read nothing and add zeros.
            n_mask = n_mask & valid
            expert = tl.where(valid, expert, 0)
            b_base += expert * stride_be
            bs_base += expert * stride_bse
        part = _fp8_block_gemv_dot(
            a_ptr,
            a_scale_ptr,
            b_base + offs_n[:, None].to(tl.int64) * stride_bn,
            bs_base + scale_row * stride_bsn,
            row // A_ROWS_PER_TOKEN,
            n_mask,
            K,
            stride_am,
            stride_asm,
            QUANT_A,
            SWIGLU,
            SWIGLU_LIMIT,
            A_GROUP_K,
            B_GROUP_K,
            GROUP_K,
            BLOCK_N,
            BLOCK_K,
            FP8_MAX,
            QUANT_EPS,
        )
        if MUL_ROUTED_WEIGHT:
            part = part * tl.load(routed_weights_ptr + row).to(tl.float32)
        acc += part
    tl.store(
        c_ptr + out_row * stride_cm + offs_n,
        acc.to(c_ptr.dtype.element_ty),
        mask=out_mask,
    )


def _groups(a_scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor):
    k = weight.shape[-1]
    group_n = triton.cdiv(weight.shape[-2], weight_scale.shape[-2])
    a_group_k = triton.cdiv(k, a_scale.shape[-1])
    b_group_k = triton.cdiv(k, weight_scale.shape[-1])
    return group_n, a_group_k, b_group_k, min(a_group_k, b_group_k)


def _weight_groups(weight: torch.Tensor, weight_scale: torch.Tensor) -> tuple[int, int]:
    group_n = triton.cdiv(weight.shape[-2], weight_scale.shape[-2])
    return group_n, triton.cdiv(weight.shape[-1], weight_scale.shape[-1])


def _weight_supported(weight: torch.Tensor, weight_scale: torch.Tensor) -> bool:
    if weight_scale.dtype != torch.float32 or weight.dtype != torch.float8_e4m3fn:
        return False
    if weight.stride(-1) != 1 or weight_scale.stride(-1) != 1:
        return False
    group_n, b_group_k = _weight_groups(weight, weight_scale)
    return (
        group_n in (1, 2, 4, 8, 16, 32, 64, 128)
        and b_group_k in (16, 32, 64, 128)
        and triton.cdiv(weight.shape[-1], b_group_k) == weight_scale.shape[-1]
        and triton.cdiv(weight.shape[-2], group_n) == weight_scale.shape[-2]
    )


def fp8_block_gemv_supported(
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> bool:
    """Whether the activation and weight scales tile K in power-of-two groups."""
    if a_scale.dtype != torch.float32 or a_scale.stride(-1) != 1:
        return False
    if not _weight_supported(weight, weight_scale):
        return False
    k = weight.shape[-1]
    _, a_group_k, b_group_k, _ = _groups(a_scale, weight, weight_scale)
    return (
        a_group_k in (16, 32, 64, 128)
        and 512 % max(a_group_k, b_group_k) == 0
        and triton.cdiv(k, a_group_k) == a_scale.shape[-1]
    )


def fp8_block_gemv_quant_supported(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> bool:
    """Whether the in-kernel activation quantization can feed this weight."""
    if not _weight_supported(weight, weight_scale):
        return False
    _, b_group_k = _weight_groups(weight, weight_scale)
    return (
        group_size in (16, 32, 64, 128)
        and b_group_k % group_size == 0
        and weight.shape[-1] % group_size == 0
    )


def _launch(
    a: torch.Tensor,
    a_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    expert_ids: torch.Tensor | None,
    routed_weights: torch.Tensor | None,
    a_rows_per_token: int,
    quant_group: int,
    sum_topk: int = 0,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    n, k = weight.shape[-2], weight.shape[-1]
    group_n, b_group_k = _weight_groups(weight, weight_scale)
    if a_scale is None:
        a_group_k = quant_group
        a_scale_stride = 0
    else:
        a_group_k = triton.cdiv(k, a_scale.shape[-1])
        a_scale_stride = a_scale.stride(0)
    has_experts = expert_ids is not None
    out_rows = rows // sum_topk if sum_topk else rows
    multi_row = None if has_experts or sum_topk else _rows_tile(n, k, rows)
    if multi_row is not None:
        rows_per_program, block_n, block_k, num_warps = multi_row
        _fp8_block_gemv_rows_kernel[
            (triton.cdiv(n, block_n), triton.cdiv(rows, rows_per_program))
        ](
            a,
            a_scale,
            weight,
            weight_scale,
            output,
            rows,
            n,
            k,
            a.stride(0),
            a_scale_stride,
            weight.stride(-2),
            weight_scale.stride(-2),
            output.stride(0),
            QUANT_A=a_scale is None,
            SWIGLU=swiglu_limit is not None,
            SWIGLU_LIMIT=float(swiglu_limit) if swiglu_limit is not None else 0.0,
            GROUP_N=group_n,
            A_GROUP_K=a_group_k,
            B_GROUP_K=b_group_k,
            GROUP_K=min(a_group_k, b_group_k),
            ROWS=rows_per_program,
            BLOCK_N=min(block_n, group_n),
            BLOCK_K=block_k,
            FP8_MAX=_FP8_MAX,
            QUANT_EPS=_QUANT_EPS,
            num_warps=num_warps,
        )
        return output
    block_n, block_k, num_warps = _tile(n, k, has_experts, out_rows)
    block_n = min(block_n, group_n)
    _fp8_block_gemv_kernel[(triton.cdiv(n, block_n), out_rows)](
        a,
        a_scale,
        weight,
        weight_scale,
        output,
        expert_ids,
        routed_weights,
        n,
        k,
        weight.shape[0] if has_experts else 1,
        a.stride(0),
        a_scale_stride,
        weight.stride(0) if has_experts else 0,
        weight.stride(-2),
        weight_scale.stride(0) if has_experts else 0,
        weight_scale.stride(-2),
        output.stride(0),
        A_ROWS_PER_TOKEN=a_rows_per_token,
        QUANT_A=a_scale is None,
        SWIGLU=swiglu_limit is not None,
        SWIGLU_LIMIT=float(swiglu_limit) if swiglu_limit is not None else 0.0,
        HAS_EXPERTS=has_experts,
        MUL_ROUTED_WEIGHT=routed_weights is not None,
        SUM_TOPK=sum_topk > 0,
        SLOTS=max(1, sum_topk),
        GROUP_N=group_n,
        A_GROUP_K=a_group_k,
        B_GROUP_K=b_group_k,
        GROUP_K=min(a_group_k, b_group_k),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        FP8_MAX=_FP8_MAX,
        QUANT_EPS=_QUANT_EPS,
        num_warps=num_warps,
    )
    return output


def _native_cfg(
    table: dict, weight: torch.Tensor, weight_scale: torch.Tensor, rows: int
) -> int | None:
    """Native GEMV config for ``rows`` activation rows, or None for the Triton GEMV."""
    if not _NATIVE_GEMV or not 0 < rows <= len(next(iter(table.values()))):
        return None
    cfgs = table.get(tuple(weight.shape))
    if cfgs is None or cfgs[rows - 1] is None:
        return None
    if (
        weight_scale.dtype != torch.float32
        or weight.stride(-1) != 1
        or weight_scale.stride(-1) != 1
        or _weight_groups(weight, weight_scale) != (32, 32)
    ):
        return None
    return cfgs[rows - 1]


def fp8_block_gemv(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """``output[m] = dequant(a[m]) @ dequant(weight).T`` for a 2D FP8 weight."""
    splitk = fp8_block_splitk_config(weight, weight_scale, a.shape[0])
    if (
        splitk is not None
        and a.dtype == torch.float8_e4m3fn
        and a.stride(-1) == 1
        and a_scale.stride(-1) == 1
        and weight.shape[1] % a_scale.shape[-1] == 0
        and weight.shape[1] // a_scale.shape[-1] in (32, 64, 128)
    ):
        return fp8_block_splitk_mm(a, a_scale, weight, weight_scale, output, splitk)
    cfg = _native_cfg(_NATIVE_FP8_CFG, weight, weight_scale, a.shape[0])
    if (
        cfg is not None
        and a.dtype == torch.float8_e4m3fn
        and a.stride(-1) == 1
        and output.stride(-1) == 1
        and a_scale.stride(-1) == 1
        and weight.shape[1] % a_scale.shape[-1] == 0
        and weight.shape[1] // a_scale.shape[-1] in (32, 64, 128)
    ):
        from vllm_musa.jit_kernel.csrc import fp8_block_gemv as native

        return native.dense_gemv(a, a_scale, weight, weight_scale, output, cfg)
    return _launch(
        a, a_scale, weight, weight_scale, output, a.shape[0], None, None, 1, 0
    )


def fp8_block_gemv_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    group_size: int,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """``fp8_block_gemv`` of ``per_token_group_quant_fp8(x, group_size)``, quantized in-kernel.

    With ``swiglu_limit``, ``x`` is a ``[gate | up]`` row and the GEMV input is its
    clamped SwiGLU.
    """
    if swiglu_limit is None and group_size == 32 and x.dtype == torch.bfloat16:
        splitk = fp8_block_splitk_config(weight, weight_scale, x.shape[0])
        if splitk is not None and x.stride(-1) == 1:
            return fp8_block_splitk_mm(x, None, weight, weight_scale, output, splitk)
    cfg = (
        _native_cfg(_NATIVE_BF16_CFG, weight, weight_scale, x.shape[0])
        if swiglu_limit is None and group_size == 32
        else None
    )
    if (
        cfg is not None
        and x.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and output.stride(-1) == 1
    ):
        from vllm_musa.jit_kernel.csrc import fp8_block_gemv as native

        xq, xs = native.act_quant(x)
        return native.dense_gemv(xq, xs, weight, weight_scale, output, cfg)
    return _launch(
        x,
        None,
        weight,
        weight_scale,
        output,
        x.shape[0],
        None,
        None,
        1,
        group_size,
        swiglu_limit=swiglu_limit,
    )


def fp8_block_moe_gemv(
    a: torch.Tensor,
    a_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    expert_ids: torch.Tensor,
    a_rows_per_token: int,
    routed_weights: torch.Tensor | None = None,
    quant_group: int = 0,
    sum_topk: int = 0,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Per routed row ``r``: ``output[r] = a[r // a_rows_per_token] @ weight[expert_ids[r]].T``.

    Rows routed to an expert outside ``[0, E)`` contribute zeros. With
    ``routed_weights`` each row is scaled by its router weight. Without
    ``a_scale``, ``a`` is a BF16 activation quantized in-kernel per
    ``quant_group`` elements. With ``sum_topk``, ``output[t]`` is the sum of the
    ``sum_topk`` consecutive routed rows of token ``t``, rounded once. With
    ``swiglu_limit``, rows of ``a`` are ``[gate | up]`` and the GEMV input is their
    clamped SwiGLU.
    """
    return _launch(
        a,
        a_scale,
        weight,
        weight_scale,
        output,
        expert_ids.numel(),
        expert_ids,
        routed_weights,
        a_rows_per_token,
        quant_group,
        sum_topk,
        swiglu_limit,
    )
