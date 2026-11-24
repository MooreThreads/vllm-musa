# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA fused Qwen4Exp mHC down projection + SiLU.

The NVIDIA path fuses the merged down+inject projection with the mHC epilogue
through a CuTe DSL kernel and gates it on SM90+. MUSA reports capability
``(3, 1)``, so it currently falls back to ``MergedColumnParallelLinear`` plus a
separate ``hc_silu`` launch.

This module implements the equivalent kernels in Triton and mirrors the
upstream dispatch:

* ``M <= 4``: one program per merged output column (the FMA-equivalent path).
  This is the decode shape the model actually runs most of the time.
* ``16 <= M <= 48``: a tiled ``tl.dot`` kernel with split-K, then a flat
  reduction/epilogue kernel. Upstream reaches the same range with its MMA +
  split-K CuTe kernel; a per-column kernel spills registers past ``M = 8`` on
  MUSA (measured: 3.7 ms at M=32, 8.0 ms at M=48).
* everything else (including 5..15, where split-K did not beat the padded
  ssgemm on S5000): the unfused ``MergedColumnParallelLinear`` + ``hc_silu``
  maths, computed here so the caller can keep a single gate.

Both fused paths keep the unfused rounding boundary: GEMM accumulator -> bf16,
then ``silu(value / hc_count)`` on the first ``rank`` columns and passthrough
on the injection columns.
"""

from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# Mirrors upstream MAX_FUSED_M; the per-M gate below decides which fused kernel
# (if any) each token count actually uses.
MAX_MUSA_FUSED_M = 48
_FMA_MAX_M = 4
_BLOCK_K = 1024
_NUM_WARPS = 8

# Split-K path (tuned on 8x S5000, K=10240): stage 1 does the tiled GEMM into an
# fp32 partial buffer, stage 2 reduces the splits and applies the epilogue.
_SPLITK_BK = 128
_SPLITK_STAGES = 2
_SPLITK_WARPS = 4
_SPLITK_SPLIT = 16
_SPLITK_MIN_M = 16
_SPLITK_BLOCK = 256


def hc_down_silu_enabled() -> bool:
    """Runtime switch used for matched fused/fallback A/B runs."""
    return os.environ.get("VLLM_MUSA_HC_DOWN_SILU", "1") != "0"


@triton.jit
def _hc_down_silu_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M: tl.constexpr,
    MP: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    SX: tl.constexpr,
    SW: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.program_id(0)
    offk = tl.arange(0, BK)
    offm = tl.arange(0, MP)
    m_mask = offm[:, None] < M
    acc = tl.zeros((MP, BK), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + offk
        mask = kk < K
        x = tl.load(
            x_ptr + offm[:, None] * SX + kk[None, :],
            mask=m_mask & mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + n * SW + kk, mask=mask, other=0.0).to(tl.float32)
        acc += x * w[None, :]

    value = tl.sum(acc, axis=1)
    # Match the unfused path exactly: GEMM accumulator -> bf16, then
    # silu(bf16_value / hc_count), then bf16 output.
    value_bf16 = value.to(tl.bfloat16).to(tl.float32)
    scaled = value_bf16 / HC
    activated = scaled * tl.sigmoid(scaled)
    value = tl.where(n < RANK, activated, value)
    tl.store(y_ptr + offm * N + n, value.to(tl.bfloat16), mask=offm < M)


@triton.jit
def _hc_down_silu_splitk_kernel(
    x_ptr,
    w_ptr,
    p_ptr,
    M: tl.constexpr,
    MP: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    KS: tl.constexpr,
    SX: tl.constexpr,
    SW: tl.constexpr,
    SN: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, MP)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((MP, BN), tl.float32)
    k_lo = pid_k * KS
    for k0 in range(k_lo, k_lo + KS, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(
            x_ptr + offs_m[:, None] * SX + offs_k[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        bt = tl.load(
            w_ptr + offs_n[:, None] * SW + offs_k[None, :],
            mask=offs_n[:, None] < N,
            other=0.0,
        )
        acc = tl.dot(a, tl.trans(bt), acc)
    tl.store(
        p_ptr + pid_k * SN + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _hc_down_silu_reduce_kernel(
    p_ptr,
    y_ptr,
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    SPLIT: tl.constexpr,
    SK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(p_ptr + s * SK + offs, mask=mask, other=0.0)
    n = offs % N
    value_bf16 = acc.to(tl.bfloat16).to(tl.float32)
    scaled = value_bf16 / HC
    activated = scaled * tl.sigmoid(scaled)
    tl.store(
        y_ptr + offs,
        tl.where(n < RANK, activated, value_bf16).to(tl.bfloat16),
        mask=mask,
    )


def _splitk_tile_n(num_tokens: int) -> int:
    """Column tile width; tuned per token-count band on S5000."""
    return 32 if num_tokens >= 25 else 64


def _splitk_usable(num_tokens: int, k: int) -> bool:
    if not _SPLITK_MIN_M <= num_tokens <= MAX_MUSA_FUSED_M:
        return False
    chunk = _SPLITK_SPLIT * _SPLITK_BK
    return k % chunk == 0 and k >= chunk


def _hc_down_silu_splitk(
    x: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    hc: int,
) -> torch.Tensor:
    num_tokens, k = x.shape
    out_n = rank + hc
    split = _SPLITK_SPLIT
    tile_n = _splitk_tile_n(num_tokens)
    mp = triton.next_power_of_2(num_tokens)
    partial = torch.empty(
        (split, mp, out_n), dtype=torch.float32, device=x.device
    )
    output = torch.empty((num_tokens, out_n), dtype=x.dtype, device=x.device)
    _hc_down_silu_splitk_kernel[(triton.cdiv(out_n, tile_n), split)](
        x,
        weight,
        partial,
        M=num_tokens,
        MP=mp,
        N=out_n,
        K=k,
        KS=k // split,
        SX=x.stride(0),
        SW=weight.stride(0),
        SN=mp * out_n,
        BN=tile_n,
        BK=_SPLITK_BK,
        num_stages=_SPLITK_STAGES,
        num_warps=_SPLITK_WARPS,
    )
    total = num_tokens * out_n
    _hc_down_silu_reduce_kernel[(triton.cdiv(total, _SPLITK_BLOCK),)](
        partial,
        output,
        TOTAL=total,
        N=out_n,
        RANK=rank,
        HC=hc,
        SPLIT=split,
        SK=mp * out_n,
        BLOCK=_SPLITK_BLOCK,
    )
    return output


def _hc_down_silu_unfused(
    x: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    hc: int,
) -> torch.Tensor:
    """Unfused maths for token counts no fused kernel wins on."""
    down = torch.nn.functional.linear(x, weight).to(torch.bfloat16)
    down = down[:, : rank + hc].contiguous()
    lora = down[:, :rank].float() / hc
    down[:, :rank] = (lora * torch.sigmoid(lora)).to(torch.bfloat16)
    return down


def _hc_down_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    hc: int,
) -> torch.Tensor:
    num_tokens, k = x.shape
    out_n = rank + hc
    assert x.ndim == 2 and weight.ndim == 2
    assert x.stride(1) == 1 and weight.stride(1) == 1
    assert weight.shape[1] == k and weight.shape[0] >= out_n
    assert 1 <= num_tokens <= MAX_MUSA_FUSED_M
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16

    if num_tokens > _FMA_MAX_M:
        if _splitk_usable(num_tokens, k):
            return _hc_down_silu_splitk(x, weight, rank, hc)
        return _hc_down_silu_unfused(x, weight, rank, hc)

    output = torch.empty((num_tokens, out_n), dtype=x.dtype, device=x.device)
    _hc_down_silu_kernel[(out_n,)](
        x,
        weight,
        output,
        M=num_tokens,
        MP=triton.next_power_of_2(num_tokens),
        K=k,
        N=out_n,
        RANK=rank,
        HC=hc,
        SX=x.stride(0),
        SW=weight.stride(0),
        BK=_BLOCK_K,
        num_warps=_NUM_WARPS,
    )
    return output


def _hc_down_silu_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    hc: int,
) -> torch.Tensor:
    del weight
    return torch.empty(
        (x.shape[0], rank + hc), dtype=x.dtype, device=x.device
    )


direct_register_custom_op(
    op_name="musa_qwen4_exp_hc_down_silu",
    op_func=_hc_down_silu,
    fake_impl=_hc_down_silu_fake,
)


def hc_down_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    hc: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(silu(down[:, :rank] / hc), down[:, rank:rank + hc])``."""
    # Dynamo needs the registered op for a graph-safe, opaque call. Eager
    # execution and CUDA-graph capture can launch the Triton kernel directly
    # and skip the dispatcher cost of the custom-op wrapper.
    if torch.compiler.is_compiling():
        output = torch.ops.vllm.musa_qwen4_exp_hc_down_silu(
            x, weight, rank, hc
        )
    else:
        output = _hc_down_silu(x, weight, rank, hc)
    return output[:, :rank], output[:, rank:]


def supports_hc_down_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    hc: int,
) -> bool:
    return (
        hc_down_silu_enabled()
        and x.ndim == 2
        and weight.ndim == 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and 1 <= x.shape[0] <= MAX_MUSA_FUSED_M
        and (
            x.shape[0] <= _FMA_MAX_M
            or _splitk_usable(x.shape[0], x.shape[1])
        )
        and x.shape[1] % 8 == 0
        and x.shape[1] == weight.shape[1]
        and weight.shape[0] >= rank + hc
        and x.stride(1) == 1
        and weight.stride(1) == 1
    )


def request_hc_down_silu_warmup(
    m_values,
    rank: int,
    hc: int,
    k: int,
) -> None:
    """Compile the decode kernel for the graph-capture token counts."""
    if not hc_down_silu_enabled():
        return
    m_set = {
        int(m) for m in m_values if 1 <= int(m) <= MAX_MUSA_FUSED_M
    }
    if not m_set:
        return
    device = torch.device("musa", torch.musa.current_device())
    weight = torch.zeros((rank + hc, k), dtype=torch.bfloat16, device=device)
    for num_tokens in sorted(m_set):
        x = torch.zeros(
            (num_tokens, k), dtype=torch.bfloat16, device=device
        )
        hc_down_silu(x, weight, rank, hc)
    torch.musa.synchronize()
