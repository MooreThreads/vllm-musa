# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA fused Qwen4Exp mHC down projection + SiLU.

The NVIDIA path fuses the merged down+inject projection with the mHC epilogue
through a CuTe DSL kernel and gates it on SM90+. MUSA reports capability
``(3, 1)``, so it currently falls back to ``MergedColumnParallelLinear`` plus a
separate ``hc_silu`` launch.

This module implements the equivalent decode kernel in Triton: one program per
merged output column, an FP32 accumulator over K, and the same bf16 rounding
boundary before ``silu(x / hc_count)``. It mirrors the upstream FMA path used
for small token counts and leaves larger M on the regular linear.
"""

from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# The production decode shape is M=1. Four tokens covers the small-M range the
# upstream FMA kernel targets; larger M stays on MergedColumnParallelLinear.
MAX_MUSA_FUSED_M = 4
_BLOCK_K = 1024
_NUM_WARPS = 4


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
