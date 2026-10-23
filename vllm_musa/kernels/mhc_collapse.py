# SPDX-License-Identifier: Apache-2.0
"""Collapse of the mHC residual streams into one hidden state.

``out[t] = sum_h mix[t, h] * residual[t, h]`` (or the mean over ``h`` without
``mix``), accumulated in FP32 and rounded once to the residual dtype, in one pass
over the BF16 streams without FP32 temporaries.
"""

from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton

_ENABLED = os.environ.get("VLLM_MUSA_MHC_COLLAPSE_TRITON", "1") != "0"
# Measured on MTT S5000 for hidden 5120 at 320-4096 tokens.
_BLOCK_D = 1024
_NUM_WARPS = 4


@triton.jit
def _mhc_collapse_kernel(
    x_ptr,
    mix_ptr,
    out_ptr,
    hidden,
    stride_xt,
    stride_xh,
    stride_mt,
    stride_ot,
    HC: tl.constexpr,
    HAS_MIX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < hidden
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for h in tl.static_range(HC):
        x = tl.load(x_ptr + t * stride_xt + h * stride_xh + d, mask=mask, other=0.0)
        if HAS_MIX:
            acc += tl.load(mix_ptr + t * stride_mt + h) * x.to(tl.float32)
        else:
            acc += x.to(tl.float32)
    if not HAS_MIX:
        acc = acc / HC
    tl.store(out_ptr + t * stride_ot + d, acc.to(out_ptr.dtype.element_ty), mask=mask)


def mhc_collapse(
    residual: torch.Tensor, mix: torch.Tensor | None = None
) -> torch.Tensor:
    """``(mix[..., None] * residual.float()).sum(1)``, or ``residual.float().mean(1)``
    without ``mix``, in the residual dtype; ``residual`` is (tokens, hc, hidden)."""
    tokens, hc, hidden = residual.shape
    supported = (
        _ENABLED
        and residual.device.type == "musa"
        and residual.dtype == torch.bfloat16
        and residual.stride(-1) == 1
        and (
            mix is None
            or (
                mix.shape == (tokens, hc)
                and mix.dtype == torch.float32
                and mix.stride(-1) == 1
            )
        )
    )
    if not supported:
        x = residual.float()
        out = x.mean(dim=1) if mix is None else (mix.unsqueeze(-1) * x).sum(dim=1)
        return out.to(residual.dtype)
    out = torch.empty((tokens, hidden), dtype=residual.dtype, device=residual.device)
    if tokens:
        _mhc_collapse_kernel[(tokens, triton.cdiv(hidden, _BLOCK_D))](
            residual,
            residual if mix is None else mix,
            out,
            hidden,
            residual.stride(0),
            residual.stride(1),
            0 if mix is None else mix.stride(0),
            out.stride(0),
            HC=hc,
            HAS_MIX=mix is not None,
            BLOCK_D=_BLOCK_D,
            num_warps=_NUM_WARPS,
        )
    return out
