# SPDX-License-Identifier: Apache-2.0
"""Single fused GDN gating kernel for the strided-QKV MATE prefill path.

Computes the two log-space GDN gating tensors in one Triton launch:

    g    = -exp(A_log) * softplus(a + dt_bias)   (log-space, no final exp)
    beta = sigmoid(b)

replacing the ~9 elementwise kernels (exp / mul / softplus / sigmoid / cast)
the strided path would otherwise issue per GDN layer. Ported from SGLang's
``fused_gdn_gating``; the wrapper returns 2D ``[L, HV]`` fp32 tensors matching
what the MATE ``chunk_gated_delta_rule`` prefill call expects.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused log-space GDN gating.

    Args:
        A_log:   [HV] log decay parameter (per v-head).
        a:       [L, HV] gating input, contiguous.
        b:       [L, HV] gating input, contiguous.
        dt_bias: [HV] dt bias (per v-head).

    Returns:
        g:    [L, HV] float32, log-space ``-exp(A_log) * softplus(a + dt_bias)``.
        beta: [L, HV] float32, ``sigmoid(b)``.
    """
    assert a.shape == b.shape and a.dim() == 2
    L, num_heads = a.shape
    g = torch.empty(L, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(L, num_heads, dtype=torch.float32, device=b.device)
    if L == 0:
        return g, beta_output
    a = a.contiguous()
    b = b.contiguous()
    grid = (L, 1, triton.cdiv(num_heads, 8))
    _fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        1,  # seq_len: each token treated independently
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
