"""Native small-M GEMVs for FP8 weights with 32x32 block scales."""

from __future__ import annotations

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
from vllm_musa.jit_kernel.utils import cache_once

GROUP = 32
MAX_ROWS = 8


@cache_once
def _module():
    return load_musa_jit(
        "vllm_musa_fp8_block_gemv",
        ("gemv/fp8_block_gemv.mu",),
        extra_musa_cflags=("-fmusa-flush-denormals-to-zero",),
    )


def _act_quant(
    x: torch.Tensor,
    xq: torch.Tensor,
    xs: torch.Tensor,
    swiglu: bool,
    swiglu_limit: float,
) -> None:
    _module().fp8_act_quant(x, xq, xs, swiglu, swiglu_limit)


def _dense_gemv(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    cfg: int,
) -> None:
    _module().fp8_dense_gemv(xq, xs, weight, weight_scale, out, cfg)


def _moe_w1_gemv(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
) -> None:
    _module().fp8_moe_w1_gemv(xq, xs, w1, w1_scale, expert_ids, out, top_k)


def _moe_w2_gemv(
    aq: torch.Tensor,
    a_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    routed_weights: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _module().fp8_moe_w2_gemv(
        aq, a_scale, w2, w2_scale, expert_ids, routed_weights, out
    )


def _no_op(*args, **kwargs) -> None:
    return None


direct_register_custom_op(
    op_name="musa_fp8_gemv_act_quant",
    op_func=_act_quant,
    mutates_args=["xq", "xs"],
    fake_impl=_no_op,
)
direct_register_custom_op(
    op_name="musa_fp8_dense_gemv",
    op_func=_dense_gemv,
    mutates_args=["out"],
    fake_impl=_no_op,
)
direct_register_custom_op(
    op_name="musa_fp8_moe_w1_gemv",
    op_func=_moe_w1_gemv,
    mutates_args=["out"],
    fake_impl=_no_op,
)
direct_register_custom_op(
    op_name="musa_fp8_moe_w2_gemv",
    op_func=_moe_w2_gemv,
    mutates_args=["out"],
    fake_impl=_no_op,
)


def act_quant(
    x: torch.Tensor, swiglu_limit: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``per_token_group_quant_fp8(x, 32)``; with ``swiglu_limit``, of the clamped
    SwiGLU of the ``[gate | up]`` rows of ``x``."""
    k = x.shape[-1] // 2 if swiglu_limit is not None else x.shape[-1]
    xq = torch.empty((x.shape[0], k), dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty((x.shape[0], k // GROUP), dtype=torch.float32, device=x.device)
    torch.ops.vllm.musa_fp8_gemv_act_quant(
        x,
        xq,
        xs,
        swiglu_limit is not None,
        float(swiglu_limit) if swiglu_limit is not None else 0.0,
    )
    return xq, xs


def dense_gemv(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    cfg: int,
) -> torch.Tensor:
    """``out = dequant(xq) @ dequant(weight).T`` for 1-8 rows."""
    torch.ops.vllm.musa_fp8_dense_gemv(xq, xs, weight, weight_scale, out, cfg)
    return out


def moe_w1_gemv(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """``out[j] = dequant(xq[j // top_k]) @ dequant(w1[expert_ids[j]]).T``."""
    torch.ops.vllm.musa_fp8_moe_w1_gemv(xq, xs, w1, w1_scale, expert_ids, out, top_k)
    return out


def moe_w2_gemv(
    aq: torch.Tensor,
    a_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    routed_weights: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """``out[t] = sum_s routed_weights[t * K + s] * dequant(aq[t * K + s]) @
    dequant(w2[expert_ids[t * K + s]]).T`` with
    ``K = expert_ids.numel() // out.shape[0]``, rounded once per token."""
    torch.ops.vllm.musa_fp8_moe_w2_gemv(
        aq, a_scale, w2, w2_scale, expert_ids, routed_weights, out
    )
    return out
