from __future__ import annotations

from typing import Optional

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
from vllm_musa.jit_kernel.utils import cache_once


@cache_once
def _topk_module():
    return load_musa_jit(
        "vllm_musa_topk_gating",
        ("topk/topk_gating.mu",),
    )


def _topk_softmax_impl(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    _topk_module().sgl_musa_topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        float(moe_softcapping),
        bias_arg,
        bool(has_correction_bias),
    )


def _topk_sigmoid_impl(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    _topk_module().sgl_musa_topk_sigmoid(
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        bias_arg,
        bool(has_correction_bias),
    )


def _topk_softmax_custom(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    has_correction_bias: bool = False,
) -> None:
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    _topk_module().sgl_musa_topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        float(moe_softcapping),
        bias_arg,
        bool(has_correction_bias),
    )


def _topk_softmax_custom_fake(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    has_correction_bias: bool = False,
) -> None:
    return


def _topk_sigmoid_custom(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    renormalize: bool = False,
    has_correction_bias: bool = False,
) -> None:
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    _topk_module().sgl_musa_topk_sigmoid(
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        bias_arg,
        bool(has_correction_bias),
    )


def _topk_sigmoid_custom_fake(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    renormalize: bool = False,
    has_correction_bias: bool = False,
) -> None:
    return


direct_register_custom_op(
    op_name="musa_topk_softmax",
    op_func=_topk_softmax_custom,
    mutates_args=["topk_weights", "topk_ids"],
    fake_impl=_topk_softmax_custom_fake,
)

direct_register_custom_op(
    op_name="musa_topk_sigmoid",
    op_func=_topk_sigmoid_custom,
    mutates_args=["topk_weights", "topk_ids"],
    fake_impl=_topk_sigmoid_custom_fake,
)


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    """sgl_kernel-compatible top-k softmax entry point."""
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    torch.ops.vllm.musa_topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        bias_arg,
        renormalize,
        moe_softcapping,
        has_correction_bias,
    )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    torch.ops.vllm.musa_topk_sigmoid(
        topk_weights,
        topk_ids,
        gating_output,
        bias_arg,
        renormalize,
        has_correction_bias,
    )
