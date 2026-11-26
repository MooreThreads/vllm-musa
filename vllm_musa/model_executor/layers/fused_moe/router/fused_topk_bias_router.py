# SPDX-License-Identifier: Apache-2.0
"""MUSA: DeepSeek-V4 sqrt-softplus routing in one kernel.

The kernel computes the same weights and ids as ``topk_hash_softplus_sqrt`` bit for
bit and also writes the constant column of a shared expert folded into the routed
expert stack, which upstream appends with separate fill and concat kernels. For
small batches it also writes the expert route map and the FP8-quantized input of
the FP8 MoE (``kernels/fp8_moe_mma.py``), which ride on the returned ids tensor.
The kernel reads FP32 logits or the BF16 gate output. Hash routing and other scoring
functions keep the upstream path, on FP32 logits.
"""

from __future__ import annotations

import os

import torch
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
    _get_padding_mask,
)

_ENABLED = (
    getattr(torch.version, "musa", None) is not None
    and os.environ.get("VLLM_MUSA_SQRTSOFTPLUS_TOPK", "1") != "0"
)
_ROUTE_MAP = os.environ.get("VLLM_MUSA_FUSED_ROUTE_MAP", "1") != "0"
_ORIGINAL_COMPUTE_ROUTING = FusedTopKBiasRouter._compute_routing


def _musa_routing_supported(
    router: FusedTopKBiasRouter,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
    input_ids: torch.Tensor | None,
) -> bool:
    from vllm_musa.jit_kernel.csrc.sqrtsoftplus_topk import MAX_EXPERTS, MAX_TOP_K

    bias = router.e_score_correction_bias
    image = router.bias_vl is not None and router.image_sentinel_lo > 0
    return (
        _ENABLED
        and router.scoring_func == "sqrtsoftplus"
        and router._hash_indices_table is None
        and (
            not image
            or (
                input_ids is not None
                and input_ids.dim() == 1
                and input_ids.dtype in (torch.int32, torch.int64)
                and input_ids.shape[0] >= router_logits.shape[0]
                and router.bias_vl.dtype == torch.float32
                and router.bias_vl.is_contiguous()
            )
        )
        and router.num_fused_shared_experts <= 1
        and indices_type in (None, torch.int32)
        and bias is not None
        and bias.dtype == torch.float32
        and bias.dim() == 1
        and bias.is_contiguous()
        and router_logits.dim() == 2
        and router_logits.dtype in (torch.float32, torch.bfloat16)
        and router_logits.stride(-1) == 1
        and bias.shape[0] == router_logits.shape[1] <= MAX_EXPERTS
        and 0 < router.top_k <= min(MAX_TOP_K, router_logits.shape[1])
    )


def _compute_routing(
    self: FusedTopKBiasRouter,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
    *,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _musa_routing_supported(self, router_logits, indices_type, input_ids):
        return _ORIGINAL_COMPUTE_ROUTING(
            self,
            hidden_states,
            router_logits.float(),
            indices_type,
            input_ids=input_ids,
        )
    from vllm_musa.jit_kernel.csrc.sqrtsoftplus_topk import sqrtsoftplus_topk

    # Fused shared experts take the ids right after the routed ones.
    shared_id = self.global_num_experts if self.num_fused_shared_experts else -1
    topk_weights, topk_ids, route_map, x_quant = sqrtsoftplus_topk(
        router_logits,
        self.e_score_correction_bias.data,
        self.top_k,
        self.renormalize,
        self.routed_scaling_factor,
        is_padding=_get_padding_mask(router_logits.shape[0]),
        shared_id=shared_id,
        shared_weight=self.shared_expert_weight,
        with_route_map=_ROUTE_MAP,
        bias_vl=self.bias_vl.data if self.bias_vl is not None else None,
        input_ids=input_ids,
        image_sentinel_lo=self.image_sentinel_lo,
        x=hidden_states,
    )
    if route_map is not None:
        topk_ids.musa_route_map = route_map
    if x_quant is not None:
        topk_ids.musa_route_x = (hidden_states, *x_quant)
    return topk_weights, topk_ids


_HOOKS_INSTALLED = False


def install_hooks() -> None:
    """Install MUSA routing override for ``FusedTopKBiasRouter`` (idempotent)."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    FusedTopKBiasRouter._compute_routing = _compute_routing
    _HOOKS_INSTALLED = True
