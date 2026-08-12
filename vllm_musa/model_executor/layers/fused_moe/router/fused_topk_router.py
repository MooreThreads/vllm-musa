# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA: route the plain (non-grouped, no-bias) top-k through the fused one-warp
topk_softmax kernel instead of the built-in ``topkGating``, matching the grouped
router. Falls back to the upstream path when the JIT kernel is unavailable or
ineligible.
"""

import torch
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
    fused_topk,
)

from vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router import (
    _musa_jit_fused_topk,
)
from vllm_musa.jit_kernel.extend_topk_shared import extend_topk_with_shared


def _compute_routing(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
    *,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scoring_func = getattr(self, "scoring_func", "softmax")
    shared_gate = getattr(self, "_musa_shared_gate", None)
    num_combined_shared_experts = int(
        getattr(self, "_musa_num_fused_shared_experts", 0)
    )
    shared_logits = None
    if shared_gate is not None:
        shared_logits, _ = shared_gate(hidden_states)

    jit_result = _musa_jit_fused_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=self.top_k,
        renormalize=self.renormalize,
        indices_type=indices_type,
        correction_bias=None,
        scoring_func=scoring_func,
        shared_expert_gate_output=shared_logits,
        num_fused_shared_experts=(
            1 if shared_logits is not None else num_combined_shared_experts
        ),
    )
    if jit_result is not None:
        return jit_result

    if num_combined_shared_experts:
        assert num_combined_shared_experts == 1
        routed_experts = self.global_num_experts
        routed_logits = router_logits[:, :routed_experts].contiguous()
        combined_shared_logits = router_logits[:, routed_experts:].contiguous()
    else:
        routed_experts = router_logits.shape[1]
        routed_logits = router_logits
        combined_shared_logits = None

    topk_weights, topk_ids, _ = fused_topk(
        hidden_states=hidden_states,
        gating_output=routed_logits,
        topk=self.top_k,
        renormalize=self.renormalize,
        indices_type=indices_type,
        scoring_func=scoring_func,
    )
    if shared_logits is not None:
        shared_expert_id = getattr(self, "_musa_shared_expert_id", None)
        assert shared_expert_id is not None
        topk_weights, topk_ids = extend_topk_with_shared(
            topk_weights,
            topk_ids,
            shared_logits,
            shared_expert_id,
        )
    elif combined_shared_logits is not None:
        topk_weights, topk_ids = extend_topk_with_shared(
            topk_weights,
            topk_ids,
            combined_shared_logits,
            routed_experts,
        )
    return topk_weights, topk_ids


FusedTopKRouter._compute_routing = _compute_routing
