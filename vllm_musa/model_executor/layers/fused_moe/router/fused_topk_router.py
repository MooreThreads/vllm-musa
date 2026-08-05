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


def _compute_routing(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
    *,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scoring_func = getattr(self, "scoring_func", "softmax")
    num_fused_shared_experts = int(
        getattr(self, "_musa_num_fused_shared_experts", 0)
    )
    jit_result = _musa_jit_fused_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=self.top_k,
        renormalize=self.renormalize,
        indices_type=indices_type,
        correction_bias=None,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    if jit_result is not None:
        return jit_result

    if num_fused_shared_experts:
        # Combined logits contain routed columns followed by one shared logit.
        # Keep the shared expert out of routed top-k selection.
        assert num_fused_shared_experts == 1
        routed_experts = self.global_num_experts
        routed_logits = router_logits[:, :routed_experts].contiguous()
        shared_logits = router_logits[:, routed_experts:].contiguous()
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states=hidden_states,
            gating_output=routed_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=scoring_func,
        )
        from vllm_musa.jit_kernel.extend_topk_shared import (
            extend_topk_with_shared,
        )

        return extend_topk_with_shared(
            topk_weights,
            topk_ids,
            shared_logits,
            routed_experts,
        )

    topk_weights, topk_ids, _ = fused_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=self.top_k,
        renormalize=self.renormalize,
        indices_type=indices_type,
        scoring_func=scoring_func,
    )
    return topk_weights, topk_ids


FusedTopKRouter._compute_routing = _compute_routing
