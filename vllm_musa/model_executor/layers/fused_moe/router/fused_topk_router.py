# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA: route the plain (non-grouped, no-bias) top-k through the fused one-warp
topk_softmax kernel instead of the built-in ``topkGating``, matching the grouped
router. Falls back to the upstream path when the JIT kernel is disabled or ineligible.
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
    jit_result = _musa_jit_fused_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=self.top_k,
        renormalize=self.renormalize,
        indices_type=indices_type,
        correction_bias=None,
        scoring_func=scoring_func,
    )
    if jit_result is not None:
        return jit_result

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
