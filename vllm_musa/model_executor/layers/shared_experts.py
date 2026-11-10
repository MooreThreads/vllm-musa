from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )


def can_overlap_shared_experts(
    config: VllmConfig, quant_method: FusedMoEMethodBase
) -> bool:
    from vllm_musa.model_executor.layers.quantization.w4afp8 import MusaW4AFp8MoEMethod
    from vllm_musa.optimization_contract import (
        OptimizationFeature,
        prefers_feature,
    )

    if not isinstance(quant_method, MusaW4AFp8MoEMethod):
        return False
    return prefers_feature(
        config,
        OptimizationFeature.DEEPSEEK_V41_W4AFP8_SHARED_EXPERT_OVERLAP,
    )
