from __future__ import annotations

import os
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

    if not isinstance(quant_method, MusaW4AFp8MoEMethod):
        return False
    parallel = config.parallel_config
    return (
        os.environ.get("VLLM_MUSA_W4AFP8_SHARED_OVERLAP", "1") == "1"
        and config.model_config.quantization == "w4afp8"
        and parallel.tensor_parallel_size == 8
        and parallel.enable_expert_parallel
        and parallel.data_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and not parallel.enable_eplb
        and not parallel.enable_dbo
    )
