import torch
from vllm.model_executor.layers.fused_moe import (
    UnquantizedFusedMoEMethod,
    fused_experts,
)
from vllm.utils.torch_utils import is_torch_equal_or_newer


@UnquantizedFusedMoEMethod.register_oot
class MusaUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def forward_oot(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        is_inplace = not is_torch_equal_or_newer("2.9")
        result = fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=is_inplace,
            activation=layer.activation,
            quant_config=self.moe_quant_config,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )

        return result
