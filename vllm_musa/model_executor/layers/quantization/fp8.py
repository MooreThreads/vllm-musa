import torch
from vllm.model_executor.layers.fused_moe import FusedMoE, fused_experts
from vllm.utils.torch_utils import is_torch_equal_or_newer


def apply(
    self,
    layer: FusedMoE,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if layer.ep_size != None and layer.ep_size <= 1:
        is_inplace = not is_torch_equal_or_newer("2.9")
        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=is_inplace,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )
    else:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=shared_experts_input,
        )


import vllm.model_executor.layers.quantization.fp8

vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.apply = apply
