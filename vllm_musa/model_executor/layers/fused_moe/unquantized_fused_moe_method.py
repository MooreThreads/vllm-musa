import inspect

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    TritonExperts,
    UnquantizedFusedMoEMethod,
    fused_experts,
)

try:
    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
        BatchedTritonExperts,
    )
except ModuleNotFoundError:
    from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
        BatchedTritonExperts,
    )

from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)

logger = init_logger(__name__)
from vllm.utils.torch_utils import is_torch_equal_or_newer

_FUSED_EXPERTS_ACCEPTS_SHARED_EXPERTS = (
    "shared_experts" in inspect.signature(fused_experts).parameters
)


@UnquantizedFusedMoEMethod.register_oot
class MusaUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    is_monolithic = False

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        from vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )

        pf = maybe_make_prepare_finalize(
            self.moe, self.moe_quant_config, routing_tables
        )
        assert pf is None or isinstance(pf, FusedMoEPrepareAndFinalizeModular)
        return pf

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> FusedMoEExpertsModular:
        assert self.moe_quant_config is not None
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            logger.debug("BatchedTritonExperts %s", self.moe)
            return BatchedTritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                max_num_tokens=self.moe.max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
            )
        else:
            logger.debug("TritonExperts %s", self.moe)
            return TritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            )

    def forward_oot(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: object | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # vLLM v0.22 passes shared experts through the MoE runner,
        # but the legacy fused_experts() helper only computes routed experts and
        # does not accept shared_experts kwargs. Return routed output here; the
        # runner computes and combines shared experts for non-overlapped MUSA
        # paths. Keep inplace disabled when shared experts exist so this legacy
        # op never mutates an input that another path may still consume.
        is_inplace = (not is_torch_equal_or_newer("2.9")) and shared_experts is None
        return fused_experts(
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
