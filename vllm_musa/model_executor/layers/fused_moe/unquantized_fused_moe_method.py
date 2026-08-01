import inspect

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    UnquantizedFusedMoEMethod,
    fused_experts,
)

try:
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonExperts,
    )
except ImportError:
    from vllm.model_executor.layers.fused_moe import TritonExperts

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

from vllm_musa.jit_kernel.extend_topk_shared import extend_topk_with_shared

logger = init_logger(__name__)

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

    def process_weights_after_loading(self, layer: "FusedMoE") -> None:  # type: ignore[name-defined] # noqa: F821
        super().process_weights_after_loading(layer)
        _fold_shared_expert_weights(layer)

    def forward_oot(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: object | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # The legacy fused_experts() helper computes routed experts only. When a
        # Qwen3.5 shared expert has been folded into the expert weights as one
        # extra always-selected slot, extend the routing by a single column so
        # the shared expert rides the same grouped GEMM instead of running as a
        # separate serial MLP. Otherwise the shared expert is combined by the
        # runner outside this call.
        if getattr(layer, "_musa_shared_folded", False):
            # The fold is only installed when the block passes shared_experts=None
            # to FusedMoE, so the runner never supplies a separate
            # shared_experts_input here; x is the shared expert's input, matching
            # the routed experts it now rides with.
            assert shared_experts_input is None, (
                "folded shared expert expects shared_experts=None"
            )
            shared_logits, _ = layer._musa_shared_gate(x)
            topk_weights, topk_ids = extend_topk_with_shared(
                topk_weights,
                topk_ids,
                shared_logits,
                layer._musa_shared_expert_id,
            )
            global_num_experts = layer._musa_shared_expert_id + 1
        else:
            global_num_experts = layer.global_num_experts
        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            quant_config=self.moe_quant_config,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=layer.expert_map,
        )


def _fold_shared_expert_weights(layer: torch.nn.Module) -> None:
    """Append a Qwen3.5 shared expert to the routed expert weights.

    The MoE block stashes its shared MLP and gate on the routed-experts layer
    when the shared and routed intermediate sizes match. Here the shared
    gate_up/down projections are concatenated as one extra expert so a single
    grouped GEMM covers routed + shared work. Runs once after load, before any
    graph capture, and replaces the routed weights in place so no full copy of
    the expert stack is kept alive.
    """
    # _musa_shared_mlp is stashed by the block on the same FusedMoE instance whose
    # quant method runs this hook and forward_oot, so the fold that consumes it
    # here and the routing extension there see the same layer. Absent it (any
    # non-folded MoE), this is inert.
    mlp = getattr(layer, "_musa_shared_mlp", None)
    if mlp is None or getattr(layer, "_musa_shared_folded", False):
        return
    num_routed = layer.w13_weight.shape[0]
    w13 = torch.cat(
        [layer.w13_weight.data, mlp.gate_up_proj.weight.data.unsqueeze(0)], dim=0
    ).contiguous()
    w2 = torch.cat(
        [layer.w2_weight.data, mlp.down_proj.weight.data.unsqueeze(0)], dim=0
    ).contiguous()
    layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    layer._musa_shared_expert_id = int(num_routed)
    layer._musa_shared_folded = True
    logger.info_once(
        "MUSA shared-expert fold active: shared expert folded into the routed "
        "grouped GEMM as slot %d (%d routed + 1 shared).",
        num_routed,
        num_routed,
    )
