import math

import torch
import vllm.model_executor.layers.quantization.fp8 as vllm_fp8
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, fused_experts
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _zero_fp8_weight(weight: torch.Tensor) -> None:
    # MUSA muDNN fill does not support FP8_E4M3 directly.
    weight.view(torch.uint8).zero_()


_ORIGINAL_FP8_MOE_MAYBE_ROUNDUP_SIZES = vllm_fp8.Fp8MoEMethod.maybe_roundup_sizes
_ORIGINAL_FP8_MOE_CREATE_WEIGHTS = vllm_fp8.Fp8MoEMethod.create_weights
_ORIGINAL_FP8_MOE_PROCESS_WEIGHTS = vllm_fp8.Fp8MoEMethod.process_weights_after_loading


def maybe_roundup_sizes(
    self,
    hidden_size: int,
    intermediate_size_per_partition: int,
    act_dtype: torch.dtype,
    moe_parallel_config,
) -> tuple[int, int]:
    hidden_size, intermediate_size_per_partition = (
        _ORIGINAL_FP8_MOE_MAYBE_ROUNDUP_SIZES(
            self,
            hidden_size,
            intermediate_size_per_partition,
            act_dtype,
            moe_parallel_config,
        )
    )

    if not (
        current_platform.is_musa()
        and getattr(self, "block_quant", False)
        and getattr(moe_parallel_config, "tp_size", 1) > 1
    ):
        return hidden_size, intermediate_size_per_partition

    weight_block_size = getattr(self, "weight_block_size", None)
    if weight_block_size is None:
        return hidden_size, intermediate_size_per_partition

    block_n, block_k = int(weight_block_size[0]), int(weight_block_size[1])
    block_multiple = math.lcm(block_n, block_k)
    padded_intermediate = _round_up(
        intermediate_size_per_partition,
        block_multiple,
    )
    if padded_intermediate != intermediate_size_per_partition:
        logger.info_once(
            "Padding MUSA FP8 MoE intermediate partition from %d to %d "
            "for block_shape=[%d, %d].",
            intermediate_size_per_partition,
            padded_intermediate,
            block_n,
            block_k,
        )
    return hidden_size, padded_intermediate


def create_weights(
    self,
    layer: torch.nn.Module,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    _ORIGINAL_FP8_MOE_CREATE_WEIGHTS(
        self,
        layer=layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
        params_dtype=params_dtype,
        **extra_weight_attrs,
    )

    if not (current_platform.is_musa() and getattr(self, "block_quant", False)):
        return

    unpadded_intermediate = getattr(
        layer.moe_config,
        "intermediate_size_per_partition_unpadded",
        intermediate_size_per_partition,
    )
    if intermediate_size_per_partition == unpadded_intermediate:
        return

    _zero_fp8_weight(layer.w13_weight.data)
    _zero_fp8_weight(layer.w2_weight.data)
    logger.debug(
        "Zero initialized padded MUSA FP8 MoE weights for %s "
        "(intermediate=%d, unpadded=%d).",
        getattr(layer, "prefix", "<unknown>"),
        intermediate_size_per_partition,
        unpadded_intermediate,
    )


def _release_cached_blocks() -> None:
    empty_cache = getattr(current_platform, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
    elif hasattr(torch, "musa"):
        torch.musa.empty_cache()


def _fold_shared_expert_weights(self, layer: FusedMoE) -> None:
    """Append the model's shared expert to the routed FP8 expert stack.

    The MoE block stashes its shared MLP and gate on the routed-experts layer
    when the two carry the same weight format. The shared gate_up/down
    projections and their block scales are concatenated as one extra expert, so
    a single grouped GEMM covers routed + shared work instead of running the
    shared expert as separate dense GEMMs on every MoE layer. Runs once after
    load, before any graph capture.
    """
    mlp = getattr(layer, "_musa_shared_mlp", None)
    if mlp is None or getattr(layer, "_musa_shared_folded", False):
        return

    scale_name = self.weight_scale_name
    parts = (
        ("w13_weight", f"w13_{scale_name}", mlp.gate_up_proj),
        ("w2_weight", f"w2_{scale_name}", mlp.down_proj),
    )
    folded: dict[str, torch.Tensor] = {}
    for weight_attr, scale_attr, proj in parts:
        routed_w = getattr(layer, weight_attr).data
        routed_s = getattr(layer, scale_attr).data
        shared_w = proj.weight.data
        shared_s = getattr(proj, scale_name).data
        if shared_w.shape != routed_w.shape[1:] or shared_w.dtype != routed_w.dtype:
            raise RuntimeError(
                f"MUSA shared-expert fold: {weight_attr} expects "
                f"{tuple(routed_w.shape[1:])}/{routed_w.dtype}, shared expert has "
                f"{tuple(shared_w.shape)}/{shared_w.dtype}"
            )
        if shared_s.shape != routed_s.shape[1:] or shared_s.dtype != routed_s.dtype:
            raise RuntimeError(
                f"MUSA shared-expert fold: {scale_attr} expects "
                f"{tuple(routed_s.shape[1:])}/{routed_s.dtype}, shared expert has "
                f"{tuple(shared_s.shape)}/{shared_s.dtype}"
            )
        folded[weight_attr] = torch.cat(
            [routed_w, shared_w.unsqueeze(0)], dim=0
        ).contiguous()
        folded[scale_attr] = torch.cat(
            [routed_s, shared_s.unsqueeze(0)], dim=0
        ).contiguous()

    num_routed = int(layer.w13_weight.shape[0])
    # Rebuild the quant config / kernel so they bind the folded tensors.
    self._setup_kernel(
        layer,
        folded["w13_weight"],
        folded["w2_weight"],
        folded[f"w13_{scale_name}"],
        folded[f"w2_{scale_name}"],
        layer.w13_input_scale,
        layer.w2_input_scale,
    )
    del folded
    # Each expert stack is concatenated while the unfolded one is still alive, so the
    # freed blocks leave expert-sized holes in the caching allocator. Left there, they
    # are reserved-but-unusable and the KV cache later fails to find contiguous space.
    _release_cached_blocks()
    layer._musa_shared_expert_id = num_routed
    layer._musa_shared_folded = True
    logger.info_once(
        "MUSA shared-expert fold active: shared expert folded into the routed "
        "FP8 grouped GEMM as slot %d (%d routed + 1 shared).",
        num_routed,
        num_routed,
    )


def process_weights_after_loading(self, layer: FusedMoE) -> None:
    _ORIGINAL_FP8_MOE_PROCESS_WEIGHTS(self, layer)
    _fold_shared_expert_weights(self, layer)


def _extend_routing_with_shared_expert(
    layer: FusedMoE,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_musa.jit_kernel.extend_topk_shared import extend_topk_with_shared

    shared_logits, _ = layer._musa_shared_gate(x)
    return extend_topk_with_shared(
        topk_weights,
        topk_ids,
        shared_logits.view(-1),
        layer._musa_shared_expert_id,
    )


def apply(
    self,
    layer: FusedMoE,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts: object | None = None,
    shared_experts_input: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    ep_size = getattr(layer, "ep_size", None)
    if ep_size is None:
        ep_size = layer.moe_config.ep_size

    # A folded shared expert rides the routed grouped GEMM as one extra
    # always-selected slot, so extend the routing by a single column here.
    folded_shared = getattr(layer, "_musa_shared_folded", False)
    if folded_shared:
        assert (
            shared_experts is None and shared_experts_input is None
        ), "folded shared expert expects shared_experts=None"
        topk_weights, topk_ids = _extend_routing_with_shared_expert(
            layer, x, topk_weights, topk_ids
        )
    global_num_experts = (
        layer._musa_shared_expert_id + 1 if folded_shared else layer.global_num_experts
    )

    if ep_size != None and ep_size <= 1:
        # the legacy fused_experts() path only computes routed
        # experts. For the no-overlap path used by DeepSeek-V2/V3 on MUSA, the
        # MoE runner computes shared experts separately and combines them with
        # this routed output. Only compute shared experts here when the runner
        # explicitly delegated them to the quant method via MK overlap.
        run_shared_in_quant_method = (
            shared_experts is not None and self.mk_can_overlap_shared_experts
        )
        if run_shared_in_quant_method:
            se_input = shared_experts_input if shared_experts_input is not None else x
        routed = fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )
        if not run_shared_in_quant_method:
            return routed
        return routed + shared_experts._layer(se_input)
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
            global_num_experts=global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


vllm_fp8.Fp8MoEMethod.maybe_roundup_sizes = maybe_roundup_sizes
vllm_fp8.Fp8MoEMethod.create_weights = create_weights
vllm_fp8.Fp8MoEMethod.process_weights_after_loading = process_weights_after_loading
vllm_fp8.Fp8MoEMethod.apply = apply
