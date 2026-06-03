import math

import torch
import vllm.model_executor.layers.quantization.fp8 as vllm_fp8
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, fused_experts
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_torch_equal_or_newer

logger = init_logger(__name__)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _zero_fp8_weight(weight: torch.Tensor) -> None:
    # MUSA muDNN fill does not support FP8_E4M3 directly.
    weight.view(torch.uint8).zero_()


_ORIGINAL_FP8_MOE_MAYBE_ROUNDUP_SIZES = vllm_fp8.Fp8MoEMethod.maybe_roundup_sizes
_ORIGINAL_FP8_MOE_CREATE_WEIGHTS = vllm_fp8.Fp8MoEMethod.create_weights


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


def apply(
    self,
    layer: FusedMoE,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts: object | None = None,
    shared_experts_input: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if layer.ep_size != None and layer.ep_size <= 1:
        # MUSA-3173: the legacy fused_experts() path only computes routed
        # experts. For the no-overlap path used by DeepSeek-V2/V3 on MUSA, the
        # MoE runner computes shared experts separately and combines them with
        # this routed output. Only compute shared experts here when the runner
        # explicitly delegated them to the quant method via MK overlap.
        run_shared_in_quant_method = (
            shared_experts is not None and self.mk_can_overlap_shared_experts
        )
        if run_shared_in_quant_method:
            se_input = shared_experts_input if shared_experts_input is not None else x
        is_inplace = (not is_torch_equal_or_newer("2.9")) and shared_experts is None
        routed = fused_experts(
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
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


vllm_fp8.Fp8MoEMethod.maybe_roundup_sizes = maybe_roundup_sizes
vllm_fp8.Fp8MoEMethod.create_weights = create_weights
vllm_fp8.Fp8MoEMethod.apply = apply
