from __future__ import annotations

import torch
from torch.nn import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

logger = init_logger(__name__)


class MusaW4AFp8Config(DeepseekV4FP8Config):
    """Packed signed INT4 experts alongside native DeepSeek FP8 linears."""

    @classmethod
    def get_name(cls) -> str:
        return "w4afp8"

    @classmethod
    def get_min_capability(cls) -> int:
        return 31

    @classmethod
    def from_config(cls, config: dict) -> MusaW4AFp8Config:
        if int(config.get("moe_group_size", 128)) != 128:
            raise ValueError("MUSA W4AFP8 requires moe_group_size=128")
        if config.get("activation_scheme", "dynamic") != "dynamic":
            raise ValueError("MUSA W4AFP8 requires dynamic FP8 activations")
        result = super().from_config(config)
        result._dense_scale_e8m0 = config.get("scale_fmt") == "ue8m0"
        return result

    @property
    def expert_dtype(self) -> str:
        return "int4"

    @property
    def is_scale_e8m0(self) -> bool:
        return getattr(self, "_dense_scale_e8m0", True)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts) and not is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return MusaW4AFp8MoEMethod(self, layer.moe_config)
        return super().get_quant_method(layer, prefix)


class MusaW4AFp8MoEMethod(FusedMoEMethodBase):
    is_monolithic = False

    def __init__(self, quant_config: MusaW4AFp8Config, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self.quant_config = quant_config

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        if hidden_size % 128 or intermediate_size_per_partition % 128:
            raise ValueError(
                "MUSA W4AFP8 requires group128-aligned expert dimensions; "
                "use --enable-expert-parallel for DeepSeek V4.1 TP8 "
                "(2304/8=288 is not group128-aligned)."
            )
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        intermediate = intermediate_size_per_partition
        shapes = {
            "w13_weight": (num_experts, 2 * intermediate, hidden_size // 2),
            "w2_weight": (num_experts, hidden_size, intermediate // 2),
            "w13_weight_scale_inv": (num_experts, 2 * intermediate, hidden_size // 128),
            "w2_weight_scale_inv": (num_experts, hidden_size, intermediate // 128),
            "w13_input_scale": (num_experts,),
            "w2_input_scale": (num_experts,),
        }
        for name, shape in shapes.items():
            if name.endswith("weight"):
                dtype = torch.int8
                attrs = extra_weight_attrs
            else:
                dtype = torch.bfloat16 if "input_scale" in name else torch.float32
                attrs = {
                    **extra_weight_attrs,
                    "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
                }
            value = (
                torch.ones(shape, dtype=dtype)
                if "input_scale" in name
                else torch.empty(shape, dtype=dtype)
            )
            parameter = Parameter(value, requires_grad=False)
            layer.register_parameter(name, parameter)
            set_weight_attrs(parameter, attrs)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            scales = getattr(layer, name).to(torch.bfloat16).contiguous()
            if not torch.isfinite(scales).all() or (scales <= 0).any():
                raise ValueError(f"Invalid W4AFP8 weight scale in {name}")
            setattr(
                layer,
                name,
                Parameter(scales, requires_grad=False),
            )
        logger.info_once(
            "MUSA W4AFP8: MATE S4FP8 masked/ragged MUBIN; dynamic per-token "
            "FP8 activations, packed signed INT4 weights, BF16 group128 scales. "
            "Checkpoint input_scale placeholders are not used for dynamic quantization."
        )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: torch.nn.Module | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_musa.kernels.w4afp8_moe import w4afp8_moe

        if layer.apply_router_weight_on_input:
            raise NotImplementedError("W4AFP8 router weights must apply to the output")
        return w4afp8_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_weights,
            topk_ids,
            layer.expert_map,
            swiglu_limit=self.moe.swiglu_limit,
        )
