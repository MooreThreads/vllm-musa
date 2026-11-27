# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.platforms import current_platform

from vllm_musa.model_executor.layers.quantization.utils import fp8_utils
from vllm_musa.optimization_contract import (
    bind_quantized_piecewise_native_route,
    should_use_native_quantized_piecewise_op,
)


@QuantFP8.register_oot
class MusaQuantFP8(QuantFP8):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        bind_quantized_piecewise_native_route(self)

    def forward_oot(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if should_use_native_quantized_piecewise_op(self):
            return self.forward_native(x, scale, scale_ub, use_triton)

        if self.is_group_quant and not self.static and current_platform.is_musa():
            assert scale is None, "Dynamic group quantization does not use scale"
            assert scale_ub is None, "Dynamic group quantization does not use scale_ub"
            if x.dim() != 2:
                return self.forward_native(x, scale, scale_ub, use_triton)
            return fp8_utils.per_token_group_quant_fp8(
                x,
                group_size=self.group_size,
                column_major_scales=self.column_major_scales,
                tma_aligned_scales=self.tma_aligned_scales,
                dtype=current_platform.fp8_dtype(),
                use_ue8m0=self.use_ue8m0,
            )

        return self.forward_native(x, scale, scale_ub, use_triton)
