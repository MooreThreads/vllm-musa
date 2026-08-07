# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from vllm.compilation.passes.fusion.matcher_utils import MatcherSiluAndMul
from vllm.compilation.passes.vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmPatternReplacement,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class MusaSiluDeepGemmPattern(VllmPatternReplacement):
    """Fuse native SwiGLU with the opaque MUSA FP8 DeepGEMM wrapper."""

    def __init__(self) -> None:
        # Quantized FULL_AND_PIECEWISE serving routes MUSA custom ops through
        # their native graph forms, so match the production SwiGLU nodes.
        self.silu_and_mul_matcher = MatcherSiluAndMul(enabled=False)

    def get_inputs(self) -> list[torch.Tensor]:
        device = current_platform.device_type
        return [
            torch.empty((5, 256), dtype=torch.bfloat16, device=device),
            torch.empty((64, 128), dtype=current_platform.fp8_dtype(), device=device),
            torch.empty((1, 1), dtype=torch.float32, device=device),
        ]

    @property
    def pattern(self):
        def _pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> torch.Tensor:
            activated = self.silu_and_mul_matcher(input)
            return auto_functionalized(
                torch.ops.vllm.musa_deepgemm_fp8_op.default,
                input=activated,
                weight=weight,
                weight_scale=weight_scale,
                group_size=128,
                use_deep_gemm_e8m0=False,
                output=None,
            )[0]

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.vllm.musa_silu_deepgemm_fp8_op(
                input,
                weight,
                weight_scale,
                128,
                False,
            )

        return _replacement


class MusaSiluDeepGemmFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "musa_silu_deepgemm_fusion_pass")
        self.register(MusaSiluDeepGemmPattern())
        self.dump_patterns(config, self.pm_pass)

    def __call__(self, graph: fx.Graph) -> None:
        super().__call__(graph)
        if self.matched_count:
            logger.info(
                "MUSA dense SwiGLU+DeepGEMM fusion matched %d pattern(s)",
                self.matched_count,
            )
