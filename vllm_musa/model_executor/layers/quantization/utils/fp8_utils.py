# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def deepgemm_post_process_fp8_weight_block(
    wq: torch.Tensor, ws: torch.Tensor, quant_block_shape: tuple[int], use_e8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    return wq, ws


import vllm.model_executor.layers.quantization.utils.fp8_utils

vllm.model_executor.layers.quantization.utils.fp8_utils.deepgemm_post_process_fp8_weight_block = (
    deepgemm_post_process_fp8_weight_block
)
