from __future__ import annotations

import torch
import vllm.envs as envs
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

from vllm_musa.optimization_contract import (
    OptimizationFeature,
    prefers_optimization,
)


def _deepgemm_block_fp8(quant_method) -> bool:
    # MUSA: the DeepGemm block-FP8 kernel writes the GEMM result into a caller
    # buffer, so out_proj/o_proj can skip the output-buffer copy at tp=1.
    if not isinstance(quant_method, Fp8LinearMethod):
        return False
    if not getattr(quant_method, "block_quant", False):
        return False
    return type(getattr(quant_method, "fp8_linear", None)).__name__ == (
        "MUSADeepGemmFp8BlockScaledMMKernel"
    )


@RowParallelLinear.register_oot
class MusaRowParallelLinear(RowParallelLinear):
    def forward_swiglu_clamp(
        self, gate_up: torch.Tensor, swiglu_limit: float
    ) -> torch.Tensor | None:
        """Run clamp-aware SwiGLU and this block-FP8 projection as one path.

        ``DeepseekV4MLP`` calls this optional OOT hook before materializing the
        activation.  Return ``None`` unless this is the exact MUSA DeepGEMM
        row-parallel configuration; the model then executes its ordinary
        activation and linear path unchanged.
        """
        fast = (
            prefers_optimization(
                self,
                OptimizationFeature.DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8,
            )
            and self.input_is_parallel
            and _deepgemm_block_fp8(self.quant_method)
            and self.bias is None
            and gate_up.dim() >= 2
            and gate_up.shape[-1] == 2 * self.input_size_per_partition
            and gate_up.dtype == torch.bfloat16
            and gate_up.is_contiguous()
            and tuple(getattr(self, "weight_block_size", None) or ()) == (128, 128)
            and swiglu_limit == 10.0
        )
        if not fast:
            return None

        weight_scale = getattr(self, "weight_scale_inv", None)
        if weight_scale is None:
            weight_scale = getattr(self, "weight_scale", None)
        if weight_scale is None:
            return None

        gate_up_2d = gate_up.view(-1, gate_up.shape[-1])
        output_parallel = torch.ops.vllm.musa_silu_clamp_deepgemm_fp8_op(
            gate_up_2d,
            self.weight,
            weight_scale,
            128,
            False,
            swiglu_limit,
        ).view(*gate_up.shape[:-1], self.output_size_per_partition)

        if self.reduce_results and self.tp_size > 1:
            return tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel

    def forward(self, input_, out=None):
        if out is None:
            return super().forward(input_)

        fast = (
            self.tp_size == 1
            and not envs.VLLM_BATCH_INVARIANT
            and self.input_is_parallel
            and _deepgemm_block_fp8(self.quant_method)
        )
        if fast:
            # tp==1 (no all-reduce), DeepGemm block-FP8: write straight into out.
            bias_ = None if self.skip_bias_add else self.bias
            result = self.quant_method.fp8_linear.apply_weights(
                self, input_, bias_, output=out
            )
            if result.data_ptr() != out.data_ptr():
                out.copy_(result)
            return (
                out
                if not self.return_bias
                else (out, (self.bias if self.skip_bias_add else None))
            )

        # Fallback: compute normally, then fill the caller buffer (stay correct
        # for non-DeepGemm / tp>1 / bias / batch-invariant).
        result = super().forward(input_)
        result_t, result_bias = result if isinstance(result, tuple) else (result, None)
        out.copy_(result_t)
        return out if not self.return_bias else (out, result_bias)
