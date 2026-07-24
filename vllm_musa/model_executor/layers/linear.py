from __future__ import annotations

import vllm.envs as envs
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod


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
