# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA model-executor extensions loaded by ``vllm_musa.register_out_of_tree``.

Split into two explicit groups:
1. Formal vLLM OOT ``CustomOp`` / ``PluggableLayer`` subclasses decorated with
   ``@<UpstreamLayer>.register_oot`` (registered on import of their owning module).
2. Compatibility adapters for upstream classes lacking a formal OOT hook yet,
   installed via explicit idempotent ``install_hooks()`` functions.
"""

# isort: off
# 1. Formal vLLM OOT CustomOp / PluggableLayer subclasses (@*.register_oot)
import vllm_musa.model_executor.layers.activation  # noqa: F401
import vllm_musa.model_executor.layers.attention.mm_encoder_attention  # noqa: F401
import vllm_musa.model_executor.layers.fused_moe.unquantized_fused_moe_method  # noqa: F401
import vllm_musa.model_executor.layers.layernorm  # noqa: F401
import vllm_musa.model_executor.layers.linear  # noqa: E402,F401
import vllm_musa.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn  # noqa: F401
import vllm_musa.model_executor.layers.quantization.input_quant_fp8  # noqa: F401
import vllm_musa.model_executor.layers.rotary_embedding.base  # noqa: F401
# isort: on

_ADAPTERS_REGISTERED = False


def register_model_executor_adapters() -> None:
    """Install idempotent compatibility hooks for model-executor modules."""
    global _ADAPTERS_REGISTERED
    if _ADAPTERS_REGISTERED:
        return

    from vllm_musa.model_executor.layers.fused_moe import fused_moe
    from vllm_musa.model_executor.layers.fused_moe.router import (
        fused_topk_bias_router,
        fused_topk_router,
        grouped_topk_router,
    )
    from vllm_musa.model_executor.layers.quantization import fp8
    from vllm_musa.model_executor.layers.quantization.utils import fp8_utils
    from vllm_musa.model_executor.warmup import deep_gemm_warmup

    fused_moe.install_hooks()
    fused_topk_bias_router.install_hooks()
    fused_topk_router.install_hooks()
    grouped_topk_router.install_hooks()
    fp8.install_hooks()
    fp8_utils.install_hooks()
    deep_gemm_warmup.install_hooks()
    _ADAPTERS_REGISTERED = True
