"""Load vLLM model-executor extensions during MUSA plugin initialization."""

from importlib import import_module

# Some layer modules also register Torch ops at import time. Keep their
# load order until those registrations can be migrated independently.
_EXTENSION_MODULES = (
    "vllm_musa.model_executor.layers.activation",
    "vllm_musa.model_executor.layers.attention.mm_encoder_attention",
    "vllm_musa.model_executor.layers.fused_moe.fused_moe",
    "vllm_musa.model_executor.layers.fused_moe.router.fused_topk_bias_router",
    "vllm_musa.model_executor.layers.fused_moe.router.fused_topk_router",
    "vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router",
    "vllm_musa.model_executor.layers.fused_moe.unquantized_fused_moe_method",
    "vllm_musa.model_executor.layers.layernorm",
    "vllm_musa.model_executor.layers.linear",
    "vllm_musa.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "vllm_musa.model_executor.layers.quantization.fp8",
    "vllm_musa.model_executor.layers.quantization.input_quant_fp8",
    "vllm_musa.model_executor.layers.quantization.utils.fp8_utils",
    "vllm_musa.model_executor.layers.rotary_embedding.base",
    "vllm_musa.model_executor.warmup.deep_gemm_warmup",
)


def register_model_executor_extensions() -> None:
    """Load OOT layers, quantization extensions and warmup hooks in order."""
    for module_name in _EXTENSION_MODULES:
        import_module(module_name)
