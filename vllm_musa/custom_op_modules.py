"""Modules that must be imported before vLLM loads its AOT compile cache."""

from importlib import import_module

CUSTOM_OP_MODULES: tuple[str, ...] = (
    "vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce",
    "vllm_musa.fused_allreduce_rmsnorm_ops",
    "vllm_musa.jit_kernel.csrc.moe",
    "vllm_musa.jit_kernel.csrc.norm",
    "vllm_musa.jit_kernel.csrc.quant",
    "vllm_musa.jit_kernel.csrc.rope",
    "vllm_musa.jit_kernel.csrc.topk",
    "vllm_musa.jit_kernel.tilelang.gdn_fused_proj",
    "vllm_musa.jit_kernel.tilelang.layernorm_gated",
    "vllm_musa.kernels.qwen3_qk_rope_kv",
    "vllm_musa.model_executor.kernels.linear.scaled_mm.deep_gemm",
    "vllm_musa.model_executor.layers.activation",
    "vllm_musa.model_executor.layers.attention.mm_encoder_attention",
    "vllm_musa.model_executor.layers.fused_moe.router.musa_router_gemm",
)


def preload_custom_op_modules() -> None:
    for module_name in CUSTOM_OP_MODULES:
        import_module(module_name)
