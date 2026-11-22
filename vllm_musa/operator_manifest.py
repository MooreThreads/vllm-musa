"""Static integration records for vLLM-MUSA operators and providers.

This module deliberately has no torch or vLLM imports.  It is used by source
checks and by the plugin preload path before the runtime stack is initialized.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = frozenset(
    {"vllm_aot_jit", "vllm_ir", "musa_native", "mate_adapter", "model_private"}
)


@dataclass(frozen=True)
class OperatorRecord:
    name: str
    category: str
    contract: str
    provider: str
    registration_module: str
    source: tuple[str, ...]
    tests: tuple[str, ...]
    fallback: str
    preload: bool = True


# This first slice describes the modules that must be imported before vLLM's
# AOT compile cache is used.  Additional records should be added as operators
# are migrated from the inventory, rather than extending an unowned tuple.
OPERATOR_RECORDS: tuple[OperatorRecord, ...] = (
    OperatorRecord("custom_all_reduce", "vllm_aot_jit", "vllm.custom_op", "vllm_musa", "vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce", ("vllm_musa/jit_kernel/csrc/allreduce.py",), ("tests/test_musa_jit_custom_all_reduce_graph.py",), "vllm.reference"),
    OperatorRecord("fused_allreduce_rmsnorm", "musa_native", "vllm.custom_op", "vllm_musa", "vllm_musa.fused_allreduce_rmsnorm_ops", ("vllm_musa/fused_allreduce_rmsnorm_ops.py",), ("tests/test_fused_add_rms_norm_compile.py",), "vllm.reference"),
    OperatorRecord("fp8_block_gemv", "vllm_aot_jit", "vllm.custom_op", "vllm_musa", "vllm_musa.jit_kernel.csrc.fp8_block_gemv", ("vllm_musa/jit_kernel/csrc/fp8_block_gemv.py",), ("tests/test_fp8_block_gemv_native.py",), "vllm.reference"),
    OperatorRecord("moe_jit", "vllm_aot_jit", "vllm.custom_op", "vllm_musa", "vllm_musa.jit_kernel.csrc.moe", ("vllm_musa/jit_kernel/csrc/moe.py",), ("tests/jit_kernel/test_csrc_jit_kernels.py",), "vllm.reference"),
    OperatorRecord("norm_jit", "vllm_aot_jit", "vllm.custom_op", "vllm_musa", "vllm_musa.jit_kernel.csrc.norm", ("vllm_musa/jit_kernel/csrc/norm.py",), ("tests/test_deepseek_v4_fused_add_rmsnorm.py",), "vllm.reference"),
    OperatorRecord("quant_jit", "vllm_aot_jit", "vllm.custom_op", "vllm_musa", "vllm_musa.jit_kernel.csrc.quant", ("vllm_musa/jit_kernel/csrc/quant.py",), ("tests/test_fp8_block_gemv_native.py",), "vllm.reference"),
    OperatorRecord("rope_jit", "vllm_aot_jit", "vllm.custom_op", "vllm_musa", "vllm_musa.jit_kernel.csrc.rope", ("vllm_musa/jit_kernel/csrc/rope.py",), ("tests/kernels/test_qwen2_rope_kv.py",), "vllm.reference"),
    OperatorRecord("sqrtsoftplus_topk", "vllm_aot_jit", "vllm.custom_op", "vllm_musa", "vllm_musa.jit_kernel.csrc.sqrtsoftplus_topk", ("vllm_musa/jit_kernel/csrc/sqrtsoftplus_topk.py",), ("tests/kernels/test_musa_sqrtsoftplus_route_map.py",), "vllm.reference"),
    OperatorRecord("topk", "vllm_aot_jit", "vllm.custom_op", "vllm_musa", "vllm_musa.jit_kernel.csrc.topk", ("vllm_musa/jit_kernel/csrc/topk.py",), ("tests/test_top_k_mask_logits.py",), "vllm.reference"),
    OperatorRecord("gdn_fused_proj", "model_private", "vllm.layer", "vllm_musa", "vllm_musa.jit_kernel.tilelang.gdn_fused_proj", ("vllm_musa/jit_kernel/tilelang/gdn_fused_proj.py",), ("tests/test_gdn_fused_zba_compile.py",), "vllm.reference"),
    OperatorRecord("layernorm_gated", "model_private", "vllm.layer", "vllm_musa", "vllm_musa.jit_kernel.tilelang.layernorm_gated", ("vllm_musa/jit_kernel/tilelang/layernorm_gated.py",), ("tests/test_grouped_topk_tilelang.py",), "vllm.reference"),
    OperatorRecord("qwen3_qk_rope_kv", "model_private", "vllm.layer", "vllm_musa", "vllm_musa.kernels.qwen3_qk_rope_kv", ("vllm_musa/kernels/qwen3_qk_rope_kv.py",), ("tests/test_qwen3_qk_rope_kv_presplit.py",), "vllm.reference"),
    OperatorRecord("deep_gemm", "mate_adapter", "vllm.linear", "mate", "vllm_musa.model_executor.kernels.linear.scaled_mm.deep_gemm", ("vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",), ("tests/kernels/test_w4afp8_moe.py",), "vllm.reference"),
    OperatorRecord("activation", "vllm_aot_jit", "vllm.custom_op", "vllm_musa", "vllm_musa.model_executor.layers.activation", ("vllm_musa/model_executor/layers/activation.py",), ("tests/jit_kernel/test_csrc_jit_kernels.py",), "vllm.reference"),
    OperatorRecord("mm_encoder_attention", "mate_adapter", "vllm.attention", "mate", "vllm_musa.model_executor.layers.attention.mm_encoder_attention", ("vllm_musa/model_executor/layers/attention/mm_encoder_attention.py",), ("tests/test_attention_backend_capabilities.py",), "vllm.reference"),
    OperatorRecord("musa_router_gemm", "mate_adapter", "vllm.moe", "mate", "vllm_musa.model_executor.layers.fused_moe.router.musa_router_gemm", ("vllm_musa/model_executor/layers/fused_moe/router/musa_router_gemm.py",), ("tests/test_musa_router_gemm.py",), "vllm.reference"),
)


def preload_modules(records: Iterable[OperatorRecord] = OPERATOR_RECORDS) -> tuple[str, ...]:
    return tuple(record.registration_module for record in records if record.preload)


def validate_records(root: Path, records: Iterable[OperatorRecord] = OPERATOR_RECORDS) -> list[str]:
    records = tuple(records)
    errors: list[str] = []
    names: set[str] = set()
    modules: set[str] = set()
    for record in records:
        if record.name in names:
            errors.append(f"duplicate operator name: {record.name}")
        names.add(record.name)
        if record.registration_module in modules:
            errors.append(f"duplicate registration module: {record.registration_module}")
        modules.add(record.registration_module)
        if record.category not in CATEGORIES:
            errors.append(f"{record.name}: invalid category {record.category!r}")
        for relative in (*record.source, *record.tests):
            if not (root / relative).is_file():
                errors.append(f"{record.name}: missing path {relative}")
        if not record.provider or not record.contract or not record.fallback:
            errors.append(f"{record.name}: provider, contract, and fallback are required")
    return errors
