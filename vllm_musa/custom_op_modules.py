# SPDX-License-Identifier: Apache-2.0
"""Modules that must be imported before vLLM loads its AOT compile cache.

Custom ops are registered as an **import side effect** of the modules below, but
vLLM loads the torch AOT autograd cache during engine init — before any compile
runs and therefore before most of these modules are imported.  When the cached
graph references an op that has not been registered yet, the load fails with a
missing-attribute error and the engine silently recompiles the whole model on
every boot.  Importing them from ``_register_ops()`` puts every
``torch.ops.vllm.*`` in the namespace before the cache is read.

The list is kept complete by ``tests/test_custom_op_preload.py``, which scans the
package with ``ast`` and fails when a module that calls
``direct_register_custom_op`` is missing here.  The requirement is registration
timing, not implementation type: JIT, TileLang and plain-Python modules all
count, because any of them can be the op the cached graph names.
"""

import logging
from importlib import import_module

try:  # keep this module importable without vLLM so the AST test can load it
    from vllm.logger import init_logger
except Exception:  # pragma: no cover - only when vllm is not installed
    init_logger = logging.getLogger

logger = init_logger(__name__)

CUSTOM_OP_MODULES: tuple[str, ...] = (
    "vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce",
    "vllm_musa.fused_allreduce_rmsnorm_ops",
    "vllm_musa.jit_kernel.csrc.fp8_block_gemv",
    "vllm_musa.jit_kernel.csrc.moe",
    "vllm_musa.jit_kernel.csrc.norm",
    "vllm_musa.jit_kernel.csrc.quant",
    "vllm_musa.jit_kernel.csrc.rope",
    "vllm_musa.jit_kernel.csrc.sqrtsoftplus_topk",
    "vllm_musa.jit_kernel.csrc.topk",
    "vllm_musa.jit_kernel.tilelang.gdn_fused_proj",
    "vllm_musa.jit_kernel.tilelang.layernorm_gated",
    "vllm_musa.kernels.qwen3_qk_rope_kv",
    "vllm_musa.model_executor.kernels.linear.scaled_mm.deep_gemm",
    "vllm_musa.model_executor.layers.activation",
    "vllm_musa.model_executor.layers.attention.mm_encoder_attention",
    "vllm_musa.model_executor.layers.fused_moe.router.musa_router_gemm",
)


def preload_custom_op_modules() -> list[str]:
    """Import every custom-op module; return the names that failed to import.

    A module that fails to import leaves its ops unregistered, which is exactly
    the failure this preload exists to prevent — so it is logged at error level
    rather than swallowed.  It is not raised: the same import would fail again
    when the op is first needed, and taking the whole platform down during plugin
    registration would lose the rest of the engine's diagnostics.
    """
    failed: list[str] = []
    for module_name in CUSTOM_OP_MODULES:
        try:
            import_module(module_name)
        except Exception:
            logger.error("Failed to preload custom-op module %s", module_name, exc_info=True)
            failed.append(module_name)
    if not failed:
        logger.debug("Preloaded %d custom-op modules", len(CUSTOM_OP_MODULES))
    return failed


def registered_vllm_ops() -> list[str]:
    """Names of the ``torch.ops.vllm.*`` entries currently registered.

    Used by the preload test to prove the preload actually put ops in the
    namespace, rather than only importing modules.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        return []
    ops = torch.ops.vllm
    return sorted(name for name in dir(ops) if not name.startswith("_"))
