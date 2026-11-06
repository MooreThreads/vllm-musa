# SPDX-License-Identifier: Apache-2.0
"""Fold a DeepSeek-V4-family shared expert into the routed FP8 expert stack.

The shared MLP has the per-rank shape and FP8 block format of a routed expert, so
after loading its projections are appended to the routed weights as one more
expert (``quantization/fp8.py``) and the router gives every token one extra slot
at weight 1.0 on that expert. The MoE then runs routed and shared work in one
grouped kernel chain, without the shared expert's separate GEMMs, activation and
output add. Only models whose optimization contract prefers the fold take it.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from vllm.config import VllmConfig

from vllm_musa.optimization_contract import (
    OptimizationFeature,
    resolve_optimization_contract,
)


def _block_size(method: object) -> list[int] | None:
    block = getattr(method, "weight_block_size", None)
    if block is None:
        block = getattr(
            getattr(method, "quant_config", None), "weight_block_size", None
        )
    return list(block) if block is not None else None


def _routed_block_size(
    config: object, quant_config: object, tp_size: int
) -> list[int] | None:
    """FP8 block of the routed experts' scales after loading: the checkpoint's expert
    block, refined like the FP8 MoE method does when it does not tile the TP shard."""
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        refine_fp8_moe_block_shape,
    )

    block = getattr(quant_config, "expert_weight_block_size", None) or getattr(
        quant_config, "weight_block_size", None
    )
    if block is None:
        return None
    shard = SimpleNamespace(
        intermediate_size_per_partition=config.moe_intermediate_size // tp_size,
        tp_size=tp_size,
        hidden_dim=config.hidden_size,
    )
    return refine_fp8_moe_block_shape(shard, list(block)) or list(block)


def deepseek_v4_shared_expert_foldable(
    vllm_config: VllmConfig, moe: nn.Module, config: object, quant_config: object
) -> bool:
    """Whether ``moe.shared_experts`` joins the routed FP8 experts: the optimization
    contract prefers the fold, routing is TP-only sqrt-softplus, and the single
    shared expert has the experts' FP8 blocks and per-rank intermediate size."""
    if not resolve_optimization_contract(vllm_config).prefers(
        OptimizationFeature.DEEPSEEK_V41_SHARED_EXPERT_FOLD
    ):
        return False
    parallel_config = vllm_config.parallel_config
    shared = getattr(moe, "shared_experts", None)
    if (
        shared is None
        or getattr(config, "n_shared_experts", None) != 1
        or parallel_config.enable_expert_parallel
        or parallel_config.enable_eplb
        or parallel_config.data_parallel_size != 1
        or getattr(parallel_config, "prefill_context_parallel_size", 1) != 1
        or getattr(moe, "use_sequence_parallel", False)
        or getattr(moe, "scoring_func", None) != "sqrtsoftplus"
        or moe.gate.tid2eid is not None
        or getattr(quant_config, "expert_dtype", None) != "fp8"
    ):
        return False
    tp_size = parallel_config.tensor_parallel_size
    inter = config.moe_intermediate_size
    if inter % tp_size:
        return False
    routed_block = _routed_block_size(config, quant_config, tp_size)
    for proj, rows in (
        (shared.gate_up_proj, 2 * inter // tp_size),
        (shared.down_proj, None),
    ):
        method = getattr(proj, "quant_method", None)
        if not getattr(method, "block_quant", False):
            return False
        if _block_size(method) != routed_block:
            return False
        if rows is not None and proj.weight.shape[0] != rows:
            return False
    return shared.down_proj.weight.shape[1] == inter // tp_size


def fold_shared_expert(experts: nn.Module, shared_experts: nn.Module) -> None:
    """Mark the routed experts to absorb ``shared_experts`` after loading and route
    every token to it."""
    routed = getattr(experts, "routed_experts", experts)
    routed._musa_shared_mlp = shared_experts
    experts.router.num_fused_shared_experts = 1


def release_shared_expert_weights(mlp: nn.Module, *scale_names: str) -> None:
    """Free a folded shared expert's projections, whose copy now lives in the routed
    expert stack, and return the cached blocks to the device.

    The stack is concatenated while the unfolded one is still alive, so the freed
    blocks leave expert-sized holes in the caching allocator. Left there, they are
    reserved-but-unusable and the KV cache later fails to find contiguous space.
    """
    for proj in (mlp.gate_up_proj, mlp.down_proj):
        for name in ("weight", *scale_names):
            tensor = getattr(proj, name, None)
            if isinstance(tensor, torch.Tensor):
                tensor.data = tensor.data.new_empty(0)

    from vllm.platforms import current_platform

    empty_cache = getattr(current_platform, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
    elif hasattr(torch, "musa"):
        torch.musa.empty_cache()
