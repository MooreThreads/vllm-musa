# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger


logger = init_logger(__name__)

_BF16_WARMUP_CACHE: set[tuple[torch.Size, torch.Size, int, int]] = set()


def _fp8_linear_may_use_deep_gemm(module: torch.nn.Module) -> bool:
    return False


import vllm.model_executor.warmup.deep_gemm_warmup

vllm.model_executor.warmup.deep_gemm_warmup._fp8_linear_may_use_deep_gemm = (
    _fp8_linear_may_use_deep_gemm
)


def _alignment_m() -> int:
    from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout

    alignment = get_mk_alignment_for_contiguous_layout()
    return int(
        alignment[0] if isinstance(alignment, (list, tuple)) else alignment
    )


def _aligned_grouped_tokens(
    num_tokens: int, top_k: int, num_experts: int, block_m: int
) -> int:
    num_slots = num_tokens * top_k
    return (
        (num_slots + num_experts * (block_m - 1) + block_m - 1) // block_m
    ) * block_m


def _bf16_warmup_token_points(max_tokens: int) -> list[int]:
    points = {1024, 2048, 4096, 8192, 12288, max(1024, int(max_tokens))}
    return sorted(point for point in points if point <= max(1024, int(max_tokens)))


def _warmup_bf16_grouped_moe(model: torch.nn.Module, max_tokens: int) -> None:
    """Precompile the MUSA BF16 grouped-MoE kernels used by large prefill.

    MUSA's ``ragged_m_moe_gemm_16bit`` backend lazily specializes its grouped
    GEMM.  A mixed request batch reaches new aligned-M shapes that are not
    covered by vLLM's decode graph capture, so the first request can spend tens
    of seconds compiling on the host.  Warm a bounded set of scheduler-sized
    shapes, matching the serving path's scheduler-sized bucket contract.
    """
    import deep_gemm

    block_m = _alignment_m()
    token_points = _bf16_warmup_token_points(max_tokens)
    for module in model.modules():
        w1 = getattr(module, "w13_weight", None)
        w2 = getattr(module, "w2_weight", None)
        if not isinstance(w1, torch.Tensor) or not isinstance(w2, torch.Tensor):
            continue
        if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
            continue
        if w1.ndim != 3 or w2.ndim != 3:
            continue
        num_experts, intermediate_size, hidden_size = map(int, w1.shape)
        if num_experts not in (256, 257):
            continue
        if (
            intermediate_size % 256 != 0
            or hidden_size % 128 != 0
            or tuple(w2.shape) != (num_experts, hidden_size, intermediate_size // 2)
        ):
            continue
        top_k = int(getattr(module, "top_k", 9))
        cache_key = (w1.shape, w2.shape, top_k, int(max_tokens))
        if cache_key in _BF16_WARMUP_CACHE:
            continue

        max_grouped_tokens = _aligned_grouped_tokens(
            max(token_points), top_k, num_experts, block_m
        )
        device = w1.device
        expert_ids = torch.randint(
            0,
            num_experts,
            (max_grouped_tokens // block_m,),
            device=device,
            dtype=torch.int32,
        ).repeat_interleave(block_m)

        logger.info(
            "Warming MUSA BF16 grouped MoE GEMM: experts=%d top_k=%d "
            "tokens=%s max_grouped_tokens=%d block_m=%d",
            num_experts,
            top_k,
            token_points,
            max_grouped_tokens,
            block_m,
        )
        with torch.inference_mode():
            for weight in (w1, w2):
                _, out_size, in_size = map(int, weight.shape)
                lhs = torch.empty(
                    (max_grouped_tokens, in_size), device=device, dtype=torch.bfloat16
                )
                out = torch.empty(
                    (max_grouped_tokens, out_size), device=device, dtype=torch.bfloat16
                )
                for num_tokens in token_points:
                    grouped_tokens = _aligned_grouped_tokens(
                        num_tokens, top_k, num_experts, block_m
                    )
                    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                        lhs[:grouped_tokens],
                        weight,
                        out[:grouped_tokens],
                        expert_ids[:grouped_tokens],
                        alignment_m=block_m,
                    )
                    if device.type == "musa":
                        torch.musa.synchronize()
                    else:
                        torch.cuda.synchronize()
                del lhs, out
        _BF16_WARMUP_CACHE.add(cache_key)


# The MUSA sparse indexer runs mate's paged MQA logits on at most 512 query rows
# per call, and mate builds one schedule-metadata module per 32-row bucket.
_MQA_LOGITS_METADATA_MAX_ROWS = 512
_MQA_LOGITS_METADATA_ROW_ALIGN = 32


def _uses_sparse_indexer(model: torch.nn.Module) -> bool:
    try:
        from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
    except ImportError:
        return False
    return any(isinstance(module, SparseAttnIndexer) for module in model.modules())


def _warmup_paged_mqa_logits_metadata(model: torch.nn.Module) -> None:
    """Build mate's paged MQA logits metadata modules before serving.

    mate compiles the module for a row bucket the first time a step reaches it,
    which stalls that step for seconds; prefill row blocks reach every bucket.
    """
    if not _uses_sparse_indexer(model):
        return
    try:
        from mate.jit.gemm.deep_gemm import get_paged_mqa_logits_metadata_module
    except ImportError:
        return
    buckets = range(
        _MQA_LOGITS_METADATA_ROW_ALIGN,
        _MQA_LOGITS_METADATA_MAX_ROWS + 1,
        _MQA_LOGITS_METADATA_ROW_ALIGN,
    )
    logger.info(
        "Building mate paged MQA logits metadata for %d-%d rows",
        buckets[0],
        buckets[-1],
    )
    for rows in buckets:
        get_paged_mqa_logits_metadata_module(rows)


def _vllm_config(model: torch.nn.Module):
    """The vLLM config ``model`` was built with, or ``None``.

    Warmup runs outside ``set_current_vllm_config()``, so fall back to the
    config a ``support_torch_compile`` module keeps as ``vllm_config``.
    """
    try:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is None:
            config = next(
                (
                    module.vllm_config
                    for module in model.modules()
                    if getattr(module, "vllm_config", None) is not None
                ),
                None,
            )
        return config
    except Exception:
        return None


def _rotary_calls_replaced(vllm_config) -> bool:
    """Whether the Qwen3 QK-RoPE-KV pre-split replaces every rotary call.

    Read from the optimization contract, which also holds when the compiled
    graph comes from the AOT cache. Where the rewrite does not match, the
    profile run has already built the one hint its 128-wide envelope can pick.
    """
    try:
        from vllm_musa.optimization_contract.policy import prefers_feature
        from vllm_musa.optimization_contract.types import OptimizationFeature

        return vllm_config is not None and prefers_feature(
            vllm_config, OptimizationFeature.QWEN3_QK_ROPE_KV_PRESPLIT
        )
    except Exception:
        return False


def _rope_kernel_rotary_dims(model: torch.nn.Module) -> set[int | None]:
    """Rotary widths a live call path hands the MUSA rotary-embedding kernel.

    A ``MusaRotaryEmbedding`` contributes its width when CustomOp dispatches it
    to ``forward_oot``, ``VLLM_MUSA_CUSTOM_OP_USE_NATIVE`` is unset and the
    config does not prefer the Qwen3 QK-RoPE-KV pre-split. A
    ``MusaVisionApplyRotaryEmb`` contributes its width unless
    ``VLLM_MUSA_CUSTOM_OP_USE_NATIVE`` is set. Empty when no call path reaches
    the kernel; ``None`` marks a width that is only known per call.
    """
    from vllm_musa.model_executor.layers.rotary_embedding.base import (
        MusaRotaryEmbedding,
    )
    from vllm_musa.optimization_contract.rotary import MusaVisionApplyRotaryEmb
    from vllm_musa.utils.environ import envs

    use_native = envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()
    calls_replaced = _rotary_calls_replaced(_vllm_config(model))
    dims: set[int | None] = set()
    for module in model.modules():
        if isinstance(module, MusaRotaryEmbedding):
            # CustomOp dispatches forward_oot only while the op is enabled; that
            # path reaches the kernel directly or through the Qwen2 RoPE+KV
            # pre-split's fused op.
            forward = getattr(module, "_forward_method", None)
            dispatches_oot = (
                getattr(forward, "__func__", None) is MusaRotaryEmbedding.forward_oot
            )
            if calls_replaced or use_native or not dispatches_oot:
                continue
            # The kernel takes the width from the cache, which partial-rotary
            # models size to head_dim * partial_rotary_factor.
            shape = getattr(getattr(module, "cos_sin_cache", None), "shape", None)
            dims.add(int(shape[-1]) if shape else None)
        elif isinstance(module, MusaVisionApplyRotaryEmb) and not use_native:
            # The adapter concatenates cos and sin into the kernel's cache.
            geometry = module.required_bf16_neox_shape
            dims.add(2 * int(geometry[1]) if geometry is not None else None)
    return dims


def _warmup_rope_store_hints(model: torch.nn.Module) -> None:
    """Build the rotary-embedding store-hint modules before graph capture.

    The kernel picks its store hint from the runtime token count and each hint
    is its own JIT module, so a hint first reached while serving or capturing
    would be built there. Every hint the kernel can pick is loaded for each
    width ``_rope_kernel_rotary_dims`` returns; a failure only logs a warning.
    """
    try:
        from vllm_musa.jit_kernel.csrc.rope import prewarm_rope_modules

        loaded: list[str] = []
        for rot_dim in _rope_kernel_rotary_dims(model):
            for name in prewarm_rope_modules(rot_dim):
                if name not in loaded:
                    loaded.append(name)
    except Exception as exc:
        logger.warning("MUSA rotary-embedding warmup skipped: %s", exc)
        return
    if loaded:
        logger.info("Loaded MUSA rotary-embedding kernels: %s", ", ".join(loaded))


_upstream_deep_gemm_warmup = (
    vllm.model_executor.warmup.deep_gemm_warmup.deep_gemm_warmup
)


def deep_gemm_warmup(model: torch.nn.Module, max_tokens: int, pbar=None):
    _upstream_deep_gemm_warmup(model, max_tokens)
    _warmup_bf16_grouped_moe(model, max_tokens)
    _warmup_paged_mqa_logits_metadata(model)
    _warmup_rope_store_hints(model)


vllm.model_executor.warmup.deep_gemm_warmup.deep_gemm_warmup = deep_gemm_warmup
