# SPDX-License-Identifier: Apache-2.0
"""Contract-backed policy queries used by platform consumers.

Model eligibility belongs to the optimization-contract package.  Keeping these
small queries here prevents the platform module from becoming a registry of
model-name and shape predicates while preserving the dynamic guards in each
consumer.
"""

from __future__ import annotations

from typing import Any

from .resolver import resolve_optimization_contract
from .types import OptimizationFeature

_DEEPSEEK_V4_SPARSE_PADDED_HEADS = 64
_DEEPSEEK_V4_SPARSE_HEAD_DIM = 512
_DEEPSEEK_V4_SPARSE_DTYPE_BYTES = 2


def prefers_feature(vllm_config: Any, feature: OptimizationFeature) -> bool:
    """Return whether a config's frozen contract prefers ``feature``."""
    return resolve_optimization_contract(vllm_config).prefers(feature)


def model_has_routed_experts(model_config: Any | None) -> bool:
    """Return whether a lightweight model config has routed experts.

    The platform no longer owns this predicate; this compatibility query is
    kept for tests and older consumers while the resolver remains the source
    of truth for complete ``VllmConfig`` objects.
    """
    if model_config is None:
        return False
    is_model_moe = getattr(model_config, "is_model_moe", None)
    if callable(is_model_moe):
        try:
            return bool(is_model_moe())
        except (AttributeError, TypeError, ValueError):
            return False
    is_moe = getattr(model_config, "is_moe", None)
    if is_moe is not None:
        return bool(is_moe)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is None:
        hf_config = getattr(model_config, "hf_config", None)
        hf_text_config = getattr(hf_config, "text_config", hf_config)
    expert_values = [
        getattr(hf_text_config, name, None)
        for name in (
            "num_experts",
            "moe_num_experts",
            "n_routed_experts",
            "num_local_experts",
        )
    ]
    if any(value is not None for value in expert_values):
        return any(bool(value) for value in expert_values)
    architectures = getattr(model_config, "architectures", None)
    if architectures is None:
        architectures = getattr(hf_text_config, "architectures", None)
    return any(
        "moe" in str(architecture).lower() for architecture in architectures or ()
    )


def deepseek_v4_flashmla_sparse_page_size(vllm_config: Any) -> int:
    """Return the contract-selected sparse FlashMLA page size."""
    if prefers_feature(
        vllm_config,
        OptimizationFeature.DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256,
    ):
        return 256
    return 64


def deepseek_v4_mtp_sparse_prefill_headroom_bytes(vllm_config: Any) -> int:
    """Return transient H64 query workspace omitted by MTP profiling."""
    if not prefers_feature(
        vllm_config,
        OptimizationFeature.DEEPSEEK_V4_TP8_MTP_SPARSE_PREFILL_HEADROOM,
    ):
        return 0
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_batched_tokens = int(
        getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
    )
    if max_num_batched_tokens <= 0:
        return 0
    return (
        max_num_batched_tokens
        * _DEEPSEEK_V4_SPARSE_PADDED_HEADS
        * _DEEPSEEK_V4_SPARSE_HEAD_DIM
        * _DEEPSEEK_V4_SPARSE_DTYPE_BYTES
    )
