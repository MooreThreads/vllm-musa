from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def candidate_context_threshold(vllm_config: VllmConfig) -> int:
    config = vllm_config.model_config.hf_text_config
    if (
        getattr(config, "model_type", None) != "deepseek_v41"
        or getattr(config, "candidate_topk_blocks", 0) <= 0
        or vllm_config.speculative_config is not None
        or not vllm_config.use_v2_model_runner
        or vllm_config.parallel_config.use_ubatching
        or vllm_config.parallel_config.data_parallel_size != 1
    ):
        return 0
    threshold = int(os.getenv("VLLM_MUSA_CANDIDATE_MIN_CONTEXT", "32768"))
    if threshold < 0:
        raise ValueError("VLLM_MUSA_CANDIDATE_MIN_CONTEXT must be nonnegative")
    return threshold


def use_candidate_indexer(
    computed_tokens: Sequence[int],
    threshold: int,
    scheduled_tokens: Sequence[int] | None = None,
) -> bool:
    """Retain candidate filtering whenever a batch contains a long request."""
    if threshold == 0:
        return True
    if scheduled_tokens is None:
        return any(int(computed) + 1 >= threshold for computed in computed_tokens)
    if len(computed_tokens) != len(scheduled_tokens):
        raise ValueError("computed and scheduled token counts must have equal length")
    return any(
        int(scheduled) > 0 and int(computed) + int(scheduled) >= threshold
        for computed, scheduled in zip(computed_tokens, scheduled_tokens)
    )


def candidate_path_enabled() -> bool:
    from vllm.forward_context import get_forward_context

    descriptor = get_forward_context().batch_descriptor
    return descriptor is None or descriptor.use_sparse_indexer_candidates


def select_candidate_graph(
    descriptor: BatchExecutionDescriptor,
    computed_tokens: Sequence[int],
    threshold: int,
    scheduled_tokens: Sequence[int] | None = None,
) -> BatchExecutionDescriptor:
    return replace(
        descriptor,
        use_sparse_indexer_candidates=use_candidate_indexer(
            computed_tokens, threshold, scheduled_tokens
        ),
    )
