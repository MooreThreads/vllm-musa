from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def use_candidate_indexer(
    computed_tokens: Sequence[int],
    threshold: int,
    scheduled_tokens: Sequence[int] | None = None,
) -> bool:
    """Select long batches; zero retains routing without context specialization."""
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
