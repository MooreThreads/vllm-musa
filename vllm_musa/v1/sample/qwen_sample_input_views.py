"""Gated views for Qwen's uniform decode sampling inputs."""

from __future__ import annotations

from typing import Any

# isort: off
import numpy as np
import torchada  # noqa: F401
import torch

# isort: on

from vllm_musa.runtime_plan import (
    RuntimeDecision,
    runtime_plan_enabled,
)


def _select_qwen_sample_input_views(
    sampler: Any,
    logits: torch.Tensor,
    input_batch: Any,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    return_logprobs: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not runtime_plan_enabled(
        sampler,
        RuntimeDecision.QWEN_SAMPLE_INPUT_VIEWS,
    ):
        return None
    if return_logprobs:
        return None

    # Import lazily to preserve vLLM sampler import order.
    from vllm_musa.v1.sample.topk_topp_sampler import (
        can_use_qwen_v2_unfiltered_gumbel,
    )

    num_reqs = input_batch.num_reqs
    if (
        num_reqs <= 0
        or input_batch.num_reqs_after_padding < num_reqs
        or input_batch.num_tokens != num_reqs
        or input_batch.num_tokens_after_padding < num_reqs
        or input_batch.num_tokens_after_padding != input_batch.num_reqs_after_padding
        or input_batch.num_draft_tokens != 0
        or input_batch.num_scheduled_tokens.shape != (num_reqs,)
        or not np.all(input_batch.num_scheduled_tokens == 1)
        or input_batch.is_prefilling_np.shape != (num_reqs,)
        or np.any(input_batch.is_prefilling_np)
        or logits.ndim != 2
        or logits.shape[0] != num_reqs
        or logits.dtype != torch.bfloat16
        or logits.stride(-1) != 1
        or input_batch.positions.ndim != 1
        or input_batch.input_ids.ndim != 1
        or input_batch.positions.shape[0] < input_batch.num_tokens_after_padding
        or input_batch.input_ids.shape[0] < input_batch.num_tokens_after_padding
    ):
        return None

    pos = input_batch.positions[:num_reqs]
    if not can_use_qwen_v2_unfiltered_gumbel(
        sampler,
        logits,
        expanded_idx_mapping,
        idx_mapping_np,
        pos,
        return_logprobs,
    ):
        return None

    return pos, input_batch.input_ids[:num_reqs]


def install_qwen_sample_input_views() -> None:
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    if not hasattr(Sampler, "_musa_select_qwen_sample_input_views"):
        Sampler._musa_select_qwen_sample_input_views = _select_qwen_sample_input_views
