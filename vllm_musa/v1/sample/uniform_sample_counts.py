# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_musa.v1.sample.topk_topp_sampler import (
    _is_qwen_sampler_vocab,
    is_musa_tensor,
)

logger = init_logger(__name__)


def select_qwen_uniform_sample_counts(
    sampler: Any, logits: torch.Tensor, input_batch: Any
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return cached one/zero counts for non-prefill Qwen decode."""
    if not getattr(sampler, "_musa_qwen_family", False):
        return None
    if not current_platform.is_musa() or not is_musa_tensor(logits):
        return None
    if (
        logits.dtype != torch.bfloat16
        or logits.ndim != 2
        or not logits.is_contiguous()
        or not _is_qwen_sampler_vocab(logits)
    ):
        return None

    try:
        num_reqs = input_batch.num_reqs
        num_scheduled_tokens = input_batch.num_scheduled_tokens
        is_prefilling = input_batch.is_prefilling_np
        num_draft_tokens = input_batch.num_draft_tokens
        seq_lens = input_batch.seq_lens
    except AttributeError:
        return None
    if (
        num_reqs <= 0
        or logits.shape[0] != num_reqs
        or num_draft_tokens != 0
        or num_scheduled_tokens.shape != (num_reqs,)
        or not np.all(num_scheduled_tokens == 1)
        or is_prefilling.shape != (num_reqs,)
        or np.any(is_prefilling)
        or seq_lens.ndim != 1
        or seq_lens.shape[0] < num_reqs
    ):
        return None

    buffers = getattr(sampler, "_musa_qwen_uniform_sample_count_buffers", None)
    cache_key = (seq_lens.device, seq_lens.dtype)
    if buffers is None or buffers[0] != cache_key or buffers[1].shape[0] < num_reqs:
        capacity = num_reqs
        num_sampled = torch.ones(
            capacity,
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )
        num_rejected = torch.zeros_like(num_sampled)
        buffers = (cache_key, num_sampled, num_rejected)
        sampler._musa_qwen_uniform_sample_count_buffers = buffers
        logger.info_once(
            "Using cached MUSA Qwen uniform decode sample counts.",
            scope="global",
        )
    return buffers[1][:num_reqs], buffers[2][:num_reqs]


def install_hooks() -> None:
    Sampler._musa_select_num_sampled_and_rejected = (  # type: ignore[attr-defined]
        select_qwen_uniform_sample_counts
    )


install_hooks()
