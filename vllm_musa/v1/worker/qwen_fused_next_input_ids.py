# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm_musa.v1.worker.qwen_identity_logits_view import (
    _is_musa_tensor,
    _is_qwen_runner,
)

logger = init_logger(__name__)

def select_qwen_fused_decode_inputs(
    runner: Any,
    req_ids: list[str],
    num_scheduled_tokens: np.ndarray,
    is_prefilling_np: np.ndarray,
    total_num_draft_tokens: int,
    total_num_logits: int,
    num_tokens: int,
    num_tokens_after_padding: int,
    num_reqs_after_padding: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Use input IDs written by the preceding post-update kernel."""
    primed_req_ids = getattr(runner, "_musa_qwen_primed_input_req_ids", None)
    runner._musa_qwen_primed_input_req_ids = None
    runner._musa_qwen_pending_input_req_ids = None

    input_ids = runner.input_buffers.input_ids
    if (
        not current_platform.is_musa()
        or not _is_musa_tensor(input_ids)
        or not _is_qwen_runner(runner)
        or getattr(runner, "use_pp", False)
    ):
        return None

    num_reqs = len(req_ids)
    if (
        num_reqs <= 0
        or input_ids.dtype != torch.int32
        or input_ids.ndim != 1
        or not input_ids.is_contiguous()
        or num_tokens != num_reqs
        or num_tokens_after_padding < num_reqs
        or num_reqs_after_padding < num_reqs
        or num_tokens_after_padding != num_reqs_after_padding
        or input_ids.shape[0] < num_tokens_after_padding
        or total_num_logits != num_reqs
        or total_num_draft_tokens != 0
        or getattr(
            getattr(runner, "model_state", None),
            "num_new_sampled_tokens_per_step",
            None,
        )
        != 1
        or num_scheduled_tokens.shape != (num_reqs,)
        or not np.all(num_scheduled_tokens == 1)
        or is_prefilling_np.shape != (num_reqs,)
        or np.any(is_prefilling_np)
    ):
        return None

    current_req_ids = tuple(req_ids)
    runner._musa_qwen_pending_input_req_ids = current_req_ids
    if primed_req_ids != current_req_ids:
        return None

    cache = getattr(runner, "_musa_qwen_decode_logits_indices", None)
    if cache is None or cache.device != input_ids.device or cache.shape[0] < num_reqs:
        cache = torch.arange(num_reqs, dtype=torch.int64, device=input_ids.device)
        runner._musa_qwen_decode_logits_indices = cache
        logger.info_once(
            "Using fused MUSA Qwen next-token input IDs for uniform decode.",
            scope="global",
        )
    return input_ids[:num_tokens_after_padding], cache[:num_reqs]


def select_qwen_next_input_ids_buffer(runner: Any) -> torch.Tensor | None:
    pending_req_ids = getattr(runner, "_musa_qwen_pending_input_req_ids", None)
    runner._musa_qwen_pending_input_req_ids = None
    if pending_req_ids is None:
        runner._musa_qwen_primed_input_req_ids = None
        return None
    runner._musa_qwen_primed_input_req_ids = pending_req_ids
    return runner.input_buffers.input_ids


def install_hooks() -> None:
    GPUModelRunner._musa_select_uniform_decode_model_inputs = (  # type: ignore[attr-defined]
        select_qwen_fused_decode_inputs
    )
    GPUModelRunner._musa_select_next_input_ids_buffer = (  # type: ignore[attr-defined]
        select_qwen_next_input_ids_buffer
    )


install_hooks()
