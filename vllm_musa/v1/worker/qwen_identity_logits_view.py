from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_musa.runtime_plan import (
    RuntimeDecision,
    runtime_plan_enabled,
)

logger = init_logger(__name__)


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "musa"


def _is_qwen_runner(runner: Any) -> bool:
    return runtime_plan_enabled(
        getattr(runner, "sampler", None),
        RuntimeDecision.QWEN_UNIFORM_DECODE_VIEWS,
    )


def select_qwen_identity_logits_view(
    runner: Any, hidden_states: torch.Tensor, input_batch: Any
) -> torch.Tensor | None:
    """Return a view when uniform decode makes logits indices the identity."""
    if not _is_qwen_runner(runner):
        return None
    if not current_platform.is_musa() or not _is_musa_tensor(hidden_states):
        return None
    if (
        hidden_states.dtype != torch.bfloat16
        or hidden_states.ndim != 2
        or not hidden_states.is_contiguous()
    ):
        return None

    num_reqs = input_batch.num_reqs
    num_scheduled_tokens = input_batch.num_scheduled_tokens
    logits_indices = input_batch.logits_indices
    if (
        num_reqs <= 0
        or hidden_states.shape[0] < num_reqs
        or logits_indices.ndim != 1
        or logits_indices.shape[0] != num_reqs
        or input_batch.num_draft_tokens != 0
        or getattr(runner.model_state, "num_new_sampled_tokens_per_step", None) != 1
        or num_scheduled_tokens.shape != (num_reqs,)
        or not np.all(num_scheduled_tokens == 1)
    ):
        return None

    logger.info_once(
        "Using MUSA identity logits view for uniform Qwen decode.",
        scope="global",
    )
    return hidden_states[:num_reqs]


def install_hooks() -> None:
    GPUModelRunner._musa_select_sample_hidden_states = (  # type: ignore[attr-defined]
        select_qwen_identity_logits_view
    )


install_hooks()
