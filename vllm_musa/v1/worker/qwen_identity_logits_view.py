from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_musa.utils.environ import envs

logger = init_logger(__name__)

_MUSA_QWEN_IDENTITY_LOGITS_VIEW_ENABLED = envs.VLLM_MUSA_QWEN_IDENTITY_LOGITS_VIEW.get()
_QWEN_ARCHITECTURES = frozenset(
    {
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
_MISSING = object()


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "musa"


def _is_qwen_runner(runner: Any) -> bool:
    cached = getattr(runner, "_musa_identity_logits_is_qwen", _MISSING)
    if cached is not _MISSING:
        return cached
    model_config = getattr(getattr(runner, "vllm_config", None), "model_config", None)
    architectures = getattr(model_config, "architectures", None) or ()
    is_qwen = bool(_QWEN_ARCHITECTURES.intersection(architectures))
    runner._musa_identity_logits_is_qwen = is_qwen
    return is_qwen


def select_qwen_identity_logits_view(
    runner: Any, hidden_states: torch.Tensor, input_batch: Any
) -> torch.Tensor | None:
    """Return a view when uniform decode makes logits indices the identity."""
    if not _MUSA_QWEN_IDENTITY_LOGITS_VIEW_ENABLED:
        return None
    if not current_platform.is_musa() or not _is_musa_tensor(hidden_states):
        return None
    if not _is_qwen_runner(runner):
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
    if not _MUSA_QWEN_IDENTITY_LOGITS_VIEW_ENABLED:
        return
    GPUModelRunner._musa_select_sample_hidden_states = (  # type: ignore[attr-defined]
        select_qwen_identity_logits_view
    )


install_hooks()
