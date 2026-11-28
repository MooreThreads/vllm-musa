# SPDX-License-Identifier: Apache-2.0
"""Expose v0.34 warmup step helpers on pinned ``vllm.v1.worker.gpu.warmup``."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

ADAPTED_RUN_STEP = "__vllm_musa_warmup_run_step_bridge__"
ADAPTED_TOKEN_IDS = "__vllm_musa_warmup_token_ids_bridge__"


def _warmup_token_ids(model_runner: Any, num_tokens: int) -> list[int]:
    vocab_size = int(model_runner.model_config.get_vocab_size())
    return [i % vocab_size for i in range(num_tokens)]


def _run_step(
    worker_execute_model: Callable[[Any], Any],
    worker_sample_tokens: Callable[[Any], Any] | None,
    scheduler_output: Any,
) -> None:
    import torch

    worker_execute_model(scheduler_output)
    if worker_sample_tokens is not None:
        worker_sample_tokens(None)
    torch.accelerator.synchronize()


def is_applied(module: ModuleType) -> list[str]:
    applied: list[str] = []
    run_step = getattr(module, ADAPTED_RUN_STEP, None)
    if run_step is not None and getattr(module, "_run_step", None) is run_step:
        applied.append("_run_step")
    token_ids = getattr(module, ADAPTED_TOKEN_IDS, None)
    if (
        token_ids is not None
        and getattr(module, "_warmup_token_ids", None) is token_ids
    ):
        applied.append("_warmup_token_ids")
    return applied


def is_satisfied(module: ModuleType) -> bool:
    return callable(getattr(module, "_run_step", None)) and callable(
        getattr(module, "_warmup_token_ids", None)
    )


def apply(module: ModuleType) -> list[str]:
    applied: list[str] = []
    if not hasattr(module, "_run_step"):
        module._run_step = _run_step
        setattr(module, ADAPTED_RUN_STEP, _run_step)
        applied.append("_run_step")
    if not hasattr(module, "_warmup_token_ids"):
        module._warmup_token_ids = _warmup_token_ids
        setattr(module, ADAPTED_TOKEN_IDS, _warmup_token_ids)
        applied.append("_warmup_token_ids")
    return applied
