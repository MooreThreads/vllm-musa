from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def preallocate_deepseek_request_token_staging(
    vllm_config: VllmConfig, model_runner: Any
) -> None:
    model_config = vllm_config.model_config
    if model_config is None or getattr(
        model_config.hf_text_config, "model_type", None
    ) not in {"deepseek_v4", "deepseek_v41", "deepseek_v41_text"}:
        return

    all_token_ids = model_runner.req_states.all_token_ids
    if all_token_ids.write_contents is None:
        return

    from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool

    all_token_ids.write_contents = UvaBufferPool(
        all_token_ids.gpu.numel(),
        all_token_ids.dtype,
        max_concurrency=all_token_ids.max_concurrency,
    )
