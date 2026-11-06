# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker


class MTGPUWorker(Worker):
    """A worker class that executes (a partition of) the model on a MTGPU.
    Each worker is associated with a single MTGPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

    def _make_model_runner(self) -> Any:
        model_runner = super()._make_model_runner()
        if not self.use_v2_model_runner:
            return model_runner
        from vllm_musa.v1.worker.request_token_staging import (
            preallocate_deepseek_request_token_staging,
        )

        preallocate_deepseek_request_token_staging(self.vllm_config, model_runner)
        return model_runner
