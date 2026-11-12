# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker

from vllm_musa.native_arch import check_device_arch


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

    def init_device(self):
        super().init_device()
        # Before the model loads: refuse a device the native build cannot run on.
        check_device_arch(self.device)
