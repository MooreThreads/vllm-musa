# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker

from vllm_musa.platform import materialize_fused_moe_runtime_policy_for_worker


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
        # RuntimePlan dispatch state is process-local.  Materialize it from the
        # serialized VllmConfig before GPUModelRunner imports/compiles the MoE
        # dispatcher; relying on parent environment inheritance is not safe
        # across vLLM's EngineCore/TP-worker spawn boundaries.
        materialize_fused_moe_runtime_policy_for_worker(
            vllm_config,
            worker_rank=rank,
        )
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(
            self.model_runner.uniform_decode_query_len,
            uniform_decode=True,
        )
