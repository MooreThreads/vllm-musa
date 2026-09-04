# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker

from vllm_musa.tuning import prime_musa_kernel_hardware


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

    def init_device(self) -> None:
        super().init_device()
        device_index = self.device.index
        prime_musa_kernel_hardware(0 if device_index is None else device_index)

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(
            self.model_runner.uniform_decode_query_len,
            uniform_decode=True,
        )

    def get_musa_cudagraph_runtime_state(self) -> dict[str, object]:
        """Return the resolved, msgpack-safe cudagraph state for evidence."""
        runner = self.model_runner
        dispatcher = getattr(runner, "cudagraph_dispatcher", None)
        configured_mode = getattr(
            getattr(runner, "compilation_config", None), "cudagraph_mode", None
        )
        if dispatcher is None:
            return {
                "rank": self.rank,
                "local_rank": self.local_rank,
                "configured_cudagraph_mode": getattr(
                    configured_mode, "name", str(configured_mode)
                ),
                "resolved_cudagraph_mode": "NONE",
                "keys_initialized": False,
                "capture_descriptors": {},
            }
        capture_descriptors: dict[str, list[dict[str, object]]] = {}
        for runtime_mode, descriptors in dispatcher.get_capture_descs():
            capture_descriptors[runtime_mode.name] = [
                {
                    "num_tokens": descriptor.num_tokens,
                    "num_reqs": descriptor.num_reqs,
                    "uniform": descriptor.uniform,
                    "has_lora": descriptor.has_lora,
                    "num_active_loras": descriptor.num_active_loras,
                }
                for descriptor in descriptors
            ]
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "configured_cudagraph_mode": getattr(
                configured_mode, "name", str(configured_mode)
            ),
            "resolved_cudagraph_mode": dispatcher.cudagraph_mode.name,
            "keys_initialized": dispatcher.keys_initialized,
            "capture_descriptors": capture_descriptors,
        }
