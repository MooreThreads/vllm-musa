# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.v1.worker.gpu_worker to support MUSA device type.

The vLLM V1 GPU worker only checks for device_config.device.type == "cuda",
which doesn't match for MUSA devices. This patch extends the check
to also accept "musa" device type.

Note: No patch is needed for torch.device("cuda:X") because torchada
automatically aliases it to torch.device("musa:X") when imported.
"""

PATCHES = [
    # Patch init_device to support musa device type (V1 worker)
    # For vLLM 0.15+
    (
        'if self.device_config.device_type == "cuda":',
        'if self.device_config.device_type in ("cuda", "musa"):',
    ),
    (
        "torch.cuda.is_available()",
        "torch.musa.is_available()",
    ),
    # Patch torch profiler activities to support musa
    (
        'activities=["CPU", "CUDA"],',
        'activities=["CPU", "MUSA"],',
    ),
    # Patch torch.accelerator.set_device_index to current_platform.set_device
    (
        'torch.accelerator.set_device_index(self.device)',
        'current_platform.set_device(self.device)'
    )
]
