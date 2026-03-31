# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration and fixtures for MUSA platform tests."""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_pymtml():
    """Fixture that mocks the pymtml module with pynvml-compatible API.

    This allows testing musa.py functionality without requiring
    actual MUSA hardware or the pymtml library.
    """
    mock_module = MagicMock()

    # Mock device handle
    mock_handle = MagicMock()

    # Mock memory info (pynvml-compatible)
    mock_memory_info = MagicMock()
    mock_memory_info.total = 80 * 1024 * 1024 * 1024  # 80GB

    # Configure pynvml-compatible mock functions
    mock_module.nvmlInit.return_value = None
    mock_module.nvmlShutdown.return_value = None
    mock_module.nvmlDeviceGetCount.return_value = 2
    mock_module.nvmlDeviceGetHandleByIndex.return_value = mock_handle
    mock_module.nvmlDeviceGetName.return_value = "MTT S80"
    mock_module.nvmlDeviceGetUUID.return_value = (
        "GPU-12345678-1234-1234-1234-123456789abc"
    )
    mock_module.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info
    mock_module.nvmlDeviceGetCudaComputeCapability.return_value = (3, 1)
    mock_module.nvmlDeviceGetP2PStatus.return_value = 0  # NVML_P2P_STATUS_OK

    # Constants
    mock_module.NVML_P2P_CAPS_INDEX_NVLINK = 0
    mock_module.NVML_P2P_STATUS_OK = 0
    mock_module.NVMLError = Exception

    # Inject mock into sys.modules before importing musa
    original_pymtml = sys.modules.get("pymtml")
    sys.modules["pymtml"] = mock_module

    yield mock_module

    # Restore original (or remove if it wasn't present)
    if original_pymtml is not None:
        sys.modules["pymtml"] = original_pymtml
    else:
        sys.modules.pop("pymtml", None)


@pytest.fixture
def mock_torch():
    """Fixture that provides a mock torch module for testing.

    This is useful for testing code that depends on torch.cuda/torch.musa
    without requiring actual GPU hardware.
    """
    mock_module = MagicMock()

    # Mock device properties
    mock_props = MagicMock()
    mock_props.total_memory = 80 * 1024 * 1024 * 1024  # 80GB
    mock_props.name = "MTT S80"

    # Mock CUDA (which torchada redirects to MUSA)
    mock_module.cuda.device_count.return_value = 2
    mock_module.cuda.get_device_capability.return_value = (3, 1)
    mock_module.cuda.get_device_name.return_value = "MTT S80"
    mock_module.cuda.get_device_properties.return_value = mock_props
    mock_module.cuda.empty_cache.return_value = None
    mock_module.cuda.reset_peak_memory_stats.return_value = None
    mock_module.cuda.max_memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB

    # Mock MUSA
    mock_module.musa = MagicMock()
    mock_module.musa.set_device.return_value = None

    # Mock device creation
    mock_module.device.return_value = MagicMock(index=0, type="musa")
    mock_module.zeros.return_value = MagicMock()

    # Mock dtypes
    mock_module.bfloat16 = "bfloat16"
    mock_module.float16 = "float16"
    mock_module.float32 = "float32"

    return mock_module


@pytest.fixture
def reset_musa_state():
    """Fixture to reset global MUSA platform state between tests."""
    import vllm_musa.platform as musa_module

    # Store original values
    original_cache_ops = musa_module._musa_cache_ops_registered

    yield

    # Restore original values
    musa_module._musa_cache_ops_registered = original_cache_ops
