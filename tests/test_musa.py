# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MUSA Platform implementation."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestMUSAPlatformBase:
    """Tests for MUSAPlatformBase class."""

    def test_device_name(self):
        """Test that device_name is set correctly."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.device_name == "musa"

    def test_device_type(self):
        """Test that device_type is set correctly."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.device_type == "musa"

    def test_dispatch_key(self):
        """Test that dispatch_key uses MUSA."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.dispatch_key == "MUSA"

    def test_dist_backend(self):
        """Test that dist_backend uses mccl."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.dist_backend == "mccl"

    def test_device_control_env_var(self):
        """Test that device_control_env_var is set correctly."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.device_control_env_var == "MUSA_VISIBLE_DEVICES"

    def test_ray_device_key(self):
        """Test that ray_device_key is set correctly."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.ray_device_key == "GPU"

    def test_is_cuda_alike_returns_true(self):
        """Test that is_cuda_alike returns True for MUSA."""
        from vllm_musa.platform import MUSAPlatformBase

        platform = MUSAPlatformBase()
        assert platform.is_cuda_alike() is True

    def test_is_sleep_mode_available_returns_true(self):
        """Test that is_sleep_mode_available returns True."""
        from vllm_musa.platform import MUSAPlatformBase

        platform = MUSAPlatformBase()
        assert platform.is_sleep_mode_available() is True

    def test_supported_dtypes(self):
        """Test that supported_dtypes includes bf16, fp16, and fp32."""
        import torch

        from vllm_musa.platform import MUSAPlatformBase

        platform = MUSAPlatformBase()
        dtypes = platform.supported_dtypes

        assert torch.bfloat16 in dtypes
        assert torch.float16 in dtypes
        assert torch.float32 in dtypes

    def test_opaque_attention_op_returns_true(self):
        """Test that opaque_attention_op returns True."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.opaque_attention_op() is True

    def test_use_custom_allreduce_returns_true(self):
        """Test that use_custom_allreduce returns True."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.use_custom_allreduce() is True

    def test_supports_fp8_for_musa_3_1(self):
        """Test that FP8 is supported on MUSA capability 3.1."""
        from vllm.platforms.interface import DeviceCapability

        from vllm_musa.platform import MUSAPlatformBase

        with patch.object(
            MUSAPlatformBase,
            "get_device_capability",
            return_value=DeviceCapability(3, 1),
        ):
            assert MUSAPlatformBase.supports_fp8() is True

    def test_supports_fp8_rejects_pre_3_1(self):
        """Test that pre-3.1 MUSA capability does not support FP8."""
        from vllm.platforms.interface import DeviceCapability

        from vllm_musa.platform import MUSAPlatformBase

        with patch.object(
            MUSAPlatformBase,
            "get_device_capability",
            return_value=DeviceCapability(3, 0),
        ):
            assert MUSAPlatformBase.supports_fp8() is False

    def test_support_hybrid_kv_cache(self):
        """Test that support_hybrid_kv_cache returns True."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.support_hybrid_kv_cache() is True

    def test_support_static_graph_mode(self):
        """Test that support_static_graph_mode returns True."""
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.support_static_graph_mode() is True

    def test_get_punica_wrapper(self):
        """Test get_punica_wrapper returns correct path."""
        from vllm_musa.platform import MUSAPlatformBase

        result = MUSAPlatformBase.get_punica_wrapper()
        assert result == "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    def test_get_device_communicator_cls(self):
        """Test get_device_communicator_cls returns CUDA communicator."""
        from vllm_musa.platform import MUSAPlatformBase

        result = MUSAPlatformBase.get_device_communicator_cls()
        expected = (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"
        )
        assert result == expected

    def test_get_static_graph_wrapper_cls(self):
        """Test get_static_graph_wrapper_cls returns CUDA graph wrapper."""
        from vllm_musa.platform import MUSAPlatformBase

        result = MUSAPlatformBase.get_static_graph_wrapper_cls()
        assert result == "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    def test_register_attention_backends_overrides_turboquant(self):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_musa.platform import register_attention_backends

        AttentionBackendEnum.TURBOQUANT.clear_override()
        register_attention_backends()

        assert (
            AttentionBackendEnum.TURBOQUANT.get_path()
            == "vllm_musa.v1.attention.backends.turboquant.MUSATurboQuantAttentionBackend"
        )

    def test_get_valid_backends_includes_turboquant_for_non_mla(self):
        from vllm.platforms.interface import DeviceCapability
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_musa.platform import _get_backend_priorities

        priorities = _get_backend_priorities(
            use_mla=False,
            device_capability=DeviceCapability(3, 1),
        )

        assert AttentionBackendEnum.TURBOQUANT in priorities
        assert AttentionBackendEnum.TURBOQUANT not in _get_backend_priorities(
            use_mla=True,
            device_capability=DeviceCapability(3, 1),
        )

    def test_turboquant_forces_fp8_e4b15_off_on_musa(self):
        from vllm.v1.attention.backends import turboquant_attn
        from vllm.v1.attention.ops import (
            triton_turboquant_decode,
            triton_turboquant_store,
        )

        import vllm_musa.v1.attention.backends.turboquant  # noqa: F401

        assert triton_turboquant_decode._use_fp8_e4b15(0) == 0
        assert triton_turboquant_store._use_fp8_e4b15(0) == 0
        assert turboquant_attn._use_fp8_e4b15(0) == 0
        assert (
            triton_turboquant_store._use_fp8_e4b15
            is triton_turboquant_decode._use_fp8_e4b15
        )
        assert turboquant_attn._use_fp8_e4b15 is triton_turboquant_decode._use_fp8_e4b15

    def test_turboquant_supports_k8v4_on_musa(self):
        import torch
        from vllm.platforms.interface import DeviceCapability

        from vllm_musa.v1.attention.backends.turboquant import (
            MUSATurboQuantAttentionBackend,
        )

        reason = MUSATurboQuantAttentionBackend.supports_combination(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="turboquant_k8v4",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            device_capability=DeviceCapability(3, 1),
        )

        assert reason is None


class TestNativeGemvSource:
    """Source-level checks for native MUSA GEMV dispatch gates."""

    def test_qwen_fp8_moe_uses_32x4_shape_gate(self):
        source = Path("csrc/musa/gemv.mu").read_text()

        assert "ShouldUseQwenFp8Moe32x4(" in source
        assert "hidden_size == 2048 && nr_n == 768" in source
        assert "hidden_size == 768 && nr_n == 2048" in source
        assert "BlockConfig qwen_fp8_moe_config{32, 4" in source
        assert "case 4: GEN_LAUNCH_KERN(32, 4)" in source

    def test_deepseek_fp8_w1_uses_32x4_shape_gate(self):
        source = Path("csrc/musa/gemv.mu").read_text()

        assert "kDeepSeekFp8W1BlockEnv" in source
        assert '"VLLM_MUSA_DEEPSEEK_FP8_W1_32X4"' in source
        assert "ShouldUseDeepSeekFp8W1Moe32x4(" in source
        assert "topk == 6" in source
        assert "hidden_size == 2048" in source
        assert "reduce_size == 2816" in source
        assert "num_experts == 64" in source
        assert "BlockConfig deepseek_fp8_w1_config{32, 4" in source
        assert "best_config = &deepseek_fp8_w1_config" in source

    def test_gemv_block_override_validates_env_config(self):
        source = Path("csrc/musa/gemv.mu").read_text()

        assert 'kGemvMoeBlockEnv = "VLLM_MUSA_GEMV_MOE_BLOCK"' in source
        assert "std::getenv(kGemvMoeBlockEnv)" in source
        assert "must use '<block_n>x<block_k>'" in source
        assert "IsForcedBlockConfigValid(forced_config" in source


class TestNonMtmlMUSAPlatform:
    """Tests for NonMtmlMUSAPlatform class."""

    def test_get_device_capability(self):
        """Test get_device_capability returns DeviceCapability."""
        with patch("torch.cuda.get_device_capability") as mock_cap:
            mock_cap.return_value = (3, 1)

            from vllm_musa.platform import NonMtmlMUSAPlatform

            # Clear cache to allow re-testing
            NonMtmlMUSAPlatform.get_device_capability.cache_clear()

            cap = NonMtmlMUSAPlatform.get_device_capability(0)

            assert cap.major == 3
            assert cap.minor == 1

    def test_get_device_name(self):
        """Test get_device_name returns device name."""
        with patch("torch.cuda.get_device_name") as mock_name:
            mock_name.return_value = "MTT S80"

            from vllm_musa.platform import NonMtmlMUSAPlatform

            name = NonMtmlMUSAPlatform.get_device_name(0)

            assert name == "MTT S80"

    def test_get_device_total_memory(self):
        """Test get_device_total_memory returns memory size."""
        mock_props = MagicMock()
        mock_props.total_memory = 80 * 1024 * 1024 * 1024  # 80GB

        with patch("torch.cuda.get_device_properties") as mock_get_props:
            mock_get_props.return_value = mock_props

            from vllm_musa.platform import NonMtmlMUSAPlatform

            memory = NonMtmlMUSAPlatform.get_device_total_memory(0)

            assert memory == 80 * 1024 * 1024 * 1024

    def test_is_fully_connected_returns_false_with_warning(self):
        """Test is_fully_connected returns False without MTML."""
        from vllm_musa.platform import NonMtmlMUSAPlatform

        result = NonMtmlMUSAPlatform.is_fully_connected([0, 1])

        assert result is False


class TestWithMtmlContext:
    """Tests for the with_mtml_context decorator."""

    def test_decorator_returns_function_result(self):
        """Test that the decorator returns the wrapped function's result."""
        from vllm_musa.platform import mtml_available, with_mtml_context

        if not mtml_available:
            pytest.skip("MTML not available")

        @with_mtml_context
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_decorator_preserves_function_name(self):
        """Test that the decorator preserves the wrapped function's name."""
        from vllm_musa.platform import with_mtml_context

        @with_mtml_context
        def my_test_function():
            return "test"

        assert my_test_function.__name__ == "my_test_function"


class TestMtmlMUSAPlatform:
    """Tests for MtmlMUSAPlatform class."""

    def test_get_device_capability_returns_3_1(self, mock_pymtml):
        """Test get_device_capability returns (3, 1) for FP8 support."""
        if "vllm_musa.platform" in sys.modules:
            del sys.modules["vllm_musa.platform"]

        from vllm_musa.platform import MtmlMUSAPlatform, mtml_available

        if not mtml_available:
            pytest.skip("MTML not available")

        # Clear cache
        MtmlMUSAPlatform.get_device_capability.cache_clear()

        cap = MtmlMUSAPlatform.get_device_capability(0)

        assert cap.major == 3
        assert cap.minor == 1

    def test_get_device_name(self):
        """Test get_device_name returns a string."""
        from vllm_musa.platform import MtmlMUSAPlatform, mtml_available

        if not mtml_available:
            pytest.skip("MTML not available")

        name = MtmlMUSAPlatform.get_device_name(0)

        assert isinstance(name, str)
        assert len(name) > 0
        # MUSA device names typically start with "MTT"
        assert "MTT" in name or len(name) > 0

    def test_get_device_uuid(self):
        """Test get_device_uuid returns a valid UUID string."""
        from vllm_musa.platform import MtmlMUSAPlatform, mtml_available

        if not mtml_available:
            pytest.skip("MTML not available")

        uuid = MtmlMUSAPlatform.get_device_uuid(0)

        assert isinstance(uuid, str)
        # UUIDs have a specific format with dashes
        assert "-" in uuid
        assert len(uuid) >= 32  # Minimum UUID length

    def test_get_device_total_memory(self):
        """Test get_device_total_memory returns a positive integer."""
        from vllm_musa.platform import MtmlMUSAPlatform, mtml_available

        if not mtml_available:
            pytest.skip("MTML not available")

        memory = MtmlMUSAPlatform.get_device_total_memory(0)

        assert isinstance(memory, int)
        assert memory > 0
        # Typical GPU memory is at least 4GB
        assert memory >= 4 * 1024 * 1024 * 1024


class TestPlatformSelection:
    """Tests for platform autodetection."""

    def test_musa_platform_is_one_of_two_options(self):
        """Test that MUSAPlatform is either MtmlMUSAPlatform or NonMtmlMUSAPlatform."""
        from vllm_musa.platform import (
            MtmlMUSAPlatform,
            MUSAPlatform,
            NonMtmlMUSAPlatform,
        )

        assert MUSAPlatform in (MtmlMUSAPlatform, NonMtmlMUSAPlatform)

    def test_platform_selection_based_on_mtml_availability(self):
        """Test that platform selection is correct based on MTML availability."""
        from vllm_musa.platform import (
            MtmlMUSAPlatform,
            MUSAPlatform,
            NonMtmlMUSAPlatform,
            mtml_available,
        )

        if mtml_available:
            assert MUSAPlatform is MtmlMUSAPlatform
        else:
            assert MUSAPlatform is NonMtmlMUSAPlatform


class TestImportTorchada:
    """Tests for torchada import handling."""

    def test_torchada_is_imported(self):
        """Test that torchada is imported when musa module loads."""
        # torchada should be available in sys.modules after importing musa
        import vllm_musa.platform  # noqa: F401

        assert "torchada" in sys.modules


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined and contains expected items."""
        from vllm_musa import platform

        assert hasattr(platform, "__all__")

        expected_exports = [
            "MUSAPlatform",
            "MUSAPlatformBase",
            "MtmlMUSAPlatform",
            "NonMtmlMUSAPlatform",
            "with_mtml_context",
            "mtml_available",
        ]

        for export in expected_exports:
            assert export in platform.__all__, f"{export} not in __all__"
            assert hasattr(platform, export), f"{export} not defined in module"

    def test_musa_platform_plugin_function_exists(self):
        """Test that musa_platform_plugin function exists for entry point."""
        from vllm_musa import musa_platform_plugin

        assert callable(musa_platform_plugin)
