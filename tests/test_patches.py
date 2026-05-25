# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MUSA platform patches module."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class TestPatchFileNaming:
    """Tests for patch file naming convention."""

    def test_get_patch_files_returns_correct_module_names(self):
        """Test that patch file names are correctly converted to module names."""
        from vllm_musa.patches import _get_patch_files

        patch_files = _get_patch_files()

        # Should find the triton unified attention patch
        module_names = [name for name, path in patch_files]

        assert "vllm.v1.attention.ops.triton_unified_attention" in module_names

    def test_naming_convention_double_underscore_to_dot(self):
        """Test that double underscores are converted to dots."""
        from vllm_musa.patches import _get_patch_files

        patch_files = _get_patch_files()

        for module_name, path in patch_files:
            # Module names should not contain double underscores
            assert "__" not in module_name
            # Should have proper Python module path format
            assert module_name.startswith("vllm")


class TestPatchFileLoading:
    """Tests for patch file content loading."""

    def test_load_patch_config_returns_patches_list(self):
        """Test that _load_patch_config extracts PATCHES list."""
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        # Find the triton patch file
        for module_name, patch_path in patch_files:
            if "triton_unified_attention" in module_name:
                patches = _load_patch_config(patch_path)

                assert isinstance(patches, list)
                assert len(patches) > 0

                # Each patch should be a tuple of (old, new)
                for old, new in patches:
                    assert isinstance(old, str)
                    assert isinstance(new, str)

    def test_load_patch_config_handles_missing_patches_list(self, tmp_path):
        """Test that _load_patch_config handles files without PATCHES."""
        from vllm_musa.patches import _load_patch_config

        # Create a temporary patch file without PATCHES
        patch_file = tmp_path / "test.patch.py"
        patch_file.write_text("# No PATCHES defined\nFOO = 'bar'\n")

        patches = _load_patch_config(patch_file)

        assert patches == []


class TestTritonPatch:
    """Tests for the Triton unified attention patch."""

    def test_patch_file_exists(self):
        """Test that the triton patch file exists."""
        from vllm_musa.patches import _get_patch_files

        patch_files = _get_patch_files()
        module_names = [name for name, path in patch_files]

        assert "vllm.v1.attention.ops.triton_unified_attention" in module_names

    def test_patch_contains_annotated_assignment_fix(self):
        """Test that patch contains the annotated assignment fix."""
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if "triton_unified_attention" in module_name:
                patches = _load_patch_config(patch_path)

                # Should have the fix for "left: tl.int32 = 0"
                old_strs = [old for old, new in patches]

                assert "left: tl.int32 = 0" in old_strs


class TestCustomAllReducePatch:
    """Tests for the MUSA custom all-reduce patch."""

    def test_patch_skips_cuda_p2p_check_on_musa(self):
        """Test that MUSA custom all-reduce does not run CUDA P2P probing."""
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if "custom_all_reduce" in module_name:
                patches = _load_patch_config(patch_path)
                old_strs = [old for old, new in patches]
                new_strs = [new for old, new in patches]

                assert (
                    "if ( not current_platform.is_rocm() or not "
                    "current_platform.is_musa() ) and not _can_p2p(rank, world_size):"
                ) in old_strs
                assert (
                    "if not current_platform.is_rocm() and not "
                    "current_platform.is_musa() and not _can_p2p(rank, world_size):"
                ) in new_strs
                assert all(
                    "not current_platform.is_rocm() or not current_platform.is_musa()"
                    not in new
                    for new in new_strs
                )
                break
        else:
            raise AssertionError("custom_all_reduce patch file was not found")


class TestSamplerPatch:
    """Tests for the MUSA top-k/top-p sampler patch."""

    def test_sampler_patch_keeps_triton_guard_and_fast_path_gate(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.v1.sample.ops.topk_topp_sampler":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "not current_platform.is_musa()" in new_source
                assert "VLLM_MUSA_SAMPLER_FAST_PATH" in new_source
                assert "_apply_top_k_top_p_musa_topk_prefilter" in new_source
                assert "logits.shape[0] >= 16" in new_source
                assert "logits.shape[1] >= 65536" in new_source
                break
        else:
            raise AssertionError("topk_topp_sampler patch file was not found")


class TestDeepGemmPatch:
    """Tests for the MUSA DeepGEMM compatibility patch."""

    def test_deep_gemm_patch_disables_unsupported_e8m0_on_musa(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.utils.deep_gemm":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "current_platform.is_musa()" in new_source
                assert "DeepGEMM E8M0 disabled on MUSA" in new_source
                assert "grouped FP8 UE8M0 cast is " in new_source
                assert "not supported by the MUSA DeepGEMM backend" in new_source
                break
        else:
            raise AssertionError("deep_gemm patch file was not found")


class TestCompilationBackendPatch:
    """Tests for MUSA torch.compile backend compatibility patches."""

    def test_vllm_backend_patch_accepts_torch_compile_options(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.compilation.backends":
                patches = _load_patch_config(patch_path)
                old_source = "\n".join(old for old, _ in patches)
                new_source = "\n".join(new for _, new in patches)

                assert "example_inputs: Sequence[Any]) -> Any" in old_source
                assert "**kwargs: Any" in new_source
                assert "autograd_cache_normalize_inputs=True" in old_source
                assert "functorch_cache_key_ctx" in new_source
                assert "hasattr(" in new_source
                break
        else:
            raise AssertionError("compilation backend patch file was not found")

    def test_live_vllm_backend_patch_ignores_options_kwarg(self, monkeypatch):
        import vllm_musa

        class DummyBackend:
            def __call__(self, graph, example_inputs):
                return graph, example_inputs

        class DummyModule:
            VllmBackend = DummyBackend

        monkeypatch.setitem(
            __import__("sys").modules,
            "vllm.compilation.backends",
            DummyModule,
        )

        vllm_musa._patch_vllm_backend_call_options()

        backend = DummyBackend()
        assert backend("graph", ["input"], options={"ignored": True}) == (
            "graph",
            ["input"],
        )

    def test_live_functorch_config_patch_skips_missing_keys(self):
        from contextlib import nullcontext

        import vllm_musa

        calls = []

        def original_patch(*args, **kwargs):
            calls.append((args, kwargs))
            return nullcontext()

        functorch_config = SimpleNamespace(existing_key=True)
        patched = vllm_musa._make_config_patch_filter(original_patch, functorch_config)

        with patched(missing_key=False):
            pass
        with patched("missing_key", True):
            pass
        with patched({"existing_key": True, "missing_key": False}):
            pass
        with patched(existing_key=True):
            pass

        assert calls == [
            (({"existing_key": True},), {}),
            ((), {"existing_key": True}),
        ]


class TestCompilationCachingPatch:
    """Tests for MUSA torch compile-cache compatibility patches."""

    def test_graph_pickler_options_patch_supports_torch_27(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.compilation.caching":
                patches = _load_patch_config(patch_path)
                old_source = "\n".join(old for old, _ in patches)
                new_source = "\n".join(new for _, new in patches)

                assert "GraphPickler, Options" in old_source
                assert 'getattr(_vllm_graph_pickler, "Options", None)' in new_source
                assert "_musa_graph_pickler_dumps" in new_source
                break
        else:
            raise AssertionError("compilation caching patch file was not found")


class TestCompilationCompilerInterfacePatch:
    """Tests for MUSA torch compiler-interface compatibility patches."""

    def test_functorch_config_patch_filters_missing_torch_keys(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.compilation.compiler_interface":
                patches = _load_patch_config(patch_path)
                old_source = "\n".join(old for old, _ in patches)
                new_source = "\n".join(new for _, new in patches)

                assert 'cfg["bundled_autograd_cache"] = False' in old_source
                assert "hasattr(functorch_config, key)" in new_source
                break
        else:
            raise AssertionError("compilation compiler_interface patch was not found")

    def test_live_functorch_config_patch_filters_missing_keys(self, monkeypatch):
        import sys

        import vllm_musa

        class DummyCompilerInterface:
            @staticmethod
            def _get_vllm_functorch_config():
                return {"existing_key": True, "missing_key": False}

        monkeypatch.setitem(
            sys.modules,
            "vllm.compilation.compiler_interface",
            DummyCompilerInterface,
        )

        dummy_functorch_config = SimpleNamespace(existing_key=True)
        monkeypatch.setattr(
            vllm_musa,
            "_filter_existing_config",
            lambda config, functorch_config: {
                key: value
                for key, value in config.items()
                if hasattr(dummy_functorch_config, key)
            },
        )

        vllm_musa._patch_vllm_functorch_config()

        config = DummyCompilerInterface._get_vllm_functorch_config()
        assert config == {"existing_key": True}


class TestCompilationPiecewiseBackendPatch:
    """Tests for MUSA piecewise backend compile-cache compatibility patches."""

    def test_piecewise_backend_patch_skips_missing_bundled_cache_key(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.compilation.piecewise_backend":
                patches = _load_patch_config(patch_path)
                old_source = "\n".join(old for old, _ in patches)
                new_source = "\n".join(new for _, new in patches)

                assert '"bundled_autograd_cache", True' in old_source
                assert "functorch_cache_ctx" in new_source
                assert "nullcontext" in new_source
                break
        else:
            raise AssertionError("compilation piecewise_backend patch was not found")


class TestAttentionCompilePatch:
    """Tests for MUSA attention torch.compile compatibility patches."""

    def test_attention_output_shape_patch_avoids_torch_size_constructor(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.model_executor.layers.attention.attention":
                patches = _load_patch_config(patch_path)
                old_source = "\n".join(old for old, _ in patches)
                new_source = "\n".join(new for _, new in patches)

                assert "output_shape = torch.Size(" in old_source
                assert "output_shape = (num_tokens," in new_source
                assert "torch.Size(" not in new_source
                break
        else:
            raise AssertionError("attention compile patch file was not found")


class TestMUSAFlashAttentionReshapeCache:
    """Tests for MUSA FlashAttention reshape+cache dispatch guards."""

    def _load_fa_utils_with_musa_platform(self, monkeypatch, musa_ops_namespace):
        import vllm
        import vllm.platforms as vllm_platforms

        import vllm_musa

        monkeypatch.setenv("VLLM_MUSA_RESHAPE_CACHE_FLASH", "1")
        monkeypatch.setattr(
            vllm_platforms,
            "current_platform",
            SimpleNamespace(is_musa=lambda: True),
        )

        flash_attn = ModuleType("flash_attn_interface")
        flash_attn.flash_attn_varlen_func = object()
        flash_attn.flash_attn_with_kvcache = object()
        flash_attn.get_scheduler_metadata = object()
        monkeypatch.setitem(sys.modules, "flash_attn_interface", flash_attn)

        vllm_ops = ModuleType("vllm._custom_ops")
        vllm_ops.reshape_and_cache_flash = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "vllm._custom_ops", vllm_ops)
        monkeypatch.setattr(vllm, "_custom_ops", vllm_ops, raising=False)

        musa_custom_ops = ModuleType("vllm_musa._custom_ops")
        musa_custom_ops.musa_reshape_and_cache_flash_nhd = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "vllm_musa._custom_ops", musa_custom_ops)
        monkeypatch.setattr(vllm_musa, "_custom_ops", musa_custom_ops, raising=False)

        if musa_ops_namespace is None:
            torch_ops = SimpleNamespace()
        else:
            torch_ops = SimpleNamespace(_C_musa_ops=musa_ops_namespace)
        monkeypatch.setattr(torch, "ops", torch_ops)

        module_name = "vllm_musa.v1.attention.backends.fa_utils"
        previous_module = sys.modules.pop(module_name, None)
        try:
            module = importlib.import_module(module_name)
        finally:
            sys.modules.pop(module_name, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module
        return module

    def test_missing_musa_ops_namespace_disables_native_cache_path(self, monkeypatch):
        module = self._load_fa_utils_with_musa_platform(
            monkeypatch, musa_ops_namespace=None
        )

        assert module._HAS_NATIVE_RESHAPE_CACHE_FLASH is False

    def test_native_cache_path_requires_matching_cache_dtypes(self, monkeypatch):
        module = self._load_fa_utils_with_musa_platform(
            monkeypatch,
            musa_ops_namespace=SimpleNamespace(
                musa_reshape_and_cache_flash_nhd=object()
            ),
        )

        key = torch.empty((2, 4, 64), dtype=torch.float16)
        value = torch.empty((2, 4, 64), dtype=torch.float16)
        key_cache = torch.empty((1, 16, 4, 64), dtype=torch.float16)
        value_cache = torch.empty((1, 16, 4, 64), dtype=torch.float16)
        slot_mapping = torch.arange(2, dtype=torch.long)
        k_scale = torch.ones(1, dtype=torch.float32)
        v_scale = torch.ones(1, dtype=torch.float32)

        assert module._can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )
        assert not module._can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value.to(torch.bfloat16),
            key_cache,
            value_cache,
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )
        assert not module._can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value,
            key_cache.to(torch.bfloat16),
            value_cache,
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )
        assert not module._can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value,
            key_cache,
            value_cache.to(torch.bfloat16),
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )


class TestMUSANativeKernelReviewHardening:
    """Source-level tests for native MUSA kernel review fixes."""

    def test_fused_rmsnorm_forced_block_env_is_validated(self):
        source = (
            Path(__file__).parents[1] / "csrc/musa/fused_add_rmsnorm.mu"
        ).read_text()

        assert "forced_block == 128" in source
        assert "forced_block == 256" in source
        assert "forced_block == 512" in source
        assert "forced_block == 1024" in source
        assert (
            "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X must be one of "
            "128, 256, 512, or 1024"
        ) in source


class TestMUSAPlatformDefaults:
    """Tests for MUSA platform-level vLLM config defaults."""

    def _make_vllm_config(
        self,
        *,
        architectures=None,
        quantization="fp8",
        quantization_config=None,
        max_cudagraph_capture_size=None,
        cudagraph_capture_sizes=None,
        cudagraph_mode=None,
        tensor_parallel_size=1,
    ):
        from types import SimpleNamespace

        hf_config = SimpleNamespace(
            architectures=architectures,
            quantization_config=quantization_config,
        )
        return SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=architectures,
                hf_config=hf_config,
                quantization=quantization,
                use_mla=False,
                is_mm_prefix_lm=False,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tensor_parallel_size,
                worker_cls="auto",
            ),
            cache_config=SimpleNamespace(block_size=16),
            scheduler_config=SimpleNamespace(
                is_multimodal_model=False,
                disable_chunked_mm_input=False,
            ),
            compilation_config=SimpleNamespace(
                custom_ops=[],
                cudagraph_mode=cudagraph_mode,
                max_cudagraph_capture_size=max_cudagraph_capture_size,
                cudagraph_capture_sizes=cudagraph_capture_sizes,
            ),
        )

    def test_qwen3_moe_fp8_caps_default_cudagraph_capture_size(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["Qwen3MoeForCausalLM"],
        )

        MUSAPlatformBase.apply_config_platform_defaults(vllm_config)

        assert vllm_config.compilation_config.max_cudagraph_capture_size == 64
        assert vllm_config.compilation_config.custom_ops == ["all"]

    def test_qwen3_moe_fp8_preserves_user_cudagraph_capture_size(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["Qwen3MoeForCausalLM"],
            max_cudagraph_capture_size=128,
        )

        MUSAPlatformBase.apply_config_platform_defaults(vllm_config)

        assert vllm_config.compilation_config.max_cudagraph_capture_size == 128

    def test_qwen3_moe_fp8_preserves_user_cudagraph_capture_sizes(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["Qwen3MoeForCausalLM"],
            cudagraph_capture_sizes=[1, 2, 4, 8],
        )

        MUSAPlatformBase.apply_config_platform_defaults(vllm_config)

        assert vllm_config.compilation_config.max_cudagraph_capture_size is None
        assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]

    def test_tp4_disables_musa_cudagraph_capture(self):
        from vllm.config import CUDAGraphMode

        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            max_cudagraph_capture_size=512,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            tensor_parallel_size=4,
        )

        MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        assert vllm_config.compilation_config.max_cudagraph_capture_size == 0
        assert vllm_config.compilation_config.cudagraph_capture_sizes == []

    def test_tp2_keeps_musa_cudagraph_capture(self):
        from vllm.config import CUDAGraphMode

        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            max_cudagraph_capture_size=512,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            tensor_parallel_size=2,
        )

        MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
        assert vllm_config.compilation_config.max_cudagraph_capture_size == 512
        assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]

    def test_dense_fp8_does_not_cap_cudagraph_capture_size(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["Qwen3ForCausalLM"],
        )

        MUSAPlatformBase.apply_config_platform_defaults(vllm_config)

        assert vllm_config.compilation_config.max_cudagraph_capture_size is None


class TestMUSAFusedMoEFP8Scales:
    """Tests for MUSA FP8 MoE scale adaptation helpers."""

    def test_static_tensor_fp8_moe_scales_expand_to_block_layout(self):
        from vllm_musa.model_executor.layers.fused_moe.fused_moe import (
            _maybe_expand_fp8_moe_per_tensor_scale,
        )

        weight = torch.empty((2, 260, 320), dtype=torch.float8_e4m3fn)
        scale = torch.tensor([0.5, 1.5], dtype=torch.float32)

        expanded = _maybe_expand_fp8_moe_per_tensor_scale(scale, weight)

        assert expanded is not None
        assert expanded.shape == (2, 5, 5)
        assert expanded.is_contiguous()
        assert torch.all(expanded[0] == scale[0])
        assert torch.all(expanded[1] == scale[1])

    def test_static_tensor_fp8_moe_scales_prefer_128_block_layout(self):
        from vllm_musa.model_executor.layers.fused_moe.fused_moe import (
            _maybe_expand_fp8_moe_per_tensor_scale,
        )

        weight = torch.empty((2, 260, 256), dtype=torch.float8_e4m3fn)
        scale = torch.tensor([0.5, 1.5], dtype=torch.float32)

        expanded = _maybe_expand_fp8_moe_per_tensor_scale(scale, weight)

        assert expanded is not None
        assert expanded.shape == (2, 3, 2)
        assert expanded.is_contiguous()

    def test_block_fp8_moe_scales_are_left_unchanged(self):
        from vllm_musa.model_executor.layers.fused_moe.fused_moe import (
            _maybe_expand_fp8_moe_per_tensor_scale,
        )

        weight = torch.empty((2, 256, 256), dtype=torch.float8_e4m3fn)
        scale = torch.ones((2, 2, 2), dtype=torch.float32)

        expanded = _maybe_expand_fp8_moe_per_tensor_scale(scale, weight)

        assert expanded is scale


class TestMUSAFp8MoEPadding:
    """Tests for MUSA FP8 MoE TP padding helpers."""

    def test_musa_fp8_moe_tp_padding_rounds_partition_to_block_lcm(self, monkeypatch):
        from vllm_musa.model_executor.layers.quantization import fp8 as musa_fp8

        monkeypatch.setattr(
            musa_fp8,
            "current_platform",
            SimpleNamespace(is_musa=lambda: True),
        )
        monkeypatch.setattr(
            musa_fp8,
            "_ORIGINAL_FP8_MOE_MAYBE_ROUNDUP_SIZES",
            lambda self, hidden_size, intermediate_size_per_partition, act_dtype, moe_parallel_config: (
                hidden_size,
                intermediate_size_per_partition,
            ),
        )

        method = SimpleNamespace(block_quant=True, weight_block_size=(128, 128))

        assert musa_fp8.maybe_roundup_sizes(
            method,
            hidden_size=2048,
            intermediate_size_per_partition=704,
            act_dtype=torch.bfloat16,
            moe_parallel_config=SimpleNamespace(tp_size=2),
        ) == (2048, 768)
        assert musa_fp8.maybe_roundup_sizes(
            method,
            hidden_size=2048,
            intermediate_size_per_partition=704,
            act_dtype=torch.bfloat16,
            moe_parallel_config=SimpleNamespace(tp_size=1),
        ) == (2048, 704)

    @pytest.mark.skipif(
        not hasattr(torch, "float8_e4m3fn"),
        reason="FP8 tensor dtype is not available in this torch build",
    )
    def test_musa_fp8_moe_padded_weights_are_zero_initialized(self, monkeypatch):
        from vllm_musa.model_executor.layers.quantization import fp8 as musa_fp8

        monkeypatch.setattr(
            musa_fp8,
            "current_platform",
            SimpleNamespace(is_musa=lambda: True),
        )

        layer = SimpleNamespace(
            moe_config=SimpleNamespace(intermediate_size_per_partition_unpadded=704),
            prefix="model.layers.0.mlp",
        )

        def fake_create_weights(self, layer, **kwargs):
            layer.w13_weight = SimpleNamespace(
                data=torch.full((8,), 42, dtype=torch.uint8).view(torch.float8_e4m3fn)
            )
            layer.w2_weight = SimpleNamespace(
                data=torch.full((8,), 43, dtype=torch.uint8).view(torch.float8_e4m3fn)
            )

        monkeypatch.setattr(
            musa_fp8,
            "_ORIGINAL_FP8_MOE_CREATE_WEIGHTS",
            fake_create_weights,
        )

        musa_fp8.create_weights(
            SimpleNamespace(block_quant=True),
            layer=layer,
            num_experts=64,
            hidden_size=2048,
            intermediate_size_per_partition=768,
            params_dtype=torch.bfloat16,
        )

        assert torch.all(layer.w13_weight.data.view(torch.uint8) == 0)
        assert torch.all(layer.w2_weight.data.view(torch.uint8) == 0)


class TestScaledMMKernelPatch:
    """Tests for the MUSA scaled-mm kernel registry patch."""

    def test_musa_deepgemm_fp8_uses_custom_op_for_compile(self):
        source = (
            Path(__file__).parents[1]
            / "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py"
        ).read_text()

        assert "torch.ops.vllm.musa_deepgemm_fp8_op" in source
        assert "direct_register_custom_op(" in source
        assert '"musa_deepgemm_fp8_op"' in source
        assert "_musa_deepgemm_fp8_op_fake" in source
        assert "VLLM_MUSA_DEEPGEMM_ROW_MAJOR_ACT_SCALES" in source

    def test_musa_deepgemm_row_major_scale_gate(self, monkeypatch):
        from vllm_musa.model_executor.kernels.linear.scaled_mm import deep_gemm

        monkeypatch.setattr(
            deep_gemm.envs, "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES", False
        )
        monkeypatch.delenv("VLLM_MUSA_DEEPGEMM_ROW_MAJOR_ACT_SCALES", raising=False)

        assert deep_gemm._use_row_major_activation_scales(False) is True
        assert deep_gemm._use_row_major_activation_scales(True) is False

        monkeypatch.setenv("VLLM_MUSA_DEEPGEMM_ROW_MAJOR_ACT_SCALES", "0")
        assert deep_gemm._use_row_major_activation_scales(False) is False

        monkeypatch.setenv("VLLM_MUSA_DEEPGEMM_ROW_MAJOR_ACT_SCALES", "1")
        monkeypatch.setattr(
            deep_gemm.envs, "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES", True
        )
        assert deep_gemm._use_row_major_activation_scales(False) is True

    def test_musa_swiglu_uses_custom_op_for_compile(self):
        source = (
            Path(__file__).parents[1] / "vllm_musa/model_executor/layers/activation.py"
        ).read_text()

        assert "torch.ops.vllm.musa_swish_glu_op" in source
        assert "direct_register_custom_op(" in source
        assert '"musa_swish_glu_op"' in source
        assert "_musa_swish_glu_op_fake" in source

    def test_musa_torch_fp8_scaled_mm_disables_fp8_output_padding(self):
        from vllm_musa.model_executor.kernels.linear.scaled_mm.torch_scaled_mm import (
            MUSAPerTensorTorchFP8ScaledMMLinearKernel,
        )

        assert (
            MUSAPerTensorTorchFP8ScaledMMLinearKernel.get_output_padding(None) is None
        )

    def test_musa_unquantized_gemm_materializes_plain_parameter_for_compile(self):
        source = (
            Path(__file__).parents[1] / "vllm_musa/model_executor/layers/utils.py"
        ).read_text()

        assert "BasevLLMParameter" in source
        assert "torch.nn.Parameter(weight.detach(), requires_grad=False)" in source
        assert "plain_weight.__dict__.update(weight.__dict__)" in source
        assert "current_platform.is_musa()" in source
        assert "process_weights_after_loading" in source
        assert "_musa_materializes_plain_parameter" in source
        assert "DisableTorchFunction" not in source

    def test_scaled_mm_patch_registers_musa_fp8_kernel_fallbacks(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.model_executor.kernels.linear":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "possible_kernels is _POSSIBLE_FP8_BLOCK_KERNELS" in new_source
                assert "MUSADeepGemmFp8BlockScaledMMKernel" in new_source
                assert "possible_kernels is _POSSIBLE_FP8_KERNELS" in new_source
                assert "MUSAPerTensorTorchFP8ScaledMMLinearKernel" in new_source
                assert "MUSAChannelWiseTorchFP8ScaledMMLinearKernel" in new_source
                break
        else:
            raise AssertionError("linear kernel patch file was not found")


class TestMUSAFP8ActivationQuant:
    """Tests for MUSA FP8 activation quantization helpers."""

    def test_per_token_group_quant_accepts_strided_2d_input(self, monkeypatch):
        from vllm_musa.model_executor.layers.quantization.utils import fp8_utils

        calls = []

        def fake_quant(
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            use_ue8m0,
            column_major_scales,
            tma_aligned_scales,
        ):
            calls.append(
                {
                    "x": x,
                    "x_q": x_q,
                    "x_s": x_s,
                    "group_size": group_size,
                    "column_major_scales": column_major_scales,
                }
            )

        monkeypatch.setattr(
            fp8_utils,
            "current_platform",
            SimpleNamespace(
                is_musa=lambda: True,
                fp8_dtype=lambda: torch.float8_e4m3fn,
            ),
        )
        monkeypatch.setattr(
            torch.ops,
            "_C_musa_ops",
            SimpleNamespace(per_token_group_fp8_quant=fake_quant),
            raising=False,
        )

        base = torch.randn(4, 3, 128, dtype=torch.float32)
        x = base[:, 1, :]
        assert x.stride(-1) == 1
        assert not x.is_contiguous()

        x_q, x_s = fp8_utils.per_token_group_quant_fp8(
            x,
            group_size=128,
            use_ue8m0=False,
        )

        assert len(calls) == 1
        assert calls[0]["x"].is_contiguous()
        assert calls[0]["x"].shape == x.shape
        assert x_q.shape == x.shape
        assert x_q.is_contiguous()
        assert x_s.shape == (4, 1)

    def test_per_token_group_quant_rejects_noncontiguous_groups(self, monkeypatch):
        from vllm_musa.model_executor.layers.quantization.utils import fp8_utils

        monkeypatch.setattr(
            fp8_utils,
            "current_platform",
            SimpleNamespace(
                is_musa=lambda: True,
                fp8_dtype=lambda: torch.float8_e4m3fn,
            ),
        )

        x = torch.randn(128, 4, dtype=torch.float32).t()
        assert x.shape == (4, 128)
        assert x.stride(-1) != 1

        with pytest.raises(AssertionError, match="groups must be contiguous"):
            fp8_utils.per_token_group_quant_fp8(
                x,
                group_size=128,
                use_ue8m0=False,
            )


class TestApplyPatches:
    """Tests for the apply_patches function."""

    def test_apply_patches_is_idempotent(self):
        """Test that apply_patches can be called multiple times safely."""
        from vllm_musa import patches

        # Reset the flag
        patches._patches_applied = False

        # First call
        patches.apply_patches()
        assert patches._patches_applied is True

        # Second call should be a no-op
        patches.apply_patches()
        assert patches._patches_applied is True

    def test_apply_patches_handles_missing_module(self):
        """Test that apply_patches handles non-existent modules gracefully."""
        from vllm_musa import patches

        # Reset state
        patches._patches_applied = False

        # Create a mock patch file for a non-existent module
        with patch.object(patches, "_get_patch_files") as mock_get:
            mock_get.return_value = [("non.existent.module", Path("/fake/path"))]

            with patch.object(patches, "_load_patch_config") as mock_load:
                mock_load.return_value = [("old", "new")]

                # Should not raise
                patches.apply_patches()


class TestPatchesReadme:
    """Tests for patches documentation."""

    def test_readme_exists(self):
        """Test that README.md exists in patches directory."""
        patches_dir = Path(__file__).parent.parent / "vllm_musa" / "patches"
        readme_path = patches_dir / "README.md"

        assert readme_path.exists()

    def test_readme_documents_naming_convention(self):
        """Test that README explains the naming convention."""
        patches_dir = Path(__file__).parent.parent / "vllm_musa" / "patches"
        readme_path = patches_dir / "README.md"

        content = readme_path.read_text()

        # Should document the double underscore convention
        assert "__" in content or "double underscore" in content.lower()
        assert ".patch.py" in content
