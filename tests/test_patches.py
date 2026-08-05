# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MUSA platform patches module."""

import importlib
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class TestPatchFileNaming:
    """Tests for patch file naming convention."""

    def test_naming_convention_double_underscore_to_dot(self):
        """Test that double underscores are converted to dots."""
        from vllm_musa.patches import _get_patch_files

        patch_files = _get_patch_files()

        for module_name, path in patch_files:
            # Module names should not contain double underscores
            assert "__" not in module_name
            # Should have proper Python module path format (torch.*
            # config shims are object patches too, alongside the vllm.* ones)
            assert module_name.startswith(("vllm", "torch"))


class TestCustomOpsRuntimePatches:
    def test_rms_norm_wrapper_is_registered_on_vllm_custom_ops(self, monkeypatch):
        # the dflash fallback is now the vllm._custom_ops cat-6 object
        # patch; load + apply() it and assert it rebinds to the _shared helpers.
        import vllm

        from vllm_musa.patches import _get_patch_files, _load_patch_module, _shared

        vllm_ops = ModuleType("vllm._custom_ops")
        monkeypatch.setitem(sys.modules, "vllm._custom_ops", vllm_ops)
        monkeypatch.setattr(vllm, "_custom_ops", vllm_ops, raising=False)

        patch_file = next(f for m, f in _get_patch_files() if m == "vllm._custom_ops")
        _load_patch_module(patch_file).apply()

        assert vllm_ops.rms_norm is _shared.musa_safe_rms_norm
        assert getattr(vllm_ops.rms_norm, "_musa_safe_rms_norm") is True
        assert vllm_ops.rotary_embedding is _shared.musa_safe_rotary_embedding
        assert getattr(vllm_ops.rotary_embedding, "_musa_safe_rotary_embedding") is True

    def test_static_fp8_fallback_is_registered_on_vllm_custom_ops(self, monkeypatch):
        import vllm

        from vllm_musa.patches import _get_patch_files, _load_patch_module

        delegated = object()
        vllm_ops = ModuleType("vllm._custom_ops")
        vllm_ops.scaled_fp8_quant = lambda *args: delegated
        monkeypatch.setitem(sys.modules, "vllm._custom_ops", vllm_ops)
        monkeypatch.setattr(vllm, "_custom_ops", vllm_ops, raising=False)

        patch_file = next(f for m, f in _get_patch_files() if m == "vllm._custom_ops")
        patch_module = _load_patch_module(patch_file)
        monkeypatch.setattr(
            patch_module,
            "current_platform",
            SimpleNamespace(fp8_dtype=lambda: torch.float8_e4m3fn),
        )
        patch_module.apply()

        assert getattr(vllm_ops.scaled_fp8_quant, "_musa_static_fp8_fallback")
        assert vllm_ops.scaled_fp8_quant(torch.ones((1, 1))) is delegated

        values = torch.tensor([[2.0, 4.0], [4.0, 4.0]])
        scale = torch.tensor([1.0, 2.0])
        quantized, returned_scale = vllm_ops.scaled_fp8_quant(
            values, scale, group_shape=(-1, 1)
        )

        assert returned_scale is scale
        assert quantized.dtype == torch.float8_e4m3fn
        torch.testing.assert_close(
            quantized.float(),
            torch.tensor([[2.0, 2.0], [4.0, 2.0]]),
            atol=0.1,
            rtol=0.1,
        )

        saturated, _ = vllm_ops.scaled_fp8_quant(
            torch.tensor([[1000.0, -1000.0]]),
            torch.tensor(1.0),
        )
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        torch.testing.assert_close(
            saturated.float(),
            torch.tensor([[fp8_info.max, fp8_info.min]]),
        )
        assert torch.isfinite(saturated.float()).all()


class TestModelOptFp8CapabilityPatch:
    def test_modelopt_fp8_uses_musa_capability_floor(self):
        patch_path = (
            Path(__file__).parent.parent
            / "vllm_musa/patches/series/0099-MUSA-allow-ModelOpt-FP8-on-MUSA-capability.patch"
        )
        source = patch_path.read_text()

        assert "vllm/model_executor/layers/quantization/modelopt.py" in source
        assert "-        return 89" in source
        assert "+        return 31" in source


class TestCompilationBackendPatch:
    """Tests for MUSA torch.compile backend compatibility patches."""

    def test_live_vllm_backend_patch_ignores_options_kwarg(self, monkeypatch):
        # now the vllm.compilation.backends cat-6 object patch.
        from vllm_musa.patches import _get_patch_files, _load_patch_module

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

        patch_file = next(
            f for m, f in _get_patch_files() if m == "vllm.compilation.backends"
        )
        _load_patch_module(patch_file).apply()

        backend = DummyBackend()
        assert backend("graph", ["input"], options={"ignored": True}) == (
            "graph",
            ["input"],
        )

    def test_qwen2_presplit_keeps_baseline_split_and_hashes_helper(self, monkeypatch):
        from vllm_musa.compilation import qwen2_rope_kv_presplit as presplit
        from vllm_musa.optimization_contract import policy
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        patch_file = next(
            f for m, f in _get_patch_files() if m == "vllm.compilation.backends"
        )
        patch_module = _load_patch_module(patch_file)
        monkeypatch.setattr(policy, "prefers_feature", lambda *_args: True)
        monkeypatch.setattr(
            presplit, "qwen2_rope_kv_backend_supported", lambda _config: True
        )
        monkeypatch.setattr(
            presplit,
            "plan_qwen2_rope_kv_presplit",
            lambda _graph: tuple(range(24)),
        )
        monkeypatch.setattr(
            presplit,
            "apply_qwen2_rope_kv_presplit",
            lambda _graph, candidates: len(candidates),
        )
        compilation_config = SimpleNamespace(
            use_inductor_graph_partition=False,
            splitting_ops=["vllm::unified_kv_cache_update", "vllm::attention"],
            traced_files=set(),
        )
        backend = SimpleNamespace(
            vllm_config=object(), compilation_config=compilation_config
        )

        assert patch_module._try_qwen2_rope_kv_presplit(backend, object()) == 24
        assert compilation_config.splitting_ops == [
            "vllm::unified_kv_cache_update",
            "vllm::attention",
            "vllm::fused_rope_and_unified_kv_cache_update",
        ]
        assert str(Path(presplit.__file__).resolve()) in compilation_config.traced_files
        assert str(Path(patch_module.__file__).resolve()) in (
            compilation_config.traced_files
        )

    def test_qwen2_presplit_mismatch_keeps_baseline_config(self, monkeypatch):
        from vllm_musa.compilation import qwen2_rope_kv_presplit as presplit
        from vllm_musa.optimization_contract import policy
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        patch_file = next(
            f for m, f in _get_patch_files() if m == "vllm.compilation.backends"
        )
        patch_module = _load_patch_module(patch_file)
        monkeypatch.setattr(policy, "prefers_feature", lambda *_args: True)
        monkeypatch.setattr(
            presplit, "qwen2_rope_kv_backend_supported", lambda _config: True
        )
        monkeypatch.setattr(
            presplit, "plan_qwen2_rope_kv_presplit", lambda _graph: None
        )
        splitting_ops = ["vllm::unified_kv_cache_update", "vllm::attention"]
        compilation_config = SimpleNamespace(
            use_inductor_graph_partition=False,
            splitting_ops=splitting_ops,
            traced_files=set(),
        )
        backend = SimpleNamespace(
            vllm_config=object(), compilation_config=compilation_config
        )

        assert patch_module._try_qwen2_rope_kv_presplit(backend, object()) == 0
        assert compilation_config.splitting_ops is splitting_ops
        assert compilation_config.traced_files == set()

    def test_qwen2_presplit_rejects_unsupported_attention_backend(self, monkeypatch):
        from vllm_musa.compilation import qwen2_rope_kv_presplit as presplit
        from vllm_musa.optimization_contract import policy
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        patch_file = next(
            f for m, f in _get_patch_files() if m == "vllm.compilation.backends"
        )
        patch_module = _load_patch_module(patch_file)
        monkeypatch.setattr(policy, "prefers_feature", lambda *_args: True)
        monkeypatch.setattr(
            presplit, "qwen2_rope_kv_backend_supported", lambda _config: False
        )
        splitting_ops = ["vllm::unified_kv_cache_update", "vllm::attention"]
        compilation_config = SimpleNamespace(
            use_inductor_graph_partition=False,
            splitting_ops=splitting_ops,
            traced_files=set(),
        )
        backend = SimpleNamespace(
            vllm_config=object(), compilation_config=compilation_config
        )

        assert patch_module._try_qwen2_rope_kv_presplit(backend, object()) == 0
        assert compilation_config.splitting_ops is splitting_ops
        assert compilation_config.traced_files == set()

    @pytest.mark.parametrize("text_config_source", ["hf_text_config", "hf_config"])
    def test_qwen3_presplit_fuses_all_sites_and_hashes_helper(
        self,
        monkeypatch,
        text_config_source: str,
    ):
        from vllm_musa.compilation import qwen3_qk_rope_kv_presplit as presplit
        from vllm_musa.optimization_contract import policy
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        patch_file = next(
            f for m, f in _get_patch_files() if m == "vllm.compilation.backends"
        )
        patch_module = _load_patch_module(patch_file)
        monkeypatch.setattr(policy, "prefers_feature", lambda *_args: True)
        monkeypatch.setattr(
            presplit,
            "qwen3_qk_rope_kv_backend_supported",
            lambda _config, expected_sites: expected_sites == 28,
        )
        monkeypatch.setattr(
            presplit,
            "plan_qwen3_qk_rope_kv_presplit",
            lambda _graph, expected_sites: tuple(range(expected_sites)),
        )
        monkeypatch.setattr(
            presplit,
            "apply_qwen3_qk_rope_kv_presplit",
            lambda _graph, candidates: len(candidates),
        )
        compilation_config = SimpleNamespace(
            use_inductor_graph_partition=False,
            splitting_ops=["vllm::unified_kv_cache_update", "vllm::attention"],
            traced_files=set(),
        )
        text_config = SimpleNamespace(num_hidden_layers=28)
        model_config = (
            SimpleNamespace(hf_text_config=text_config)
            if text_config_source == "hf_text_config"
            else SimpleNamespace(hf_config=SimpleNamespace(text_config=text_config))
        )
        backend = SimpleNamespace(
            vllm_config=SimpleNamespace(model_config=model_config),
            compilation_config=compilation_config,
        )

        assert patch_module._try_qwen3_qk_rope_kv_presplit(backend, object()) == 28
        assert compilation_config.splitting_ops == [
            "vllm::unified_kv_cache_update",
            "vllm::attention",
            "vllm::musa_qwen3_qk_rope_and_unified_kv_cache_update",
        ]
        assert str(Path(presplit.__file__).resolve()) in compilation_config.traced_files
        assert str(Path(patch_module.__file__).resolve()) in (
            compilation_config.traced_files
        )

    def test_qwen3_presplit_missing_layer_count_is_fail_closed(self, monkeypatch):
        from vllm_musa.compilation import qwen3_qk_rope_kv_presplit as presplit
        from vllm_musa.optimization_contract import policy
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        patch_file = next(
            f for m, f in _get_patch_files() if m == "vllm.compilation.backends"
        )
        patch_module = _load_patch_module(patch_file)
        monkeypatch.setattr(policy, "prefers_feature", lambda *_args: True)

        def unexpected_backend_check(*_args, **_kwargs):
            pytest.fail("backend support must not run without a layer count")

        monkeypatch.setattr(
            presplit,
            "qwen3_qk_rope_kv_backend_supported",
            unexpected_backend_check,
        )
        splitting_ops = ["vllm::unified_kv_cache_update", "vllm::attention"]
        compilation_config = SimpleNamespace(
            use_inductor_graph_partition=False,
            splitting_ops=splitting_ops,
            traced_files=set(),
        )
        backend = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=SimpleNamespace())
            ),
            compilation_config=compilation_config,
        )

        assert patch_module._try_qwen3_qk_rope_kv_presplit(backend, object()) == 0
        assert compilation_config.splitting_ops is splitting_ops
        assert compilation_config.traced_files == set()

    def test_qwen3_presplit_site_mismatch_is_atomic(self, monkeypatch):
        from vllm_musa.compilation import qwen3_qk_rope_kv_presplit as presplit
        from vllm_musa.optimization_contract import policy
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        patch_file = next(
            f for m, f in _get_patch_files() if m == "vllm.compilation.backends"
        )
        patch_module = _load_patch_module(patch_file)
        monkeypatch.setattr(policy, "prefers_feature", lambda *_args: True)
        monkeypatch.setattr(
            presplit,
            "qwen3_qk_rope_kv_backend_supported",
            lambda _config, expected_sites: expected_sites == 36,
        )
        monkeypatch.setattr(
            presplit,
            "plan_qwen3_qk_rope_kv_presplit",
            lambda _graph, _expected_sites: None,
        )

        def fail_if_applied(_graph, _candidates):
            raise AssertionError("an incomplete Qwen3 plan must not be applied")

        monkeypatch.setattr(
            presplit, "apply_qwen3_qk_rope_kv_presplit", fail_if_applied
        )
        splitting_ops = ["vllm::unified_kv_cache_update", "vllm::attention"]
        compilation_config = SimpleNamespace(
            use_inductor_graph_partition=False,
            splitting_ops=splitting_ops,
            traced_files=set(),
        )
        backend = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(num_hidden_layers=36)
                )
            ),
            compilation_config=compilation_config,
        )

        assert patch_module._try_qwen3_qk_rope_kv_presplit(backend, object()) == 0
        assert compilation_config.splitting_ops is splitting_ops
        assert compilation_config.traced_files == set()

    def test_qwen3_presplit_rejects_unsupported_attention_backend(self, monkeypatch):
        from vllm_musa.compilation import qwen3_qk_rope_kv_presplit as presplit
        from vllm_musa.optimization_contract import policy
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        patch_file = next(
            f for m, f in _get_patch_files() if m == "vllm.compilation.backends"
        )
        patch_module = _load_patch_module(patch_file)
        monkeypatch.setattr(policy, "prefers_feature", lambda *_args: True)
        monkeypatch.setattr(
            presplit,
            "qwen3_qk_rope_kv_backend_supported",
            lambda _config, _expected_sites: False,
        )

        def fail_if_planned(_graph, _expected_sites):
            raise AssertionError("unsupported backends must not plan a Qwen3 rewrite")

        monkeypatch.setattr(presplit, "plan_qwen3_qk_rope_kv_presplit", fail_if_planned)
        splitting_ops = ["vllm::unified_kv_cache_update", "vllm::attention"]
        compilation_config = SimpleNamespace(
            use_inductor_graph_partition=False,
            splitting_ops=splitting_ops,
            traced_files=set(),
        )
        backend = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(num_hidden_layers=28)
                )
            ),
            compilation_config=compilation_config,
        )

        assert patch_module._try_qwen3_qk_rope_kv_presplit(backend, object()) == 0
        assert compilation_config.splitting_ops is splitting_ops
        assert compilation_config.traced_files == set()

    def test_live_functorch_config_patch_skips_missing_keys(self):
        from contextlib import nullcontext

        from vllm_musa.patches._shared import make_config_patch_filter

        calls = []

        def original_patch(*args, **kwargs):
            calls.append((args, kwargs))
            return nullcontext()

        functorch_config = SimpleNamespace(existing_key=True)
        patched = make_config_patch_filter(original_patch, functorch_config)

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


class TestCompilationCompilerInterfacePatch:
    """Tests for MUSA torch compiler-interface compatibility patches."""

    def test_live_functorch_config_patch_filters_missing_keys(self, monkeypatch):
        # now the vllm.compilation.compiler_interface cat-6 object
        # patch; it filters _get_vllm_functorch_config() against the live
        # torch._functorch.config, so stub that config to only have existing_key.
        import sys

        import torch._functorch

        from vllm_musa.patches import _get_patch_files, _load_patch_module

        class DummyCompilerInterface:
            @staticmethod
            def _get_vllm_functorch_config():
                return {"existing_key": True, "missing_key": False}

        monkeypatch.setitem(
            sys.modules,
            "vllm.compilation.compiler_interface",
            DummyCompilerInterface,
        )
        monkeypatch.setattr(
            torch._functorch,
            "config",
            SimpleNamespace(existing_key=True),
            raising=False,
        )

        patch_file = next(
            f
            for m, f in _get_patch_files()
            if m == "vllm.compilation.compiler_interface"
        )
        _load_patch_module(patch_file).apply()

        config = DummyCompilerInterface._get_vllm_functorch_config()
        assert config == {"existing_key": True}


class TestMUSAFlashAttentionReshapeCache:
    """Tests for MUSA FlashAttention reshape+cache dispatch guards.

    Both tests are SKIPPED (not xfail): their mock-reimport of ``fa_utils`` trips
    the v0.22 safetensors PyO3 single-init wall, which poisons the rest of the
    process and makes the whole suite order-dependent. Skipping keeps plain
    ``pytest`` deterministic; the real fix (assert the guard without reimporting
    ``fa_utils``) is tracked as.
    """

    def _load_fa_utils_with_musa_platform(self, monkeypatch, musa_ops_namespace):
        import vllm
        import vllm.platforms as vllm_platforms

        import vllm_musa

        monkeypatch.setattr(
            vllm_platforms,
            "current_platform",
            # v0.22 quant_utils.py reads current_platform.fp8_dtype() at import,
            # so the mock must provide it.
            SimpleNamespace(
                is_musa=lambda: True,
                fp8_dtype=lambda: torch.float8_e4m3fn,
                dispatch_key="MUSA",
            ),
        )

        # v0.22's import machinery (importlib.util) requires a real
        # __spec__ on injected modules, else it raises "<name>.__spec__ is None".
        from importlib.machinery import ModuleSpec

        def _mock_module(name):
            mod = ModuleType(name)
            mod.__spec__ = ModuleSpec(name, None)
            return mod

        flash_attn = _mock_module("flash_attn_interface")
        flash_attn.flash_attn_varlen_func = object()
        flash_attn.flash_attn_with_kvcache = object()
        flash_attn.get_scheduler_metadata = object()
        monkeypatch.setitem(sys.modules, "flash_attn_interface", flash_attn)

        vllm_ops = _mock_module("vllm._custom_ops")
        vllm_ops.reshape_and_cache_flash = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "vllm._custom_ops", vllm_ops)
        monkeypatch.setattr(vllm, "_custom_ops", vllm_ops, raising=False)

        musa_custom_ops = _mock_module("vllm_musa._custom_ops")
        musa_custom_ops.musa_reshape_and_cache_flash_nhd = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "vllm_musa._custom_ops", musa_custom_ops)
        monkeypatch.setattr(vllm_musa, "_custom_ops", musa_custom_ops, raising=False)

        if musa_ops_namespace is None:
            torch_ops = SimpleNamespace()
        else:
            torch_ops = SimpleNamespace(_C_musa_ops=musa_ops_namespace)
        monkeypatch.setattr(torch, "ops", torch_ops)

        module_name = "vllm_musa.v1.attention.backends.fa_utils"
        # Full sys.modules isolation: v0.22 fa_utils pulls in a deep
        # import chain (mla.common, fp8/quant utils, ...). Snapshot before the
        # reimport and drop EVERYTHING imported during it, so a partially-imported
        # module from this mocked environment does not leak into later tests and
        # cause order-dependent failures across the suite.
        mod_snapshot = set(sys.modules)
        previous_module = sys.modules.pop(module_name, None)
        try:
            module = importlib.import_module(module_name)
        finally:
            for _leaked in set(sys.modules) - mod_snapshot:
                sys.modules.pop(_leaked, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module
        return module

    @pytest.mark.skip(
        reason="the mock-reimport approach hits v0.22's deep "
        "fa_utils import chain (mla.common, quant utils, then a safetensors PyO3 "
        "single-init wall). xfail is not enough here: RUNNING the body trips that "
        "single-init wall and POISONS every later test in the process (the source "
        "of the order-dependent failures this suite used to mask with --forked). "
        "Skip until the rewrite asserts the guard without reimporting "
        "fa_utils."
    )
    def test_missing_musa_ops_namespace_disables_native_cache_path(self, monkeypatch):
        module = self._load_fa_utils_with_musa_platform(
            monkeypatch, musa_ops_namespace=None
        )

        assert module._HAS_NATIVE_RESHAPE_CACHE_FLASH is False

    @pytest.mark.skip(
        reason="same v0.22 fa_utils reimport-chain "
        "incompatibility as test_missing_musa_ops_namespace_disables_native_cache_"
        "path — RUNNING the body poisons the process (safetensors single-init), so "
        "skip (not xfail) until the rewrite avoids reimporting fa_utils."
    )
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
        # The error message is split across two adjacent C++ string literals in
        # the .mu source (a line break between "...must be one of " and the
        # value list), so match the parts rather than the concatenation
        # (avoid brittle cross-literal substring matching).
        assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X must be one of" in source
        assert "128, 256, 512, or 1024" in source
        assert "int requested_block" in source
        assert "env_forced_block > 0 ? env_forced_block : requested_block" in source

    def test_fused_rmsnorm_qwen2_small_hidden_uses_256_threads(self):
        source = (
            Path(__file__).parents[1] / "csrc/musa/fused_add_rmsnorm.mu"
        ).read_text()

        gate_start = source.index(
            "std::is_same<T, __mt_bfloat16>::value && rows > 0 &&"
        )
        gate = source[gate_start : source.index("} else", gate_start)]
        assert "block_x = 256" in gate
        assert "std::is_same<T, __mt_bfloat16>::value" in gate
        assert "rows > 0" in gate
        assert "rows <= 16 && hidden_size == 896" in gate
        assert "use_qwen2_small_hidden_launch" not in source
        assert "getMUSAArch" not in source


class TestMUSAPlatformDefaults:
    """Tests for MUSA platform-level vLLM config defaults.

    apply_config_platform_defaults() writes real os.environ vars (not
    via monkeypatch); the autouse fixture below restores os.environ per test so
    they don't leak across tests in this class.
    """

    @pytest.fixture(autouse=True)
    def _restore_environ(self):
        saved = dict(os.environ)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(saved)

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
        use_mla=False,
        index_topk=None,
        attention_backend=None,
        cache_block_size=16,
    ):
        from types import SimpleNamespace

        from vllm.config import CUDAGraphMode
        from vllm.config.compilation import CompilationMode

        is_deepseek_v4 = architectures == ["DeepseekV4ForCausalLM"]
        if is_deepseek_v4 and attention_backend is None:
            attention_backend = "FLASHMLA"
        if is_deepseek_v4 and cudagraph_mode is None:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        if is_deepseek_v4 and quantization_config is None:
            quantization_config = {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
        hf_config = SimpleNamespace(
            architectures=architectures,
            model_type="deepseek_v4" if is_deepseek_v4 else None,
            quantization_config=quantization_config,
            hidden_size=4096 if is_deepseek_v4 else None,
            num_hidden_layers=43 if is_deepseek_v4 else None,
            num_attention_heads=64 if is_deepseek_v4 else None,
            num_key_value_heads=1 if is_deepseek_v4 else None,
            head_dim=512 if is_deepseek_v4 else None,
            vocab_size=129280 if is_deepseek_v4 else None,
            n_routed_experts=256 if is_deepseek_v4 else None,
            num_experts_per_tok=6 if is_deepseek_v4 else None,
            n_shared_experts=1 if is_deepseek_v4 else None,
            moe_intermediate_size=2048 if is_deepseek_v4 else None,
            expert_dtype="fp8" if is_deepseek_v4 else None,
            hidden_act="silu" if is_deepseek_v4 else None,
            swiglu_limit=10.0 if is_deepseek_v4 else None,
        )
        if index_topk is not None:
            hf_config.index_topk = index_topk
        return SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=architectures,
                hf_config=hf_config,
                hf_text_config=hf_config,
                dtype="bfloat16",
                quantization=quantization,
                use_mla=use_mla,
                is_hybrid=False,
                is_moe=is_deepseek_v4,
                enforce_eager=False,
                is_mm_prefix_lm=False,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                decode_context_parallel_size=1,
                worker_cls="auto",
            ),
            cache_config=SimpleNamespace(
                block_size=cache_block_size,
                cache_dtype="fp8" if is_deepseek_v4 else "auto",
            ),
            scheduler_config=SimpleNamespace(
                is_multimodal_model=False,
                disable_chunked_mm_input=False,
                max_num_seqs=1 if is_deepseek_v4 else 64,
            ),
            attention_config=SimpleNamespace(backend=attention_backend),
            quant_config=SimpleNamespace(weight_block_size=[128, 128])
            if is_deepseek_v4
            else None,
            compilation_config=SimpleNamespace(
                custom_ops=[],
                mode=CompilationMode.NONE if is_deepseek_v4 else "NONE",
                cudagraph_mode=cudagraph_mode,
                max_cudagraph_capture_size=max_cudagraph_capture_size,
                cudagraph_capture_sizes=cudagraph_capture_sizes,
            ),
        )

    def test_qwen3_moe_does_not_cap_default_cudagraph_capture_size(self):
        # The FP8-only cudagraph capture-size cap was removed; the platform no
        # longer caps Qwen3-MoE capture size. Defaults leave it unset (use
        # cudagraph_mode=FULL_DECODE_ONLY for large MoE instead).
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["Qwen3MoeForCausalLM"],
        )

        MUSAPlatformBase.apply_config_platform_defaults(vllm_config)

        assert vllm_config.compilation_config.max_cudagraph_capture_size is None
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

    def test_tp4_keeps_musa_cudagraph_capture(self):
        """Tensor parallelism alone must not turn cudagraph capture off.

        An earlier platform default forced ``cudagraph_mode=NONE`` above TP=2,
        on the reading that MCCL collectives could not run under stream
        capture. That reading was wrong: capture broke because collectives fell
        back to ``torch.distributed.all_reduce``, whose MCCL ProcessGroup
        watchdog called into the driver inside the capture window. Routing them
        through the captureable, stream-bound custom all-reduce fixed the cause
        and the TP guard came out with it. Capture stays on at TP=4, so a
        reintroduced TP gate fails here.
        """
        from vllm.config import CUDAGraphMode

        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            max_cudagraph_capture_size=512,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            tensor_parallel_size=4,
        )

        MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
        assert vllm_config.compilation_config.max_cudagraph_capture_size == 512
        assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]

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

    def test_deepseek_v4_does_not_default_generic_kernel_envs(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_MUSA_GEMV_MOE_BLOCK", None)
            os.environ.pop("VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X", None)

            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert "VLLM_MUSA_GEMV_MOE_BLOCK" not in os.environ
            assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X" not in os.environ

    def test_deepseek_v4_preserves_user_moe_gemv_block(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
        )

        with patch.dict(os.environ, {"VLLM_MUSA_GEMV_MOE_BLOCK": "16x8"}):
            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert os.environ["VLLM_MUSA_GEMV_MOE_BLOCK"] == "16x8"

    def test_deepseek_v4_preserves_empty_user_moe_gemv_block(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
        )

        with patch.dict(os.environ, {"VLLM_MUSA_GEMV_MOE_BLOCK": ""}):
            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert os.environ["VLLM_MUSA_GEMV_MOE_BLOCK"] == ""

    def test_non_deepseek_v4_does_not_default_moe_gemv_block(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["Qwen3ForCausalLM"],
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_MUSA_GEMV_MOE_BLOCK", None)

            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert "VLLM_MUSA_GEMV_MOE_BLOCK" not in os.environ

    def test_deepseek_v4_tp8_uses_contract_without_profile_env(self):
        from vllm_musa.optimization_contract import (
            OptimizationFeature,
            resolve_optimization_contract,
        )
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            tensor_parallel_size=8,
            use_mla=True,
            index_topk=512,
        )

        with patch.dict(os.environ, {}, clear=False):
            for key in (
                "VLLM_MUSA_DEEPSEEK_V4_TP8_PROFILE",
                "VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE",
                "VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE",
                "VLLM_MUSA_DEEPSEEK_V4_TILELANG_DISABLE_HOST_ASSERTS",
                "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS",
            ):
                os.environ.pop(key, None)
            os.environ.pop("VLLM_MUSA_GEMV_MOE_BLOCK", None)
            os.environ.pop("VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X", None)

            MUSAPlatformBase.check_and_update_config(vllm_config)

            contract = resolve_optimization_contract(vllm_config)
            assert contract.prefers(
                OptimizationFeature.DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256
            )
            assert contract.prefers(
                OptimizationFeature.DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256
            )
            assert "VLLM_MUSA_GEMV_MOE_BLOCK" not in os.environ
            assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X" not in os.environ
            assert "VLLM_MUSA_DEEPSEEK_V4_TP8_PROFILE" not in os.environ
            assert (
                os.environ.get("VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE")
                is None
            )

    def test_non_deepseek_v4_keeps_generic_envs_untouched(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(architectures=["Qwen3ForCausalLM"])

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_MUSA_GEMV_MOE_BLOCK", None)
            os.environ.pop("VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X", None)
            MUSAPlatformBase.check_and_update_config(vllm_config)
            assert "VLLM_MUSA_GEMV_MOE_BLOCK" not in os.environ
            assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X" not in os.environ

    def test_deepseek_v4_tp8_flashmla_uses_validated_page256(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            tensor_parallel_size=8,
            use_mla=True,
            index_topk=512,
            cache_block_size=64,
        )

        # A retired gate must not be able to turn the validated path off.
        with patch.dict(
            os.environ,
            {"VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE": "64"},
            clear=False,
        ):
            MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.cache_config.block_size == 256

    def test_deepseek_v4_non_tp8_flashmla_keeps_generic_page64(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            tensor_parallel_size=4,
            use_mla=True,
            index_topk=512,
            cache_block_size=256,
        )

        MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.cache_config.block_size == 64

    def test_non_deepseek_sparse_mla_keeps_generic_page64(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["OtherSparseMLAForCausalLM"],
            tensor_parallel_size=8,
            use_mla=True,
            index_topk=512,
            cache_block_size=256,
        )

        MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.cache_config.block_size == 64

    def test_deepseek_v4_tp8_preserves_explicit_generic_overrides(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            tensor_parallel_size=8,
        )

        with patch.dict(
            os.environ,
            {
                "VLLM_MUSA_GEMV_MOE_BLOCK": "32x8",
                "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X": "128",
            },
            clear=False,
        ):
            MUSAPlatformBase.check_and_update_config(vllm_config)
            assert os.environ["VLLM_MUSA_GEMV_MOE_BLOCK"] == "32x8"
            assert os.environ["VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X"] == "128"

    def test_update_block_size_for_backend_defaults_and_hybrid_modes(self, monkeypatch):
        # The MUSA platform seeds a 64-element KV page for non-hybrid,
        # non-user-specified configs so paged FMHA/MLA decode takes the TME
        # bulk-gather path. Fixed-page kernels and explicit user overrides are
        # preserved; hybrid models defer to the upstream mamba-aligned selection.
        from types import SimpleNamespace

        from vllm.v1.attention.backend import AttentionBackend, MultipleOf

        from vllm_musa.platform import MUSAPlatformBase

        class _MultipleOf16Backend(AttentionBackend):
            @staticmethod
            def get_supported_kernel_block_sizes():
                return [MultipleOf(16)]

            @staticmethod
            def get_name():
                return "STUB_M16"

        class _FixedPage256Backend(AttentionBackend):
            @staticmethod
            def get_supported_kernel_block_sizes():
                return [256]

            @staticmethod
            def get_name():
                return "STUB_256"

        def _cfg(
            *,
            is_hybrid=False,
            user_specified=False,
            block_size=16,
            mamba_cache_mode="none",
        ):
            return SimpleNamespace(
                model_config=SimpleNamespace(is_hybrid=is_hybrid),
                cache_config=SimpleNamespace(
                    block_size=block_size,
                    user_specified_block_size=user_specified,
                    mamba_cache_mode=mamba_cache_mode,
                    mamba_page_size_padded=1056,
                ),
            )

        def _use_backend(backend):
            monkeypatch.setattr(
                MUSAPlatformBase,
                "_find_non_ssm_backend",
                classmethod(lambda cls, vllm_config: backend),
            )

        # non-hybrid, default block size, kernel accepts multiples of 16 -> 64
        _use_backend(_MultipleOf16Backend)
        cfg = _cfg()
        MUSAPlatformBase.update_block_size_for_backend(cfg)
        assert cfg.cache_config.block_size == 64

        # kernel that cannot take 64 keeps its own required page (256)
        _use_backend(_FixedPage256Backend)
        cfg = _cfg()
        MUSAPlatformBase.update_block_size_for_backend(cfg)
        assert cfg.cache_config.block_size == 256

        # an explicit --block-size is never overridden
        _use_backend(_MultipleOf16Backend)
        cfg = _cfg(user_specified=True, block_size=32)
        MUSAPlatformBase.update_block_size_for_backend(cfg)
        assert cfg.cache_config.block_size == 32

        # Separate-pool hybrid models keep 64-token attention pages.
        _use_backend(_MultipleOf16Backend)
        cfg = _cfg(is_hybrid=True)
        MUSAPlatformBase.update_block_size_for_backend(cfg)
        assert cfg.cache_config.block_size == 64
        assert cfg.cache_config.mamba_page_size_padded is None

        # The legacy aligned-cache mode still follows the upstream page size.
        monkeypatch.setattr(
            MUSAPlatformBase,
            "_align_hybrid_block_size",
            classmethod(lambda cls, vllm_config, backend_cls: None),
        )
        cfg = _cfg(is_hybrid=True, mamba_cache_mode="align")
        MUSAPlatformBase.update_block_size_for_backend(cfg)
        assert cfg.cache_config.block_size == 16
        assert cfg.cache_config.mamba_page_size_padded == 1056

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

        weight = torch.empty((2, 256, 320), dtype=torch.float8_e4m3fn)
        scale = torch.tensor([0.5, 1.5], dtype=torch.float32)

        expanded = _maybe_expand_fp8_moe_per_tensor_scale(scale, weight)

        assert expanded is not None
        assert expanded.shape == (2, 4, 5)
        assert expanded.is_contiguous()
        assert torch.all(expanded[0] == scale[0])
        assert torch.all(expanded[1] == scale[1])

    def test_static_tensor_fp8_moe_scales_prefer_128_block_layout(self):
        from vllm_musa.model_executor.layers.fused_moe.fused_moe import (
            _maybe_expand_fp8_moe_per_tensor_scale,
        )

        weight = torch.empty((2, 384, 256), dtype=torch.float8_e4m3fn)
        scale = torch.tensor([0.5, 1.5], dtype=torch.float32)

        expanded = _maybe_expand_fp8_moe_per_tensor_scale(scale, weight)

        assert expanded is not None
        assert expanded.shape == (2, 3, 2)
        assert expanded.is_contiguous()

    def test_static_tensor_fp8_moe_scales_reject_partial_output_blocks(self):
        from vllm_musa.model_executor.layers.fused_moe.fused_moe import (
            _maybe_expand_fp8_moe_per_tensor_scale,
        )

        weight = torch.empty((2, 260, 320), dtype=torch.float8_e4m3fn)
        scale = torch.tensor([0.5, 1.5], dtype=torch.float32)

        with pytest.raises(
            ValueError,
            match="requires the weight output dimension to be divisible",
        ):
            _maybe_expand_fp8_moe_per_tensor_scale(scale, weight)

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
            lambda self,
            hidden_size,
            intermediate_size_per_partition,
            act_dtype,
            moe_parallel_config: (
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
        assert "torch.ops.vllm.musa_fp8_small_m_gemv_op" in source
        assert "direct_register_custom_op(" in source
        assert '"musa_deepgemm_fp8_op"' in source
        assert '"musa_fp8_small_m_gemv_op"' in source
        assert "_musa_deepgemm_fp8_op_fake" in source
        assert "_musa_fp8_small_m_gemv_op_fake" in source
        assert "VLLM_MUSA_DEEPGEMM_ROW_MAJOR_ACT_SCALES" in source
        assert "VLLM_MUSA_FP8_SMALL_M_GEMV" in source

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
                dispatch_key="MUSA",
            ),
        )
        # float32 input takes the fallback branch, which calls the op out of
        # the _C namespace rather than _C_musa_ops.
        monkeypatch.setattr(
            torch.ops,
            "_C",
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
                dispatch_key="MUSA",
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


class TestPatchManifest:
    """deterministic patch discovery + read-only patch_report()."""

    def test_get_patch_files_sorted_and_unique(self):
        from vllm_musa.patches import _get_patch_files

        names = [f.name for _, f in _get_patch_files()]
        assert names, "no patch files discovered"
        assert names == sorted(names), "patch discovery order is not deterministic"
        assert len(names) == len(set(names)), "duplicate patch files"

    def test_patch_report_structure_and_determinism(self):
        import vllm_musa
        from vllm_musa.patches import _get_patch_files

        report = vllm_musa.patch_report()
        assert isinstance(report, list)
        assert len(report) == len(_get_patch_files()), "report must cover every patch"

        # Source patches are applied at build time; the runtime report only ever
        # sees the object patches (side-effect) plus anomaly states.
        allowed_status = {
            "side-effect",
            "load-failed",
            "misplaced-source-patch",
            "error",
            "unknown",
        }
        allowed_kind = {"source-transform", "side-effect", "load-failed", "unknown"}
        for e in report:
            assert {"module", "file", "kind", "target_resolved", "status"} <= set(e), e
            assert e["status"] in allowed_status, e
            assert e["kind"] in allowed_kind, e
            assert isinstance(e["target_resolved"], bool)

        # Deterministic across calls and sorted by file (ordering).
        report2 = vllm_musa.patch_report()
        assert [e["file"] for e in report] == [e["file"] for e in report2]
        assert [e["file"] for e in report] == sorted(e["file"] for e in report)

    def test_patch_report_has_spec_fields(self):
        # every entry carries the derived PatchSpec metadata.
        import vllm_musa

        for e in vllm_musa.patch_report():
            assert e.get("id"), e
            assert e["phase"], e
            assert e["process_scope"] in {"disk-persistent", "process-local"}, e
            assert isinstance(e["required"], bool), e
            assert isinstance(e["is_failure"], bool), e
            assert "version_range" in e


class TestObjectPatchPhase:
    """side-effect patches are applied by an explicit, ordered,
    idempotent object-patch phase via a module-level ``apply()`` — not as an
    import-time side effect of the patch loader."""

    # The in-process object/monkey patches in this tree (+.
    _OBJECT_PATCH_MODULES = {
        "vllm.v1.spec_decode.eagle",  # kernel prime
        "vllm.distributed.parallel_state",  # draft-TP=1 wiring
        # torch/vLLM config compat shims migrated from vllm_musa/__init__.py
        "torch._functorch.config",
        "torch._inductor.config",
        "torch._inductor.aot_cache_safelist",
        "vllm.compilation.backends",
        "vllm.compilation.compiler_interface",
        "vllm._custom_ops",
    }

    def test_object_patch_files_expose_callable_apply(self):
        # Each side-effect patch must define a top-level ``apply()`` so the
        # object-patch phase can call it explicitly (no import-time side effect).
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        found = {}
        for module_name, patch_file in _get_patch_files():
            mod = _load_patch_module(patch_file)
            if mod is not None and callable(getattr(mod, "apply", None)):
                found[module_name] = patch_file.name
        assert self._OBJECT_PATCH_MODULES <= set(found), (
            f"object-patch modules missing apply(): "
            f"{self._OBJECT_PATCH_MODULES - set(found)}"
        )

    def test_object_patch_files_keep_empty_patches_list(self):
        # An object patch must NOT also be a source transform: empty PATCHES and
        # no normalize_source. Source edits live in series/ (build-time); a
        # non-empty PATCHES here would be flagged misplaced-source-patch.
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        for module_name, patch_file in _get_patch_files():
            if module_name not in self._OBJECT_PATCH_MODULES:
                continue
            mod = _load_patch_module(patch_file)
            assert getattr(mod, "PATCHES", None) == [], module_name
            assert not callable(getattr(mod, "normalize_source", None)), module_name

    def test_loading_object_patch_module_has_no_import_side_effect(self):
        # Regression for importing the eagle shim must NOT prime the
        # kernel by itself — only calling apply() may. We assert the module body
        # defines apply but does not import the prime at top level by checking
        # the source has no module-level ``import vllm_musa.v1.spec_decode.utils``
        # outside the apply() function.
        from vllm_musa.patches import _get_patch_files

        eagle = next(
            f for m, f in _get_patch_files() if m == "vllm.v1.spec_decode.eagle"
        )
        src = eagle.read_text()
        assert "def apply(" in src, "eagle shim lost its apply()"
        # The prime import must be indented (inside apply), never at column 0.
        for line in src.splitlines():
            if line.startswith("import vllm_musa.v1.spec_decode.utils"):
                raise AssertionError(
                    "eagle prime is a module-level import-side-effect again; "
                    "it must live inside apply()"
                )

    def test_apply_object_patches_is_idempotent(self):
        # force=True always runs; calling twice must not raise (each apply() is
        # individually idempotent), and every result row has the expected shape.
        from vllm_musa.patches import apply_object_patches

        r1 = apply_object_patches(force=True)
        r2 = apply_object_patches(force=True)
        assert isinstance(r1, list) and isinstance(r2, list)
        applied_mods = {e["module"] for e in r1}
        assert self._OBJECT_PATCH_MODULES <= applied_mods, applied_mods
        for e in r1 + r2:
            assert {"module", "file", "status"} <= set(e), e
            assert e["status"] in {"applied", "error"}, e
            assert e["status"] == "applied", e  # none should error in this env

    def test_apply_object_patches_guard_skips_repeat(self):
        # Without force, the module-level guard returns [] after the first run.
        import vllm_musa.patches as P

        P.apply_object_patches(force=True)  # ensure the guard is set
        assert P._object_patches_applied is True
        assert P.apply_object_patches() == []

    def test_report_flags_object_patches(self):
        # The read-only report marks side-effect patches that expose apply().
        import vllm_musa

        by_mod = {e["module"]: e for e in vllm_musa.patch_report()}
        for module_name in self._OBJECT_PATCH_MODULES:
            e = by_mod[module_name]
            assert e["kind"] == "side-effect", e
            assert e.get("object_patch") is True, e


class TestBuildTimeSeries:
    """vLLM source edits ship as a build-time ``git format-patch``
    series in ``vllm_musa/patches/series/`` and are applied to the cloned vLLM at
    build time by ``build_apply.py``. These assert the series is present and
    well-formed and that the applier honours its idempotent/loud-conflict
    contract — without needing a real vLLM checkout."""

    _PATCHES_DIR = Path(__file__).parent.parent / "vllm_musa" / "patches"
    _SERIES_DIR = _PATCHES_DIR / "series"

    @staticmethod
    def _read_patch(path: Path) -> str:
        return path.read_bytes().decode("utf-8", errors="surrogateescape")

    def test_read_patch_tolerates_non_utf8_payload(self, tmp_path):
        patch = tmp_path / "non-utf8.patch"
        patch.write_bytes(b"From: musa <musa@local>\ninvalid: \xff\xfe\n")

        text = self._read_patch(patch)

        assert text.startswith("From: musa <musa@local>")
        assert text.encode("utf-8", errors="surrogateescape") == patch.read_bytes()

    def _load_build_apply(self):
        # Load by file path exactly as setup.py does: build_apply is stdlib-only
        # so this needs neither a vllm nor a vllm_musa import.
        spec = importlib.util.spec_from_file_location(
            "musa_build_apply_under_test", self._PATCHES_DIR / "build_apply.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_series_dir_present_and_nonempty(self):
        assert self._SERIES_DIR.is_dir(), "build-time series/ dir is missing"
        assert sorted(self._SERIES_DIR.glob("*.patch")), "no .patch files in series/"

    def test_series_files_are_wellformed_format_patches(self):
        for p in sorted(self._SERIES_DIR.glob("*.patch")):
            text = self._read_patch(p)
            assert text.startswith("From "), f"{p.name}: not a git format-patch"
            assert "Subject:" in text, f"{p.name}: missing Subject line"
            assert "diff --git" in text, f"{p.name}: no diff hunk"

    def test_build_apply_exposes_contract(self):
        ba = self._load_build_apply()
        for fn in ("apply_patch", "series_files", "apply_patch_series", "main"):
            assert callable(getattr(ba, fn, None)), f"build_apply.{fn} missing"

    def test_series_files_is_deterministic_and_covers_dir(self):
        ba = self._load_build_apply()
        got = [p.name for p in ba.series_files(self._SERIES_DIR)]
        assert got, "series_files returned nothing"
        assert got == sorted(got), "series_files order is not deterministic"
        # With no quilt manifest, it must equal the sorted glob exactly.
        assert got == sorted(p.name for p in self._SERIES_DIR.glob("*.patch"))

    def test_gemma_norm_series_patch_uses_generic_inplace_ir_contract(self):
        series = (
            self._SERIES_DIR / "0079-MUSA-allow-Gemma-RMSNorm-residual-donation.patch"
        )
        series = self._read_patch(series)

        assert "fused_add_rms_norm.maybe_inplace" in series
        assert "allow_inplace" not in series
        assert "vllm/model_executor/models/qwen3_5.py" not in series
        assert "vllm/model_executor/models/qwen3_5_mtp.py" not in series

    def test_gated_qkv_series_patch_uses_generic_ir_contract(self):
        series = self._SERIES_DIR / "0080-MUSA-add-gated-QKV-RMSNorm-MRoPE-IR.patch"
        series = self._read_patch(series)

        assert '@register_op(activations=["packed_qkv"])' in series
        assert "def gated_qkv_rms_norm_rope(" in series
        assert "ir.ops.gated_qkv_rms_norm_rope(" in series
        assert "q, k = ir.ops.gated_qkv_rms_norm_rope(" in series
        assert "q, k, gate = ir.ops.gated_qkv_rms_norm_rope(" not in series
        assert "attn_output.view_as(gate) * torch.sigmoid(gate)" in series
        assert "gate = gate.reshape(*orig_shape, -1)" not in series
        assert "vllm/model_executor/models/qwen3_next.py" in series
        assert "VLLM_MUSA_EXPERIMENTAL_GATED_QKV_MROPE" not in series
        assert "current_platform.is_musa()" not in series
        assert "override_tolerance(torch.bfloat16, atol=2e-2, rtol=1.6e-2)" in series

    def test_apply_patch_series_missing_dir_is_noop(self, tmp_path):
        ba = self._load_build_apply()
        assert ba.apply_patch_series(tmp_path, tmp_path / "nope") == []

    def test_deepseek_v4_spec_metadata_upload_series_patch_present(self):
        series = self._SERIES_DIR / "0035-MUSA-vllm.v1.worker.gpu_model_runner.patch"
        series = self._read_patch(series)

        assert "_spec_cu_num_draft_tokens" in series
        assert "_spec_cu_num_sampled_tokens" in series
        assert "_spec_logits_indices" in series
        assert "_spec_target_logits_indices" in series
        assert "_spec_bonus_logits_indices" in series
        assert (
            "self._spec_target_logits_indices = self._make_buffer(\n"
            "+            self.max_num_tokens, dtype=torch.int64\n"
            "+        )" in series
        )
        assert "current_platform.is_musa()" in series
        assert "copy_to_gpu(num_sampled_total)" in series
        musa_branch = series.split("MUSA-3406: copy through", 1)[1].split("else:", 1)[0]
        assert "torch.from_numpy(cu_num_draft_tokens).to" not in musa_branch
        assert "torch.from_numpy(cu_num_sampled_tokens).to" not in musa_branch
        assert "torch.from_numpy(logits_indices).to" not in musa_branch
        assert "torch.from_numpy(target_logits_indices).to" not in musa_branch
        assert "torch.from_numpy(bonus_logits_indices).to" not in musa_branch

    def test_deepseek_v4_materialized_indexer_series_patch_present(self):
        series = (
            self._SERIES_DIR
            / "0013-MUSA-vllm.model_executor.layers.sparse_attn_indexer.patch"
        )
        series = self._read_patch(series)

        assert "VLLM_MUSA_DEEPSEEK_V4_INDEXER_TOPK_PREFILL" not in series
        assert "return True" in series
        assert "width = 640" in series
        assert "chunk_rows = 512" in series
        assert "sorted=materialized_topk_sorted" in series
        assert "materialized_direct_topk" in series
        assert "direct_local" in series
        assert "deepseek_v4_indexer_rerank_prefill" in series
        assert "Paged-MQA logits use absolute prefix coordinates" in series
        assert "context_lens = row_ends.reshape(rows, 1).contiguous()" in series
        assert "_musa_try_fill_prefill_topk_from_materialized_logits" in series

    def test_deepseek_v4_rerank_prefill_requires_enough_candidates(self):
        source = (
            Path(__file__).parents[1]
            / "csrc/musa/attention/deepseek_v4_indexer_topk.mu"
        ).read_text()

        assert "topk <= candidate_abs_indices.size(1)" in source
        assert "candidate_abs_indices width must be at least topk" in source

    @pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
    def test_apply_patch_idempotent_and_loud_on_conflict(self, tmp_path):
        # End-to-end on a throwaway git repo: first apply == "applied", re-apply
        # == "already-applied" (idempotent re-apply), and a patch that does not
        # apply == "conflict" (caller fails the build loud).
        import subprocess

        ba = self._load_build_apply()
        repo = tmp_path / "repo"
        repo.mkdir()

        def git(*args):
            subprocess.run(
                ["git", "-C", str(repo), *args], check=True, capture_output=True
            )

        git("init", "-q")
        git("config", "user.email", "t@t")
        git("config", "user.name", "t")
        target = repo / "f.txt"
        target.write_text("line1\nline2\n")
        git("add", "f.txt")
        git("commit", "-q", "-m", "base")
        base = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # Build a patch that edits f.txt, capture it, then reset to base.
        target.write_text("line1\nCHANGED\n")
        git("commit", "-aqm", "change")
        patch_text = subprocess.run(
            ["git", "-C", str(repo), "format-patch", "-1", "--stdout"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git("reset", "-q", "--hard", base)
        patch_file = tmp_path / "0001-change.patch"
        patch_file.write_text(patch_text)

        assert ba.apply_patch(repo, patch_file) == "applied"
        assert "CHANGED" in target.read_text()
        assert ba.apply_patch(repo, patch_file) == "already-applied"

        # A patch whose context no longer exists is a loud conflict.
        target.write_text("totally different content\n")
        git("commit", "-aqm", "drift")
        assert ba.apply_patch(repo, patch_file) == "conflict"


class TestMUSAGroupedTopKRouter:
    def _router_stub(self, routed_scaling_factor: float = 2.0):
        return SimpleNamespace(
            num_expert_group=8,
            e_score_correction_bias=None,
            top_k=2,
            renormalize=True,
            scoring_func="softmax",
            routed_scaling_factor=routed_scaling_factor,
            topk_group=1,
            num_fused_shared_experts=0,
        )

    def test_no_bias_jit_path_applies_routed_scaling_factor(self, monkeypatch):
        import vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router as router_mod

        hidden_states = torch.zeros((2, 4), dtype=torch.float32)
        router_logits = torch.zeros((2, 4), dtype=torch.float32)
        base_weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)
        topk_ids = torch.tensor([[1, 2], [0, 3]], dtype=torch.int32)

        def fake_jit(**kwargs):
            return base_weights.clone(), topk_ids.clone()

        monkeypatch.setattr(router_mod, "_musa_jit_fused_topk", fake_jit)

        weights, ids = router_mod._compute_routing(
            self=self._router_stub(routed_scaling_factor=2.0),
            hidden_states=hidden_states,
            router_logits=router_logits,
            indices_type=torch.int32,
        )

        assert torch.equal(ids, topk_ids)
        assert torch.equal(weights, base_weights * 2.0)

    def test_no_bias_fused_topk_fallback_applies_routed_scaling_factor(
        self, monkeypatch
    ):
        import vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router as router_mod

        hidden_states = torch.zeros((2, 4), dtype=torch.float32)
        router_logits = torch.zeros((2, 4), dtype=torch.float32)
        base_weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)
        topk_ids = torch.tensor([[1, 2], [0, 3]], dtype=torch.int32)
        token_expert_indices = torch.empty_like(topk_ids)

        def fake_fused_topk(**kwargs):
            return base_weights.clone(), topk_ids.clone(), token_expert_indices

        monkeypatch.setattr(router_mod, "_musa_jit_fused_topk", lambda **kwargs: None)
        monkeypatch.setattr(router_mod, "fused_topk", fake_fused_topk)

        weights, ids = router_mod._compute_routing(
            self=self._router_stub(routed_scaling_factor=2.0),
            hidden_states=hidden_states,
            router_logits=router_logits,
            indices_type=torch.int32,
        )

        assert torch.equal(ids, topk_ids)
        assert torch.equal(weights, base_weights * 2.0)
