# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MUSA Platform implementation."""

import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
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

    def test_uses_musa_post_grad_pass_manager(self):
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.get_pass_manager_cls() == (
            "vllm_musa.compilation.passes.MusaPostGradPassManager"
        )

    def test_runner_kv_caches_support_multiple_layers_per_index(self):
        from vllm_musa.platform import MUSAPlatformBase

        assert MUSAPlatformBase.check_runner_kv_caches_multi_layer() is None

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

    def test_fused_add_ir_priority_honors_native_safety_route(self, monkeypatch):
        from vllm.config import CUDAGraphMode
        from vllm.config.compilation import CompilationMode

        from vllm_musa.platform import MUSAPlatformBase
        from vllm_musa.utils.environ import envs

        config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            quant_config=None,
            compilation_config=SimpleNamespace(
                backend="inductor",
                mode=CompilationMode.VLLM_COMPILE,
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
            ),
        )
        monkeypatch.delenv(envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.name, raising=False)

        priority = MUSAPlatformBase.get_default_ir_op_priority(config)
        assert priority.fused_add_rms_norm == [
            "musa",
            "native",
        ]

        # Priority is installed before check_and_update_config(). The safety
        # route must therefore be derived directly from the quantized PIECEWISE
        # config while the process-wide env is still unset.
        config.quant_config = object()
        priority = MUSAPlatformBase.get_default_ir_op_priority(config)
        assert priority.fused_add_rms_norm == ["native"]

        # Explicit 0 remains the documented force-kernel escape hatch.
        with envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.override(False):
            priority = MUSAPlatformBase.get_default_ir_op_priority(config)
            assert priority.fused_add_rms_norm == [
                "musa",
                "native",
            ]
        with envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.override(True):
            priority = MUSAPlatformBase.get_default_ir_op_priority(config)
            assert priority.fused_add_rms_norm == ["native"]

    def test_gated_qkv_inductor_priority_excludes_routed_moe(self):
        from vllm.config.compilation import CompilationMode

        from vllm_musa.optimization_contract.policy import model_has_routed_experts
        from vllm_musa.platform import MUSAPlatformBase

        compilation_config = SimpleNamespace(
            backend="inductor",
            mode=CompilationMode.VLLM_COMPILE,
        )
        dense_model = SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="example_dense")
        )
        dense_config = SimpleNamespace(
            compilation_config=compilation_config,
            model_config=dense_model,
        )
        priority = MUSAPlatformBase.get_default_ir_op_priority(dense_config)
        assert priority.gated_qkv_rms_norm_rope == ["musa_inductor", "native"]
        assert not model_has_routed_experts(dense_model)

        authoritative_moe = SimpleNamespace(
            is_moe=True,
            hf_text_config=SimpleNamespace(model_type="heterogeneous_blocks"),
        )
        assert model_has_routed_experts(authoritative_moe)
        authoritative_dense = SimpleNamespace(
            is_moe=False,
            hf_text_config=SimpleNamespace(num_experts=8),
        )
        assert not model_has_routed_experts(authoritative_dense)

        for expert_count_name in (
            "num_experts",
            "moe_num_experts",
            "n_routed_experts",
            "num_local_experts",
        ):
            moe_model = SimpleNamespace(
                hf_text_config=SimpleNamespace(**{expert_count_name: 8})
            )
            moe_config = SimpleNamespace(
                compilation_config=compilation_config,
                model_config=moe_model,
            )
            priority = MUSAPlatformBase.get_default_ir_op_priority(moe_config)
            assert priority.gated_qkv_rms_norm_rope == ["native"]
            assert model_has_routed_experts(moe_model)

        compilation_config.mode = CompilationMode.NONE
        priority = MUSAPlatformBase.get_default_ir_op_priority(dense_config)
        assert priority.gated_qkv_rms_norm_rope == ["native"]

    def test_native_safety_route_is_quantized_piecewise_only(self):
        from vllm.config import CUDAGraphMode
        from vllm.config.compilation import CompilationMode

        from vllm_musa.platform import (
            _should_route_quantized_piecewise_ops_native,
        )

        config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            compilation_config=SimpleNamespace(
                mode=CompilationMode.VLLM_COMPILE,
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
            ),
            quant_config=None,
        )
        assert not _should_route_quantized_piecewise_ops_native(config)
        config.quant_config = object()
        assert _should_route_quantized_piecewise_ops_native(config)
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
        assert not _should_route_quantized_piecewise_ops_native(config)

    def test_fused_add_profit_boundary_splits_eligible_compile_range(self):
        import torch

        from vllm_musa.platform import (
            _configure_fused_add_rmsnorm_compile_range,
        )
        from vllm_musa.tuning import FUSED_ADD_RMSNORM_MIN_ROWS

        config = SimpleNamespace(
            model_config=SimpleNamespace(
                get_hidden_size=lambda: 5120,
                dtype=torch.bfloat16,
                enforce_eager=False,
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
            compilation_config=SimpleNamespace(compile_ranges_endpoints=[2048]),
        )
        assert _configure_fused_add_rmsnorm_compile_range(
            config, native_custom_ops=False
        )
        assert config.compilation_config.compile_ranges_endpoints == [
            FUSED_ADD_RMSNORM_MIN_ROWS - 1,
            2048,
        ]
        assert not _configure_fused_add_rmsnorm_compile_range(
            config, native_custom_ops=False
        )

        config.compilation_config.compile_ranges_endpoints = []
        assert not _configure_fused_add_rmsnorm_compile_range(
            config, native_custom_ops=True
        )
        assert config.compilation_config.compile_ranges_endpoints == []

        config.kernel_config = SimpleNamespace(
            ir_op_priority=SimpleNamespace(fused_add_rms_norm=["native"])
        )
        assert not _configure_fused_add_rmsnorm_compile_range(
            config, native_custom_ops=False
        )
        assert config.compilation_config.compile_ranges_endpoints == []

    def test_qwen2_rope_kv_fusion_exact_config_gate(self):
        import torch

        from vllm_musa.optimization_contract import policy
        from vllm_musa.optimization_contract.types import OptimizationFeature

        def is_eligible(config):
            return policy.prefers_feature(
                config, OptimizationFeature.QWEN2_ROPE_KV_PRESPLIT
            )

        def make_config(model_type: str, kv_heads, intermediate_size):
            return SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(
                        model_type=model_type,
                        hidden_size=896,
                        intermediate_size=intermediate_size,
                        num_hidden_layers=24,
                        num_attention_heads=14,
                        num_key_value_heads=kv_heads,
                    ),
                    dtype=torch.bfloat16,
                    quantization=None,
                    enforce_eager=False,
                ),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                    decode_context_parallel_size=1,
                ),
                quant_config=None,
                speculative_config=None,
            )

        assert not is_eligible(make_config("qwen2", None, None))
        assert is_eligible(make_config("cosyvoice3", None, None))
        config = make_config("qwen2", 2, 4864)
        assert is_eligible(config)
        config.cache_config = SimpleNamespace(cache_dtype="bfloat16", block_size=64)
        assert is_eligible(config)
        config.cache_config.block_size = 128
        assert not is_eligible(config)
        config.cache_config.block_size = 64
        config.quant_config = object()
        assert not is_eligible(config)

    def test_qwen3_qk_rope_kv_fusion_exact_config_gate(self):
        import torch

        from vllm_musa.optimization_contract import policy
        from vllm_musa.optimization_contract.types import OptimizationFeature

        def is_eligible(config):
            return policy.prefers_feature(
                config, OptimizationFeature.QWEN3_QK_ROPE_KV_PRESPLIT
            )

        def make_config(
            *,
            hidden_size: int,
            intermediate_size: int,
            num_hidden_layers: int,
            num_attention_heads: int,
            num_key_value_heads: int = 8,
            head_dim: int = 128,
            model_type: str = "qwen3",
            architecture: str = "Qwen3ForCausalLM",
            tensor_parallel_size: int = 1,
            quantization=None,
            quant_config=None,
            block_size: int = 64,
        ):
            return SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(
                        model_type=model_type,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        num_hidden_layers=num_hidden_layers,
                        num_attention_heads=num_attention_heads,
                        num_key_value_heads=num_key_value_heads,
                        head_dim=head_dim,
                    ),
                    architectures=(architecture,),
                    dtype=torch.bfloat16,
                    quantization=quantization,
                    enforce_eager=False,
                ),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                    decode_context_parallel_size=1,
                ),
                cache_config=SimpleNamespace(
                    cache_dtype="bfloat16", block_size=block_size
                ),
                quant_config=quant_config,
                speculative_config=None,
            )

        qwen3_0_6b = make_config(
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=28,
            num_attention_heads=16,
        )
        qwen3_8b = make_config(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_attention_heads=32,
        )

        assert is_eligible(qwen3_0_6b)
        assert is_eligible(qwen3_8b)

        assert not is_eligible(
            make_config(
                hidden_size=1024,
                intermediate_size=3072,
                num_hidden_layers=28,
                num_attention_heads=16,
                model_type="qwen2",
                architecture="Qwen2ForCausalLM",
            )
        )
        assert not is_eligible(
            make_config(
                hidden_size=2048,
                intermediate_size=6144,
                num_hidden_layers=24,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=256,
                model_type="qwen3_5",
                architecture="Qwen3_5ForCausalLM",
            )
        )
        assert not is_eligible(
            make_config(
                hidden_size=4096,
                intermediate_size=12288,
                num_hidden_layers=36,
                num_attention_heads=32,
                tensor_parallel_size=2,
            )
        )
        assert not is_eligible(
            make_config(
                hidden_size=4096,
                intermediate_size=12288,
                num_hidden_layers=36,
                num_attention_heads=32,
                quantization="fp8",
            )
        )
        assert not is_eligible(
            make_config(
                hidden_size=4096,
                intermediate_size=12288,
                num_hidden_layers=36,
                num_attention_heads=32,
                quant_config=object(),
            )
        )
        assert not is_eligible(
            make_config(
                hidden_size=4096,
                intermediate_size=12288,
                num_hidden_layers=36,
                num_attention_heads=32,
                block_size=128,
            )
        )

    @pytest.mark.parametrize(
        ("layout", "shape", "expected"),
        [
            ("NHD", (16, 8, 128), True),
            ("NHD", (32, 8, 128), True),
            ("NHD", (4, 1, 256), True),
            ("NHD", (32, 2, 256), True),
            ("HND", (32, 2, 256), False),
            ("NHD", (32, 4, 256), False),
        ],
    )
    def test_qwen3_qk_rope_kv_provider_layout_gate(
        self,
        monkeypatch,
        layout: str,
        shape: tuple[int, int, int],
        expected: bool,
    ) -> None:
        from vllm_musa.v1.attention.backends import flash_attn

        num_heads, num_kv_heads, head_size = shape
        impl = SimpleNamespace(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            attn_type=flash_attn.AttentionType.DECODER,
            kv_cache_dtype="auto",
            alibi_slopes=None,
            sliding_window=(-1, -1),
            logits_soft_cap=0.0,
            sinks=None,
            kv_sharing_target_layer_name=None,
        )
        monkeypatch.setattr(flash_attn, "get_flash_attn_version", lambda: 3)
        monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: layout)
        assert (
            flash_attn.FlashAttentionImpl.qwen3_qk_rope_kvcache_supported(impl)
            is expected
        )

    def test_qwen_fa3_scheduler_lookup_config_gate(self):
        from vllm_musa.v1.attention.backends.flash_attn import (
            _is_qwen_family_scheduler_lookup_config,
        )

        def make_config(
            model_type: str,
            architecture: str,
            *,
            max_num_seqs: int = 1,
            speculative_config=None,
            pipeline_parallel_size: int = 1,
            decode_context_parallel_size: int = 1,
            tensor_parallel_size: int = 1,
        ):
            return SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(
                        model_type=model_type,
                        architectures=[architecture],
                    ),
                    architectures=[architecture],
                ),
                scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    decode_context_parallel_size=decode_context_parallel_size,
                ),
                speculative_config=speculative_config,
            )

        assert _is_qwen_family_scheduler_lookup_config(
            make_config("qwen3", "Qwen3ForCausalLM")
        )
        assert _is_qwen_family_scheduler_lookup_config(
            make_config("qwen2", "Qwen2ForCausalLM")
        )
        assert not _is_qwen_family_scheduler_lookup_config(
            make_config("cosyvoice3", "CosyVoice3ForConditionalGeneration")
        )
        assert _is_qwen_family_scheduler_lookup_config(
            make_config("cosyvoice3", "CosyVoice3Model")
        )
        assert not _is_qwen_family_scheduler_lookup_config(
            make_config("qwen2_vl", "Qwen2VLForConditionalGeneration")
        )
        assert not _is_qwen_family_scheduler_lookup_config(
            make_config("qwen3_omni_moe", "Qwen3OmniMoeForConditionalGeneration")
        )
        assert _is_qwen_family_scheduler_lookup_config(
            make_config("qwen2", "Qwen2ForCausalLM", max_num_seqs=8)
        )
        assert not _is_qwen_family_scheduler_lookup_config(
            make_config("llama", "LlamaForCausalLM")
        )
        assert not _is_qwen_family_scheduler_lookup_config(
            make_config("deepseek_v3", "DeepseekV3ForCausalLM")
        )
        assert not _is_qwen_family_scheduler_lookup_config(
            make_config("qwen3", "Qwen3ForCausalLM", max_num_seqs=0)
        )
        assert not _is_qwen_family_scheduler_lookup_config(
            make_config(
                "qwen3",
                "Qwen3ForCausalLM",
                speculative_config=object(),
            )
        )
        assert not _is_qwen_family_scheduler_lookup_config(
            make_config(
                "qwen3",
                "Qwen3ForCausalLM",
                decode_context_parallel_size=2,
            )
        )
        assert not _is_qwen_family_scheduler_lookup_config(
            make_config(
                "qwen3",
                "Qwen3ForCausalLM",
                tensor_parallel_size=2,
            )
        )
        incomplete_parallel_config = make_config("qwen3", "Qwen3ForCausalLM")
        del incomplete_parallel_config.parallel_config.pipeline_parallel_size
        assert not _is_qwen_family_scheduler_lookup_config(incomplete_parallel_config)
        incomplete_scheduler_config = make_config("qwen3", "Qwen3ForCausalLM")
        del incomplete_scheduler_config.scheduler_config.max_num_seqs
        assert not _is_qwen_family_scheduler_lookup_config(incomplete_scheduler_config)

    def test_qwen_fa3_scheduler_lookup_layout_version_gate(self, monkeypatch):
        from importlib.metadata import PackageNotFoundError

        from vllm_musa.v1.attention.backends import flash_attn

        versions = {"mate": "0.2.4", "flash_attn_3": "0.2.4+musa"}
        monkeypatch.setattr(flash_attn, "version", versions.__getitem__)
        assert flash_attn._has_supported_fa3_scheduler_layout()

        versions["mate"] = "0.2.5"
        assert not flash_attn._has_supported_fa3_scheduler_layout()

        versions["mate"] = "0.2.4"
        versions["flash_attn_3"] = "0.2.4+custom"
        assert not flash_attn._has_supported_fa3_scheduler_layout()

        def missing_package(_package):
            raise PackageNotFoundError

        monkeypatch.setattr(flash_attn, "version", missing_package)
        assert not flash_attn._has_supported_fa3_scheduler_layout()

    @staticmethod
    def _spec(arch_name: str, arch_code: int):
        from torchada.musa.device import MusaDeviceSpec

        return MusaDeviceSpec(
            device_capability=(arch_code // 100, (arch_code % 100) // 10),
            arch_code=arch_code,
            arch_name=arch_name,
            mp_count=56,
            warp_size=32 if arch_code == 310 else 128,
            total_memory=1 << 30,
            l2_cache_size=1 << 20,
            shared_memory_per_block=1 << 16,
            shared_memory_per_multiprocessor=1 << 16,
            regs_per_multiprocessor=1 << 16,
            max_threads_per_block=1024,
            max_threads_per_multiprocessor=2048,
            device_name="test device",
        )

    def _supports_fp8(self, arch_name: str, arch_code: int, stamp):
        """Ask the platform with the device and the build stamp both pinned.

        ``supports_fp8`` answers from the arch's explicit capability set and the
        build stamp, never from a capability ordering, so the test has to drive
        those two rather than ``get_device_capability``.
        """
        from unittest.mock import patch

        from vllm_musa import device_spec
        from vllm_musa.platform import MUSAPlatformBase

        with (
            patch.object(
                device_spec, "current_device_spec", return_value=self._spec(arch_name, arch_code)
            ),
            patch.object(device_spec, "read_build_stamp", return_value=stamp),
        ):
            return MUSAPlatformBase.supports_fp8()

    def test_supports_fp8_for_musa_3_1(self):
        """mp_31 has native FP8, so a build that carries it advertises FP8."""
        from torchada.musa.build import BuildStamp

        assert self._supports_fp8("mp_31", 310, BuildStamp(("mp_31",), fp8=True)) is True

    def test_supports_fp8_needs_an_arch_that_has_it(self):
        """An arch without the instructions cannot answer True for any build."""
        from torchada.musa.build import BuildStamp

        assert self._supports_fp8("mp_30", 300, BuildStamp(("mp_30",), fp8=True)) is False

    def test_supports_fp8_needs_a_build_that_carries_it(self):
        """Advertise FP8 only when both the arch and the build have it."""
        from torchada.musa.build import BuildStamp

        assert self._supports_fp8("mp_31", 310, BuildStamp(("mp_31",), fp8=False)) is False

    def test_supports_fp8_without_a_stamp_answers_from_the_arch(self):
        """An editable install has no stamp; the arch still decides."""
        assert self._supports_fp8("mp_31", 310, None) is True

    def test_supports_fp4_is_false_without_fp4_hardware(self):
        """FP4 is an empty set, so the answer is False on every known arch.

        The question is "does this arch execute FP4 instructions"; an MX FP4
        checkpoint still runs, through dequantisation, on any arch.
        """
        from unittest.mock import patch

        from vllm_musa import device_spec
        from vllm_musa.platform import MUSAPlatformBase

        spec = self._spec("mp_31", 310)
        with patch.object(device_spec, "current_device_spec", return_value=spec):
            assert MUSAPlatformBase.supports_fp4() is False

    def test_the_validated_target_question_asks_the_current_device(self):
        """No argument means "the device this process runs on", like supports_fp8."""
        from unittest.mock import patch

        from vllm_musa import device_spec
        from vllm_musa.platform import MUSAPlatformBase

        spec = self._spec("mp_31", 310)
        seen: list = []

        def current():
            seen.append("current")
            return spec

        with (
            patch.object(device_spec, "current_device_spec", current),
            patch.object(
                device_spec, "get_device_spec", lambda device_id: seen.append(device_id)
            ),
        ):
            assert MUSAPlatformBase.is_validated_musa_target() is True
        assert seen == ["current"], "an indexed probe would answer for another GPU"

    def test_supports_fp4_is_false_even_on_a_future_fp8_arch(self):
        """No FP8-to-FP4 implication: the sets are independent."""
        from unittest.mock import patch

        from vllm_musa import device_spec
        from vllm_musa.platform import MUSAPlatformBase

        spec = self._spec("mp_31", 310)
        assert spec.supports_native_fp8 is True
        with patch.object(device_spec, "current_device_spec", return_value=spec):
            assert MUSAPlatformBase.supports_fp4() is False

    def test_supports_fp4_answers_the_same_two_halves_as_fp8(self):
        """The stamp records what the .so carries, so the arch alone cannot answer."""
        from unittest.mock import patch

        from torchada.musa.build import BuildStamp

        from vllm_musa import device_spec
        from vllm_musa.platform import MUSAPlatformBase

        # An arch with FP4 and a build without it: the device can, this wheel cannot.
        # A stand-in, because no arch in the table has FP4 yet and the real spec
        # derives that answer from the table rather than carrying it as a field.
        from types import SimpleNamespace

        spec = SimpleNamespace(
            arch_name="mp_41", arch_code=410, supports_native_fp4=True
        )
        with (
            patch.object(device_spec, "current_device_spec", return_value=spec),
            patch.object(
                device_spec,
                "read_build_stamp",
                return_value=BuildStamp(("mp_41",), fp4=False),
            ),
        ):
            assert MUSAPlatformBase.supports_fp4() is False
        with (
            patch.object(device_spec, "current_device_spec", return_value=spec),
            patch.object(
                device_spec,
                "read_build_stamp",
                return_value=BuildStamp(("mp_41",), fp4=True),
            ),
        ):
            assert MUSAPlatformBase.supports_fp4() is True

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
            use_mm_prefix=False,
            device_capability=DeviceCapability(3, 1),
        )

        assert reason is None


class TestNativeGemvSource:
    """Source-level checks for native MUSA GEMV dispatch gates."""

    def test_qwen_fp8_moe_uses_32x4_shape_gate(self):
        source = Path("csrc/musa/gemv.mu").read_text()

        assert "ShouldUseQwenFp8Moe32x4(" in source
        assert "nr_n == 768 || nr_n == 512 || nr_n == 384" in source
        assert "nr_n == 256 || nr_n == 128" in source
        assert "use_swigelu && hidden_size == 2048 && qwen_intermediate_size" in source
        assert "!use_swigelu && nr_n == 2048" in source
        assert "hidden_size == 512 || hidden_size == 384" in source
        assert "hidden_size == 256 || hidden_size == 128" in source
        assert "BlockConfig qwen_fp8_moe_config{32, 4" in source
        assert "case 4: GEN_LAUNCH_KERN(32, 4)" in source

    def test_deepseek_fp8_w1_uses_32x4_shape_gate(self):
        source = Path("csrc/musa/gemv.mu").read_text()

        assert "kDeepSeekFp8W1BlockEnv" not in source
        assert '"VLLM_MUSA_DEEPSEEK_FP8_W1_32X4"' not in source
        assert "ShouldUseDeepSeekFp8W1Moe32x4(" in source
        assert "topk == 6" in source
        assert "hidden_size == 2048" in source
        assert "reduce_size == 2816" in source
        assert "num_experts == 64" in source
        assert "BlockConfig deepseek_fp8_w1_config{32, 4" in source
        assert "best_config = &deepseek_fp8_w1_config" in source

    def test_deepseek_v4_split_tile_precedes_generic_override(self):
        source = Path("csrc/musa/gemv.mu").read_text()

        # A stale generic A/B override must not suppress the validated
        # DeepSeek-V4 split-tile production path.
        split_pos = source.index("if (ShouldUseDeepSeekV4Fp8MoeSplitTile(")
        forced_pos = source.index("} else if (ParseForcedBlockConfig(&forced_config))")
        assert split_pos < forced_pos
        assert "VLLM_MUSA_DEEPSEEK_FP8_W1_32X4" not in source

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

    def test_decorator_balances_context_and_returns_result(self, mock_pymtml):
        """Successful calls release the MTML lifecycle reference."""
        if "vllm_musa.platform" in sys.modules:
            del sys.modules["vllm_musa.platform"]

        from vllm_musa.platform import with_mtml_context

        mock_pymtml.reset_mock()

        @with_mtml_context
        def test_func():
            return "success"

        result = test_func()

        assert result == "success"
        assert mock_pymtml.nvmlInit.call_count == 1
        assert mock_pymtml.nvmlShutdown.call_count == 1

    def test_decorator_balances_context_when_wrapped_call_raises(self, mock_pymtml):
        """Exceptional calls release the MTML lifecycle reference."""
        if "vllm_musa.platform" in sys.modules:
            del sys.modules["vllm_musa.platform"]

        from vllm_musa.platform import with_mtml_context

        mock_pymtml.reset_mock()

        @with_mtml_context
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()

        assert mock_pymtml.nvmlInit.call_count == 1
        assert mock_pymtml.nvmlShutdown.call_count == 1

    def test_nested_decorators_balance_each_context(self, mock_pymtml):
        """Nested MTML users retain and release separate lifecycle handles."""
        if "vllm_musa.platform" in sys.modules:
            del sys.modules["vllm_musa.platform"]

        from vllm_musa.platform import with_mtml_context

        mock_pymtml.reset_mock()

        @with_mtml_context
        def inner():
            return "nested"

        @with_mtml_context
        def outer():
            return inner()

        assert outer() == "nested"
        assert mock_pymtml.nvmlInit.call_count == 2
        assert mock_pymtml.nvmlShutdown.call_count == 2

    def test_decorator_preserves_function_name(self):
        """Test that the decorator preserves the wrapped function's name."""
        from vllm_musa.platform import with_mtml_context

        @with_mtml_context
        def my_test_function():
            return "test"

        assert my_test_function.__name__ == "my_test_function"


class TestMtmlMUSAPlatform:
    """Tests for MtmlMUSAPlatform class."""

    def test_has_device_capability_reuses_cached_mtml_query(self, mock_pymtml):
        """Capability comparisons must not reinitialize MTML on every call."""
        if "vllm_musa.platform" in sys.modules:
            del sys.modules["vllm_musa.platform"]

        from vllm_musa.platform import MtmlMUSAPlatform

        MtmlMUSAPlatform.get_device_capability.cache_clear()
        mock_pymtml.reset_mock()

        results = [MtmlMUSAPlatform.has_device_capability((3, 1)) for _ in range(60)]
        results.append(MtmlMUSAPlatform.has_device_capability((4, 0)))

        assert results == [True] * 60 + [False]
        assert mock_pymtml.nvmlInit.call_count == 1
        assert mock_pymtml.nvmlShutdown.call_count == 1

    def test_mtml_availability_probe_balances_context(self, mock_pymtml):
        if "vllm_musa.platform" in sys.modules:
            del sys.modules["vllm_musa.platform"]

        mock_pymtml.reset_mock()
        platform = import_module("vllm_musa.platform")

        assert platform.mtml_available is True
        # One pair probes MTML and one runs MUSAPlatform.log_warnings().
        assert mock_pymtml.nvmlInit.call_count == 2
        assert mock_pymtml.nvmlShutdown.call_count == 2

    def test_mtml_availability_probe_does_not_shutdown_failed_init(self, mock_pymtml):
        if "vllm_musa.platform" in sys.modules:
            del sys.modules["vllm_musa.platform"]

        mock_pymtml.reset_mock()
        mock_pymtml.nvmlInit.side_effect = RuntimeError("MTML unavailable")

        platform = import_module("vllm_musa.platform")

        assert platform.mtml_available is False
        assert mock_pymtml.nvmlInit.call_count == 1
        assert mock_pymtml.nvmlShutdown.call_count == 0

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
