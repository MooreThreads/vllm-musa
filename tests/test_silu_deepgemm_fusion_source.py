# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source checks for dense MUSA SwiGLU plus DeepGEMM fusion."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fused_deepgemm_op_uses_shipped_silu_group_quant_kernel():
    source = (
        ROOT
        / "vllm_musa"
        / "model_executor"
        / "kernels"
        / "linear"
        / "scaled_mm"
        / "deep_gemm.py"
    ).read_text()

    assert "def _musa_silu_deepgemm_fp8_op(" in source
    assert "silu_and_mul_per_token_group_fp8_quant(" in source
    assert "fp8_gemm_nt(" in source
    assert '"musa_silu_deepgemm_fp8_op"' in source
    assert "fake_impl=_musa_silu_deepgemm_fp8_op_fake" in source


def test_fusion_matches_the_production_native_swiglu_deepgemm_pair():
    source = (
        ROOT / "vllm_musa" / "compilation" / "passes" / "silu_deepgemm_fusion.py"
    ).read_text()

    assert "MatcherSiluAndMul(enabled=False)" in source
    assert "from torch._higher_order_ops.auto_functionalize import" in source
    assert "return auto_functionalized(" in source
    assert "torch.ops.vllm.musa_deepgemm_fp8_op.default," in source
    assert "group_size=128," in source
    assert "use_deep_gemm_e8m0=False," in source
    assert "output=None," in source
    assert ")[0]" in source
    assert "torch.ops.vllm.musa_silu_deepgemm_fp8_op(" in source
    assert source.count("\n                128,") == 1
    assert source.count("\n                False,") == 1


def test_pass_manager_keeps_the_experimental_scope_narrow():
    source = (
        ROOT / "vllm_musa" / "compilation" / "passes" / "pass_manager.py"
    ).read_text()

    assert "_silu_deepgemm_fusion_requested(config)" in source
    assert "VLLM_MUSA_CUSTOM_OP_USE_NATIVE" not in source
    assert "_is_dense_model(config)" in source
    assert "_use_row_major_activation_scales(False)" in source
    assert "self.pass_config.fuse_act_quant" in source
    assert "self.passes.append(MusaSiluDeepGemmFusionPass(config))" in source


def test_pass_manager_uses_independent_standard_fusion_flags():
    source = (
        ROOT / "vllm_musa" / "compilation" / "passes" / "pass_manager.py"
    ).read_text()

    silu_block, rms_block = source.split(
        "        if (\n            _rms_deepgemm_fusion_requested(config)", 1
    )
    assert "self.pass_config.fuse_act_quant" in silu_block
    assert "self.pass_config.fuse_norm_quant" not in silu_block
    assert "self.pass_config.fuse_norm_quant" in rms_block
    assert "self.pass_config.fuse_act_quant" not in rms_block


def test_runtime_plan_auto_enablement_is_limited_to_the_validated_scope():
    platform_source = (ROOT / "vllm_musa" / "platform.py").read_text()
    qwen_source = (ROOT / "vllm_musa" / "runtime_plan" / "qwen.py").read_text()
    policy_source = (ROOT / "vllm_musa" / "runtime_plan" / "policy.py").read_text()

    assert "def _is_validated_qwen3_8b_fp8_single_gpu(" not in platform_source
    assert "def _is_qwen2_rope_kv_fusion_config(" not in platform_source
    assert "def _is_qwen3_qk_rope_kv_fusion_config(" not in platform_source
    assert "def _has_routed_experts(" not in platform_source
    assert "def _deepseek_v4_flashmla_sparse_block_size(" not in platform_source
    assert "def runtime_plan_enabled(" in policy_source
    assert "def deepseek_v4_flashmla_sparse_page_size(" in policy_source
    assert "== (4096, 12288, 36)" in qwen_source
    assert "model.hidden_size == 2048" in qwen_source
    assert "VLLM_MUSA_SILU_DEEPGEMM_FUSION" not in platform_source

    environ_source = (ROOT / "vllm_musa" / "utils" / "environ.py").read_text()
    assert "VLLM_MUSA_SILU_DEEPGEMM_FUSION" not in environ_source

    pass_manager_source = (
        ROOT / "vllm_musa" / "compilation" / "passes" / "pass_manager.py"
    ).read_text()
    assert "def _silu_deepgemm_fusion_requested(" in pass_manager_source
    assert "VLLM_MUSA_SILU_DEEPGEMM_FUSION" not in pass_manager_source
    assert "VLLM_MUSA_CUSTOM_OP_USE_NATIVE" not in pass_manager_source
    assert "policy.runtime_plan_enabled(" in pass_manager_source
    assert "QWEN3_DENSE_FP8_POST_GRAD_FUSIONS" in pass_manager_source
    assert "_has_validated_musa_device_capability()" in pass_manager_source
    assert "torch.musa.current_device()" in pass_manager_source
    assert "torch.musa.get_device_capability(device_id)" in pass_manager_source
    assert "current_platform.is_device_capability" not in pass_manager_source
