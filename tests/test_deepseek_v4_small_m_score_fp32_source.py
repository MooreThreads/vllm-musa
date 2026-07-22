# SPDX-License-Identifier: Apache-2.0
"""Source contracts for DeepSeek-V4 small-M FP32 score GEMM dispatch."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    REPO_ROOT
    / "vllm_musa/patches/series/0090-MUSA-DeepSeek-V4-small-M-score-FP32-DeepGEMM.patch"
)


def test_patch_uses_fixed_small_m_policy_without_ab_env_controls():
    source = PATCH.read_text()

    assert "VLLM_MUSA_DEEPSEEK_V4_SCORE_FP32_GEMM_IMPL" not in source
    assert "VLLM_MUSA_DEEPSEEK_V4_SCORE_FP32_DEEPGEMM_MAX_TOKENS" not in source
    assert "VLLM_USE_DEEP_GEMM" not in source
    assert "REUSE_SCORE_FP32_CAST" not in source
    assert "_MUSA_DEEPSEEK_V4_SCORE_FP32_DEEPGEMM_MAX_TOKENS = 16" in source
    assert '== "torch"' not in source
    assert '== "deepgemm"' not in source
    assert "deep_gemm.bf16_gemm_nt(a, weight, output)" in source
    assert "From: musa <musa@local>" in source


def test_patch_helper_requires_bf16_inputs_and_fp32_output():
    source = PATCH.read_text()

    assert "a.dtype == torch.bfloat16" in source
    assert "weight.dtype == torch.bfloat16" in source
    assert "out_dtype == torch.float32" in source
    assert "a.shape[0]" in source
    assert "_MUSA_DEEPSEEK_V4_SCORE_FP32_GEMM_IMPL" not in source
    assert "_MUSA_DEEPSEEK_V4_USE_DEEP_GEMM" not in source


def test_patch_has_no_score_cast_reuse_dependency():
    source = PATCH.read_text()

    assert "REUSE_SCORE_FP32_CAST" not in source
    assert "hidden_states_fp32" not in source
