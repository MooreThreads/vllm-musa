# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the BF16 DeepGEMM MoE prefill path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MOE_SOURCE = (
    ROOT
    / "vllm_musa"
    / "model_executor"
    / "layers"
    / "fused_moe"
    / "fused_moe.py"
)
PREPROCESS_SOURCE = (
    ROOT
    / "vllm_musa"
    / "jit_kernel"
    / "tilelang"
    / "deep_gemm_contig_preprocess.py"
)
WARMUP_SOURCE = (
    ROOT / "vllm_musa" / "model_executor" / "warmup" / "deep_gemm_warmup.py"
)


def test_bf16_deepgemm_prefill_uses_native_bf16_gemm_by_default():
    source = MOE_SOURCE.read_text()

    assert "VLLM_MUSA_MOE_BF16_DEEPGEMM_PREFILL" not in source
    assert "_DEEPGEMM_BF16_PREFILL_MIN_TOKENS = 1024" in source
    assert "def _can_use_moe_deepgemm_bf16_prefill(" in source
    assert source.count("deep_gemm.m_grouped_bf16_gemm_nt_contiguous(") == 2
    assert "torch.ops._C.silu_and_mul(act_out, mm1_out)" in source
    assert "falling back to the upstream Triton path" in source


def test_bf16_tilelang_preprocess_has_compaction_and_shape_gate():
    source = PREPROCESS_SOURCE.read_text()

    assert "def _bf16_assign_compact_kernel(" in source
    assert "def deep_gemm_contig_preprocess_bf16_tilelang(" in source
    assert "def can_use_bf16_tilelang(" in source
    assert '"sgl_tl_copy_bf16x8"' in source


def test_bf16_deepgemm_warmup_covers_scheduler_shapes_by_default():
    source = WARMUP_SOURCE.read_text()

    assert "def _warmup_bf16_grouped_moe(" in source
    assert "VLLM_MUSA_MOE_BF16_DEEPGEMM_PREFILL" not in source
    assert "_aligned_grouped_tokens" in source
    assert "1024, 2048, 4096, 8192, 12288" in source
    assert "m_grouped_bf16_gemm_nt_contiguous" in source
