# SPDX-License-Identifier: Apache-2.0
"""Packaging contracts for the Qwen3.5/3.6 return-through patch."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa/patches/series/0099-perf-musa-return-Qwen-attention-projection-outputs.patch"
)
MUSA_GDN = ROOT / "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"


def test_patch_uses_canonical_format_patch_headers() -> None:
    source = PATCH.read_text()

    assert source.startswith(
        "From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001\n"
    )
    assert (
        "Subject: [PATCH] perf(musa): return Qwen attention projection outputs"
        in source
    )
    assert "Subject: [PATCH " not in source
    assert not source.rstrip().endswith("2.34.1")


def test_patch_carries_phase_gate_and_both_qwen_initializers() -> None:
    source = PATCH.read_text()

    assert "vllm/model_executor/models/qwen3_5.py" in source
    assert "VLLM_MUSA_QWEN_NEXT_RETURN_ATTN_OUTPUT" in source
    assert "current_platform.is_musa()" in source
    assert "model_config.dtype == torch.bfloat16" in source
    assert "quant_config is None" in source
    assert 'getattr(vllm_config, "lora_config", None) is None' in source
    assert "config.hidden_size == 5120" in source
    assert "get_tensor_model_parallel_world_size() == 2" in source
    assert "hidden_states.shape[0] >= 1024" in source
    assert "_is_musa_qwen_next_prefill(self._musa_attention_metadata_prefix)" in source
    assert source.count("_use_musa_qwen_next_return_attention_output(") >= 3


def test_musa_gdn_oot_propagates_returned_projection() -> None:
    source = MUSA_GDN.read_text()

    assert "output: torch.Tensor | None" in source
    assert (
        "return self._output_projection(core_attn_out, z, output, num_tokens)" in source
    )
