# SPDX-License-Identifier: Apache-2.0
"""Source contract for the Qwen3.5/3.6 prefill return-through path."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa/patches/series/0099-perf-musa-return-Qwen-attention-projection-outputs.patch"
)
MUSA_GDN = (
    ROOT / "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
)


def test_return_through_is_exactly_gated() -> None:
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
    assert source.count("_use_musa_qwen_next_return_attention_output(") >= 3


def test_full_and_gdn_attention_return_projection_without_copy() -> None:
    source = PATCH.read_text()

    assert source.count("if output is None:") == 2
    assert source.count("return projected") == 2
    assert "None if return_attention_output else torch.empty_like(hidden_states)" in source
    assert "returned_attention_output" in source
    assert "output[:] = projected" in source
    assert "output[:num_tokens] = projected" in source


def test_musa_gdn_oot_propagates_returned_projection() -> None:
    source = MUSA_GDN.read_text()

    assert "output: torch.Tensor | None" in source
    assert "return self._output_projection(core_attn_out, z, output, num_tokens)" in source
