# SPDX-License-Identifier: Apache-2.0
"""Source contract for the Qwen3.6 BF16 C25 W1-only MoE tile."""

from pathlib import Path


PATCH = (
    Path(__file__).resolve().parent.parent
    / "vllm_musa"
    / "patches"
    / "series"
    / "0105-perf-specialize-Qwen3.6-BF16-W1-tile-at-C25.patch"
)


def test_qwen36_bf16_c25_specializes_only_w1_config():
    source = PATCH.read_text()

    assert "current_platform.is_musa()" in source
    assert "hidden_states.dtype == torch.bfloat16" in source
    assert "M == 25" in source
    assert "tuple(w1.shape) == (257, 1024, 2048)" in source
    assert "tuple(w2.shape) == (257, 2048, 512)" in source
    assert "top_k_num == 9" in source
    assert "global_num_experts == 257" in source
    assert "expert_map is None" in source
    assert "block_shape is None" in source

    assert 'config.get("BLOCK_SIZE_M") == 32' in source
    assert 'config.get("BLOCK_SIZE_N") == 32' in source
    assert 'config.get("BLOCK_SIZE_K") == 64' in source
    assert 'config.get("num_warps") == 4' in source
    assert "BLOCK_SIZE_N=64" in source
    assert "BLOCK_SIZE_K=128" in source
    assert "num_warps=8" in source

    # One use aligns experts for W1 and one dispatches W1. The untouched W2
    # dispatch remains on ``config`` because it is outside this narrow diff.
    assert source.count("+        w1_config,") == 2
    assert "VLLM_MUSA" not in source
