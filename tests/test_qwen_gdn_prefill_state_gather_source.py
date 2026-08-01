# SPDX-License-Identifier: Apache-2.0
"""Source contract for the fused Qwen GDN prefill state gather."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "vllm_musa/jit_kernel/gdn_state_gather_mask.py"
CALLER = ROOT / "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"


def test_fused_gdn_state_gather_has_narrow_shape_and_dtype_gate() -> None:
    source = KERNEL.read_text()
    assert "_QWEN_GDN_LOCAL_HEAD_COUNTS = (8, 16, 32)" in source
    assert "tuple(state.shape[2:]) == (128, 128)" in source
    assert "state.dtype == torch.float32" in source
    assert "state_indices.dtype == torch.int32" in source
    assert "has_initial_state.dtype == torch.bool" in source
    assert "num_sequences == 64" in source
    assert "_SMALL_HEAD_BLOCK_SIZE if state.shape[1] == 8" in source
    assert "if state.shape[1] == 8" in source


def test_qwen_gdn_prefill_preserves_fallback_and_scatter() -> None:
    source = CALLER.read_text()
    assert "if fused_initial_state is None:" in source
    assert "initial_state = ssm_state[state_indices].to(torch.float32)" in source
    assert "initial_state[~has_initial_state, ...] = 0" in source
    assert "ssm_state.index_copy_(" in source
