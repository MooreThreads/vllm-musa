# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

PATCH = (
    Path(__file__).parents[1]
    / "vllm_musa/patches/series/0095-perf-fuse-qwen-next-input-ids.patch"
)


def test_qwen_fused_next_input_ids_patch_wires_both_boundaries() -> None:
    source = PATCH.read_text()
    assert 'getattr(self, "_musa_select_uniform_decode_model_inputs", None)' in source
    assert 'getattr(self, "_musa_select_next_input_ids_buffer", None)' in source
    assert "next_input_ids=next_input_ids" in source
    assert "WRITE_NEXT_INPUT_IDS=next_input_ids is not None" in source
    assert "tl.store(next_input_ids_ptr + req_id, token_id)" in source
    assert "input_ids=input_ids" in source
    assert "idx_mapping_np," not in source
