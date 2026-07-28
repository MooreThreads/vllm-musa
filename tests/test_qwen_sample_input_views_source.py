from pathlib import Path


def test_sample_input_view_patch_has_fallback_and_gate():
    root = Path(__file__).parents[1]
    source = (
        root / "vllm_musa/patches/series/0096-perf-view-qwen-sampling-inputs.patch"
    ).read_text()
    assert "_musa_select_qwen_sample_input_views" in source
    assert "input_batch.positions[input_batch.logits_indices]" in source
    assert "input_batch.input_ids[input_batch.logits_indices]" in source
    assert "sample_inputs is None" in source
    assert "return_logprobs" in source
