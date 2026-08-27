# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source contracts for MUSA sampled speculative rejection support."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"
REJECTION_PATCH = (
    SERIES / "0028-MUSA-vllm.v1.sample.rejection_sampler-v0.28-reanchor.patch"
)
MODEL_RUNNER_PATCH = (
    SERIES / "0035-MUSA-vllm.v1.worker.gpu_model_runner-v0.28-reanchor.patch"
)


def _series_patch(number: int) -> str:
    matches = list(SERIES.glob(f"{number:04d}-*.patch"))
    assert len(matches) == 1
    return matches[0].read_text()


def test_sampled_spec_decode_has_no_environment_guard() -> None:
    patch_sources = "\n".join(
        path.read_text() for path in sorted(SERIES.glob("*.patch"))
    )

    assert "VLLM_MUSA_SPEC_DECODE_RANDOM_FALLBACK" not in patch_sources
    assert "_musa_sample_first_target_token" not in patch_sources


def test_rejection_sampler_keeps_musa_random_kernel_compatibility() -> None:
    source = REJECTION_PATCH.read_text()

    assert ").to(torch.int32)" in source
    assert "if synthetic_conditional_rates is None:" in source
    assert "if is_greedy == 0:" in source
    assert "if is_greedy != 0:" in source
    assert "rejection_random_sample_kernel" in source


def test_rejection_sampler_uses_correct_top_p_mask_on_musa() -> None:
    source = REJECTION_PATCH.read_text()

    assert "if current_platform.is_musa():" in source
    assert "return apply_top_k_top_p_pytorch(logits, top_k, top_p)" in source


def test_model_runner_does_not_skip_non_greedy_drafter() -> None:
    source = MODEL_RUNNER_PATCH.read_text()

    assert "not self.input_batch.sampling_metadata.all_greedy" not in source
    assert "input_fits_in_drafter = False" not in source


def test_mrv2_rejection_predicates_keep_musa_triton_rank_stable() -> None:
    greedy = _series_patch(125)
    block_verification = _series_patch(128)
    stochastic = _series_patch(129)

    for source in (greedy, block_verification, stochastic):
        assert "accepted |= tl.zeros((1,), tl.int1)" in source
        assert "accepted = tl.sum(accepted.to(tl.int32), axis=0) != 0" in source


def test_mrv2_rejection_avoids_chained_runtime_predicates() -> None:
    source = _series_patch(126)

    assert "if USE_BLOCK_VERIFICATION and (" in source
    assert "(not is_greedy) & (accepted_length < num_draft_tokens)" in source


def test_mrv2_rejection_resampler_uses_v028_gumbel_signature() -> None:
    source = _series_patch(127)

    assert "residual_logits,\n         block,\n+        block," in source
