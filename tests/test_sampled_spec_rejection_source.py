# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source contracts for MUSA sampled speculative rejection support."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"
REJECTION_PATCH = SERIES / "0028-MUSA-vllm.v1.sample.rejection_sampler.patch"
MODEL_RUNNER_PATCH = SERIES / "0035-MUSA-vllm.v1.worker.gpu_model_runner.patch"


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
