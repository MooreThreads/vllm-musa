# SPDX-License-Identifier: Apache-2.0
"""Source contract for the Qwen unit-temperature divide skip."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0090-perf-musa-unify-Qwen-runtime-fast-paths.patch"
)
SAMPLER = ROOT / "vllm_musa" / "v1" / "sample" / "topk_topp_sampler.py"


def test_qwen_unit_temperature_metadata_gate_is_fail_closed() -> None:
    source = PATCH.read_text()

    assert "uniform_temperature: float | None = None" in source
    assert "self.vocab_size in (151936, 248320)" in source
    assert "self.all_random" in source
    assert "num_reqs > 0" in source
    assert "temperature_cpu == np.float32(1.0)" in source
    assert "VLLM_MUSA_QWEN_SKIP_UNIT_TEMPERATURE" not in source
    assert "uniform_temperature=uniform_temperature" in source
    assert "resolve_optimization_contract" in source
    assert "QWEN_LEGACY_SAMPLING" in source


def test_qwen_unit_temperature_sampler_keeps_original_fallback() -> None:
    source = SAMPLER.read_text()

    assert "def _can_skip_legacy_qwen_unit_temperature" in source
    assert "prefers_optimization" in source
    assert "QWEN_LEGACY_SAMPLING" in source
    assert 'getattr(sampling_metadata, "all_random", False)' in source
    assert "_is_qwen_sampler_vocab(logits)" in source
    assert 'sampling_metadata, "uniform_temperature", None' in source
    assert "if not _can_skip_legacy_qwen_unit_temperature" in source
    assert "logits = self.apply_temperature(" in source
