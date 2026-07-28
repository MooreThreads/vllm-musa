"""Source contract for attribute-gated Qwen sampler fast paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0090-perf-musa-unify-Qwen-runtime-fast-paths.patch"
)
AUTO_FAST_PATH_SOURCES = (
    ROOT / "vllm_musa/v1/sample/topk_topp_sampler.py",
    ROOT / "vllm_musa/v1/sample/uniform_sample_counts.py",
    ROOT / "vllm_musa/v1/sample/qwen_sample_input_views.py",
    ROOT / "vllm_musa/v1/worker/qwen_identity_logits_view.py",
    ROOT / "vllm_musa/v1/worker/qwen_fused_next_input_ids.py",
)


def test_sampler_traits_are_derived_from_model_architecture() -> None:
    source = PATCH.read_text()

    assert "VLLM_MUSA_QWEN" not in source
    assert (
        "+        self.sampler._musa_qwen_family = is_musa_qwen_text_generation"
        in source
    )
    assert "+            qwen_sampling_architectures = {" in source
    assert "+            self.sampler._musa_qwen_family = any(" in source
    for architecture in (
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        assert architecture in source


def test_both_runner_generations_propagate_qwen_traits() -> None:
    source = PATCH.read_text()

    assert (
        "diff --git a/vllm/v1/worker/gpu_model_runner.py "
        "b/vllm/v1/worker/gpu_model_runner.py"
    ) in source
    assert (
        "diff --git a/vllm/v1/worker/gpu/model_runner.py "
        "b/vllm/v1/worker/gpu/model_runner.py"
    ) in source


def test_followup_qwen_fast_paths_use_model_traits_without_env_flags() -> None:
    sources = {path: path.read_text() for path in AUTO_FAST_PATH_SOURCES}

    assert all("VLLM_MUSA_QWEN" not in source for source in sources.values())
    assert "_musa_qwen_family" in sources[AUTO_FAST_PATH_SOURCES[0]]
    assert "_musa_qwen_family" in sources[AUTO_FAST_PATH_SOURCES[1]]
    assert "_musa_qwen_family" in sources[AUTO_FAST_PATH_SOURCES[2]]
    assert "_musa_qwen_family" in sources[AUTO_FAST_PATH_SOURCES[3]]
    assert "_is_qwen_runner(runner)" in sources[AUTO_FAST_PATH_SOURCES[4]]
    assert "worker_cls.sample = _worker_sample" in sources[AUTO_FAST_PATH_SOURCES[0]]
    assert "_worker_sample_unfiltered_gumbel" not in sources[AUTO_FAST_PATH_SOURCES[0]]
