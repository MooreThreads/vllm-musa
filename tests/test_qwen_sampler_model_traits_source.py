"""Source contract for contract-bound Qwen sampler fast paths."""

import ast

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0091-perf-musa-unify-Qwen-runtime-fast-paths.patch"
)
AUTO_FAST_PATH_SOURCES = (
    ROOT / "vllm_musa/v1/sample/topk_topp_sampler.py",
    ROOT / "vllm_musa/v1/sample/uniform_sample_counts.py",
    ROOT / "vllm_musa/v1/sample/qwen_sample_input_views.py",
    ROOT / "vllm_musa/v1/worker/qwen_identity_logits_view.py",
    ROOT / "vllm_musa/v1/worker/qwen_fused_next_input_ids.py",
)


def test_sampler_traits_are_resolved_by_the_contract() -> None:
    source = PATCH.read_text()

    assert "VLLM_MUSA_QWEN" not in source
    assert "resolve_optimization_contract" in source
    assert "_musa_optimization_contract" in source
    assert "QWEN_LEGACY_SAMPLING" in source
    assert "_musa_qwen_family" not in source


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
    assert "prefers_optimization" in sources[AUTO_FAST_PATH_SOURCES[0]]
    assert "QWEN_UNIFORM_SAMPLE_COUNTS" in sources[AUTO_FAST_PATH_SOURCES[1]]
    assert "QWEN_SAMPLE_INPUT_VIEWS" in sources[AUTO_FAST_PATH_SOURCES[2]]
    assert "QWEN_UNIFORM_DECODE_VIEWS" in sources[AUTO_FAST_PATH_SOURCES[3]]
    assert "_is_qwen_runner" in sources[AUTO_FAST_PATH_SOURCES[4]]
    assert "_is_qwen_runner(runner)" in sources[AUTO_FAST_PATH_SOURCES[4]]
    assert "worker_cls.sample = _worker_sample" in sources[AUTO_FAST_PATH_SOURCES[0]]
    assert "_worker_sample_unfiltered_gumbel" not in sources[AUTO_FAST_PATH_SOURCES[0]]


def test_worker_sampler_signature_matches_v028_sampler_call() -> None:
    """The v0.28 Sampler.__call__ passes idx_mapping before idx_mapping_np."""
    source = (ROOT / "vllm_musa/v1/sample/topk_topp_sampler.py").read_text()
    tree = ast.parse(source)
    worker = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_worker_sample"
    )
    positional_names = [arg.arg for arg in worker.args.args]
    assert positional_names[:8] == [
        "self",
        "logits",
        "expanded_idx_mapping",
        "idx_mapping",
        "idx_mapping_np",
        "pos",
        "input_ids",
        "expanded_local_pos",
    ]
    fallback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "original_sample"
    )
    assert [keyword.arg for keyword in fallback.keywords] == ["return_logprobs"]
    assert [arg.id for arg in fallback.args[:4]] == [
        "self",
        "logits",
        "expanded_idx_mapping",
        "idx_mapping",
    ]
