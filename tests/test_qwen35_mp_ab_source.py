from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "serving"
    / "benchmark_qwen35_mp_ab.py"
)


def test_qwen35_ab_uses_compiled_capture_path() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "tensor_parallel_size=4" in source
    assert "enforce_eager=False" in source
    assert "TokensPrompt(prompt_token_ids=prompt_ids)" in source
    assert "encoded.input_ids" in source
    assert "ignore_eos=True" in source
    assert "semantic_pass" in source


def test_qwen35_ab_requires_exact_output_and_semantics() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert 'parser.add_argument("--warmup", type=int, default=4)' in source
    assert 'parser.add_argument("--repeats", type=int, default=5)' in source
    assert "generated != args.batch_size * args.output_tokens" in source
    assert '"beijing" not in semantic_text.lower()' in source
