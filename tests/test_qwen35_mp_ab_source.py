from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "serving"
    / "benchmark_qwen35_mp_ab.py"
)
REPLAY_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "serving"
    / "benchmark_qwen35_graph_bucket_replay.py"
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
    assert 'parser.add_argument("--max-num-seqs", type=int)' in source
    assert "max_num_seqs = args.max_num_seqs or args.batch_size" in source
    assert "max_num_seqs=max_num_seqs" in source
    assert 'dispatch_policy = {"baseline": "upstream", "candidate": "auto"}' in source
    assert "os.environ[DISPATCH_ENV] = dispatch_policy" in source
    assert "generated != args.batch_size * args.output_tokens" in source
    assert '"beijing" not in semantic_text.lower()' in source


def test_qwen35_graph_bucket_replay_uses_one_general_engine() -> None:
    source = REPLAY_SOURCE.read_text(encoding="utf-8")
    assert 'parser.add_argument("--max-num-seqs", type=int, default=64)' in source
    assert 'default="1,16,4,16,1,2,16"' in source
    assert "llm = LLM(" in source
    assert "max_num_seqs=args.max_num_seqs" in source
    assert '"--cudagraph-capture-sizes"' in source
    assert 'default="1,2,4,16,64"' in source
    assert "required_capture_sizes = {1, 2, 4}" in source
    assert '"cudagraph_capture_sizes": args.cudagraph_capture_sizes' in source
    assert 'dispatch_policy = {"baseline": "upstream", "candidate": "auto"}' in source
    assert "os.environ[DISPATCH_ENV] = dispatch_policy" in source
    assert "for step, batch_size in enumerate(args.batch_sequence):" in source
    assert "enforce_eager=False" in source
    assert "inspect_compile_state(llm)" in source
    assert '"compile_state": compile_state' in source
    assert "del llm" in source
    assert "gc.collect()" in source
