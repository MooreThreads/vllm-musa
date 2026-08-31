from pathlib import Path

BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "fused_moe"
    / "benchmark_dispatch_crossover.py"
)


def _source() -> str:
    return BENCHMARK.read_text(encoding="utf-8")


def test_bf16_crossover_uses_production_native_gate() -> None:
    source = _source()
    assert '"--weight-dtype"' in source
    assert 'choices=("fp8", "bf16")' in source
    assert "_can_use_musa_native_bf16_moe_gemv" in source
    assert '"BF16 requires exactly gemv and upstream backends"' in source


def test_bf16_crossover_has_upstream_correctness_reference() -> None:
    source = _source()
    assert 'oracle_outputs["upstream"].float()' in source
    assert 'oracle_reference = "upstream-bf16"' in source
    assert "w1_scale = None" in source
    assert "w2_scale = None" in source


def test_bf16_crossover_primes_hardware_and_records_archive_source() -> None:
    source = _source()
    assert "prime_musa_kernel_hardware" in source
    assert "frozen MUSA hardware fingerprint changed" in source
    assert '"--source-revision"' in source
    assert '"source_kind": "archive"' in source


def test_bf16_crossover_accepts_intentional_prefix_fallback() -> None:
    source = _source()
    assert "expected_backend(requested: str)" in source
    assert "shape_matcher_max_tokens" in source
    assert '"expected_backend": resolved' in source
