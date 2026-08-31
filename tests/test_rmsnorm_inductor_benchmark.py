from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "kernel_tactics" / "benchmark_rmsnorm_inductor.py"
CANDIDATE = ROOT / "benchmarks" / "kernel_tactics" / "_rmsnorm_inductor_candidate.py"


def test_inductor_rmsnorm_benchmark_is_paired_cold_l2_and_fenced() -> None:
    source = BENCHMARK.read_text(encoding="utf-8")
    assert '"--num-warps"' in source
    assert 'choices=("op", "scale")' in source
    assert "verify_lease_device_fence(" in source
    assert '"paired_alternating_order": True' in source
    assert '"flush_before_every_timed_launch": True' in source
    assert '"inductor_cache_disabled": True' in source
    assert '"candidate_kind": "exploratory-rejected"' in source
    assert "vllm_musa.kernels.rmsnorm_inductor" not in source


def test_inductor_rmsnorm_candidate_has_no_ir_registration_side_effect() -> None:
    source = CANDIDATE.read_text(encoding="utf-8")
    assert "wrap_triton(_rms_norm_kernel)[grid](" in source
    assert "register_impl" not in source
    assert "direct_register_custom_op" not in source
    assert "vllm_musa.kernels" not in source
