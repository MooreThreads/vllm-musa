import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

BENCH_PATH = (
    Path(__file__).parents[1]
    / "benchmarks/kernel_tactics/benchmark_dsv4_weighted_rmsnorm_jit.py"
)
sys.path.insert(0, str(BENCH_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "benchmark_dsv4_weighted_rmsnorm_jit", BENCH_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


@pytest.mark.parametrize("value", ["64", "128", "256"])
def test_parse_threads_accepts_supported_launch_geometry(value):
    assert BENCH.parse_threads(value) == int(value)


@pytest.mark.parametrize("value", ["bad", "32", "192", "512"])
def test_parse_threads_rejects_unqualified_launch_geometry(value):
    with pytest.raises(argparse.ArgumentTypeError):
        BENCH.parse_threads(value)


def test_production_threads_match_current_provider():
    assert BENCH.PRODUCTION_THREADS == 128


def test_candidate_kernel_has_complete_256_thread_reduction_stage():
    source = (
        Path(__file__).parents[1] / "vllm_musa/deepseek_v4_jit/tilelang_kernels.py"
    ).read_text()
    assert "assert threads in (64, 128, 256)" in source
    assert "if threads >= 256:" in source
    assert "sumsq[0] += shared[tx + 128]" in source


def test_weighted_rmsnorm_benchmark_requires_lease_device_fence():
    source = BENCH_PATH.read_text(encoding="utf-8")
    assert '"--expected-physical-device", type=int, required=True' in source
    assert '"--expected-device-uuid", required=True' in source
    assert '"lease_device_fence": lease_device_fence' in source
