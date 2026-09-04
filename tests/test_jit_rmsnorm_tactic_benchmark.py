import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BENCH_PATH = ROOT / "benchmarks/kernel_tactics/benchmark_jit_rmsnorm_threads.py"
API_PATH = ROOT / "vllm_musa/jit_kernel/csrc/norm.py"
sys.path.insert(0, str(BENCH_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "benchmark_jit_rmsnorm_threads", BENCH_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


@pytest.mark.parametrize("value", ["32", "128", "640", "1024"])
def test_parse_threads_accepts_compiled_launch_geometry(value):
    assert BENCH.parse_threads(value) == int(value)


@pytest.mark.parametrize("value", ["bad", "0", "31", "129", "768"])
def test_parse_threads_rejects_unlisted_geometry(value):
    with pytest.raises(argparse.ArgumentTypeError):
        BENCH.parse_threads(value)


@pytest.mark.parametrize(
    ("mode", "rows", "hidden_size", "expected"),
    [
        ("plain", 1, 1024, None),
        ("plain", 16, 2048, None),
        ("plain", 32, 4096, 256),
        ("plain", 512, 4096, 128),
        ("plain", 1, 5120, 512),
        ("fused", 64, 5120, 640),
        ("fused_gemma", 4, 1536, 192),
    ],
)
def test_production_thread_resolver(mode, rows, hidden_size, expected):
    assert BENCH.production_threads(mode, rows, hidden_size) == expected


def test_jit_rmsnorm_contract_exposes_default_preserving_requested_threads():
    wrapper = (ROOT / "vllm_musa/jit_kernel/csrc/norm.py").read_text()
    kernel = (ROOT / "vllm_musa/jit_kernel/csrc/norm/rmsnorm.mu").read_text()

    assert wrapper.count("block_threads: int = 0") >= 3
    assert "int requested_threads" in kernel
    assert "check_requested_threads" in kernel
    assert "requested_threads == 0 ||" in kernel
    assert "requested_threads > 0 ? requested_threads" in kernel
    assert "requested_threads == 0 && rows <= 16 && hidden == 1024" in kernel


def test_jit_rmsnorm_timing_resets_mutable_inputs_outside_events():
    source = BENCH_PATH.read_text()
    launch = source[
        source.index("            def launch(") : source.index(
            "            reset_mutable_inputs()",
            source.index("            def launch("),
        )
    ]
    assert "copy_(" not in launch
    assert "reset_mutable_inputs()\n                        flush.zero_()" in source
    assert '"--expected-physical-device", type=int, required=True' in source
    assert '"--expected-device-uuid", required=True' in source
    assert '"lease_device_fence": lease_device_fence' in source


def test_jit_rmsnorm_benchmark_primes_hardware_for_production_baselines():
    benchmark = BENCH_PATH.read_text(encoding="utf-8")
    assert "prime_musa_kernel_hardware" in benchmark
    assert '"primed_hardware": {' in benchmark


def test_jit_rmsnorm_runtime_uses_exact_primed_hardware_tactic():
    source = API_PATH.read_text(encoding="utf-8")
    assert "get_primed_musa_kernel_hardware" in source
    assert "select_jit_rmsnorm_tactic" in source
    assert 'mode="fused_gemma" if gemma else "fused"' in source
    assert "return tactic.block_threads if tactic is not None else 0" in source


@pytest.mark.parametrize(
    ("mode", "rows"),
    [
        ("plain", 512),
        ("gemma", 512),
        ("gemma", 4096),
        ("fused_gemma", 512),
    ],
)
def test_production_thread_resolver_uses_exact_mp60_h5120_tactic(mode, rows):
    assert BENCH.production_threads(mode, rows, 5120, 60) == 320
    assert BENCH.production_threads(mode, rows, 5120, 48) != 320
    if rows == 512:
        assert BENCH.production_threads(mode, rows, 5120, 56) == 320
    else:
        assert BENCH.production_threads(mode, rows, 5120, 56) != 320


@pytest.mark.parametrize("mode", ["plain", "gemma", "fused", "fused_gemma"])
def test_production_thread_resolver_uses_exact_mp56_h5120_tactic(mode):
    assert BENCH.production_threads(mode, 512, 5120, 56) == 320
    assert BENCH.production_threads(mode, 512, 5120, 48) != 320


def test_production_thread_resolver_uses_exact_mp48_plain_h5120_tactic():
    assert BENCH.production_threads("plain", 512, 5120, 48) == 256


@pytest.mark.parametrize(
    ("mode", "fallback_threads"),
    [("gemma", 512), ("fused", 640), ("fused_gemma", 640)],
)
def test_production_thread_resolver_rejects_non_plain_mp48_tactic(
    mode, fallback_threads
):
    assert BENCH.production_threads(mode, 512, 5120, 48) == fallback_threads
