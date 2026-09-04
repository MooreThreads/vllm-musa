import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

BENCH_PATH = (
    Path(__file__).parents[1]
    / "benchmarks/kernel_tactics/benchmark_dsv4_mhc_post_jit.py"
)
sys.path.insert(0, str(BENCH_PATH.parent))
SPEC = importlib.util.spec_from_file_location("benchmark_dsv4_mhc_post_jit", BENCH_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


def test_parse_config_accepts_full_write_geometry():
    assert BENCH.parse_config("64x64") == (64, 64)
    assert BENCH.parse_config("128X256") == (128, 256)


@pytest.mark.parametrize("value", ["bad", "0x64", "128x64", "64x0"])
def test_parse_config_rejects_invalid_or_partial_write_geometry(value):
    with pytest.raises(argparse.ArgumentTypeError):
        BENCH.parse_config(value)


def test_production_config_matches_current_provider_defaults():
    source = (
        Path(__file__).parents[1] / "vllm_musa/deepseek_v4_jit/tilelang_kernels.py"
    ).read_text(encoding="utf-8")
    assert BENCH.PRODUCTION_CONFIG == (256, 256)
    assert (
        "def mhc_post_kernel(hidden_size: int, hidden_block: int = 256, threads: int = 256)"
        in source
    )


def test_default_tokens_are_production_standalone_post_buckets(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BENCH_PATH),
            "--expected-physical-device",
            "0",
            "--expected-device-uuid",
            "test-uuid",
        ],
    )
    assert BENCH.parse_args().tokens == [20, 80, 320]


def test_mhc_post_benchmark_requires_lease_device_fence():
    source = BENCH_PATH.read_text(encoding="utf-8")
    assert '"--expected-physical-device", type=int, required=True' in source
    assert '"--expected-device-uuid", required=True' in source
    assert '"lease_device_fence": lease_device_fence' in source
