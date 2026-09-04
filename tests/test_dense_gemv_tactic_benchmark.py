import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BENCH_PATH = ROOT / "benchmarks/kernel_tactics/benchmark_dense_fp8_gemv_blocks.py"
sys.path.insert(0, str(BENCH_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "benchmark_dense_fp8_gemv_blocks", BENCH_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


@pytest.mark.parametrize("value", ["4x32", "8X16", "32x4", "128x1"])
def test_parse_block_accepts_compiled_dense_gemv_variants(value):
    assert BENCH.parse_block(value) in BENCH.SUPPORTED_BLOCKS


@pytest.mark.parametrize("value", ["bad", "0x8", "4x4", "64x8"])
def test_parse_block_rejects_uncompiled_variants(value):
    with pytest.raises(argparse.ArgumentTypeError):
        BENCH.parse_block(value)


@pytest.mark.parametrize(
    ("tokens", "block"),
    [
        (1, (4, 32)),
        (2, (4, 32)),
        (4, (8, 16)),
        (8, (4, 32)),
        (16, (32, 4)),
        (32, (8, 16)),
        (64, (8, 16)),
    ],
)
def test_dsv4_o_proj_production_ladder(tokens, block):
    assert BENCH.production_block("dsv4_o_proj", tokens) == block


@pytest.mark.parametrize(
    ("mp", "tokens", "block"),
    [
        (48, 1, (8, 32)),
        (48, 8, (16, 8)),
        (48, 32, (32, 4)),
        (48, 64, (32, 4)),
        (56, 8, (16, 16)),
        (56, 64, (32, 4)),
        (60, 8, (8, 16)),
        (60, 64, (32, 4)),
    ],
)
def test_dsv4_o_proj_exact_mp_ladder(mp, tokens, block):
    assert BENCH.production_block("dsv4_o_proj", tokens, mp) == block


def test_dense_gemv_exposes_default_preserving_per_call_blocks():
    bindings = (ROOT / "csrc/musa/torch_bindings.cpp").read_text()
    header = (ROOT / "csrc/musa/musa_ops.h").read_text()
    kernel = (ROOT / "csrc/musa/gemv.mu").read_text()
    wrapper = (ROOT / "vllm_musa/_custom_ops.py").read_text()

    assert 'float eps, int block_n=0, int block_k=0) -> ()"' in bindings
    assert "int64_t block_n" in header
    assert "int64_t block_k" in header
    assert "int64_t requested_block_n" in kernel
    assert "int64_t requested_block_k" in kernel
    assert "if (requested_block_n > 0)" in kernel
    assert "best_config = &requested_config" in kernel
    assert "block_n: int = 0" in wrapper
    assert "block_k: int = 0" in wrapper


def test_dense_gemv_production_direct_call_keeps_auto_tactic():
    source = (
        ROOT / "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py"
    ).read_text()
    start = source.index("torch.ops._C_musa_ops.musa_fused_gemv(")
    call = source[start : source.index("return output", start)]
    assert call.rstrip().endswith("0,\n        0,\n    )")


def test_dense_gemv_requires_lease_device_fence():
    source = BENCH_PATH.read_text()
    assert '"--expected-physical-device", type=int, required=True' in source
    assert '"--expected-device-uuid", required=True' in source
    assert '"lease_device_fence": lease_device_fence' in source
