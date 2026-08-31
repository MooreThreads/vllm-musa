import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BENCH_PATH = ROOT / "benchmarks/kernel_tactics/benchmark_reshape_cache_nhd_blocks.py"
sys.path.insert(0, str(BENCH_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "benchmark_reshape_cache_nhd_blocks",
    BENCH_PATH,
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


@pytest.mark.parametrize("value", ["128", "256", "512", "1024"])
def test_parse_block_x_accepts_compiled_launch_geometry(value):
    assert BENCH.parse_block_x(value) == int(value)


@pytest.mark.parametrize("value", ["bad", "0", "32", "64", "2048"])
def test_parse_block_x_rejects_uncompiled_launch_geometry(value):
    with pytest.raises(argparse.ArgumentTypeError):
        BENCH.parse_block_x(value)


@pytest.mark.parametrize("value", ["1", "6", "64", "256", "4096"])
def test_parse_tokens_accepts_checked_production_buckets(value):
    assert BENCH.parse_production_tokens(value) == int(value)


@pytest.mark.parametrize("value", ["bad", "0", "7", "65", "8192"])
def test_parse_tokens_rejects_untracked_buckets(value):
    with pytest.raises(argparse.ArgumentTypeError):
        BENCH.parse_production_tokens(value)


@pytest.mark.parametrize(
    ("shape_name", "expected_vecs", "expected_tokens_per_block"),
    [
        ("qwen2-kv2-d64", 16, 8),
        ("qwen3-kv8-d128", 128, 4),
        ("qwen3-kv1-d256", 32, 8),
        ("qwen3-kv2-d256", 64, 8),
    ],
)
def test_production_shapes_match_native_tokens_per_block_selector(
    shape_name,
    expected_vecs,
    expected_tokens_per_block,
):
    shape = BENCH.PRODUCTION_SHAPES[shape_name]
    assert shape.vecs_per_token == expected_vecs
    assert BENCH.production_tokens_per_block(shape) == expected_tokens_per_block


def test_native_op_exposes_default_preserving_per_call_block_x():
    bindings = (ROOT / "csrc/musa/torch_bindings.cpp").read_text()
    header = (ROOT / "csrc/musa/musa_ops.h").read_text()
    kernel = (ROOT / "csrc/musa/cache_kernels.mu").read_text()
    wrapper = (ROOT / "vllm_musa/_custom_ops.py").read_text()
    production_caller = (
        ROOT / "vllm_musa/v1/attention/backends/fa_utils.py"
    ).read_text()
    call_start = production_caller.index("musa_ops.musa_reshape_and_cache_flash_nhd(")
    call = production_caller[
        call_start : production_caller.index("\n            return", call_start)
    ]

    assert "int block_x=0) -> ()" in bindings
    assert "int64_t block_x" in header
    assert "int64_t requested_block_x" in kernel
    assert "requested_block_x == 0 ? 512 : requested_block_x" in kernel
    assert "block_x: int = 0" in wrapper
    assert "block_x" not in call


def test_native_op_rejects_every_uncompiled_block_before_empty_return():
    kernel = (ROOT / "csrc/musa/cache_kernels.mu").read_text()
    public_start = kernel.index("void musa_reshape_and_cache_flash_nhd(")
    public_body = kernel[public_start:]

    for block_x in BENCH.SUPPORTED_BLOCK_X:
        assert f"block_x == {block_x}" in public_body
    assert "block_x == 0" in public_body
    assert "block_x must be 0 (production default)" in public_body
    assert public_body.index("block_x == 0") < public_body.index("num_tokens == 0")


def test_benchmark_has_paired_cold_l2_correctness_and_device_fence():
    source = BENCH_PATH.read_text()

    assert '"--expected-physical-device", type=int, required=True' in source
    assert '"--expected-device-uuid", required=True' in source
    assert '"lease_device_fence": lease_device_fence' in source
    assert '"inner_iters": 1' in source
    assert '"paired_alternating_order": True' in source
    assert "flush.zero_()" in source
    assert "torch.equal(expected_key, actual_key)" in source
    assert "negative_slot_unchanged" in source
    assert "changed_from_poison" in source
    assert "VLLM_MUSA_RESHAPE_CACHE_TOKENS_PER_BLOCK" in source
