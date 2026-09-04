# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BENCH_PATH = ROOT / "benchmarks/kernel_tactics/benchmark_jit_topk_warps.py"
KERNEL_PATH = ROOT / "vllm_musa/jit_kernel/csrc/topk/topk_gating.mu"
sys.path.insert(0, str(BENCH_PATH.parent))
SPEC = importlib.util.spec_from_file_location("benchmark_jit_topk_warps", BENCH_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


def test_topk_families_follow_checked_production_buckets() -> None:
    families = BENCH.load_production_families()

    qwen = families["qwen_softmax_e256_k8"]
    assert (qwen.input_experts, qwen.routed_topk, qwen.output_topk) == (256, 8, 8)
    assert qwen.production_rows == (1, 2, 4, 8, 12, 16, 32, 36, 64)
    assert tuple(row // qwen.routed_topk for row in qwen.routed_rows) == (
        qwen.production_rows
    )

    folded = families["qwen35_folded_softmax_e257_k9"]
    assert (folded.input_experts, folded.routed_topk, folded.output_topk) == (
        257,
        8,
        9,
    )
    assert folded.num_fused_shared_experts == 1
    assert folded.production_rows == qwen.production_rows

    deepseek = families["deepseek_v4_sigmoid_e256_k6_local_no_bias"]
    assert (deepseek.input_experts, deepseek.routed_topk) == (256, 6)
    assert deepseek.production_reachable is False
    assert "MATE grouped correction-bias router is excluded" in deepseek.scope_note
    assert deepseek.production_rows == (
        1,
        2,
        4,
        5,
        8,
        16,
        20,
        32,
        64,
        80,
        128,
        256,
        320,
    )
    biased = families["deepseek_v4_sigmoid_e256_k6_correction_bias"]
    assert (biased.input_experts, biased.routed_topk) == (256, 6)
    assert biased.production_reachable is True
    assert biased.has_correction_bias is True


@pytest.mark.parametrize("value", ["1", "2", "4", "8"])
def test_parse_warps_accepts_compiled_tactics(value: str) -> None:
    assert BENCH.parse_warps_per_cta(value) == int(value)


@pytest.mark.parametrize("value", ["bad", "0", "3", "16"])
def test_parse_warps_rejects_uncompiled_tactics(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        BENCH.parse_warps_per_cta(value)


def test_launch_geometry_maps_one_warp_to_one_row() -> None:
    assert BENCH.launch_geometry(65, 1) == {
        "warps_per_cta": 1,
        "threads_per_cta": 32,
        "grid_ctas": 65,
    }
    assert BENCH.launch_geometry(65, 8) == {
        "warps_per_cta": 8,
        "threads_per_cta": 256,
        "grid_ctas": 9,
    }


def test_row_override_must_stay_inside_each_family() -> None:
    family = BENCH.load_production_families()["qwen_softmax_e256_k8"]
    assert BENCH.selected_rows(family, [1, 64, 1]) == (1, 64)
    with pytest.raises(ValueError, match="production buckets"):
        BENCH.selected_rows(family, [5])


@pytest.mark.parametrize("repeats", [2, 4, 32])
def test_balanced_pairing_requires_positive_even_repeats(repeats: int) -> None:
    BENCH.validate_timing_args(dry_runs=1, repeats=repeats)


@pytest.mark.parametrize("repeats", [-2, 0, 1, 3, 31])
def test_unbalanced_repeat_counts_are_rejected(repeats: int) -> None:
    with pytest.raises(ValueError, match="balanced AB/BA"):
        BENCH.validate_timing_args(dry_runs=1, repeats=repeats)


def test_native_seam_is_benchmark_only_and_default_preserving() -> None:
    source = KERNEL_PATH.read_text(encoding="utf-8")

    assert "constexpr int kWarpsPerCta = 4;" in source
    assert "int WarpsPerCta = kWarpsPerCta" in source
    assert "if (warps_per_cta == 0)" in source
    assert "dispatch_topk<IsSoftmax>(" in source
    for warps in BENCH.VALID_WARPS_PER_CTA:
        assert f"warps_per_cta == {warps}" in source
    assert "Softmax benchmark tactics exclude correction-bias routing" in source
    assert "Benchmark topk tactics exclude separate shared-gate routing" in source
    assert "topk_sigmoid_warp_kernel<T, 256, 8, WarpsPerCta>" in source
    assert "#if defined(VLLM_MUSA_TOPK_BENCHMARK_TACTICS)" in source
    first_guard = source.index("#if defined(VLLM_MUSA_TOPK_BENCHMARK_TACTICS)")
    launch_helper = source.index("void launch_topk_benchmark_tactic")
    first_close = source.index("#endif", launch_helper)
    assert first_guard < launch_helper < first_close
    assert source.count("TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_topk_softmax") == 2
    assert source.count("TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_topk_sigmoid") == 2
    assert "sgl_musa_topk_softmax_benchmark_tactic" in source
    assert "sgl_musa_topk_sigmoid_benchmark_tactic" in source


def test_benchmark_has_cold_paired_poisoned_device_fence_contract() -> None:
    source = BENCH_PATH.read_text(encoding="utf-8")

    assert BENCH.L2_FLUSH_BYTES == 8_000_000_000
    assert '"--expected-physical-device", type=int, required=True' in source
    assert '"--expected-device-uuid", required=True' in source
    assert '"lease_device_fence": lease_device_fence' in source
    assert 'weights.fill_(float("nan"))' in source
    assert "ids.fill_(-1)" in source
    assert "flush.zero_()" in source
    assert '"paired_alternating_order": True' in source
    assert '"cache_policy": "cold-l2-per-sample"' in source
    assert '"implementation": "vllm-musa local native JIT only"' in source
    assert "descending=True" in source
    assert "stable=True" in source
    assert 'else "local-native-default"' in source
    assert '"-DVLLM_MUSA_TOPK_BENCHMARK_TACTICS=1"' in source
    assert "module.sgl_musa_topk_softmax(" in source
    assert "module.sgl_musa_topk_sigmoid(" in source
    assert "deepseek_v4_sigmoid_e256_k6_correction_bias" in source
    assert "selection_scores = scores + correction_bias" in source
    assert "from mate" not in source
