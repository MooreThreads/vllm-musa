import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BENCH_PATH = ROOT / "benchmarks/kernel_tactics/benchmark_fp8_quant_groups.py"
BENCH_DIR = BENCH_PATH.parent
sys.path.insert(0, str(BENCH_DIR))
SPEC = importlib.util.spec_from_file_location("benchmark_fp8_quant_groups", BENCH_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)

HEADER = ROOT / "csrc/musa/musa_ops.h"
BINDINGS = ROOT / "csrc/musa/torch_bindings.cpp"
QUANT = ROOT / "csrc/musa/quantization/per_token_group_quant_8bit_vec.cu"
SILU = ROOT / "csrc/musa/quantization/silu_and_mul_per_token_group_fp8_quant.cu"
SHAPES = ROOT / "benchmarks/kernel_tactics/qwen_dsv4_fp8_quant_shapes.json"


@pytest.mark.parametrize(
    ("num_groups", "expected"),
    [(1, 1), (8, 8), (12, 4), (24, 8), (32, 16), (40, 8), (56, 8), (64, 16)],
)
def test_production_groups_per_block(num_groups, expected):
    assert BENCH.production_groups_per_block(num_groups) == expected


def test_quant_schemas_keep_default_preserving_benchmark_override():
    bindings = BINDINGS.read_text(encoding="utf-8")
    assert bindings.count("int groups_per_block=0") == 3
    header = HEADER.read_text(encoding="utf-8")
    assert header.count("int64_t groups_per_block") == 3
    assert "requested_groups_per_block" in QUANT.read_text(encoding="utf-8")
    assert "requested_groups_per_block" in SILU.read_text(encoding="utf-8")


def test_quant_override_validates_supported_geometry():
    quant = QUANT.read_text(encoding="utf-8")
    silu = SILU.read_text(encoding="utf-8")
    for value in (1, 2, 4, 8, 16):
        assert f"requested_groups_per_block == {value}" in quant
        assert f"requested_groups_per_block == {value}" in silu
    assert "num_groups % requested_groups_per_block == 0" in quant


def test_fp8_quant_benchmark_has_required_evidence_contract():
    source = BENCH_PATH.read_text(encoding="utf-8")
    assert '"--expected-physical-device", type=int, required=True' in source
    assert '"--expected-device-uuid", required=True' in source
    assert '"lease_device_fence": lease_device_fence' in source
    assert "candidate_q.fill_(fp8_info.min)" in source
    assert "torch.equal(reference_q.float(), candidate_q.float())" in source
    assert '"paired_alternating_order": True' in source
    assert "flush.zero_()" in source


def test_quant_shape_manifest_distinguishes_model_and_intermediate_widths():
    payload = json.loads(SHAPES.read_text(encoding="utf-8"))
    families = payload["families"]

    assert families["activation_quant"]["deepseek_v4_tp8"]["hidden_sizes"] == [4096]
    assert families["silu_clamp_shared_mlp"]["deepseek_v4_tp8"]["hidden_sizes"] == [256]
    assert families["silu_routed_moe"]["qwen_tp8_topk8"]["hidden_sizes"] == [128]
    assert (
        320 in families["silu_dense_fallback"]["m2_5_local_after_sphere_gate"]["rows"]
    )

    for family in families.values():
        for model in family.values():
            assert set(model["rows"]) <= set(BENCH.PRODUCTION_TOKENS)
            assert set(model["hidden_sizes"]) <= set(BENCH.PRODUCTION_HIDDEN_SIZES)
