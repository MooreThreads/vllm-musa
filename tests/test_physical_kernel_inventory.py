import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "benchmarks/kernel_tactics/inventory_vllm_musa_physical_kernels.py"
SPEC = importlib.util.spec_from_file_location(
    "inventory_vllm_musa_physical_kernels", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
INVENTORY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INVENTORY
SPEC.loader.exec_module(INVENTORY)

CATALOG_SCRIPT = ROOT / "benchmarks/kernel_tactics/kernel_sweep_catalog.py"
CATALOG_SPEC = importlib.util.spec_from_file_location(
    "physical_kernel_sweep_catalog", CATALOG_SCRIPT
)
assert CATALOG_SPEC is not None and CATALOG_SPEC.loader is not None
CATALOG = importlib.util.module_from_spec(CATALOG_SPEC)
sys.modules[CATALOG_SPEC.name] = CATALOG
CATALOG_SPEC.loader.exec_module(CATALOG)

MANIFEST_PATH = ROOT / "benchmarks/kernel_tactics/full_kernel_sweep.json"


def test_physical_inventory_is_unique_and_source_complete():
    entries = INVENTORY.discover(ROOT)
    assert entries
    assert len({entry.id for entry in entries}) == len(entries)
    for entry in entries:
        assert (ROOT / entry.source).is_file(), entry


def test_physical_inventory_covers_every_backend_class():
    payload = INVENTORY.payload(ROOT)
    assert payload["schema"] == INVENTORY.SCHEMA
    assert payload["counts"] == {
        "aot-native-physical": 36,
        "jit-native-physical": 77,
        "jit-tilelang-physical": 32,
        "jit-triton-physical": 7,
    }


def test_known_physical_kernels_are_discovered():
    symbols = {entry.symbol for entry in INVENTORY.discover(ROOT)}
    assert {
        "musa_gemv_kernel",
        "deepseek_v4_indexer_topk_prefill_q_cache_partialsort_kernel",
        "fused_add_rmsnorm_vec8_kernel",
        "topk_softmax_no_bias_warp_kernel_fixed_k",
        "deep_gemm_contig_preprocess_prefix_counts_no_pad",
        "main",
        "_qwen2_rope_kv_cache_kernel",
    } <= symbols


def test_plain_tilelang_primfunc_helper_is_not_a_callable_entrypoint():
    callable_entries = CATALOG.INVENTORY.discover(ROOT)
    physical_entries = INVENTORY.discover(ROOT)

    assert not any(
        entry.symbol == "_prefix_counts_no_pad_kernel" for entry in callable_entries
    )
    assert any(
        entry.owner == "_prefix_counts_no_pad_kernel" for entry in physical_entries
    )


def test_every_physical_kernel_maps_to_a_manifest_family():
    entries = INVENTORY.discover(ROOT)
    coverage = CATALOG.physical_coverage(ROOT)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    family_ids = {family["id"] for family in manifest["families"]}

    assert set(coverage) == {entry.id for entry in entries}
    assert all(families for families in coverage.values())
    assert all(set(families) <= family_ids for families in coverage.values())
