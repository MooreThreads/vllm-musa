import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "benchmarks/kernel_tactics/inventory_vllm_musa_kernels.py"
SPEC = importlib.util.spec_from_file_location("inventory_vllm_musa_kernels", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
INVENTORY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INVENTORY
SPEC.loader.exec_module(INVENTORY)

CATALOG_SCRIPT = ROOT / "benchmarks/kernel_tactics/kernel_sweep_catalog.py"
CATALOG_SPEC = importlib.util.spec_from_file_location(
    "kernel_sweep_catalog", CATALOG_SCRIPT
)
assert CATALOG_SPEC is not None and CATALOG_SPEC.loader is not None
CATALOG = importlib.util.module_from_spec(CATALOG_SPEC)
sys.modules[CATALOG_SPEC.name] = CATALOG
CATALOG_SPEC.loader.exec_module(CATALOG)

MANIFEST_PATH = ROOT / "benchmarks/kernel_tactics/full_kernel_sweep.json"


def test_inventory_is_unique_and_source_complete():
    entries = INVENTORY.discover(ROOT)
    assert entries
    assert len({entry.id for entry in entries}) == len(entries)
    for entry in entries:
        assert (ROOT / entry.source).is_file(), entry


def test_inventory_covers_all_owned_backend_classes():
    payload = INVENTORY.payload(ROOT)
    assert payload["schema"] == INVENTORY.SCHEMA
    counts = payload["counts"]
    assert counts["aot-native"] >= 30
    assert counts["jit-native"] >= 10
    assert counts["jit-native-ffi"] >= 15
    assert counts["jit-triton"] >= 7
    assert counts["jit-tilelang"] >= 20
    assert counts == {
        "aot-native": 34,
        "jit-native": 10,
        "jit-native-ffi": 19,
        "jit-tilelang": 31,
        "jit-triton": 7,
    }


def test_known_production_kernels_are_discovered():
    ids = {entry.id for entry in INVENTORY.discover(ROOT)}
    expected_suffixes = {
        "musa_fused_gemv_moe",
        "musa_csrc_fused_add_rmsnorm",
        "vllm_musa_custom_ar_launch_registered",
        "_gated_qk_norm_rope_token_kernel",
        "_qwen2_rope_kv_cache_kernel",
        "mhc_weighted_rmsnorm_mudnn_like_kernel",
        "musa_sparse_attention_fwd_kernel_v1",
    }
    for suffix in expected_suffixes:
        assert any(entry.endswith(f":{suffix}") for entry in ids), suffix


def test_full_sweep_manifest_covers_every_entry_exactly_once():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    discovered = INVENTORY.discover(ROOT)
    discovered_ids = {entry.id for entry in discovered}
    manifest_members = [
        member for family in manifest["families"] for member in family["members"]
    ]

    assert manifest["schema"] == CATALOG.SCHEMA
    assert len(manifest_members) == len(set(manifest_members))
    assert set(manifest_members) == discovered_ids
    assert len(manifest["families"]) == len(
        {family["id"] for family in manifest["families"]}
    )


def test_manifest_family_grouping_matches_reviewed_catalog_rules():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_families = {
        family["id"]: set(family["members"]) for family in manifest["families"]
    }
    expected = {
        family: {entry.id for entry in entries}
        for family, entries in CATALOG.grouped_families(ROOT).items()
    }

    assert manifest_families == expected


def test_every_family_has_an_explicit_public_disposition():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    allowed = set(manifest["allowed_dispositions"])
    for family in manifest["families"]:
        assert family["disposition"] in allowed
        if family["disposition"] == "no-tunable-seam":
            # A fixed-geometry kernel can still be production reachable on
            # every device; no-tunable means there is no safe selector seam,
            # not that the family is out of scope.
            assert family["tunable_parameters"] == []
            if family.get("production_consumers"):
                assert family["production_consumers"]


def test_public_catalog_contains_no_fleet_evidence():
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["schema"] == CATALOG.SCHEMA
    assert "created_against" not in manifest
    assert "target_mp_bins" not in manifest["scope"]
    assert "evidence" not in manifest_text
    assert "generated/MUSA-" not in manifest_text
    assert "archive_sha256" not in manifest_text
    assert "model_e2e" not in manifest_text
