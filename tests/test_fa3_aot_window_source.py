"""Source contracts for the per-attention-window AOT scheduler metadata.

These run in the series CI job, which has no vLLM/torch installed, so they read
the applied tree instead of importing the backend. The behavioural suite lives
in `tests/test_flash_attn_scheduler_metadata_kinds.py` and needs a MUSA image.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "vllm_musa" / "v1" / "attention" / "backends" / "flash_attn.py"
DRIFT = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "module-drift"
    / "v1.attention.backends.flash_attn.diff"
)


def _source() -> str:
    return BACKEND.read_text()


def test_the_model_wide_aot_kill_switch_is_gone() -> None:
    """A second attention window must not disable AOT for the whole model."""
    assert "_get_sliding_window_configs" not in _source()


def test_one_slot_per_attention_window() -> None:
    source = _source()
    tree = ast.parse(source)
    fields = {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "kind_scheduler_metadata" in fields
    methods = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "scheduler_metadata_for" in methods
    assert "_normalize_attention_window" in methods
    assert "_get_attention_window_layouts" in methods


def test_a_missing_window_never_borrows_another_schedule() -> None:
    """The lookup returns None on a map miss so MATE self-schedules."""
    tree = ast.parse(_source())
    lookup = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "scheduler_metadata_for"
    )
    returns = [
        node.value
        for node in ast.walk(lookup)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    assert any(
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "get"
        for value in returns
    ), "a map miss must go through dict.get (None), not a fallback to another slot"


def test_per_window_aot_stays_opt_in() -> None:
    source = _source()
    assert 'VLLM_MUSA_FA3_AOT_PER_WINDOW", "0"' in source, (
        "the extra scheduler slot must default to off: it refreshes one plan per "
        "window per step, which is a measured decode regression on small models"
    )
    assert "_AOT_PER_WINDOW_ENABLED" in source


def test_dcp_keeps_the_single_slot_behaviour() -> None:
    source = _source()
    assert "dcp_world_size > 1 or not _AOT_PER_WINDOW_ENABLED" in source


def test_module_drift_tripwire_follows_the_refactor() -> None:
    """The shadow is a tracked copy of upstream; its diff must show the change."""
    diff = DRIFT.read_text()
    assert "_get_attention_window_layouts" in diff
    assert "kind_scheduler_metadata" in diff
    # the deleted helper may only survive as a removed line in the diff
    assert all(
        line.startswith("-")
        for line in diff.splitlines()
        if "_get_sliding_window_configs" in line
    )
