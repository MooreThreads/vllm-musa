"""Guards for the consolidated fused all-reduce RMSNorm dispatch table.

The rule table is pure data plus a matcher, so it needs no MUSA device: every
check here runs on a CPU-only host. The guards cover four properties that must
not drift once the table drives a launch decision:

* the table stays complete and every rule name stays unique,
* dispatch is inert unless ``VLLM_MUSA_CAR_FUSED_DISPATCH`` is set,
* the measured decode shapes still resolve to the pinned launch parameters,
* only an actual v2 launcher call records the fused host-launch counter.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
POLICY = ROOT / "vllm_musa/optimization_contract/Fuse_CAR_RMSnorm.py"
COMM = ROOT / "vllm_musa/distributed/device_communicators/musa_jit_custom_all_reduce.py"

_EXPECTED_RULE_COUNT = 154


def _policy():
    spec = importlib.util.spec_from_file_location("fuse_car_rmsnorm_policy", POLICY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rule_table_is_complete_and_named_uniquely() -> None:
    policy = _policy()
    rules = policy.CUSTOM_AR_DISPATCH_RULES
    assert len(rules) == _EXPECTED_RULE_COUNT, len(rules)
    names = [rule.name for rule in rules]
    assert len(set(names)) == len(names), "duplicate rule name"
    assert all(rule.config is not None or rule.patch for rule in rules), "empty rule"
    source = POLICY.read_text(encoding="utf-8")
    assert "sorted(CUSTOM_AR_DISPATCH_RULES, key=lambda item: item.order)" in source


def test_dispatch_is_inert_unless_switched_on() -> None:
    policy = _policy()
    assert policy.dispatch_enabled({}) is False
    assert policy.dispatch_enabled({policy.DISPATCH_ENV: "0"}) is False
    assert policy.dispatch_enabled({policy.DISPATCH_ENV: "1"}) is True
    disabled = policy.resolve_fused_car_rmsnorm(
        mp_count=60, world_size=4, rows=1, hidden_size=2048, environ={}
    )
    assert disabled is None


def test_measured_decode_shapes_keep_their_launch_parameters() -> None:
    policy = _policy()
    environ = {policy.DISPATCH_ENV: "1"}
    cases = {
        2048: (256, 1),
        5376: (672, 1),
    }
    for hidden, (threads, blocks) in cases.items():
        config = policy.resolve_fused_car_rmsnorm(
            mp_count=60,
            world_size=4,
            rows=1,
            hidden_size=hidden,
            environ=environ,
        )
        assert config is not None, hidden
        assert config.kernel is policy.CustomARKernel.ROW_ONE, hidden
        assert (config.threads, config.blocks) == (threads, blocks), hidden


def test_unmeasured_shape_falls_through() -> None:
    policy = _policy()
    config = policy.resolve_fused_car_rmsnorm(
        mp_count=60,
        world_size=4,
        rows=7,
        hidden_size=1234,
        environ={policy.DISPATCH_ENV: "1"},
    )
    assert config is None


def test_v2_trace_follows_the_actual_launcher() -> None:
    import ast

    source = COMM.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MusaCustomAllReduceV2RMSNorm"
    )
    run = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
    launch_line = next(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "launcher"
    )
    record_line = next(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "record"
    )
    assert launch_line < record_line
    assert '"host_launch"' in ast.get_source_segment(source, run)
    assert "_record_fused_rmsnorm_decision" not in source


def test_rule_table_never_reaches_a_launch_on_its_own() -> None:
    policy = _policy()
    source = POLICY.read_text(encoding="utf-8")
    for forbidden in ("torch", "ctypes", "subprocess", "vllm_musa.jit_kernel"):
        assert f"import {forbidden}" not in source, forbidden
    assert policy.decision_label(None) == "no-rule"


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
            else:
                print(f"ok   {name}")
    print(f"{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
