"""CPU checks for the distributed CAR fused runtime runner."""

from __future__ import annotations

from typing import Any

from test_musa_car_v2_fused_runtime import (
    Case,
    _case_label,
    _cases,
    _dispatch_snapshot,
    _execution_mode,
    _fp32_weight_cases,
    _parser,
    _reference_from_reduced,
    _save_trace,
)


def test_execution_mode_uses_actual_v2_dispatch() -> None:
    from types import SimpleNamespace

    import pytest

    class LocalGroup:
        @staticmethod
        def get_world_size(group: Any) -> int:
            return 2

        @staticmethod
        def all_gather_object(results: list[Any], value: Any, group: Any) -> None:
            results[:] = [value, value]

    comm = SimpleNamespace(supports_fused_rmsnorm=True)
    with pytest.raises(AssertionError, match="missing actual fused execution mode"):
        _execution_mode(LocalGroup, None, comm, "v2")
    comm._last_fused_execution = "fallback"
    assert _execution_mode(LocalGroup, None, comm, "v2", "fallback") == "fallback"
    comm._last_fused_execution = "fused"
    assert _execution_mode(LocalGroup, None, comm, "v2") == "fused"
    with pytest.raises(AssertionError, match="expected fallback execution"):
        _execution_mode(LocalGroup, None, comm, "v2", "fallback")
    assert _execution_mode(LocalGroup, None, SimpleNamespace(), "v1") == "fused"


def test_fp32_weight_fallback_plan_keeps_target_matrix() -> None:
    args = _parser().parse_args([])
    assert len(_cases(args)) == 96
    extras = _fp32_weight_cases(args)
    assert len(extras) == 3
    assert {case.variant for case in extras} == {"norm", "residual", "no_raw"}
    assert all(case.impl == "v2" and case.weight_dtype == "float32" for case in extras)
    args.skip_fallback = True
    assert not _fp32_weight_cases(args)


def test_reference_uses_ir_rounding_order() -> None:
    from types import SimpleNamespace

    class Value:
        def __init__(self, expr: str, dtype: str = "bf16") -> None:
            self.expr, self.dtype = expr, dtype

        def float(self) -> Any:
            return Value(f"float({self.expr})", "float32")

        def to(self, dtype: str) -> Any:
            return Value(f"cast[{dtype}]({self.expr})", dtype)

        def square(self) -> Any:
            return Value(f"square({self.expr})", self.dtype)

        def mean(self, *args: Any, **kwargs: Any) -> Any:
            return Value(f"mean({self.expr})", self.dtype)

        def __add__(self, other: Any) -> Any:
            return Value(
                f"add({self.expr},{getattr(other, 'expr', other)})", self.dtype
            )

        def __mul__(self, other: Any) -> Any:
            return Value(f"mul({self.expr},{other.expr})", self.dtype)

    torch = SimpleNamespace(
        rsqrt=lambda value: Value(f"rsqrt({value.expr})", "float32")
    )
    reduced, residual = Value("reduced"), Value("residual")
    for dtype in ("bf16", "float32"):
        weight = Value("weight", dtype)
        for variant in ("norm", "residual", "no_raw"):
            out = _reference_from_reduced(
                torch, reduced, residual, weight, variant, 1e-6
            )
            assert out[0].expr.startswith(f"cast[bf16](mul(cast[{dtype}](mul(")
            assert out[0].expr.endswith(",weight))")
            if variant != "norm":
                unrounded = "add(float(reduced),float(residual))"
                assert f"mean(square({unrounded}))" in out[0].expr
                assert out[1].expr == f"cast[bf16]({unrounded})"
                alternate = _reference_from_reduced(
                    torch, reduced, residual, weight, variant, 1e-6, separate_add=True
                )
                assert (
                    f"mean(square(float(cast[bf16]({unrounded}))))" in alternate[0].expr
                )


def test_dispatch_snapshot_copies_and_checks_call_contract() -> None:
    from types import SimpleNamespace

    import pytest

    case = Case("v2", "bfloat16", "same", 2048, 1, "residual")
    value = dict(
        backend="v2",
        execution="fused",
        rows=1,
        hidden=2048,
        need_raw=True,
        has_residual=True,
        phase="eager",
        registered=False,
        policy_label="one-shot",
        launcher_symbol="fused_kernel",
        effective_config={"threads": 256},
        jit_cflags=["-O3"],
    )
    comm = SimpleNamespace(_last_fused_dispatch=value)
    snapshot = _dispatch_snapshot(comm, case, "eager", "fused", True)
    value["effective_config"]["threads"] = 512
    value["jit_cflags"].append("-DTEST")
    assert snapshot["effective_config"] == {"threads": 256}
    assert snapshot["jit_cflags"] == ["-O3"]
    with pytest.raises(AssertionError, match="contradicts call"):
        _dispatch_snapshot(comm, case, "capture", "fused", True)
    value.update(
        phase="capture",
        execution="fallback",
        launcher_symbol="unfused_v2_ir",
        effective_config={},
        jit_cflags=[],
    )
    assert (
        _dispatch_snapshot(comm, case, "capture", "fallback", True)["execution"]
        == "fallback"
    )
    with pytest.raises(AssertionError, match="missing v2 dispatch"):
        _dispatch_snapshot(SimpleNamespace(), case, "eager", "fused", True)
    v1 = Case("v1", "bfloat16", "same", 2048, 1, "residual")
    assert (
        _dispatch_snapshot(SimpleNamespace(), v1, "eager", "fused", True)["status"]
        == "NOT_AVAILABLE"
    )
    assert (
        _case_label(2, case)
        == "CAR_CASE:tp2:impl=v2:dtype=bfloat16:h2048:r1:variant=residual:weight=same"
    )


def test_failed_trace_export_preserves_original_failure(tmp_path: Any) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pytest

    profiler = SimpleNamespace(
        export_chrome_trace=Mock(side_effect=ValueError("trace failed"))
    )
    report = {"status": "FAIL", "error": "numerical error"}
    _save_trace(profiler, tmp_path / "trace.json", report)
    assert report["error"] == "numerical error"
    assert report["trace_export_error"] == "ValueError: trace failed"
    with pytest.raises(ValueError, match="trace failed"):
        _save_trace(profiler, tmp_path / "trace.json", {"status": "PASS"})
