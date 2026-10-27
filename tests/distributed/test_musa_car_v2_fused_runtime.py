"""Run with torchrun on TP2/TP4; --help and --list-cases need no GPU imports.

Example: torchrun --standalone --nproc_per_node=2 \
    tests/distributed/test_musa_car_v2_fused_runtime.py --output-dir /tmp/car-fused-tp2
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    impl: str
    dtype: str
    weight_dtype: str
    hidden: int
    rows: int
    variant: str
    gemma: bool = False


def _csv(value: str) -> list[str]:
    values = value.split(",")
    if not values or any(not item for item in values):
        raise argparse.ArgumentTypeError("expected a nonempty comma-separated list")
    return values


def _positive_csv(value: str) -> list[int]:
    try:
        values = [int(item) for item in _csv(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected positive integers") from error
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive integers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gemma", action="store_true", help="raw zero-centered weights; v2 only"
    )
    parser.add_argument("--impls", type=_csv, default=["v2", "v1"])
    parser.add_argument("--dtypes", type=_csv, default=["bfloat16", "float16"])
    parser.add_argument("--weight-dtypes", type=_csv, default=["same"])
    parser.add_argument("--hidden", type=_positive_csv, default=[2048, 5120])
    parser.add_argument("--rows", type=_positive_csv, default=[1, 16, 128, 256])
    parser.add_argument("--variants", type=_csv, default=["norm", "residual", "no_raw"])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--skip-fallback", action="store_true")
    parser.add_argument("--max-size-mib", type=int, default=32)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--fused-dispatch", choices=["0", "1"], default="1")
    parser.add_argument("--output-dir", type=Path, default=Path("car-v2-fused-results"))
    parser.add_argument(
        "--profile", action="store_true", help="export per-rank device traces"
    )
    parser.add_argument(
        "--list-cases", action="store_true", help="print plan without execution"
    )
    return parser


def _cases(args: argparse.Namespace) -> list[Case]:
    allowed = {
        "impls": {"v1", "v2"},
        "dtypes": {"float16", "bfloat16"},
        "weight_dtypes": {"same", "float32"},
        "variants": {"norm", "residual", "no_raw"},
    }
    for name, choices in allowed.items():
        if not set(getattr(args, name)) <= choices:
            raise ValueError(f"unsupported {name}: {getattr(args, name)}")
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            raise ValueError(f"duplicate {name} entries")
    counts = (
        args.iterations,
        args.graph_replays,
        args.max_size_mib,
        args.timeout_seconds,
    )
    if min(counts) < 1:
        raise ValueError(
            "iteration, replay, buffer and timeout values must be positive"
        )
    if args.eps <= 0:
        raise ValueError("eps must be positive")
    if max(args.rows) * max(args.hidden) * 2 > args.max_size_mib * 1024**2:
        raise ValueError("target shapes exceed --max-size-mib")
    if args.gemma and args.impls != ["v2"]:
        raise ValueError("--gemma requires --impls v2; v1 Gemma fusion is unsupported")
    axes = [
        getattr(args, name)
        for name in ("impls", "dtypes", "weight_dtypes", "hidden", "rows", "variants")
    ]
    return [Case(*values, gemma=args.gemma) for values in itertools.product(*axes)]


def _runtime() -> tuple[Any, Any, Any]:
    # isort: off
    import torchada  # noqa: F401
    import torch
    import torch.distributed as dist

    # isort: on

    if getattr(torch.version, "musa", None) is None or not hasattr(torch, "musa"):
        raise RuntimeError("real MUSA torch runtime is required; no GPU checks ran")
    if not torch.musa.is_available():
        raise RuntimeError("MUSA devices are unavailable; no GPU checks ran")
    from vllm_musa.distributed.device_communicators import musa_jit_custom_all_reduce

    return torch, dist, musa_jit_custom_all_reduce


def _fp32_weight_cases(args: argparse.Namespace) -> list[Case]:
    if args.skip_fallback or "v2" not in args.impls or "float32" in args.weight_dtypes:
        return []
    return [
        Case("v2", "bfloat16", "float32", 2048, 1, variant, args.gemma)
        for variant in ("norm", "residual", "no_raw")
    ]


def _inputs(torch: Any, case: Case, rank: int, step: int, seed: int) -> tuple[Any, ...]:
    generator = torch.Generator(device="cpu").manual_seed(
        seed + rank * 1009 + step * 37
    )
    dtype = getattr(torch, case.dtype)
    shape = (case.rows, case.hidden)
    x = torch.randn(shape, generator=generator).to(dtype)
    residual = torch.randn(shape, generator=generator).to(dtype)
    # All ranks use the same normalization weights.
    generator.manual_seed(seed + step * 53)
    weight_dtype = dtype if case.weight_dtype == "same" else torch.float32
    weight = (
        (0.0 if case.gemma else 1.0)
        + 0.1 * torch.randn((case.hidden,), generator=generator)
    ).to(weight_dtype)
    return x, residual, weight


def _reference_from_reduced(
    torch: Any,
    reduced: Any,
    residual: Any,
    weight: Any,
    variant: str,
    eps: float,
    *,
    separate_add: bool = False,
    gemma: bool = False,
) -> tuple[Any, ...]:
    fp32 = reduced.float()
    residual_out = None
    if variant != "norm":
        fp32 = fp32 + residual.float()
        residual_out = fp32.to(reduced.dtype)
        if separate_add:
            fp32 = residual_out.float()
    normalized = fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + eps)
    norm = (
        (normalized * (weight.float() + 1.0)).to(reduced.dtype)
        if gemma
        else (normalized.to(weight.dtype) * weight).to(reduced.dtype)
    )
    if variant == "norm":
        return norm, reduced
    if variant == "residual":
        return norm, residual_out, reduced
    return norm, residual_out


def _reference(
    torch: Any,
    dist: Any,
    group: Any,
    values: tuple[Any, ...],
    variant: str,
    eps: float,
    audit: list[dict[str, Any]] | None = None,
    gemma: bool = False,
) -> tuple[Any, ...]:
    x, residual, weight = values
    if x.device.type != "cpu":
        raise ValueError("the high-precision all-reduce reference requires CPU inputs")
    # Host fp64 avoids rank-order rounding at activation-dtype midpoints.
    reduced = x.double().contiguous().clone()
    dist.all_reduce(reduced, group=group)
    reduced = reduced.to(x.dtype)
    result = _reference_from_reduced(
        torch, reduced, residual, weight, variant, eps, gemma=gemma
    )
    if audit is not None and variant != "norm":
        alternate = _reference_from_reduced(
            torch,
            reduced,
            residual,
            weight,
            variant,
            eps,
            separate_add=True,
            gemma=gemma,
        )
        difference = (result[0].float() - alternate[0].float()).abs()
        audit.append(
            {
                "max_abs_norm_difference": difference.max().item(),
                "different_norm_elements": int((difference != 0).sum().item()),
            }
        )
    return result


def _call(
    comm: Any, values: tuple[Any, ...], variant: str, eps: float, gemma: bool = False
) -> Any:
    x, residual, weight = values
    kwargs = {"gemma": True} if gemma else {}
    if variant == "norm":
        return comm.fused_allreduce_rmsnorm(x, weight, eps, **kwargs)
    if variant == "residual":
        return comm.fused_allreduce_residual_rmsnorm(x, residual, weight, eps, **kwargs)
    return comm.fused_allreduce_residual_rmsnorm_no_raw(
        x, residual, weight, eps, **kwargs
    )


def _eligible(comm: Any, values: tuple[Any, ...], variant: str) -> bool:
    x, residual, weight = values
    if variant == "norm":
        return comm.should_fused_allreduce_rmsnorm(x, weight)
    return comm.should_fused_allreduce_residual_rmsnorm(x, residual, weight)


def _validate(
    torch: Any,
    outputs: Any,
    expected: tuple[Any, ...],
    inputs: tuple[Any, ...],
    original: tuple[Any, ...],
) -> float:
    if not isinstance(outputs, tuple) or len(outputs) != len(expected):
        raise AssertionError(
            f"fused output ABI mismatch: {type(outputs)}, expected {len(expected)}"
        )
    storage = [value.untyped_storage().data_ptr() for value in (*inputs, *outputs)]
    if len(storage) != len(set(storage)):
        raise AssertionError("fused outputs must not alias inputs or each other")
    for value, snapshot in zip(inputs, original):
        torch.testing.assert_close(value.cpu(), snapshot, rtol=0, atol=0)
    if inputs[0].dtype == torch.bfloat16:
        rtol, atol = 0.016, 1e-3
    elif inputs[0].dtype == torch.float16:
        rtol, atol = 0.002, 1e-2
    else:
        rtol, atol = 1.3e-6, 1e-5
    max_error = 0.0
    for value, target in zip(outputs, expected):
        actual = value.cpu()
        torch.testing.assert_close(actual, target.cpu(), rtol=rtol, atol=atol)
        max_error = max(
            max_error, (actual.float() - target.cpu().float()).abs().max().item()
        )
    return max_error


def _consensus(dist: Any, group: Any, error: str | None) -> None:
    errors: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    failures = [
        f"rank {rank}: {message}" for rank, message in enumerate(errors) if message
    ]
    if failures:
        raise AssertionError("; ".join(failures))


def _checked(
    torch: Any,
    dist: Any,
    group: Any,
    outputs: Any,
    expected: tuple[Any, ...],
    device_inputs: tuple[Any, ...],
    original: tuple[Any, ...],
) -> float:
    error = None
    max_error = 0.0
    try:
        torch.musa.synchronize()
        max_error = _validate(torch, outputs, expected, device_inputs, original)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _consensus(dist, group, error)
    return max_error


def _execution_mode(
    dist: Any, group: Any, comm: Any, impl: str, expected: str | None = None
) -> str:
    mode = getattr(comm, "_last_fused_execution", None) if impl == "v2" else "fused"
    modes: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(modes, mode, group=group)
    if any(value not in ("fused", "fallback") for value in modes):
        raise AssertionError(f"missing actual fused execution mode: {modes}")
    if any(value != modes[0] for value in modes):
        raise AssertionError(f"execution modes differ between ranks: {modes}")
    if expected is not None and modes[0] != expected:
        raise AssertionError(f"expected {expected} execution, got {modes[0]}")
    return str(modes[0])


def _dispatch_snapshot(
    comm: Any, case: Case, phase: str, execution: str, required: bool
) -> dict[str, Any]:
    if case.impl != "v2" or not required:
        return {"status": "NOT_AVAILABLE", "backend": case.impl, "phase": phase}
    snapshot = json.loads(json.dumps(getattr(comm, "_last_fused_dispatch", None)))
    if not isinstance(snapshot, dict):
        raise AssertionError("missing v2 dispatch snapshot while tracing")
    expected = {
        "backend": "v2",
        "execution": execution,
        "rows": case.rows,
        "hidden": case.hidden,
        "need_raw": case.variant != "no_raw",
        "has_residual": case.variant != "norm",
        "gemma": case.gemma,
        "phase": phase,
    }
    if any(snapshot.get(key) != value for key, value in expected.items()):
        raise AssertionError(
            f"v2 dispatch snapshot contradicts call: {snapshot}, expected {expected}"
        )
    if not isinstance(snapshot.get("registered"), bool):
        raise AssertionError("dispatch snapshot must identify registered/staging mode")
    if not all(
        isinstance(snapshot.get(key), str) and snapshot[key]
        for key in ("policy_label", "launcher_symbol")
    ):
        raise AssertionError("dispatch snapshot must name policy and launcher")
    if not isinstance(snapshot.get("effective_config"), dict) or not isinstance(
        snapshot.get("jit_cflags"), list
    ):
        raise AssertionError(
            "dispatch snapshot must record effective config and JIT flags"
        )
    return snapshot


def _record_dispatch(
    dist: Any,
    group: Any,
    comm: Any,
    case: Case,
    phase: str,
    execution: str,
    snapshots: list[dict[str, Any]],
) -> None:
    error = None
    try:
        required = os.environ.get("VLLM_MUSA_CAR_TRACE") == "1"
        snapshot = _dispatch_snapshot(comm, case, phase, execution, required)
        snapshots.append(snapshot)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _consensus(dist, group, error)


def _case_label(tp: int, case: Case) -> str:
    return (
        f"CAR_CASE:tp{tp}:impl={case.impl}:dtype={case.dtype}:h{case.hidden}:"
        f"r{case.rows}:variant={case.variant}:weight={case.weight_dtype}"
        + (":gemma=1" if case.gemma else "")
    )


def _phase_context(
    torch: Any, tp: int, case: Case, phase: str, index: int, enabled: bool
) -> Any:
    label = f"CAR_PHASE:{_case_label(tp, case)}:phase={phase}:index={index}"
    return torch.profiler.record_function(label) if enabled else nullcontext()


def _run_case(
    torch: Any,
    dist: Any,
    group: Any,
    comm: Any,
    device: Any,
    case: Case,
    args: argparse.Namespace,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    rank = dist.get_rank(group)
    errors: list[float] = []
    reference_audit: list[dict[str, Any]] = []
    eager_modes: list[str] = []
    eligible_call = True
    for step in range(args.iterations):
        with _phase_context(
            torch, dist.get_world_size(group), case, "eager", step, args.profile
        ):
            original = _inputs(torch, case, rank, step, args.seed)
            expected = _reference(
                torch,
                dist,
                group,
                original,
                case.variant,
                args.eps,
                audit=reference_audit,
                gemma=case.gemma,
            )
            inputs = tuple(value.to(device) for value in original)
            eligible: list[Any] = [None] * dist.get_world_size(group)
            dist.all_gather_object(
                eligible, _eligible(comm, inputs, case.variant), group=group
            )
            if len(set(eligible)) != 1:
                raise AssertionError(
                    f"communicator eligibility differs between ranks: {eligible}"
                )
            eligible_call = bool(eligible[0])
            expected_fallback = case.impl == "v2" and case.weight_dtype == "float32"
            if case.impl == "v2" and not eligible_call:
                raise AssertionError(
                    f"v2 eligibility violates supported shape contract: {case}"
                )
            if eligible_call:
                if case.impl == "v2":
                    comm._last_fused_execution = None
                    comm._last_fused_dispatch = None
                outputs = _call(comm, inputs, case.variant, args.eps, case.gemma)
                execution = _execution_mode(
                    dist,
                    group,
                    comm,
                    case.impl,
                    "fallback" if expected_fallback else None,
                )
                eager_modes.append(execution)
            else:
                result = _call(comm, inputs, case.variant, args.eps, case.gemma)
                _consensus(
                    dist,
                    group,
                    None if result is None else "expected None fallback sentinel",
                )
                outputs = _unfused(torch, dist, inputs, case.variant, args.eps)
                eager_modes.append("fallback")
            _record_dispatch(
                dist, group, comm, case, "eager", eager_modes[-1], snapshots
            )
            errors.append(
                _checked(torch, dist, group, outputs, expected, inputs, original)
            )

    replays = 0
    graph_mode = None
    capture_snapshot_index = None
    if not args.skip_graph and eligible_call:
        graph = torch.musa.MUSAGraph()
        stream = torch.musa.Stream()
        stream.wait_stream(torch.musa.current_stream())
        with _phase_context(
            torch, dist.get_world_size(group), case, "capture", 0, args.profile
        ):
            with comm.capture():
                with torch.musa.stream(stream):
                    # Prepare graph launch variants without active stream capture.
                    _call(comm, inputs, case.variant, args.eps, case.gemma)
                torch.musa.current_stream().wait_stream(stream)
                torch.musa.synchronize()
                dist.barrier(group=group)
                if case.impl == "v2":
                    comm._last_fused_execution = None
                    comm._last_fused_dispatch = None
                with torch.musa.graph(graph, stream=stream):
                    outputs = _call(comm, inputs, case.variant, args.eps, case.gemma)
        graph_mode = _execution_mode(
            dist, group, comm, case.impl, "fallback" if expected_fallback else None
        )
        capture_snapshot_index = len(snapshots)
        _record_dispatch(dist, group, comm, case, "capture", graph_mode, snapshots)
        for replay in range(args.graph_replays):
            with _phase_context(
                torch, dist.get_world_size(group), case, "replay", replay, args.profile
            ):
                original = _inputs(
                    torch, case, rank, args.iterations + replay, args.seed
                )
                expected = _reference(
                    torch,
                    dist,
                    group,
                    original,
                    case.variant,
                    args.eps,
                    audit=reference_audit,
                    gemma=case.gemma,
                )
                for destination, source in zip(inputs, original):
                    destination.copy_(source)
                torch.musa.synchronize()
                dist.barrier(group=group)
                graph.replay()
                errors.append(
                    _checked(torch, dist, group, outputs, expected, inputs, original)
                )
                replays += 1
        torch.musa.synchronize()
    executed_modes = set(eager_modes)
    if graph_mode is not None:
        executed_modes.add(graph_mode)
    return {
        **asdict(case),
        "status": "PASS",
        "kind": (
            next(iter(executed_modes)).upper() if len(executed_modes) == 1 else "MIXED"
        ),
        "eager_calls": args.iterations,
        "eager_fused_calls": eager_modes.count("fused"),
        "eager_fallback_calls": eager_modes.count("fallback"),
        "graph_replays": replays,
        "graph_execution": graph_mode,
        "graph_fused_replays": replays if graph_mode == "fused" else 0,
        "graph_fallback_replays": replays if graph_mode == "fallback" else 0,
        "max_abs_error": max(errors),
        "dispatch_snapshots": snapshots,
        "graph_replay_dispatch": {
            "capture_snapshot_index": capture_snapshot_index,
            "checked_replays": replays,
            "replay_calls_python_dispatch": False,
        },
        "add_then_rms_reference_audit": {
            "observations": len(reference_audit),
            "max_abs_norm_difference": max(
                (row["max_abs_norm_difference"] for row in reference_audit), default=0.0
            ),
            "different_norm_elements": sum(
                row["different_norm_elements"] for row in reference_audit
            ),
        },
    }


def _unfused(
    torch: Any,
    dist: Any,
    values: tuple[Any, ...],
    variant: str,
    eps: float,
    gemma: bool = False,
) -> tuple[Any, ...]:
    from vllm.ir import ops

    x, residual, weight = values
    if gemma:
        weight = weight.float() + 1.0
    reduced = x.float().contiguous().clone()
    dist.all_reduce(reduced)
    reduced = reduced.to(x.dtype)
    if variant == "norm":
        norm = ops.rms_norm(reduced.clone(), weight, eps)
        return norm, reduced
    norm, residual_out = ops.fused_add_rms_norm(
        reduced.clone(), residual.clone(), weight, eps
    )
    return (
        (norm, residual_out, reduced) if variant == "residual" else (norm, residual_out)
    )


def _fallback_cases(
    torch: Any, device: Any, impl: str, args: argparse.Namespace
) -> list[tuple[str, tuple[Any, ...]]]:
    x = torch.ones((2, 2048), device=device, dtype=torch.bfloat16)
    residual = torch.ones_like(x)
    weight = torch.ones((2048,), device=device, dtype=x.dtype)
    odd = torch.ones((2, 17), device=device, dtype=x.dtype)
    oversized = torch.ones(
        (args.max_size_mib * 1024**2 // (2048 * 2) + 1, 2048),
        device=device,
        dtype=x.dtype,
    )
    cases = [
        ("float32-input", (x.float(), residual.float(), weight.float())),
        (
            "unaligned-hidden",
            (odd, odd.clone(), torch.ones(17, device=device, dtype=x.dtype)),
        ),
        ("noncontiguous-input", (x.t().contiguous().t(), residual, weight)),
        ("oversized-input", (oversized, oversized.clone(), weight)),
    ]
    return cases


def _run_fallbacks(
    torch: Any,
    dist: Any,
    group: Any,
    comm: Any,
    device: Any,
    impl: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results = []
    for name, inputs in _fallback_cases(torch, device, impl, args):
        for variant in args.variants:
            original = tuple(value.cpu().clone() for value in inputs)
            expected = _reference(
                torch, dist, group, original, variant, args.eps, gemma=args.gemma
            )
            error = (
                "unsupported input accepted"
                if _eligible(comm, inputs, variant)
                else None
            )
            _consensus(dist, group, error)
            result = _call(comm, inputs, variant, args.eps, args.gemma)
            _consensus(
                dist,
                group,
                None if result is None else "expected None fallback sentinel",
            )
            outputs = _unfused(torch, dist, inputs, variant, args.eps, args.gemma)
            error = _checked(torch, dist, group, outputs, expected, inputs, original)
            results.append(
                {
                    "impl": impl,
                    "case": name,
                    "variant": variant,
                    "status": "PASS",
                    "kind": "FALLBACK",
                    "max_abs_error": error,
                }
            )
    return results


def _profiler(torch: Any, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    available = torch.profiler.supported_activities()
    device_activities = [
        activity
        for activity in available
        if activity != torch.profiler.ProfilerActivity.CPU
    ]
    if not device_activities:
        raise RuntimeError(
            "device profiler activity unavailable; refusing a CPU-only trace"
        )
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, *device_activities],
        record_shapes=True,
    )


def _save_trace(profiler: Any, path: Path, report: dict[str, Any]) -> None:
    try:
        profiler.export_chrome_trace(str(path))
        report["trace"] = str(path)
    except Exception as exc:
        report["trace_export_error"] = f"{type(exc).__name__}: {exc}"
        if report.get("status") != "FAIL":
            report["status"] = "FAIL"
            raise


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = _cases(args)
    fp32_weight_cases = _fp32_weight_cases(args)
    if args.list_cases:
        plan = {
            "status": "NOT_RUN",
            "planned_cases": len(cases),
            "cases": [asdict(case) for case in cases],
            "fp32_weight_fallback_cases": [asdict(case) for case in fp32_weight_cases],
        }
        print(json.dumps(plan, indent=2))
        return 0
    os.environ["VLLM_MUSA_CUSTOM_AR_V2"] = "1"
    os.environ["VLLM_MUSA_CAR_FUSED_DISPATCH"] = args.fused_dispatch
    if args.profile:
        os.environ["VLLM_MUSA_CAR_TRACE"] = "1"
    torch, dist, custom_ar = _runtime()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (2, 4):
        raise RuntimeError("launch using torchrun with TP2 or TP4; no checks ran")
    device = torch.device("musa", int(os.environ["LOCAL_RANK"]))
    torch.musa.set_device(device)
    dist.init_process_group(
        backend="cpu:gloo,musa:mccl", timeout=timedelta(seconds=args.timeout_seconds)
    )
    group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=args.timeout_seconds)
    )
    rank = dist.get_rank()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"rank-{rank}.json"
    report: dict[str, Any] = {
        "status": "RUNNING",
        "rank": rank,
        "world_size": world_size,
        "torch_version": torch.__version__,
        "musa_version": torch.version.musa,
        "fused_dispatch": args.fused_dispatch,
        "cases": [],
        "fallbacks": [],
        "fp32_weight_fallbacks": [],
        "planned_cases": len(cases),
        "planned_fp32_weight_fallbacks": len(fp32_weight_cases),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "tolerances": {
            "bfloat16": {"rtol": 0.016, "atol": 1e-3},
            "float16": {"rtol": 0.002, "atol": 1e-2},
            "float32": {"rtol": 1.3e-6, "atol": 1e-5},
        },
        "reference_contract": {
            "vllm_revision": "5df63a7203cdb9328d824a27b682e77c7b9928be",
            "source": "vllm/ir/ops/layernorm.py",
            "all_reduce": "CPU Gloo fp64 sum, then activation dtype cast",
            "residual_primary": "fused_add_rms_norm",
            "residual_alternative_audit": "add_then_rms_norm",
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    profiler = None
    try:
        with _profiler(torch, args.profile) as profiler:
            for impl in args.impls:
                cls = (
                    custom_ar._MusaCustomAllreduceV2Impl
                    if impl == "v2"
                    else custom_ar._MusaJitCustomAllreduceImpl
                )
                comm = cls(
                    group=group, device=device, max_size=args.max_size_mib * 1024**2
                )
                _consensus(
                    dist,
                    group,
                    "communicator initialization disabled" if comm.disabled else None,
                )
                tagged_cases = [("cases", case) for case in cases] + [
                    ("fp32_weight_fallbacks", case) for case in fp32_weight_cases
                ]
                for bucket, case in tagged_cases:
                    if case.impl != impl:
                        continue
                    report["active_case"] = asdict(case)
                    snapshots: list[dict[str, Any]] = []
                    report["active_dispatch_snapshots"] = snapshots
                    label = _case_label(world_size, case)
                    report["active_trace_label"] = label
                    report_path.write_text(json.dumps(report, indent=2) + "\n")
                    context = (
                        torch.profiler.record_function(label)
                        if args.profile
                        else nullcontext()
                    )
                    with context:
                        row = _run_case(
                            torch, dist, group, comm, device, case, args, snapshots
                        )
                    row["trace_label"] = label
                    report[bucket].append(row)
                    report_path.write_text(json.dumps(report, indent=2) + "\n")
                if not args.skip_fallback:
                    report["fallbacks"].extend(
                        _run_fallbacks(torch, dist, group, comm, device, impl, args)
                    )
                torch.musa.synchronize()
                dist.barrier(group=group)
                comm.close()
            torch.musa.synchronize()
        report["status"] = "PASS"
        report.pop("active_case", None)
        report.pop("active_dispatch_snapshots", None)
        report.pop("active_trace_label", None)
        report["passed_cases"] = len(report["cases"])
        report["passed_fp32_weight_fallbacks"] = len(report["fp32_weight_fallbacks"])
        execution_rows = report["cases"] + report["fp32_weight_fallbacks"]
        report["passed_fused_cases"] = sum(
            row["kind"] == "FUSED" for row in report["cases"]
        )
        report["passed_fallbacks"] = len(report["fallbacks"]) + sum(
            row["kind"] == "FALLBACK" for row in execution_rows
        )
        report["checked_fused_calls"] = sum(
            row["eager_fused_calls"] + row["graph_fused_replays"]
            for row in execution_rows
        )
        report["checked_fallback_calls"] = len(report["fallbacks"]) + sum(
            row["eager_fallback_calls"] + row["graph_fallback_replays"]
            for row in execution_rows
        )
        checked_replays = sum(row["graph_replays"] for row in execution_rows)
        report["graph_status"] = "PASS" if checked_replays else "NOT_RUN"
        report["fallback_status"] = "NOT_RUN" if args.skip_fallback else "PASS"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as exc:
        report.update(
            status="FAIL",
            error=f"{type(exc).__name__}: {exc}",
            passed_cases=len(report["cases"]),
            passed_fallbacks=len(report["fallbacks"]),
            passed_fp32_weight_fallbacks=len(report["fp32_weight_fallbacks"]),
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        try:
            if args.profile and profiler is not None:
                _save_trace(
                    profiler, args.output_dir / f"rank-{rank}.trace.json", report
                )
        finally:
            try:
                dist.destroy_process_group()
            except Exception as exc:
                report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if report.get("status") != "FAIL":
                    report["status"] = "FAIL"
                    raise
            finally:
                report_path.write_text(json.dumps(report, indent=2) + "\n")
    if rank == 0:
        fields = (
            "status",
            "world_size",
            "passed_fused_cases",
            "passed_fallbacks",
            "checked_fused_calls",
            "graph_status",
        )
        print(
            json.dumps({key: report[key] for key in fields}, sort_keys=True), flush=True
        )
    return 0


def test_musa_car_v2_fused_runtime() -> None:
    """Pytest entry requires explicit distributed execution authorization."""
    import pytest

    if os.environ.get("RUN_MUSA_CAR_V2_FUSED_RUNTIME") != "1":
        pytest.skip("run this file with torchrun; GPU runtime verification not run")
    assert main([]) == 0


if __name__ == "__main__":
    raise SystemExit(main())
