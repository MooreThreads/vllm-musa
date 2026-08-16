# SPDX-License-Identifier: Apache-2.0

"""Live MUSA adapters for the first engine-plan AutoTuner vertical slice."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_io import load_json_object_file, write_json_object_file
from .artifacts import (
    BenchmarkCase,
    ObservationStatus,
    PlanTarget,
    TacticDefinition,
    TimingCache,
    compute_artifact_fingerprint,
    required_power2_bound_rows,
)
from .autotuner import (
    AUTOTUNE_CANDIDATE_SET_KEY,
    AUTOTUNE_MEASUREMENT_KEY,
    AUTOTUNE_TOOLCHAIN_KEY,
    CandidateMeasurement,
    TunableRunner,
    TuningContext,
    collect_operation,
    validate_resume_target,
)
from .core import EnginePlanError
from .tuning_domains import (
    FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES,
    FUSED_ADD_RMS_NORM_DOMAIN_ID,
    FUSED_ADD_RMS_NORM_OPERATION,
    TUNING_DOMAIN_PROVENANCE_KEY,
    validate_tuning_domain_target,
)


class AutoTuneRuntimeError(EnginePlanError):
    """Raised when a live MUSA tuning contract cannot be executed safely."""


@dataclass(frozen=True, slots=True)
class FusedAddRmsNormCollectConfig:
    warmups: int
    iterations: int
    cold_cache_bytes: int
    graph_capture: bool

    def __post_init__(self) -> None:
        if self.warmups <= 0 or self.iterations <= 0:
            raise AutoTuneRuntimeError(
                "AutoTuner warmups and iterations must be positive"
            )
        if self.cold_cache_bytes < 0:
            raise AutoTuneRuntimeError(
                "AutoTuner cold-cache bytes must not be negative"
            )


@dataclass(frozen=True, slots=True)
class FusedAddRmsNormRunner:
    runner_id: str
    implementation_fingerprint: str
    function: Callable[..., object]

    def supports(self, case: BenchmarkCase) -> bool:
        return case.hidden_size in set(
            FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES.values()
        ) and case.dtype in {"bfloat16", "bf16"}

    def build_callable(self) -> Callable[..., object]:
        return self.function


class FusedAddRmsNormCollector:
    """Measure JIT and C-extension runners once, then project thresholds."""

    def __init__(self, config: FusedAddRmsNormCollectConfig) -> None:
        self.config = config
        self.physical_measurements = 0
        self._runner_cache: dict[str, TunableRunner] | None = None

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - live runtime only
            raise AutoTuneRuntimeError(f"torch import failed: {exc}") from exc
        if not hasattr(torch, "musa") or not torch.musa.is_available():
            raise AutoTuneRuntimeError("AutoTuner collect requires a live MUSA device")
        return torch

    def _runners(self) -> dict[str, TunableRunner]:
        if self._runner_cache is not None:
            return self._runner_cache
        try:
            from vllm.ir.op import IrOp
            from vllm.platforms import current_platform

            from vllm_musa.kernels.musa_ops import (
                _run_musa_fused_add_rms_norm_impl,
            )

            current_platform.import_ir_kernels()
            implementations = IrOp.registry["fused_add_rms_norm"].impls
            native = implementations["native"]
            musa = implementations["musa"]
        except Exception as exc:  # pragma: no cover - live runtime only
            raise AutoTuneRuntimeError(
                f"fused-add RMSNorm providers are unavailable: {exc}"
            ) from exc

        def native_function(x: Any, residual: Any, weight: Any) -> Any:
            return native.impl_fn(x, residual, weight, 1e-6, None)

        def runner_function(runner_id: str) -> Callable[..., object]:
            def function(x: Any, residual: Any, weight: Any) -> Any:
                return _run_musa_fused_add_rms_norm_impl(
                    runner_id, x, residual, weight, 1e-6
                )

            return function

        native_fingerprint = str(native.uuid())
        musa_fingerprint = str(musa.uuid())
        self._runner_cache = {
            "native": FusedAddRmsNormRunner(
                runner_id="native",
                implementation_fingerprint=native_fingerprint,
                function=native_function,
            ),
            **{
                runner_id: FusedAddRmsNormRunner(
                    runner_id=runner_id,
                    implementation_fingerprint=compute_artifact_fingerprint(
                        {
                            "provider": musa_fingerprint,
                            "runner": runner_id,
                        }
                    ),
                    function=runner_function(runner_id),
                )
                for runner_id in ("c_ext", "jit")
            },
        }
        return self._runner_cache

    @staticmethod
    def _inputs(case: BenchmarkCase) -> tuple[Any, Any, Any]:
        torch = FusedAddRmsNormCollector._torch()
        torch.musa.manual_seed(20260821 + case.rows)
        x = torch.randn(
            case.rows,
            case.hidden_size,
            device="musa",
            dtype=torch.bfloat16,
        )
        residual = torch.randn(
            case.rows,
            case.hidden_size,
            device="musa",
            dtype=torch.bfloat16,
        )
        weight = torch.randn(
            case.hidden_size,
            device="musa",
            dtype=torch.bfloat16,
        )
        return x, residual, weight

    def _compile(self, runner_id: str, rows: int) -> Any:
        torch = self._torch()
        runner = self._runners()[runner_id]
        compiled = torch.compile(runner.build_callable(), dynamic=True, fullgraph=True)
        try:
            from vllm.compilation.passes.inductor_pass import pass_context
            from vllm.config.utils import Range
        except Exception as exc:
            raise AutoTuneRuntimeError(
                "vLLM compile-range context is unavailable; refusing to "
                "collect timing evidence outside the production lowering context"
            ) from exc

        # vLLM's Range is inclusive at both ends.  Each physical runner is
        # compiled for the exact row bucket being measured so its IR support
        # predicates observe the same pass context as production compilation.
        return compiled, pass_context(Range(rows, rows))

    def _run_runner(
        self,
        runner_id: str,
        case: BenchmarkCase,
        base_inputs: tuple[Any, Any, Any],
    ) -> tuple[tuple[float, ...], tuple[Any, Any]]:
        torch = self._torch()
        x_base, residual_base, weight = base_inputs
        x = x_base.clone()
        residual = residual_base.clone()
        compiled, context = self._compile(runner_id, case.rows)
        with context:
            for _ in range(self.config.warmups):
                x.copy_(x_base)
                residual.copy_(residual_base)
                compiled(x, residual, weight)
            torch.musa.synchronize()

            graph = None
            if self.config.graph_capture:
                graph = torch.musa.MUSAGraph()
                x.copy_(x_base)
                residual.copy_(residual_base)
                torch.musa.synchronize()
                # The MUSA graph context owns capture teardown, including the
                # generator bookkeeping that permits later eager RNG calls.
                # Calling capture_begin/capture_end directly leaves that
                # bookkeeping active on current torch-musa and breaks the next
                # rows bucket while it creates deterministic random inputs.
                with torch.musa.graph(graph):
                    outputs = compiled(x, residual, weight)
            else:
                outputs = compiled(x, residual, weight)
            torch.musa.synchronize()

            flush = None
            if self.config.cold_cache_bytes:
                elements = max(1, self.config.cold_cache_bytes // 4)
                flush = torch.empty(elements, dtype=torch.float32, device="musa")

            samples: list[float] = []
            for _ in range(self.config.iterations):
                x.copy_(x_base)
                residual.copy_(residual_base)
                if flush is not None:
                    flush.zero_()
                torch.musa.synchronize()
                start = torch.musa.Event(enable_timing=True)
                end = torch.musa.Event(enable_timing=True)
                start.record()
                if graph is not None:
                    graph.replay()
                else:
                    outputs = compiled(x, residual, weight)
                end.record()
                torch.musa.synchronize()
                samples.append(float(start.elapsed_time(end)))
        self.physical_measurements += 1
        return tuple(samples), (
            outputs[0].detach().clone(),
            outputs[1].detach().clone(),
        )

    def _run_reference(
        self,
        case: BenchmarkCase,
        base_inputs: tuple[Any, Any, Any],
    ) -> tuple[Any, Any]:
        torch = self._torch()
        x_base, residual_base, weight = base_inputs
        x = x_base.clone()
        residual = residual_base.clone()
        compiled, context = self._compile("native", case.rows)
        with context:
            for _ in range(self.config.warmups):
                x.copy_(x_base)
                residual.copy_(residual_base)
                outputs = compiled(x, residual, weight)
            torch.musa.synchronize()
        return outputs[0].detach().clone(), outputs[1].detach().clone()

    @staticmethod
    def _correct(actual: tuple[Any, Any], expected: tuple[Any, Any]) -> bool:
        torch = FusedAddRmsNormCollector._torch()
        return bool(
            torch.allclose(actual[0], expected[0], rtol=2e-2, atol=2e-2)
            and torch.allclose(actual[1], expected[1], rtol=2e-2, atol=2e-2)
        )

    def measure_missing(
        self,
        context: TuningContext,
        missing: tuple[TacticDefinition, ...],
    ) -> dict[str, CandidateMeasurement]:
        case = context.case
        if case.hidden_size not in set(
            FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES.values()
        ) or case.dtype not in {"bfloat16", "bf16"}:
            raise AutoTuneRuntimeError(
                "fused-add RMSNorm threshold tuning requires a registered "
                "BF16 hidden size"
            )
        inputs = self._inputs(case)
        try:
            expected = self._run_reference(case, inputs)
        except Exception as exc:
            raise AutoTuneRuntimeError(
                f"native correctness reference failed for rows={case.rows}: {exc}"
            ) from exc

        required_runners = {
            "jit" if case.rows >= int(definition.choice) else "c_ext"
            for definition in missing
        }
        measured_runners: dict[str, CandidateMeasurement] = {}
        for runner_id in sorted(required_runners):
            runner = self._runners()[runner_id]
            if not runner.supports(case):
                measured_runners[runner_id] = CandidateMeasurement(
                    values_ms=(),
                    status=ObservationStatus.FAILED,
                    correctness="failed",
                    provenance=(
                        ("autotune.runner", runner_id),
                        (
                            "autotune.runner_implementation",
                            runner.implementation_fingerprint,
                        ),
                        ("autotune.error", "runner does not support case"),
                    ),
                )
                continue
            try:
                values, outputs = self._run_runner(runner_id, case, inputs)
                correctness = self._correct(outputs, expected)
                measured_runners[runner_id] = CandidateMeasurement(
                    values_ms=values if correctness else (),
                    status=(
                        ObservationStatus.PASSED
                        if correctness
                        else ObservationStatus.FAILED
                    ),
                    correctness="passed" if correctness else "failed",
                    provenance=(
                        ("autotune.runner", runner_id),
                        (
                            "autotune.runner_implementation",
                            self._runners()[runner_id].implementation_fingerprint,
                        ),
                    ),
                )
            except Exception as exc:
                measured_runners[runner_id] = CandidateMeasurement(
                    values_ms=(),
                    status=ObservationStatus.FAILED,
                    correctness="failed",
                    provenance=(
                        ("autotune.runner", runner_id),
                        (
                            "autotune.runner_implementation",
                            self._runners()[runner_id].implementation_fingerprint,
                        ),
                        ("autotune.error", f"{type(exc).__name__}: {exc}"),
                    ),
                )

        return {
            definition.tactic_id: measured_runners[
                "jit" if case.rows >= int(definition.choice) else "c_ext"
            ]
            for definition in missing
        }


def _load_target(path: str | Path) -> PlanTarget:
    document = load_json_object_file(path)
    target_document = document.get("target", document)
    return PlanTarget.from_document(target_document)


def _load_resume(path: str | Path | None) -> TimingCache | None:
    if path is None:
        return None
    return TimingCache.from_document(
        load_json_object_file(path), require_fingerprint=False
    )


def collect_fused_add_rms_norm(args: Any) -> dict[str, object]:
    """Execute the explicit offline collect mode used by the CLI and skill."""

    from .cli import _runtime_catalog
    from .runtime import diff_environment_identity

    target = _load_target(args.target)
    domain_id = getattr(args, "domain", FUSED_ADD_RMS_NORM_DOMAIN_ID)
    if domain_id not in FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES:
        raise AutoTuneRuntimeError(
            f"collect adapter does not implement tuning domain {domain_id!r}; "
            "supported="
            f"{tuple(FUSED_ADD_RMS_NORM_DOMAIN_HIDDEN_SIZES)!r}"
        )
    try:
        validate_tuning_domain_target(domain_id, target)
    except ValueError as exc:
        raise AutoTuneRuntimeError(str(exc)) from exc
    resume = _load_resume(args.resume)
    # Run this before the live catalog is enumerated.  collect_operation repeats
    # the check so future adapters cannot accidentally bypass the core guard.
    validate_resume_target(target, resume)
    if resume is not None:
        resume_domain = dict(resume.provenance).get(TUNING_DOMAIN_PROVENANCE_KEY)
        if resume_domain is not None and resume_domain != domain_id:
            raise AutoTuneRuntimeError(
                "resume cache tuning domain does not match the requested "
                f"domain: cache={resume_domain!r}, requested={domain_id!r}"
            )
    differences = diff_environment_identity(target)
    if differences:
        raise AutoTuneRuntimeError(
            "AutoTuner target identity does not match the live environment; "
            "refusing to measure candidates: " + "; ".join(differences)
        )
    operation = FUSED_ADD_RMS_NORM_OPERATION
    definitions = tuple(
        TacticDefinition.from_document(item, index)
        for index, item in enumerate(_runtime_catalog(include_tuning=True))
        if item["operation"] == operation
    )
    if not definitions:
        raise AutoTuneRuntimeError(
            "the live runtime did not expose fused-add RMSNorm tuning candidates"
        )
    rows = (
        tuple(sorted(set(args.rows)))
        if args.rows
        else required_power2_bound_rows(target.workload.max_num_batched_tokens)
    )
    invalid_rows = [
        row
        for row in rows
        if row < target.workload.tokens.minimum
        or row > target.workload.tokens.maximum
        or row > target.workload.max_num_batched_tokens
    ]
    if invalid_rows:
        raise AutoTuneRuntimeError(
            "requested rows fall outside the target token envelope: "
            f"{invalid_rows!r}"
        )
    cases = tuple(
        BenchmarkCase.create(
            operation=operation,
            phase="operator",
            batch_size=1,
            tokens=row,
            rows=row,
            hidden_size=target.model.hidden_size,
            dtype=target.model.dtype,
        )
        for row in rows
    )
    graph_capture = not args.eager and target.workload.graph_mode.lower() not in {
        "none",
        "eager",
    }
    collector = FusedAddRmsNormCollector(
        FusedAddRmsNormCollectConfig(
            warmups=args.warmups,
            iterations=args.iterations,
            cold_cache_bytes=args.cold_cache_bytes,
            graph_capture=graph_capture,
        )
    )
    started = time.monotonic()
    document, stats = collect_operation(
        target=target,
        definitions=definitions,
        cases=cases,
        measure_missing=collector.measure_missing,
        software_dependencies=(
            "driver",
            "musa",
            "torch",
            "torch-musa",
            "vllm",
            "vllm-musa",
        ),
        source_dependencies=("vllm", "vllm-musa"),
        existing_cache=resume,
        provenance={
            "collector": "vllm_musa.engine_plan.autotune_runtime",
            TUNING_DOMAIN_PROVENANCE_KEY: domain_id,
            "measurement.graph_capture": str(graph_capture).lower(),
            "measurement.cold_cache_bytes": str(args.cold_cache_bytes),
            "measurement.warmups": str(args.warmups),
            "measurement.iterations": str(args.iterations),
        },
        measurement_identity={
            "adapter": "fused_add_rms_norm.threshold.v1",
            "discovery_source": "target_envelope",
            "graph_capture": graph_capture,
            "cold_cache_bytes": args.cold_cache_bytes,
            "warmups": args.warmups,
            "iterations": args.iterations,
        },
        seal=False,
    )
    write_json_object_file(args.output, document)
    summary: dict[str, object] = stats.to_document()
    summary.update(
        {
            "status": "written",
            "output": str(args.output),
            "discovery_source": "target_envelope",
            "tuning_domain": domain_id,
            "captured_events": 0,
            "unique_keys": len(cases),
            "scheduled_trials": stats.measured,
            "rows": list(rows),
            "physical_measurements": collector.physical_measurements,
            "wall_time_seconds": round(time.monotonic() - started, 6),
            "evidence_fingerprint": compute_artifact_fingerprint(document),
            "candidate_set_fingerprint": document["provenance"][
                AUTOTUNE_CANDIDATE_SET_KEY
            ],
            "toolchain_fingerprint": document["provenance"][AUTOTUNE_TOOLCHAIN_KEY],
            "measurement_fingerprint": document["provenance"][AUTOTUNE_MEASUREMENT_KEY],
        }
    )
    if args.summary_output:
        write_json_object_file(args.summary_output, summary)
    return summary
