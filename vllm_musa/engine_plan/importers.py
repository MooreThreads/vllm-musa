# SPDX-License-Identifier: Apache-2.0

"""Adapters from existing vLLM-MUSA benchmark evidence into timing caches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .artifacts import (
    LEGACY_TIMING_CACHE_SCHEMA_VERSION,
    BenchmarkCase,
    PlanningArtifactError,
    PlanTarget,
    TimingCacheBuilder,
    seal_timing_cache_document,
)
from .json_utils import dumps as dump_json

OPERATOR_INTEGRATION_SCHEMA = "vllm_musa.operator_integration.campaign.v1"


def _case_dimension(
    case: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    names: tuple[str, ...],
    field: str,
) -> int:
    parameters = case.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise PlanningArtifactError("campaign case parameters must be an object")
    values = [
        source[name]
        for source in (parameters, metadata)
        for name in names
        if name in source
    ]
    if not values:
        raise PlanningArtifactError(
            f"campaign must provide typed {field} in case.parameters or metadata"
        )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in values
    ):
        raise PlanningArtifactError(f"campaign {field} must be a positive integer")
    if len(set(values)) != 1:
        raise PlanningArtifactError(f"campaign {field} declarations disagree")
    return values[0]


def _typed_benchmark_case(
    case: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    target: PlanTarget,
    operation: str,
) -> BenchmarkCase:
    if metadata.get("operator") != operation:
        raise PlanningArtifactError(
            "campaign metadata.operator must exactly match the imported operation"
        )
    dtype = metadata.get("dtype")
    if dtype != target.model.dtype:
        raise PlanningArtifactError(
            "campaign metadata.dtype must exactly match target.model.dtype"
        )
    rows = _case_dimension(
        case,
        metadata,
        names=("rows",),
        field="rows",
    )
    hidden_size = _case_dimension(
        case,
        metadata,
        names=("hidden_size", "hidden"),
        field="hidden_size",
    )
    if hidden_size != target.model.hidden_size:
        raise PlanningArtifactError(
            "campaign hidden_size must exactly match target.model.hidden_size"
        )
    return BenchmarkCase.create(
        operation=operation,
        phase="operator",
        batch_size=1,
        tokens=rows,
        rows=rows,
        hidden_size=hidden_size,
        dtype=dtype,
    )


def _stage_for_choice(choice: str) -> str:
    if choice == "native":
        return "native_chain"
    if choice == "musa":
        return "musa_provider_chain"
    raise PlanningArtifactError(
        f"No operator-integration stage mapping exists for provider {choice!r}; "
        "pass an explicit stage map through the Python SDK"
    )


def _correctness_status(case: Mapping[str, Any], stage: str) -> str:
    if stage == "native_chain":
        return "passed"
    correctness = case.get("correctness", {})
    if not isinstance(correctness, Mapping):
        return "missing"
    comparison = correctness.get(stage)
    if not isinstance(comparison, Mapping):
        return "missing"
    close = comparison.get("close")
    if not isinstance(close, list) or not close:
        return "missing"
    return "passed" if all(value is True for value in close) else "failed"


def _stage_values_ms(
    case: Mapping[str, Any],
    *,
    stage: str,
    mode: str,
) -> tuple[str, list[float], str]:
    stages = case.get("stages")
    if not isinstance(stages, Mapping):
        return "failed", [], "missing_stages"
    stage_result = stages.get(stage)
    if not isinstance(stage_result, Mapping):
        return "failed", [], "missing_stage"
    if "setup_error" in stage_result:
        return "failed", [], f"setup_error:{stage_result['setup_error']}"
    mode_result = stage_result.get(mode)
    if not isinstance(mode_result, Mapping):
        return "failed", [], "missing_mode"
    if "error" in mode_result:
        return "failed", [], f"measurement_error:{mode_result['error']}"
    event = mode_result.get("event_us")
    if not isinstance(event, Mapping):
        return "failed", [], "missing_event_us"
    samples = event.get("samples")
    if not isinstance(samples, list) or not samples:
        return "failed", [], "missing_samples"
    values: list[float] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, (int, float)) or isinstance(sample, bool):
            raise PlanningArtifactError(
                f"{stage}.{mode}.event_us.samples[{index}] is not numeric"
            )
        values.append(float(sample) / 1000.0)
    return "passed", values, "imported_event_us"


def import_operator_integration_campaign(
    campaign: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    operation: str,
    case_index: int = 0,
    mode: str = "graph_replay_1",
    stage_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Import one harness case without duplicating its measurement runner."""

    if campaign.get("schema_version") != OPERATOR_INTEGRATION_SCHEMA:
        raise PlanningArtifactError(
            f"campaign.schema_version must be {OPERATOR_INTEGRATION_SCHEMA!r}"
        )
    results = campaign.get("results")
    if not isinstance(results, list) or not results:
        raise PlanningArtifactError("campaign.results must be a non-empty list")
    if not isinstance(case_index, int) or isinstance(case_index, bool):
        raise PlanningArtifactError("case_index must be an integer")
    try:
        case = results[case_index]
    except IndexError as exc:
        raise PlanningArtifactError(
            f"case_index {case_index} is outside campaign.results"
        ) from exc
    if not isinstance(case, Mapping):
        raise PlanningArtifactError(f"campaign.results[{case_index}] must be an object")

    operation_catalog = [
        dict(item) for item in catalog if item.get("operation") == operation
    ]
    mapping = dict(stage_map or {})
    selected_catalog = [
        item
        for item in operation_catalog
        if item.get("choice") in {"native", "musa"} or item.get("id") in mapping
    ]
    if not selected_catalog:
        raise PlanningArtifactError(
            f"catalog contains no tactics for operation {operation!r}"
        )
    campaign_metadata = campaign.get("metadata", {})
    if not isinstance(campaign_metadata, Mapping):
        raise PlanningArtifactError("campaign.metadata must be an object")
    parsed_target = PlanTarget.from_document(dict(target))
    benchmark_case = _typed_benchmark_case(
        case,
        campaign_metadata,
        target=parsed_target,
        operation=operation,
    )
    case_name = str(case.get("name", f"case-{case_index}"))
    try:
        campaign_metadata_json = dump_json(
            campaign_metadata,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PlanningArtifactError(
            "campaign metadata must contain only finite JSON values"
        ) from exc
    collector = TimingCacheBuilder.from_documents(
        target=target,
        catalog=selected_catalog,
        provenance={
            "importer": OPERATOR_INTEGRATION_SCHEMA,
            "case": case_name,
            "case_index": str(case_index),
            "mode": mode,
            "case_fingerprint": benchmark_case.fingerprint,
            "campaign_metadata": campaign_metadata_json,
            "excluded_unmeasured_tactics": ",".join(
                sorted(
                    str(item.get("id"))
                    for item in operation_catalog
                    if item not in selected_catalog
                )
            )
            or "none",
        },
    )
    for tactic in collector.catalog:
        stage = mapping.get(tactic.tactic_id)
        if stage is None:
            stage = _stage_for_choice(tactic.choice)
        status, values, import_reason = _stage_values_ms(
            case,
            stage=stage,
            mode=mode,
        )
        correctness = _correctness_status(case, stage)
        if correctness != "passed":
            status = "failed"
            values = []
        collector.add_observation(
            tactic.tactic_id,
            values,
            case=benchmark_case,
            status=status,
            correctness=correctness,
            provenance={
                "case": case_name,
                "stage": stage,
                "mode": mode,
                "import_reason": import_reason,
            },
        )
    # Campaign-v1 names/metadata were designed for reporting, not as a
    # machine-bound shape manifest emitted by the tensor-construction path.
    # Preserve the imported measurements as inspectable legacy evidence, but
    # deliberately remove the typed activation binding so selection must stay
    # on the native fallback. A future harness schema can emit BenchmarkCase
    # fingerprints from the same code that constructs the measured tensors.
    document = collector.build(seal=False)
    document["schema_version"] = LEGACY_TIMING_CACHE_SCHEMA_VERSION
    document["target"]["model"].pop("hidden_size")
    for observation in document["observations"]:
        observation.pop("case")
    document["provenance"]["coverage_binding"] = "unbound:campaign-v1"
    return seal_timing_cache_document(document)
