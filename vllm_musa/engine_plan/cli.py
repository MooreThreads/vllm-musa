# SPDX-License-Identifier: Apache-2.0

"""Command-line UX for building and understanding MUSA engine plans."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from vllm_musa.runtime_plan.catalog import list_runtime_decision_specs
from vllm_musa.runtime_plan.declarative import (
    DeclarativeProfileError,
    declarative_profile_catalog,
    load_declarative_profile,
)

from .artifact_io import (
    ArtifactFileError,
    load_json_object_file,
    write_json_object_file,
)
from .artifacts import (
    PlanningArtifactError,
    PlanTarget,
    TacticDefinition,
    TacticKind,
    TimingCache,
    merge_timing_cache_documents,
    seal_timing_cache_document,
)
from .core import EnginePlanError, load_plan, parse_plan_document
from .importers import import_operator_integration_campaign
from .json_utils import dumps as dump_json
from .planner import (
    BuildPolicy,
    build_plan_document,
    explain_plan,
    inspect_plan,
)
from .tactic_fingerprints import runtime_decision_implementation
from .tuning_domains import (
    FUSED_ADD_RMS_NORM_DOMAIN_ID,
    FUSED_MOE_FP8_BLOCK_DOMAIN_ID,
    TUNING_DOMAIN_SCHEMA_VERSION,
    fused_moe_tactic_definitions,
    get_tuning_domain,
    list_tuning_domains,
    resolve_timing_cache_domain,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return load_json_object_file(path)


def _write_json(path: str | Path, document: dict[str, Any]) -> None:
    write_json_object_file(path, document)


def _paths_alias(first: str | Path, second: str | Path) -> bool:
    first_path = Path(first)
    second_path = Path(second)
    if first_path.resolve() == second_path.resolve():
        return True
    try:
        return first_path.samefile(second_path)
    except OSError:
        return False


def _load_context(path: str | Path) -> dict[str, Any]:
    document = _read_json(path)
    if "target" in document:
        target = document["target"]
    else:
        target = document
    if not isinstance(target, dict):
        raise EnginePlanError("Runtime context must be a target JSON object")
    return target


def _cmd_build(args: argparse.Namespace) -> int:
    timings = [_read_json(path) for path in args.timings]
    runtime_decisions = (
        _read_json(args.runtime_decisions)
        if args.runtime_decisions is not None
        else None
    )
    policy = BuildPolicy(
        min_samples=args.min_samples,
        min_speedup_pct=args.min_speedup_pct,
        tie_tolerance_pct=args.tie_tolerance_pct,
    )
    plan = build_plan_document(
        timings,
        plan_id=args.plan_id,
        policy=policy,
        runtime_decisions=runtime_decisions,
    )
    _write_json(args.output, plan)
    parsed = parse_plan_document(plan)
    print(
        dump_json(
            {
                "status": "built",
                "output": str(Path(args.output)),
                "plan_id": parsed.plan_id,
                "fingerprint": parsed.fingerprint,
                "variants": len(parsed.variants),
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_cache_seal(args: argparse.Namespace) -> int:
    sealed = seal_timing_cache_document(_read_json(args.input))
    _write_json(args.output, sealed)
    print(
        dump_json(
            {
                "status": "sealed",
                "output": args.output,
                "fingerprint": sealed["fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_cache_merge(args: argparse.Namespace) -> int:
    merged = merge_timing_cache_documents([_read_json(path) for path in args.inputs])
    _write_json(args.output, merged)
    print(
        dump_json(
            {
                "status": "merged",
                "output": args.output,
                "inputs": len(args.inputs),
                "fingerprint": merged["fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_import_harness(args: argparse.Namespace) -> int:
    catalog_document = _read_json(args.catalog)
    catalog = catalog_document.get("catalog")
    if not isinstance(catalog, list):
        raise EnginePlanError("catalog JSON must contain a catalog list")
    timing_cache = import_operator_integration_campaign(
        _read_json(args.campaign),
        target=_load_context(args.target),
        catalog=catalog,
        operation=args.operation,
        case_index=args.case_index,
        mode=args.mode,
    )
    _write_json(args.output, timing_cache)
    print(
        dump_json(
            {
                "status": "imported",
                "output": args.output,
                "fingerprint": timing_cache["fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


def _key_value_pairs(values: list[str], *, field: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise EnginePlanError(f"{field} values must use KEY=VALUE")
        if key in result:
            raise EnginePlanError(f"{field} contains duplicate key {key!r}")
        result[key] = item
    return result


def _cmd_target(args: argparse.Namespace) -> int:
    # This command intentionally imports the runtime probe lazily. All offline
    # artifact commands remain usable without torch, vLLM, or MUSA.
    from .runtime import collect_environment_identity

    environment = collect_environment_identity(
        device_count=args.tensor_parallel_size * args.pipeline_parallel_size
    )
    explicit_source_revisions = _key_value_pairs(
        args.source_revision,
        field="--source-revision",
    )
    detected_source_revisions = environment.get("source_revisions", {})
    if (
        explicit_source_revisions
        and detected_source_revisions
        and explicit_source_revisions != detected_source_revisions
    ):
        raise EnginePlanError(
            "Explicit source revisions do not match the runtime build manifest"
        )
    source_revisions = detected_source_revisions or explicit_source_revisions
    document = {
        "model": {
            "profile": args.profile,
            "architecture": args.architecture,
            "model_id": args.model_id,
            "hidden_size": args.hidden_size,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
        },
        "hardware": environment["hardware"],
        "software": {
            "versions": environment["software_versions"],
            "source_revisions": source_revisions,
            "image_digest": args.image_digest,
        },
        "workload": {
            "phase": args.phase,
            "batch_size": {"min": args.batch_min, "max": args.batch_max},
            "tokens": {"min": args.tokens_min, "max": args.tokens_max},
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "compile_mode": args.compile_mode,
            "graph_mode": args.graph_mode,
        },
    }
    if args.cudagraph_capture_sizes is not None:
        document["workload"]["cudagraph_capture_sizes"] = args.cudagraph_capture_sizes
    target = PlanTarget.from_document(document)
    _write_json(args.output, target.to_document())
    print(
        dump_json(
            {
                "status": "captured",
                "output": args.output,
                "target_fingerprint": target.fingerprint,
            },
            sort_keys=True,
        )
    )
    return 0


def _human_inspect(summary: dict[str, Any]) -> str:
    lines = [
        f"Plan: {summary['plan_id']}",
        f"Schema: {summary['schema_version']}",
        f"Fingerprint: {summary['fingerprint']}",
    ]
    for variant in summary["variants"]:
        model = variant["model"]
        hardware = variant["hardware"]
        workload = variant["workload"]
        lines.extend(
            [
                "",
                f"Variant: {variant['variant_id']}",
                (
                    "  Target: "
                    f"{model['architecture']} {model['dtype']} "
                    f"TP{model['tensor_parallel_size']} "
                    f"PP{model['pipeline_parallel_size']}"
                ),
                (
                    "  Hardware: "
                    f"{hardware['device_name']} capability="
                    f"{hardware['device_capability']} "
                    f"MP={hardware['multiprocessor_count']}"
                ),
                (
                    "  Workload: "
                    f"{workload['phase']} batch={workload['batch_size']} "
                    f"tokens={workload['tokens']} compile="
                    f"{workload['compile_mode']} graph={workload['graph_mode']}"
                ),
                f"  Timing cache: {variant['timing_fingerprint']}",
                (
                    "  Tuning domain: "
                    f"{variant['tuning_domain']['id']} "
                    f"({variant['tuning_domain']['source']})"
                ),
            ]
        )
        for selection in variant["selections"]:
            contexts = selection.get("contexts")
            if isinstance(contexts, list):
                lines.append(
                    "  Select "
                    f"{selection['operation']}: contextual "
                    f"(fallback={selection['fallback']}, "
                    f"contexts={len(contexts)})"
                )
                for context in contexts:
                    token_bucket = context["token_bucket"]
                    shape = context["shape"]
                    lines.append(
                        "    Context "
                        f"{context['context_id']}: {context['winner']} "
                        f"(graph={shape['graph_mode']}, "
                        f"tokens={token_bucket['min']}-{token_bucket['max']}, "
                        f"fallback={context['fallback']}, "
                        f"speedup={context['speedup_pct']:.3f}%, "
                        f"samples={context['samples']}, "
                        f"reason={context['reason']})"
                    )
                continue
            lines.append(
                "  Select "
                f"{selection['operation']}: {selection['winner']} "
                f"(fallback={selection['fallback']}, "
                f"median={selection['winner_median_ms']}, "
                f"p95={selection['winner_p95_ms']}, "
                f"speedup={selection['speedup_pct']:.3f}%, "
                f"samples={selection['samples']}, "
                f"reason={selection['reason']}, "
                f"coverage={selection['coverage']['status']} "
                f"{len(selection['coverage']['observed_buckets'])}/"
                f"{len(selection['coverage']['required_buckets'])})"
            )
        projection = variant["runtime_decisions"]
        lines.append(f"  Runtime profile: {projection['profile']}")
        for key, value in sorted(projection["values"].items()):
            lines.append(
                f"  Decide {key}: "
                f"{dump_json(value, sort_keys=True, ensure_ascii=False)}"
            )
    return "\n".join(lines)


def _cmd_inspect(args: argparse.Namespace) -> int:
    summary = inspect_plan(load_plan(args.plan))
    if args.json:
        print(dump_json(summary, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(_human_inspect(summary))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    document = _read_json(args.artifact)
    schema_version = document.get("schema_version")
    if isinstance(schema_version, int):
        plan = parse_plan_document(document)
        result: dict[str, Any] = {
            "status": "valid",
            "kind": "engine_plan",
            "schema_version": plan.schema_version,
            "fingerprint": plan.fingerprint,
        }
        if args.context:
            explanation = explain_plan(
                plan,
                runtime_target=_load_context(args.context),
                final=not args.early,
            )
            decision = explanation["runtime_decision"]
            result["runtime_decision"] = decision
            if decision["status"] != "selected":
                print(dump_json(result, indent=2, sort_keys=True))
                return 2
    else:
        timing = TimingCache.from_document(document, require_fingerprint=True)
        domain, domain_source = resolve_timing_cache_domain(timing)
        result = {
            "status": "valid",
            "kind": "timing_cache",
            "schema_version": schema_version,
            "fingerprint": timing.fingerprint,
        }
        if domain is not None:
            result["tuning_domain"] = domain.domain_id
            result["tuning_domain_source"] = domain_source
    print(dump_json(result, indent=2, sort_keys=True))
    return 0


def _human_explain(explanation: dict[str, Any]) -> str:
    decision = explanation["runtime_decision"]
    lines = [
        f"Plan: {explanation['plan_id']}",
        f"Decision: {decision['status']}",
        f"Reason: {decision['reason']}",
    ]
    if decision.get("selected_variant"):
        lines.append(f"Selected variant: {decision['selected_variant']}")
    for variant in decision.get("variants", []):
        lines.append(
            f"Variant {variant['variant_id']}: "
            f"{'match' if variant['matches'] else 'rejected'}"
        )
        for difference in variant["differences"]:
            lines.append(f"  mismatch: {difference}")
        for selection in variant["selections"]:
            if "contexts" in selection:
                lines.append(
                    f"  {selection['operation']}: contextual winners="
                    f"{len(selection['contexts'])} fallback={selection['fallback']}"
                )
                for context in selection["contexts"]:
                    bucket = context["token_bucket"]
                    lines.append(
                        "    "
                        f"tokens=[{bucket['min']},{bucket['max']}]: "
                        f"winner={context['winner']} reason={context['reason']}"
                    )
                continue
            lines.append(
                f"  {selection['operation']}: winner={selection['winner']} "
                f"fallback={selection['fallback']} reason={selection['reason']}"
            )
            for rejected in selection["rejected"]:
                lines.append(
                    f"    rejected {rejected['tactic_id']}: {rejected['reason']}"
                )
    return "\n".join(lines)


def _cmd_explain(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    context = _load_context(args.context) if args.context else None
    explanation = explain_plan(
        plan,
        runtime_target=context,
        final=not args.early,
    )
    if args.json:
        print(dump_json(explanation, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(_human_explain(explanation))
    return 0


def _runtime_catalog(*, include_tuning: bool = False) -> list[dict[str, Any]]:
    """Project the existing vLLM IR registry; never create a second registry."""

    from vllm_musa.engine_plugins import list_engine_ir_providers

    providers = list_engine_ir_providers()
    native_operations = {
        item.operation for item in providers if item.provider == "native"
    }
    catalog: list[dict[str, Any]] = []
    for item in providers:
        if item.operation not in native_operations:
            continue
        definition = TacticDefinition(
            tactic_id=f"vllm.ir.{item.operation}:{item.provider}",
            kind=TacticKind.VLLM_IR_PROVIDER,
            operation=item.operation,
            choice=item.provider,
            fallback_id=f"vllm.ir.{item.operation}:native",
            implementation_fingerprint=item.implementation_fingerprint,
            description=(
                f"Existing vLLM IR provider {item.provider!r} "
                f"for {item.operation!r}"
            ),
        )
        catalog.append(definition.to_document())
    if include_tuning:
        from vllm_musa.runtime_plan import RuntimeDecision
        from vllm_musa.tuning import (
            DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS,
            FUSED_ADD_RMSNORM_THRESHOLD_CHOICES,
        )

        provider_fingerprints = {
            (item.operation, item.provider): item.implementation_fingerprint
            for item in providers
        }
        operation = RuntimeDecision.MUSA_FUSED_ADD_RMSNORM_MIN_ROWS.value
        implementation = runtime_decision_implementation(operation)
        provider_fingerprint = (
            provider_fingerprints.get(implementation.provider_key)
            if implementation is not None
            else None
        )
        if implementation is not None and provider_fingerprint is not None:
            prefix = "runtime.musa.fused_add_rms_norm.min_rows"
            fallback_id = f"{prefix}:{DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS}"
            selector_fingerprint = implementation.fingerprint(provider_fingerprint)
            for threshold in FUSED_ADD_RMSNORM_THRESHOLD_CHOICES:
                catalog.append(
                    TacticDefinition(
                        tactic_id=f"{prefix}:{threshold}",
                        kind=TacticKind.RUNTIME_DECISION,
                        operation=operation,
                        choice=threshold,
                        fallback_id=fallback_id,
                        implementation_fingerprint=selector_fingerprint,
                        description=(
                            "Select the first row count for a registered BF16 "
                            "hidden size using the MUSA JIT fused-add RMSNorm "
                            f"runner ({threshold})"
                        ),
                    ).to_document()
                )
        catalog.extend(
            definition.to_document() for definition in fused_moe_tactic_definitions()
        )
    if not catalog:
        raise EnginePlanError("The live vLLM runtime exposed no supported IR tactics")
    return catalog


def _cmd_catalog(args: argparse.Namespace) -> int:
    document = {
        "schema_version": "musa.engine_tactic_catalog.v1",
        "catalog": _runtime_catalog(include_tuning=args.include_tuning),
    }
    if args.output:
        _write_json(args.output, document)
        print(dump_json({"status": "written", "output": args.output}))
    else:
        print(dump_json(document, indent=2, sort_keys=True))
    return 0


def _cmd_profiles(args: argparse.Namespace) -> int:
    if args.profile:
        if args.output and _paths_alias(args.profile, args.output):
            raise DeclarativeProfileError(
                "profile validation output must not overwrite its input"
            )
        profile = load_declarative_profile(Path(args.profile))
        document = {
            "schema_version": "musa.runtime_profile_validation.v1",
            "status": "valid",
            "id": profile.identifier,
            "priority": profile.priority,
            "fingerprint": profile.fingerprint,
            "decisions": [
                {
                    "decision": rule.decision.value,
                    "tunability": rule.tunability,
                    "requires": [dependency.value for dependency in rule.requires],
                }
                for rule in profile.decisions
            ],
        }
    else:
        document = {
            "schema_version": "musa.runtime_profile_catalog.v1",
            "profiles": list(declarative_profile_catalog()),
            "decision_catalog": [
                {
                    "decision": spec.key.value,
                    "kind": spec.kind.value,
                    "phase": spec.phase.value,
                    "fallback": spec.fallback,
                    "choices": list(spec.choices),
                    "external_only": spec.external_only,
                    "profile_families": list(spec.profile_families),
                    "tunability": spec.tunability.value,
                }
                for spec in list_runtime_decision_specs()
            ],
        }
    if args.output:
        _write_json(args.output, document)
    else:
        print(dump_json(document, indent=2, sort_keys=True))
    return 0


def _cmd_autotune_collect(args: argparse.Namespace) -> int:
    from .autotune_runtime import collect_fused_add_rms_norm

    summary = collect_fused_add_rms_norm(args)
    print(dump_json(summary, indent=2, sort_keys=True))
    return 0


def _import_tuning_domain(args: argparse.Namespace, *, domain_id: str) -> int:
    domain = get_tuning_domain(domain_id)
    if domain.evidence_mode != "import":
        raise EnginePlanError(
            f"tuning domain {domain_id!r} uses {domain.evidence_mode!r} evidence; "
            "it cannot be imported"
        )
    if domain_id != FUSED_MOE_FP8_BLOCK_DOMAIN_ID:
        raise EnginePlanError(
            f"no evidence importer is registered for tuning domain {domain_id!r}"
        )
    from .fused_moe_evidence import import_fused_moe_crossover_evidence

    summary = import_fused_moe_crossover_evidence(
        target_path=args.target,
        evidence_paths=args.evidence,
        output_path=args.output,
        image_digest=args.image_digest,
    )
    print(dump_json(summary, indent=2, sort_keys=True))
    return 0


def _cmd_autotune_import(args: argparse.Namespace) -> int:
    return _import_tuning_domain(args, domain_id=args.domain)


def _cmd_autotune_import_fused_moe(args: argparse.Namespace) -> int:
    return _import_tuning_domain(args, domain_id=FUSED_MOE_FP8_BLOCK_DOMAIN_ID)


def _cmd_autotune_domains(args: argparse.Namespace) -> int:
    del args
    print(
        dump_json(
            {
                "schema_version": TUNING_DOMAIN_SCHEMA_VERSION,
                "domains": [domain.to_document() for domain in list_tuning_domains()],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vllm_musa.engine_plan",
        description=(
            "Build, inspect, validate, and explain sealed vLLM-MUSA "
            "performance plans."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="select tactics from normalized timing caches and seal a plan",
    )
    build.add_argument("--timings", action="append", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--plan-id", required=True)
    build.add_argument("--min-samples", type=int, default=3)
    build.add_argument("--min-speedup-pct", type=float, default=1.0)
    build.add_argument("--tie-tolerance-pct", type=float, default=0.5)
    build.add_argument(
        "--runtime-decisions",
        help=(
            "JSON profile map of typed runtime decisions; values are sealed "
            "into the exact matching plan variant and cannot override "
            "timing-derived selections"
        ),
    )
    build.set_defaults(func=_cmd_build)

    cache = subparsers.add_parser(
        "cache",
        help="seal or merge mutable timing evidence before plan construction",
    )
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_seal = cache_commands.add_parser(
        "seal",
        help="canonicalize and fingerprint one timing cache",
    )
    cache_seal.add_argument("input")
    cache_seal.add_argument("--output", required=True)
    cache_seal.set_defaults(func=_cmd_cache_seal)
    cache_merge = cache_commands.add_parser(
        "merge",
        help="merge repeated observations for one exact target/catalog",
    )
    cache_merge.add_argument("inputs", nargs="+")
    cache_merge.add_argument("--output", required=True)
    cache_merge.set_defaults(func=_cmd_cache_merge)

    import_harness = subparsers.add_parser(
        "import-harness",
        help="convert an operator-integration campaign into a timing cache",
    )
    import_harness.add_argument("--campaign", required=True)
    import_harness.add_argument("--target", required=True)
    import_harness.add_argument("--catalog", required=True)
    import_harness.add_argument("--operation", required=True)
    import_harness.add_argument("--case-index", type=int, default=0)
    import_harness.add_argument("--mode", default="graph_replay_1")
    import_harness.add_argument("--output", required=True)
    import_harness.set_defaults(func=_cmd_import_harness)

    target = subparsers.add_parser(
        "target",
        help="capture MUSA hardware/software and combine explicit serving scope",
    )
    target.add_argument("--profile", required=True)
    target.add_argument("--architecture", required=True)
    target.add_argument("--model-id", required=True)
    target.add_argument("--hidden-size", type=int, required=True)
    target.add_argument("--dtype", required=True)
    target.add_argument("--quantization", default="none")
    target.add_argument("--tensor-parallel-size", type=int, default=1)
    target.add_argument("--pipeline-parallel-size", type=int, default=1)
    target.add_argument(
        "--source-revision",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "declare target source identity when no build manifest is present; "
            "autotune collect still requires a live build manifest"
        ),
    )
    target.add_argument("--image-digest", required=True)
    target.add_argument(
        "--phase",
        choices=("prefill", "decode", "mixed", "operator", "serving"),
        required=True,
    )
    target.add_argument("--batch-min", type=int, required=True)
    target.add_argument("--batch-max", type=int, required=True)
    target.add_argument("--tokens-min", type=int, required=True)
    target.add_argument("--tokens-max", type=int, required=True)
    target.add_argument("--max-model-len", type=int, required=True)
    target.add_argument("--max-num-batched-tokens", type=int, required=True)
    target.add_argument("--max-num-seqs", type=int, required=True)
    target.add_argument("--compile-mode", required=True)
    target.add_argument("--graph-mode", required=True)
    target.add_argument(
        "--cudagraph-capture-sizes",
        type=int,
        nargs="+",
        help=(
            "final ordered graph capture ladder; required when contextual "
            "capture evidence is promoted into a RuntimePlan"
        ),
    )
    target.add_argument("--output", required=True)
    target.set_defaults(func=_cmd_target)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="show target, evidence, winner, fallback, and provenance",
    )
    inspect_parser.add_argument("plan")
    inspect_parser.add_argument("--json", action="store_true")
    inspect_parser.set_defaults(func=_cmd_inspect)

    validate = subparsers.add_parser(
        "validate",
        help="verify schema, fingerprints, and optional runtime compatibility",
    )
    validate.add_argument("artifact")
    validate.add_argument("--context")
    validate.add_argument("--early", action="store_true")
    validate.set_defaults(func=_cmd_validate)

    explain = subparsers.add_parser(
        "explain",
        help="explain selection, rejection, invalidation, and fallback",
    )
    explain.add_argument("plan")
    explain.add_argument("--context")
    explain.add_argument("--early", action="store_true")
    explain.add_argument("--json", action="store_true")
    explain.set_defaults(func=_cmd_explain)

    catalog = subparsers.add_parser(
        "catalog",
        help="export supported tactics from the existing vLLM IR registry",
    )
    catalog.add_argument("--output")
    catalog.add_argument(
        "--include-tuning",
        action="store_true",
        help="include first-party runtime-decision tuning candidates",
    )
    catalog.set_defaults(func=_cmd_catalog)

    profiles = subparsers.add_parser(
        "profiles",
        help="list declarative per-model RuntimePlan defaults and tunability",
    )
    profiles.add_argument(
        "--profile",
        help="validate one declarative profile file instead of listing built-ins",
    )
    profiles.add_argument("--output")
    profiles.set_defaults(func=_cmd_profiles)

    autotune = subparsers.add_parser(
        "autotune",
        help="collect incremental MUSA tactic evidence outside serving",
    )
    autotune_commands = autotune.add_subparsers(dest="autotune_command", required=True)
    collect = autotune_commands.add_parser(
        "collect",
        help=(
            "measure cache misses after live build-manifest verification and "
            "emit mutable timing-v2 evidence"
        ),
    )
    collect.add_argument("--target", required=True)
    collect.add_argument(
        "--domain",
        default=FUSED_ADD_RMS_NORM_DOMAIN_ID,
        help=(
            "operator capability domain to collect; defaults to the legacy "
            "fused-add RMSNorm vertical slice"
        ),
    )
    collect.add_argument("--output", required=True)
    collect.add_argument("--summary-output")
    collect.add_argument("--resume")
    collect.add_argument("--rows", type=int, action="append", default=[])
    collect.add_argument("--warmups", type=int, default=5)
    collect.add_argument("--iterations", type=int, default=30)
    collect.add_argument("--cold-cache-bytes", type=int, default=256 * 1024 * 1024)
    collect.add_argument(
        "--eager",
        action="store_true",
        help="diagnostic only: measure without graph capture",
    )
    collect.set_defaults(func=_cmd_autotune_collect)
    domains = autotune_commands.add_parser(
        "domains",
        help="list model-independent operator tuning domains and contracts",
    )
    domains.set_defaults(func=_cmd_autotune_domains)
    import_domain = autotune_commands.add_parser(
        "import",
        help="normalize qualified external evidence for one tuning domain",
    )
    import_domain.add_argument("--domain", required=True)
    import_domain.add_argument("--target", required=True)
    import_domain.add_argument("--evidence", action="append", required=True)
    import_domain.add_argument("--image-digest", required=True)
    import_domain.add_argument("--output", required=True)
    import_domain.set_defaults(func=_cmd_autotune_import)
    import_fused_moe = autotune_commands.add_parser(
        "import-fused-moe",
        help=(
            "compatibility alias for 'autotune import --domain " "fused_moe.fp8_block'"
        ),
    )
    import_fused_moe.add_argument("--target", required=True)
    import_fused_moe.add_argument("--evidence", action="append", required=True)
    import_fused_moe.add_argument("--image-digest", required=True)
    import_fused_moe.add_argument("--output", required=True)
    import_fused_moe.set_defaults(func=_cmd_autotune_import_fused_moe)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (
        ArtifactFileError,
        DeclarativeProfileError,
        EnginePlanError,
        PlanningArtifactError,
        OSError,
    ) as exc:
        print(f"vllm_musa.engine_plan: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
