from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .catalog import (
    RuntimeDecisionKind,
    runtime_decision_spec,
    validate_runtime_decision_value,
)
from .types import (
    ExecutionSignature,
    ModelFamily,
    ModelRole,
    ModelSignature,
    RuntimeDecision,
    RuntimePlan,
    canonical_fingerprint,
)

PROFILE_SCHEMA_VERSION = "musa.runtime_profile.v1"
PROFILE_DIRECTORY = Path(__file__).with_name("profiles")
_TUNABILITY_VALUES = frozenset({"fixed", "profile", "autotune"})
_MODEL_PATHS = frozenset(field.name for field in fields(ModelSignature)) | {
    "effective_architectures"
}
_EXECUTION_PATHS = frozenset(field.name for field in fields(ExecutionSignature))
_PATH_SCHEMAS: dict[str, tuple[str, bool]] = {
    "model.family": ("string", False),
    "model.role": ("string", False),
    "model.architectures": ("string_collection", False),
    "model.effective_architectures": ("string_collection", False),
    "model.outer_architectures": ("string_collection", False),
    "model.text_architectures": ("string_collection", False),
    "model.model_type": ("string", True),
    "model.dtype": ("string", True),
    "model.quantization": ("string", True),
    "model.hidden_size": ("integer", True),
    "model.intermediate_size": ("integer", True),
    "model.num_hidden_layers": ("integer", True),
    "model.num_attention_heads": ("integer", True),
    "model.num_key_value_heads": ("integer", True),
    "model.head_dim": ("integer", True),
    "model.vocab_size": ("integer", True),
    "model.num_experts": ("integer", True),
    "model.num_experts_per_tok": ("integer", True),
    "model.num_shared_experts": ("integer", True),
    "model.moe_intermediate_size": ("integer", True),
    "model.expert_dtype": ("string", True),
    "model.hidden_act": ("string", True),
    "model.swiglu_limit": ("number", True),
    "model.gdn_conv_width": ("integer", True),
    "model.gdn_conv_dim": ("integer", True),
    "model.has_routed_experts": ("boolean", True),
    "model.enforce_eager": ("boolean", False),
    "model.index_topk": ("integer", True),
    "model.outer_model_type": ("string", True),
    "model.uses_mla": ("boolean", True),
    "model.quant_block_shape": ("integer_collection", True),
    "model.is_hybrid": ("boolean", True),
    "execution.tensor_parallel_size": ("integer", False),
    "execution.pipeline_parallel_size": ("integer", False),
    "execution.data_parallel_size": ("integer", False),
    "execution.decode_context_parallel_size": ("integer", False),
    "execution.has_speculative_config": ("boolean", False),
    "execution.has_quant_config": ("boolean", False),
    "execution.is_pooling_model": ("boolean", False),
    "execution.has_parallel_config": ("boolean", False),
    "execution.cache_dtype": ("string", True),
    "execution.cache_block_size": ("integer", True),
    "execution.max_num_seqs": ("integer", True),
    "execution.attention_backend": ("string", True),
    "execution.compilation_mode": ("string", True),
    "execution.cudagraph_mode": ("string", True),
    "execution.batch_invariant_enabled": ("boolean", False),
    "execution.speculative_method": ("string", True),
    "execution.async_scheduling": ("boolean", False),
}
if set(_PATH_SCHEMAS) != {
    *(f"model.{name}" for name in _MODEL_PATHS),
    *(f"execution.{name}" for name in _EXECUTION_PATHS),
}:  # pragma: no cover - import-time guard for signature evolution
    raise RuntimeError("declarative condition path schemas are incomplete")


class DeclarativeProfileError(ValueError):
    """Raised when a declarative RuntimePlan profile is invalid."""


Condition = Mapping[str, Any] | bool


def _validate_json_tree(
    value: object,
    *,
    field: str = "profile document",
    depth: int = 0,
    budget: list[int] | None = None,
) -> None:
    if budget is None:
        budget = [4096]
    budget[0] -= 1
    if budget[0] < 0:
        raise DeclarativeProfileError("profile document exceeds 4096 JSON nodes")
    if depth > 32:
        raise DeclarativeProfileError("profile document exceeds depth 32")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DeclarativeProfileError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(
                item,
                field=f"{field}[{index}]",
                depth=depth + 1,
                budget=budget,
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DeclarativeProfileError(f"{field} contains a non-string key")
            _validate_json_tree(
                item,
                field=f"{field}.{key}",
                depth=depth + 1,
                budget=budget,
            )
        return
    raise DeclarativeProfileError(
        f"{field} contains unsupported value type {type(value).__name__}"
    )


def _expect_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DeclarativeProfileError(f"{field} must be an object")
    return dict(value)


def _expect_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise DeclarativeProfileError(f"{field} must be an array")
    return value


def _expect_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DeclarativeProfileError(f"{field} must be a non-empty string")
    return value


def _expect_exact_keys(
    document: Mapping[str, Any],
    *,
    field: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(document))
    extra = sorted(set(document) - required - optional)
    if missing or extra:
        raise DeclarativeProfileError(
            f"{field} has invalid fields: missing={missing}, extra={extra}"
        )


def _normalize_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, frozenset):
        return tuple(sorted(_normalize_value(item) for item in value))
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _context_value(
    path: str,
    model: ModelSignature,
    execution: ExecutionSignature,
) -> object:
    root, separator, name = path.partition(".")
    if not separator:
        raise DeclarativeProfileError(
            f"condition path must use model.* or execution.*: {path!r}"
        )
    if root == "model":
        if name not in _MODEL_PATHS:
            raise DeclarativeProfileError(f"unknown model condition path {path!r}")
        if name == "effective_architectures":
            return tuple(model.outer_architectures or model.architectures)
        return _normalize_value(getattr(model, name))
    if root == "execution":
        if name not in _EXECUTION_PATHS:
            raise DeclarativeProfileError(f"unknown execution condition path {path!r}")
        return _normalize_value(getattr(execution, name))
    raise DeclarativeProfileError(
        f"condition path must use model.* or execution.*: {path!r}"
    )


def _validate_path(path: object, field: str) -> str:
    path = _expect_string(path, field)
    root, separator, name = path.partition(".")
    if not separator or (
        root == "model"
        and name not in _MODEL_PATHS
        or root == "execution"
        and name not in _EXECUTION_PATHS
        or root not in {"model", "execution"}
    ):
        raise DeclarativeProfileError(f"{field} names unsupported path {path!r}")
    return path


def _validate_typed_operand(
    value: object,
    *,
    kind: str,
    nullable: bool,
    field: str,
) -> None:
    if value is None:
        if nullable:
            return
        raise DeclarativeProfileError(f"{field} must not be null")
    if kind == "string":
        valid = isinstance(value, str)
    elif kind == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif kind == "boolean":
        valid = isinstance(value, bool)
    elif kind in {"string_collection", "integer_collection"}:
        if not isinstance(value, list):
            valid = False
        else:
            element_kind = kind.removesuffix("_collection")
            for index, item in enumerate(value):
                _validate_typed_operand(
                    item,
                    kind=element_kind,
                    nullable=False,
                    field=f"{field}[{index}]",
                )
            valid = True
    else:  # pragma: no cover - _PATH_SCHEMAS is closed above
        raise AssertionError(f"unknown declarative operand kind {kind!r}")
    if not valid:
        raise DeclarativeProfileError(f"{field} must be a {kind.replace('_', ' ')}")


def _validate_condition_operand(
    path: str,
    operation: str,
    value: object,
    *,
    field: str,
) -> None:
    kind, nullable = _PATH_SCHEMAS[path]
    if operation in {"gt", "ge", "lt", "le"}:
        if kind not in {"integer", "number"}:
            raise DeclarativeProfileError(
                f"{field} operation {operation!r} requires a numeric path"
            )
        _validate_typed_operand(
            value,
            kind=kind,
            nullable=False,
            field=f"{field}.value",
        )
        return
    if operation in {"contains_any", "contains_all"}:
        if kind not in {"string_collection", "integer_collection"}:
            raise DeclarativeProfileError(
                f"{field} operation {operation!r} requires a collection path"
            )
        values = _expect_list(value, f"{field}.value")
        element_kind = kind.removesuffix("_collection")
        for index, item in enumerate(values):
            _validate_typed_operand(
                item,
                kind=element_kind,
                nullable=False,
                field=f"{field}.value[{index}]",
            )
        return
    if operation in {"in", "not_in"}:
        values = _expect_list(value, f"{field}.value")
        for index, item in enumerate(values):
            _validate_typed_operand(
                item,
                kind=kind,
                nullable=nullable,
                field=f"{field}.value[{index}]",
            )
        return
    _validate_typed_operand(
        value,
        kind=kind,
        nullable=nullable,
        field=f"{field}.value",
    )


def _condition_references(condition: Condition, *, field: str) -> set[str]:
    if isinstance(condition, bool):
        return set()
    document = _expect_mapping(condition, field)
    if "ref" in document:
        _expect_exact_keys(document, field=field, required={"ref"})
        return {_expect_string(document["ref"], f"{field}.ref")}
    if "all" in document or "any" in document:
        key = "all" if "all" in document else "any"
        _expect_exact_keys(document, field=field, required={key})
        children = _expect_list(document[key], f"{field}.{key}")
        if not children:
            raise DeclarativeProfileError(f"{field}.{key} must not be empty")
        references: set[str] = set()
        for index, child in enumerate(children):
            references.update(
                _condition_references(child, field=f"{field}.{key}[{index}]")
            )
        return references
    if "not" in document:
        _expect_exact_keys(document, field=field, required={"not"})
        return _condition_references(document["not"], field=f"{field}.not")
    if "path" in document:
        _expect_exact_keys(
            document,
            field=field,
            required={"path", "op", "value"},
        )
        path = _validate_path(document["path"], f"{field}.path")
        operation = _expect_string(document["op"], f"{field}.op")
        if operation not in {
            "eq",
            "ne",
            "in",
            "not_in",
            "gt",
            "ge",
            "lt",
            "le",
            "contains_any",
            "contains_all",
        }:
            raise DeclarativeProfileError(
                f"{field}.op has unsupported operation {operation!r}"
            )
        if operation in {"in", "not_in", "contains_any", "contains_all"}:
            _expect_list(document["value"], f"{field}.value")
        _validate_condition_operand(
            path,
            operation,
            document["value"],
            field=field,
        )
        return set()
    if "paths" in document:
        _expect_exact_keys(
            document,
            field=field,
            required={"paths", "op", "value"},
        )
        paths = _expect_list(document["paths"], f"{field}.paths")
        if not paths:
            raise DeclarativeProfileError(f"{field}.paths must not be empty")
        validated_paths = [
            _validate_path(path, f"{field}.paths[{index}]")
            for index, path in enumerate(paths)
        ]
        operation = _expect_string(document["op"], f"{field}.op")
        if operation != "tuple_in":
            raise DeclarativeProfileError(
                f"{field}.op has unsupported multi-path operation {operation!r}"
            )
        values = _expect_list(document["value"], f"{field}.value")
        for index, candidate in enumerate(values):
            candidate = _expect_list(candidate, f"{field}.value[{index}]")
            if len(candidate) != len(paths):
                raise DeclarativeProfileError(
                    f"{field}.value[{index}] must have {len(paths)} items"
                )
            for path, item in zip(validated_paths, candidate):
                _validate_condition_operand(
                    path,
                    "eq",
                    item,
                    field=f"{field}.value[{index}]",
                )
        return set()
    raise DeclarativeProfileError(
        f"{field} must be a boolean, ref, all/any/not, or typed comparison"
    )


def _validate_condition_definitions(definitions: Mapping[str, Condition]) -> None:
    graph: dict[str, set[str]] = {}
    for name, condition in definitions.items():
        _expect_string(name, "conditions key")
        graph[name] = _condition_references(
            condition,
            field=f"conditions.{name}",
        )
    for name, references in graph.items():
        missing = sorted(references - set(graph))
        if missing:
            raise DeclarativeProfileError(
                f"conditions.{name} references unknown conditions: {missing}"
            )

    active: list[str] = []
    complete: set[str] = set()

    def visit(name: str) -> None:
        if name in complete:
            return
        if name in active:
            cycle = active[active.index(name) :] + [name]
            raise DeclarativeProfileError(
                f"condition reference cycle: {' -> '.join(cycle)}"
            )
        if len(active) >= 32:
            raise DeclarativeProfileError("condition reference chain exceeds depth 32")
        active.append(name)
        for reference in sorted(graph[name]):
            visit(reference)
        active.pop()
        complete.add(name)

    for name in sorted(graph):
        visit(name)


def _evaluate_condition(
    condition: Condition,
    *,
    model: ModelSignature,
    execution: ExecutionSignature,
    definitions: Mapping[str, Condition],
) -> bool:
    if isinstance(condition, bool):
        return condition
    document = dict(condition)
    if "ref" in document:
        return _evaluate_condition(
            definitions[str(document["ref"])],
            model=model,
            execution=execution,
            definitions=definitions,
        )
    if "all" in document:
        return all(
            _evaluate_condition(
                child,
                model=model,
                execution=execution,
                definitions=definitions,
            )
            for child in document["all"]
        )
    if "any" in document:
        return any(
            _evaluate_condition(
                child,
                model=model,
                execution=execution,
                definitions=definitions,
            )
            for child in document["any"]
        )
    if "not" in document:
        return not _evaluate_condition(
            document["not"],
            model=model,
            execution=execution,
            definitions=definitions,
        )
    operation = str(document["op"])
    expected = document["value"]
    if "paths" in document:
        actual = tuple(
            _context_value(str(path), model, execution) for path in document["paths"]
        )
        return operation == "tuple_in" and actual in expected
    actual = _context_value(str(document["path"]), model, execution)
    if operation == "eq":
        return actual == expected
    if operation == "ne":
        return actual != expected
    if operation == "in":
        return actual in expected
    if operation == "not_in":
        return actual not in expected
    try:
        if operation == "gt":
            return actual is not None and actual > expected
        if operation == "ge":
            return actual is not None and actual >= expected
        if operation == "lt":
            return actual is not None and actual < expected
        if operation == "le":
            return actual is not None and actual <= expected
    except (TypeError, ValueError):
        return False
    if operation == "contains_any":
        return isinstance(actual, tuple) and bool(set(actual) & set(expected))
    if operation == "contains_all":
        return isinstance(actual, tuple) and set(expected).issubset(actual)
    raise AssertionError(f"validated condition has unknown operation {operation!r}")


@dataclass(frozen=True, slots=True)
class ClassificationRule:
    when: Condition
    family: ModelFamily
    role: ModelRole


@dataclass(frozen=True, slots=True)
class ProfileRule:
    when: Condition
    profile: str


@dataclass(frozen=True, slots=True)
class DecisionRule:
    decision: RuntimeDecision
    supported_when: Condition
    value_when: Condition
    value: object
    requires: tuple[RuntimeDecision, ...]
    tunability: str


def _runtime_profile_family(profile: str) -> ModelFamily | None:
    families = sorted(
        (family for family in ModelFamily if family is not ModelFamily.UNKNOWN),
        key=lambda family: len(family.value),
        reverse=True,
    )
    return next(
        (
            family
            for family in families
            if profile == family.value or profile.startswith(f"{family.value}.")
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class DeclarativeRuntimeProfile:
    identifier: str
    priority: int
    reason: str
    provider_when: Condition
    conditions: Mapping[str, Condition]
    classifications: tuple[ClassificationRule, ...]
    profiles: tuple[ProfileRule, ...]
    decisions: tuple[DecisionRule, ...]
    runtime_profiles: frozenset[str]
    runtime_profile_families: frozenset[ModelFamily]
    fingerprint: str

    def matches(
        self,
        model: ModelSignature,
        execution: ExecutionSignature,
    ) -> bool:
        return _evaluate_condition(
            self.provider_when,
            model=model,
            execution=execution,
            definitions=self.conditions,
        )

    def resolve(
        self,
        model: ModelSignature,
        execution: ExecutionSignature,
    ) -> RuntimePlan | None:
        if not self.matches(model, execution):
            return None
        classification = next(
            (
                rule
                for rule in self.classifications
                if _evaluate_condition(
                    rule.when,
                    model=model,
                    execution=execution,
                    definitions=self.conditions,
                )
            ),
            None,
        )
        if classification is None:
            raise DeclarativeProfileError(
                f"profile {self.identifier!r} matched without a classification"
            )
        model = replace(
            model,
            family=classification.family,
            role=classification.role,
        )
        profile = next(
            (
                rule.profile
                for rule in self.profiles
                if _evaluate_condition(
                    rule.when,
                    model=model,
                    execution=execution,
                    definitions=self.conditions,
                )
            ),
            None,
        )
        if profile is None:
            raise DeclarativeProfileError(
                f"profile {self.identifier!r} matched without a profile rule"
            )
        profile_family = _runtime_profile_family(profile)
        if profile_family is not model.family:
            raise DeclarativeProfileError(
                f"profile {profile!r} does not match classified family "
                f"{model.family.value!r}"
            )

        supported: set[RuntimeDecision] = set()
        values: dict[RuntimeDecision, object] = {}
        for rule in self.decisions:
            dependencies_satisfied = all(
                dependency in supported
                and bool(
                    values.get(
                        dependency,
                        runtime_decision_spec(dependency).fallback,
                    )
                )
                for dependency in rule.requires
            )
            if not dependencies_satisfied:
                continue
            if not _evaluate_condition(
                rule.supported_when,
                model=model,
                execution=execution,
                definitions=self.conditions,
            ):
                continue
            if not runtime_decision_spec(rule.decision).supports_profile(profile):
                raise DeclarativeProfileError(
                    f"decision {rule.decision.value!r} does not support "
                    f"profile {profile!r}"
                )
            supported.add(rule.decision)
            if _evaluate_condition(
                rule.value_when,
                model=model,
                execution=execution,
                definitions=self.conditions,
            ):
                values[rule.decision] = rule.value
        return RuntimePlan(
            model=model,
            execution=execution,
            profile=profile,
            supported_decisions=frozenset(supported),
            decision_values=tuple(values.items()),
            reason=self.reason,
            profile_config_id=self.identifier,
            profile_config_fingerprint=self.fingerprint,
        )


def _parse_classification_rule(
    value: object,
    *,
    index: int,
    definitions: Mapping[str, Condition],
) -> ClassificationRule:
    field = f"classifications[{index}]"
    document = _expect_mapping(value, field)
    _expect_exact_keys(
        document,
        field=field,
        required={"when", "family", "role"},
    )
    references = _condition_references(document["when"], field=f"{field}.when")
    missing = sorted(references - set(definitions))
    if missing:
        raise DeclarativeProfileError(f"{field}.when has unknown refs: {missing}")
    try:
        family = ModelFamily(_expect_string(document["family"], f"{field}.family"))
        role = ModelRole(_expect_string(document["role"], f"{field}.role"))
    except ValueError as exc:
        raise DeclarativeProfileError(f"{field} has unknown family or role") from exc
    if family is ModelFamily.UNKNOWN or role is ModelRole.UNKNOWN:
        raise DeclarativeProfileError(f"{field} cannot classify an unknown family/role")
    return ClassificationRule(_freeze_json(document["when"]), family, role)


def _parse_profile_rule(
    value: object,
    *,
    index: int,
    definitions: Mapping[str, Condition],
) -> ProfileRule:
    field = f"profiles[{index}]"
    document = _expect_mapping(value, field)
    _expect_exact_keys(document, field=field, required={"when", "profile"})
    references = _condition_references(document["when"], field=f"{field}.when")
    missing = sorted(references - set(definitions))
    if missing:
        raise DeclarativeProfileError(f"{field}.when has unknown refs: {missing}")
    return ProfileRule(
        _freeze_json(document["when"]),
        _expect_string(document["profile"], f"{field}.profile"),
    )


def _parse_decision_rules(
    values: object,
    *,
    definitions: Mapping[str, Condition],
) -> tuple[DecisionRule, ...]:
    documents = _expect_list(values, "decisions")
    parsed: dict[RuntimeDecision, DecisionRule] = {}
    for index, value in enumerate(documents):
        field = f"decisions[{index}]"
        document = _expect_mapping(value, field)
        _expect_exact_keys(
            document,
            field=field,
            required={
                "decision",
                "supported_when",
                "value_when",
                "value",
                "tunability",
            },
            optional={"requires"},
        )
        raw_decision = _expect_string(document["decision"], f"{field}.decision")
        try:
            decision = RuntimeDecision(raw_decision)
        except ValueError as exc:
            raise DeclarativeProfileError(
                f"{field}.decision names unknown decision {raw_decision!r}"
            ) from exc
        if decision in parsed:
            raise DeclarativeProfileError(
                f"decisions contains duplicate decision {decision.value!r}"
            )
        if runtime_decision_spec(decision).external_only:
            raise DeclarativeProfileError(
                f"profile defaults cannot set external-only decision {decision.value!r}"
            )
        for condition_field in ("supported_when", "value_when"):
            references = _condition_references(
                document[condition_field],
                field=f"{field}.{condition_field}",
            )
            missing = sorted(references - set(definitions))
            if missing:
                raise DeclarativeProfileError(
                    f"{field}.{condition_field} has unknown refs: {missing}"
                )
        try:
            validated_value = validate_runtime_decision_value(
                decision, document["value"]
            )
        except (TypeError, ValueError) as exc:
            raise DeclarativeProfileError(
                f"{field}.value is invalid for {decision.value!r}: {exc}"
            ) from exc
        tunability = _expect_string(document["tunability"], f"{field}.tunability")
        if tunability not in _TUNABILITY_VALUES:
            raise DeclarativeProfileError(
                f"{field}.tunability must be one of {sorted(_TUNABILITY_VALUES)}"
            )
        catalog_tunability = runtime_decision_spec(decision).tunability.value
        if tunability != catalog_tunability:
            raise DeclarativeProfileError(
                f"{field}.tunability {tunability!r} does not match host catalog "
                f"{catalog_tunability!r}"
            )
        requires: list[RuntimeDecision] = []
        for dependency_index, dependency in enumerate(document.get("requires", [])):
            raw_dependency = _expect_string(
                dependency,
                f"{field}.requires[{dependency_index}]",
            )
            try:
                dependency_key = RuntimeDecision(raw_dependency)
            except ValueError as exc:
                raise DeclarativeProfileError(
                    f"{field}.requires names unknown decision {raw_dependency!r}"
                ) from exc
            if dependency_key is decision:
                raise DeclarativeProfileError(
                    f"{field}.requires cannot reference itself"
                )
            if (
                runtime_decision_spec(dependency_key).kind
                is not RuntimeDecisionKind.BOOLEAN
            ):
                raise DeclarativeProfileError(
                    f"{field}.requires may name only boolean decisions"
                )
            requires.append(dependency_key)
        parsed[decision] = DecisionRule(
            decision=decision,
            supported_when=_freeze_json(document["supported_when"]),
            value_when=_freeze_json(document["value_when"]),
            value=validated_value,
            requires=tuple(requires),
            tunability=tunability,
        )

    missing_dependencies = sorted(
        {
            dependency.value
            for rule in parsed.values()
            for dependency in rule.requires
            if dependency not in parsed
        }
    )
    if missing_dependencies:
        raise DeclarativeProfileError(
            f"decision rules require undeclared decisions: {missing_dependencies}"
        )
    ordered: list[DecisionRule] = []
    pending = dict(parsed)
    while pending:
        ready = sorted(
            (
                rule
                for rule in pending.values()
                if all(dependency not in pending for dependency in rule.requires)
            ),
            key=lambda rule: rule.decision.value,
        )
        if not ready:
            cycle = sorted(decision.value for decision in pending)
            raise DeclarativeProfileError(f"decision dependency cycle among: {cycle}")
        for rule in ready:
            ordered.append(rule)
            del pending[rule.decision]
    return tuple(ordered)


def parse_declarative_profile(document: Mapping[str, Any]) -> DeclarativeRuntimeProfile:
    document = dict(document)
    _validate_json_tree(document)
    _expect_exact_keys(
        document,
        field="profile document",
        required={
            "schema_version",
            "id",
            "priority",
            "reason",
            "provider_when",
            "conditions",
            "classifications",
            "profiles",
            "decisions",
        },
    )
    if document["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise DeclarativeProfileError(
            f"unsupported profile schema {document['schema_version']!r}"
        )
    identifier = _expect_string(document["id"], "id")
    priority = document["priority"]
    if not isinstance(priority, int) or isinstance(priority, bool) or priority < 0:
        raise DeclarativeProfileError("priority must be a non-negative integer")
    reason = _expect_string(document["reason"], "reason")
    definitions = _expect_mapping(document["conditions"], "conditions")
    _validate_condition_definitions(definitions)
    provider_references = _condition_references(
        document["provider_when"], field="provider_when"
    )
    missing = sorted(provider_references - set(definitions))
    if missing:
        raise DeclarativeProfileError(f"provider_when has unknown refs: {missing}")
    classifications = tuple(
        _parse_classification_rule(
            value,
            index=index,
            definitions=definitions,
        )
        for index, value in enumerate(
            _expect_list(document["classifications"], "classifications")
        )
    )
    profiles = tuple(
        _parse_profile_rule(value, index=index, definitions=definitions)
        for index, value in enumerate(_expect_list(document["profiles"], "profiles"))
    )
    if not classifications or not profiles:
        raise DeclarativeProfileError(
            "profile document requires classifications and profiles"
        )
    classification_families = frozenset(rule.family for rule in classifications)
    runtime_profile_families: set[ModelFamily] = set()
    for index, rule in enumerate(profiles):
        family = _runtime_profile_family(rule.profile)
        if family is None:
            raise DeclarativeProfileError(
                f"profiles[{index}].profile has unknown family prefix"
            )
        if family not in classification_families:
            raise DeclarativeProfileError(
                f"profiles[{index}].profile family {family.value!r} is not "
                "declared by classifications"
            )
        runtime_profile_families.add(family)
    decisions = _parse_decision_rules(
        document["decisions"],
        definitions=definitions,
    )
    for rule in decisions:
        spec = runtime_decision_spec(rule.decision)
        if not any(
            spec.supports_profile(family.value) for family in runtime_profile_families
        ):
            raise DeclarativeProfileError(
                f"decision {rule.decision.value!r} does not support any "
                "classified profile family"
            )
    return DeclarativeRuntimeProfile(
        identifier=identifier,
        priority=priority,
        reason=reason,
        provider_when=_freeze_json(document["provider_when"]),
        conditions=MappingProxyType(
            {name: _freeze_json(condition) for name, condition in definitions.items()}
        ),
        classifications=classifications,
        profiles=profiles,
        decisions=decisions,
        runtime_profiles=frozenset(rule.profile for rule in profiles),
        runtime_profile_families=frozenset(runtime_profile_families),
        fingerprint=canonical_fingerprint(document),
    )


def load_declarative_profile(path: Path) -> DeclarativeRuntimeProfile:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                DeclarativeProfileError(
                    f"profile {path} contains non-finite JSON constant {value!r}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DeclarativeProfileError(
            f"failed to load declarative runtime profile {path}: {exc}"
        ) from exc
    if not isinstance(document, Mapping):
        raise DeclarativeProfileError(f"profile {path} must contain one object")
    return parse_declarative_profile(document)


@lru_cache(maxsize=1)
def builtin_declarative_profiles() -> tuple[DeclarativeRuntimeProfile, ...]:
    paths = sorted(PROFILE_DIRECTORY.glob("*.json"))
    if not paths:
        raise DeclarativeProfileError(
            f"no declarative runtime profiles found under {PROFILE_DIRECTORY}"
        )
    profiles = tuple(load_declarative_profile(path) for path in paths)
    identifiers = [profile.identifier for profile in profiles]
    if len(set(identifiers)) != len(identifiers):
        raise DeclarativeProfileError(
            f"duplicate declarative runtime profile IDs: {identifiers}"
        )
    return tuple(
        sorted(profiles, key=lambda profile: (-profile.priority, profile.identifier))
    )


def resolve_declarative_runtime_plan(
    model: ModelSignature,
    execution: ExecutionSignature,
    *,
    identifier: str | None = None,
) -> RuntimePlan | None:
    matches = [
        profile
        for profile in builtin_declarative_profiles()
        if (identifier is None or profile.identifier == identifier)
        and profile.matches(model, execution)
    ]
    if not matches:
        return None
    highest_priority = matches[0].priority
    highest = [profile for profile in matches if profile.priority == highest_priority]
    if len(highest) != 1:
        raise DeclarativeProfileError(
            "ambiguous declarative runtime profiles: "
            + ", ".join(profile.identifier for profile in highest)
        )
    return highest[0].resolve(model, execution)


def declarative_profile_identity(
    runtime_profile: str,
) -> tuple[str, str] | None:
    matches = [
        profile
        for profile in builtin_declarative_profiles()
        if runtime_profile in profile.runtime_profiles
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise DeclarativeProfileError(
            f"runtime profile {runtime_profile!r} has ambiguous declarative "
            "configuration owners: "
            + ", ".join(profile.identifier for profile in matches)
        )
    return matches[0].identifier, matches[0].fingerprint


def declarative_profile_catalog() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "id": profile.identifier,
            "priority": profile.priority,
            "fingerprint": profile.fingerprint,
            "runtime_profiles": sorted(profile.runtime_profiles),
            "runtime_profile_families": sorted(
                family.value for family in profile.runtime_profile_families
            ),
            "decisions": [
                {
                    "decision": rule.decision.value,
                    "tunability": rule.tunability,
                    "requires": [dependency.value for dependency in rule.requires],
                }
                for rule in profile.decisions
            ],
        }
        for profile in builtin_declarative_profiles()
    )
