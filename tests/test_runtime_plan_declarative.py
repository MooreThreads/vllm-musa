from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import vllm_musa.runtime_plan.declarative as declarative
from vllm_musa.engine_plan.cli import main as engine_plan_main
from vllm_musa.runtime_plan.declarative import (
    DeclarativeProfileError,
    builtin_declarative_profiles,
    declarative_profile_catalog,
    declarative_profile_identity,
    load_declarative_profile,
    parse_declarative_profile,
)
from vllm_musa.runtime_plan.types import (
    ExecutionSignature,
    ModelSignature,
    RuntimeDecision,
    RuntimePlan,
)


def _minimal_profile_document() -> dict[str, object]:
    return {
        "schema_version": "musa.runtime_profile.v1",
        "id": "test-profile",
        "priority": 1,
        "reason": "test declarative profile",
        "provider_when": True,
        "conditions": {},
        "classifications": [
            {
                "when": True,
                "family": "qwen3",
                "role": "text_generation",
            }
        ],
        "profiles": [{"when": True, "profile": "qwen3.text_generation"}],
        "decisions": [
            {
                "decision": "qwen.v2_sampling",
                "supported_when": True,
                "value_when": True,
                "value": True,
                "tunability": "profile",
            }
        ],
    }


def _qwen3_rope_model() -> ModelSignature:
    return replace(
        RuntimePlan.unknown().model,
        architectures=("Qwen3ForCausalLM",),
        outer_architectures=("Qwen3ForCausalLM",),
        model_type="qwen3",
        dtype="bfloat16",
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        has_routed_experts=False,
    )


def _qwen3_rope_execution() -> ExecutionSignature:
    return replace(
        RuntimePlan.unknown().execution,
        has_parallel_config=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        has_quant_config=False,
        has_speculative_config=False,
        cache_dtype="bfloat16",
        cache_block_size=64,
    )


def test_builtin_profile_catalog_is_versioned_and_complete() -> None:
    profiles = builtin_declarative_profiles()
    assert [profile.identifier for profile in profiles] == ["deepseek_v4", "qwen"]
    assert [len(profile.decisions) for profile in profiles] == [11, 18]
    assert all(profile.fingerprint.startswith("sha256:") for profile in profiles)
    assert declarative_profile_catalog() == tuple(
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
        for profile in profiles
    )
    assert declarative_profile_identity("qwen3.text_generation") == (
        profiles[1].identifier,
        profiles[1].fingerprint,
    )
    assert declarative_profile_identity("qwen3.typo") is None


def test_profile_config_controls_default_without_changing_resolver_code() -> None:
    path = Path(__file__).parents[1] / "vllm_musa/runtime_plan/profiles/qwen.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    profile = parse_declarative_profile(document)
    original = profile.resolve(_qwen3_rope_model(), _qwen3_rope_execution())
    assert original is not None
    assert original.enabled(RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT)
    assert (
        original.decision_source(RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT)
        == "profile_default"
    )

    changed = deepcopy(document)
    decision = next(
        value
        for value in changed["decisions"]
        if value["decision"] == "qwen3.qk_rope_kv_presplit"
    )
    decision["value"] = False
    overridden = parse_declarative_profile(changed).resolve(
        _qwen3_rope_model(), _qwen3_rope_execution()
    )

    assert overridden is not None
    assert overridden.supports(RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT)
    assert not overridden.enabled(RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT)
    assert (
        overridden.decision_source(RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT)
        == "profile_default"
    )
    assert original.profile_config_fingerprint != overridden.profile_config_fingerprint


def test_generic_provider_uses_priority_and_rejects_ambiguity(monkeypatch) -> None:
    first_document = _minimal_profile_document()
    first_document["id"] = "first"
    first = parse_declarative_profile(first_document)
    second_document = deepcopy(first_document)
    second_document["id"] = "second"
    second = parse_declarative_profile(second_document)
    monkeypatch.setattr(
        declarative,
        "builtin_declarative_profiles",
        lambda: (first, second),
    )

    with pytest.raises(DeclarativeProfileError, match="ambiguous"):
        declarative.resolve_declarative_runtime_plan(
            RuntimePlan.unknown().model,
            RuntimePlan.unknown().execution,
        )

    higher_document = deepcopy(second_document)
    higher_document["priority"] = 2
    higher = parse_declarative_profile(higher_document)
    monkeypatch.setattr(
        declarative,
        "builtin_declarative_profiles",
        lambda: (higher, first),
    )
    resolved = declarative.resolve_declarative_runtime_plan(
        RuntimePlan.unknown().model,
        RuntimePlan.unknown().execution,
    )
    assert resolved is not None
    assert resolved.profile_config_id == "second"


def test_runtime_plan_fingerprint_binds_profile_config() -> None:
    kwargs = dict(
        model=RuntimePlan.unknown().model,
        execution=RuntimePlan.unknown().execution,
        profile="unknown",
        supported_decisions=frozenset(),
        decision_values=(),
    )
    first = RuntimePlan(
        **kwargs,
        profile_config_id="test",
        profile_config_fingerprint="sha256:" + "1" * 64,
    )
    second = RuntimePlan(
        **kwargs,
        profile_config_id="test",
        profile_config_fingerprint="sha256:" + "2" * 64,
    )
    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    ("profile_config_id", "profile_config_fingerprint"),
    [("test", None), (None, "sha256:" + "1" * 64)],
)
def test_runtime_plan_requires_complete_profile_provenance(
    profile_config_id: str | None,
    profile_config_fingerprint: str | None,
) -> None:
    with pytest.raises(ValueError, match="must be set together"):
        replace(
            RuntimePlan.unknown(),
            profile_config_id=profile_config_id,
            profile_config_fingerprint=profile_config_fingerprint,
        )


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    [
        (lambda doc: doc.update({"extra": True}), "extra"),
        (
            lambda doc: doc.update({"schema_version": "future"}),
            "unsupported profile schema",
        ),
        (
            lambda doc: doc["provider_when"].update(  # type: ignore[union-attr]
                {"path": "model.not_a_field", "op": "eq", "value": 1}
            ),
            "unsupported path",
        ),
    ],
)
def test_profile_schema_fails_closed(mutation, pattern: str) -> None:
    document = _minimal_profile_document()
    if not isinstance(document["provider_when"], dict):
        document["provider_when"] = {}
    mutation(document)
    with pytest.raises(DeclarativeProfileError, match=pattern):
        parse_declarative_profile(document)


def test_unknown_condition_operator_is_rejected() -> None:
    document = _minimal_profile_document()
    document["provider_when"] = {
        "path": "model.dtype",
        "op": "exec",
        "value": "bfloat16",
    }
    with pytest.raises(DeclarativeProfileError, match="unsupported operation"):
        parse_declarative_profile(document)


@pytest.mark.parametrize(
    "provider_when",
    [
        {"path": "model.hidden_size", "op": "gt", "value": "x"},
        {"path": "model.hidden_size", "op": "gt", "value": {}},
        {"path": "model.hidden_size", "op": "eq", "value": "4096"},
        {"path": "model.dtype", "op": "gt", "value": "bfloat16"},
        {"path": "model.hidden_size", "op": "contains_any", "value": [1]},
        {
            "paths": ["model.hidden_size", "model.dtype"],
            "op": "tuple_in",
            "value": [[4096, 8]],
        },
    ],
)
def test_condition_operand_types_fail_at_parse_time(provider_when) -> None:
    document = _minimal_profile_document()
    document["provider_when"] = provider_when
    with pytest.raises(DeclarativeProfileError):
        parse_declarative_profile(document)


def test_condition_reference_cycle_is_rejected() -> None:
    document = _minimal_profile_document()
    document["conditions"] = {"a": {"ref": "b"}, "b": {"ref": "a"}}
    with pytest.raises(DeclarativeProfileError, match="condition reference cycle"):
        parse_declarative_profile(document)


def test_decision_dependency_cycle_is_rejected() -> None:
    document = _minimal_profile_document()
    document["decisions"] = [
        {
            "decision": "qwen.v2_sampling",
            "supported_when": True,
            "value_when": True,
            "value": True,
            "tunability": "profile",
            "requires": ["qwen.legacy_sampling"],
        },
        {
            "decision": "qwen.legacy_sampling",
            "supported_when": True,
            "value_when": True,
            "value": True,
            "tunability": "profile",
            "requires": ["qwen.v2_sampling"],
        },
    ]
    with pytest.raises(DeclarativeProfileError, match="dependency cycle"):
        parse_declarative_profile(document)


def test_decision_dependency_must_be_boolean() -> None:
    document = _minimal_profile_document()
    document["decisions"][0]["requires"] = ["deepseek_v4.flashmla_sparse_page_size"]
    document["decisions"].append(
        {
            "decision": "deepseek_v4.flashmla_sparse_page_size",
            "supported_when": True,
            "value_when": True,
            "value": 256,
            "tunability": "fixed",
        }
    )
    with pytest.raises(DeclarativeProfileError, match="only boolean decisions"):
        parse_declarative_profile(document)


def test_external_only_decision_cannot_be_a_profile_default() -> None:
    document = _minimal_profile_document()
    document["decisions"][0]["decision"] = "musa.fused_moe.dispatch_policy"
    document["decisions"][0]["value"] = []
    with pytest.raises(DeclarativeProfileError, match="external-only"):
        parse_declarative_profile(document)


def test_profile_cannot_reclassify_host_catalog_tunability() -> None:
    document = _minimal_profile_document()
    document["decisions"][0]["tunability"] = "autotune"
    with pytest.raises(DeclarativeProfileError, match="does not match host catalog"):
        parse_declarative_profile(document)


def test_profile_cannot_attach_cross_family_decision() -> None:
    document = _minimal_profile_document()
    document["decisions"][0].update(
        {
            "decision": "deepseek_v4.native_sparse_indexer",
            "tunability": "profile",
        }
    )
    with pytest.raises(DeclarativeProfileError, match="classified profile family"):
        parse_declarative_profile(document)


def test_runtime_profile_name_must_match_classification_family() -> None:
    document = _minimal_profile_document()
    document["profiles"][0]["profile"] = "deepseek_v4.text_generation"
    with pytest.raises(DeclarativeProfileError, match="not declared"):
        parse_declarative_profile(document)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_non_finite_profile_values_are_rejected(value: float) -> None:
    document = _minimal_profile_document()
    document["conditions"] = {
        "bad": {"path": "model.swiglu_limit", "op": "eq", "value": value}
    }
    with pytest.raises(DeclarativeProfileError, match="non-finite"):
        parse_declarative_profile(document)


def test_profile_document_depth_is_bounded() -> None:
    document = _minimal_profile_document()
    condition: dict[str, object] = {
        "path": "model.dtype",
        "op": "eq",
        "value": "bfloat16",
    }
    for _ in range(40):
        condition = {"not": condition}
    document["provider_when"] = condition
    with pytest.raises(DeclarativeProfileError, match="exceeds depth"):
        parse_declarative_profile(document)


@pytest.mark.parametrize(
    "payload",
    [b"\xff", b"[" * 1500 + b"0" + b"]" * 1500],
)
def test_profile_file_decode_and_parser_failures_are_wrapped(
    tmp_path: Path,
    payload: bytes,
) -> None:
    profile = tmp_path / "bad-profile.json"
    profile.write_bytes(payload)

    with pytest.raises(DeclarativeProfileError, match="failed to load"):
        load_declarative_profile(profile)


def test_parsed_profile_is_immutable_from_input_and_public_view() -> None:
    document = _minimal_profile_document()
    document["conditions"] = {
        "dtype": {"path": "model.dtype", "op": "eq", "value": "bfloat16"}
    }
    document["provider_when"] = {"ref": "dtype"}
    profile = parse_declarative_profile(document)
    fingerprint = profile.fingerprint

    document["conditions"]["dtype"]["value"] = "float16"
    assert profile.fingerprint == fingerprint
    assert profile.matches(
        replace(RuntimePlan.unknown().model, dtype="bfloat16"),
        RuntimePlan.unknown().execution,
    )
    with pytest.raises(TypeError):
        profile.conditions["dtype"]["value"] = "float16"


def test_profile_parser_has_no_dynamic_execution_entrypoint() -> None:
    source = (
        Path(__file__).parents[1] / "vllm_musa/runtime_plan/declarative.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("eval(", "exec(", "importlib", "__import__"):
        assert forbidden not in source


def test_setup_packages_declarative_profile_documents() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"runtime_plan/profiles/*.json"' in pyproject


def test_engine_plan_cli_lists_declarative_profiles(capsys) -> None:
    assert engine_plan_main(["profiles"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema_version"] == "musa.runtime_profile_catalog.v1"
    assert [profile["id"] for profile in document["profiles"]] == [
        "deepseek_v4",
        "qwen",
    ]
    qwen = document["profiles"][1]
    assert qwen["runtime_profile_families"] == [
        "qwen2",
        "qwen3",
        "qwen3.5_3.6",
    ]
    assert len(qwen["decisions"]) == 18
    assert {decision["tunability"] for decision in qwen["decisions"]} == {"profile"}
    assert len(document["decision_catalog"]) == 33
    assert {
        tunability: sum(
            decision["tunability"] == tunability
            for decision in document["decision_catalog"]
        )
        for tunability in ("fixed", "profile", "autotune")
    } == {"fixed": 3, "profile": 27, "autotune": 3}


def test_engine_plan_cli_validates_one_profile(capsys) -> None:
    profile = Path(__file__).parents[1] / "vllm_musa/runtime_plan/profiles/qwen.json"
    assert engine_plan_main(["profiles", "--profile", str(profile)]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema_version"] == "musa.runtime_profile_validation.v1"
    assert document["status"] == "valid"
    assert document["id"] == "qwen"
    assert document["fingerprint"].startswith("sha256:")


@pytest.mark.parametrize("alias_kind", ["same_path", "hardlink"])
def test_profile_validation_refuses_output_alias(
    tmp_path: Path,
    capsys,
    alias_kind: str,
) -> None:
    packaged = Path(__file__).parents[1] / "vllm_musa/runtime_plan/profiles/qwen.json"
    source = tmp_path / "profile.json"
    original = packaged.read_text(encoding="utf-8")
    source.write_text(original, encoding="utf-8")
    output = source
    if alias_kind == "hardlink":
        output = tmp_path / "validation.json"
        output.hardlink_to(source)

    assert (
        engine_plan_main(
            ["profiles", "--profile", str(source), "--output", str(output)]
        )
        == 2
    )
    assert "must not overwrite its input" in capsys.readouterr().err
    assert source.read_text(encoding="utf-8") == original
