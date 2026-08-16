# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
import multiprocessing
import os
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_musa.engine_plugins import (
    ENGINE_PLAN_ENV,
    ENGINE_PLAN_FINGERPRINT_ENV,
    ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV,
    EnginePluginActivationError,
    apply_engine_plugin_defaults,
    resolve_engine_plugin_runtime_decisions,
    validate_engine_plugin_runtime,
)
from vllm_musa.engine_plugins.registry import _reset_engine_plugin_registry_for_test

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm_musa.engine_plan import BenchmarkCase, TimingCacheBuilder  # noqa: E402
from vllm_musa.engine_plan import plugin as example_plugin  # noqa: E402
from vllm_musa.engine_plan.artifact_io import write_json_object_file  # noqa: E402
from vllm_musa.engine_plan.artifacts import required_power2_bound_rows  # noqa: E402
from vllm_musa.engine_plan.core import (  # noqa: E402
    EnginePlanError,
    load_plan,
    parse_plan_document,
    seal_plan_document,
)
from vllm_musa.engine_plan.planner import build_plan_document  # noqa: E402
from vllm_musa.engine_plan.runtime import RuntimeVariantDecision  # noqa: E402
from vllm_musa.runtime_plan.declarative import (  # noqa: E402
    declarative_profile_identity,
)


class WeakConfig:
    pass


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    _reset_engine_plugin_registry_for_test()
    monkeypatch.delenv(ENGINE_PLAN_ENV, raising=False)
    monkeypatch.delenv(ENGINE_PLAN_FINGERPRINT_ENV, raising=False)
    monkeypatch.delenv(
        ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV,
        raising=False,
    )
    example_plugin._PLUGIN._plans.clear()
    yield
    _reset_engine_plugin_registry_for_test()
    example_plugin._PLUGIN._plans.clear()


def _target() -> dict:
    return {
        "model": {
            "profile": "qwen3.text_generation",
            "architecture": "Qwen3ForCausalLM",
            "model_id": "Qwen/Qwen3-8B",
            "hidden_size": 4096,
            "dtype": "bfloat16",
            "quantization": "none",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        },
        "hardware": {
            "platform": "musa",
            "device_name": "MTT S5000",
            "device_uuid": "GPU-test-s5000-64mp",
            "device_capability": "3.1",
            "multiprocessor_count": 64,
            "device_count": 1,
        },
        "software": {
            "versions": {
                "driver": "5.2.0-server",
                "flash-attn-3": "0.2.4+musa",
                "flash-mla": "0.2.4+musa",
                "mate": "0.2.4",
                "mthreads-ml-py": "2.4.1",
                "musa": "5.2.0",
                "torch": "2.11.0.post1",
                "torch-musa": "2.11.0.post1",
                "torchada": "0.1.81",
                "vllm": "0.24.0",
                "vllm-musa": "0.1.25.dev0",
            },
            "source_revisions": {
                "vllm": "ee0da84ab",
                "vllm-musa": "96a30cac6",
            },
            "image_digest": "sha256:test-image",
        },
        "workload": {
            "phase": "serving",
            "batch_size": {"min": 1, "max": 256},
            "tokens": {"min": 1, "max": 8192},
            "max_model_len": 32768,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 256,
            "compile_mode": "VLLM_COMPILE",
            "graph_mode": "FULL_AND_PIECEWISE",
        },
    }


def _timing_document() -> dict:
    catalog = [
        {
            "id": "vllm.ir.fused_add_rms_norm:native",
            "kind": "vllm_ir_provider",
            "operation": "fused_add_rms_norm",
            "choice": "native",
            "fallback_id": "vllm.ir.fused_add_rms_norm:native",
            "implementation_fingerprint": "sha256:native",
            "description": "native fallback",
        },
        {
            "id": "vllm.ir.fused_add_rms_norm:musa",
            "kind": "vllm_ir_provider",
            "operation": "fused_add_rms_norm",
            "choice": "musa",
            "fallback_id": "vllm.ir.fused_add_rms_norm:native",
            "implementation_fingerprint": "sha256:musa",
            "description": "measured MUSA provider",
        },
    ]
    builder = TimingCacheBuilder.from_documents(
        target=_target(),
        catalog=catalog,
        provenance={"run": "unit-test"},
    )
    for rows in required_power2_bound_rows(8192):
        case = BenchmarkCase.create(
            operation="fused_add_rms_norm",
            phase="operator",
            batch_size=1,
            tokens=rows,
            rows=rows,
            hidden_size=4096,
            dtype="bfloat16",
        )
        builder.add_observation(
            "vllm.ir.fused_add_rms_norm:native",
            [1.0, 1.0, 1.0],
            case=case,
        )
        builder.add_observation(
            "vllm.ir.fused_add_rms_norm:musa",
            [0.8, 0.8, 0.8],
            case=case,
        )
    return builder.build()


def _plan(
    *,
    plan_id: str = "schema-v5-test",
    decisions: dict | None = None,
) -> dict:
    return build_plan_document(
        [_timing_document()],
        plan_id=plan_id,
        runtime_decisions=decisions,
    )


def _write_plan(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "plan.json"
    write_json_object_file(path, document)
    return path


def _spawn_plan_load_result(queue) -> None:
    """Spawn target proving workers honor the parent-pinned plan identity."""

    try:
        plan = example_plugin.JsonPlanRuntimePlugin()._load_selected_plan()
    except Exception as exc:  # pragma: no cover - asserted through child result
        queue.put(("error", str(exc)))
    else:  # pragma: no cover - failure branch asserted by the parent
        queue.put(("loaded", plan.fingerprint))


def _config() -> WeakConfig:
    config = WeakConfig()
    config.kernel_config = SimpleNamespace(
        ir_op_priority=SimpleNamespace(rms_norm=[], fused_add_rms_norm=[]),
    )
    return config


def _activate_exact_variant(monkeypatch, document: dict) -> dict:
    parsed = parse_plan_document(document)
    variant = dict(parsed.variants[0])
    runtime_target = _target()
    monkeypatch.setattr(
        example_plugin,
        "select_runtime_variant",
        lambda plan, config, final=False: RuntimeVariantDecision(
            variant=variant,
            runtime_target=runtime_target,
            reason="runtime_compatibility_key_early",
            differences=(),
        ),
    )
    monkeypatch.setattr(
        example_plugin,
        "validate_runtime_variant",
        lambda selected, config: (runtime_target, ()),
    )
    return variant


def test_schema_v5_seal_and_load_round_trip(tmp_path):
    document = _plan()
    path = _write_plan(tmp_path, document)

    loaded = load_plan(path)

    assert loaded.plan_id == "schema-v5-test"
    assert loaded.schema_version == 5
    assert loaded.fingerprint.startswith("sha256:")


def test_schema_v5_unifies_provider_and_model_decisions():
    document = _plan(
        decisions={
            "qwen3.text_generation": {
                "qwen3.qk_rope_kv_presplit": False,
                "qwen.v2_sampling": True,
            }
        }
    )

    projection = document["variants"][0]["runtime_decisions"]
    profile_config = declarative_profile_identity("qwen3.text_generation")
    assert profile_config is not None

    assert projection == {
        "profile": "qwen3.text_generation",
        "profile_config_id": profile_config[0],
        "profile_config_fingerprint": profile_config[1],
        "values": {
            "qwen.v2_sampling": True,
            "qwen3.qk_rope_kv_presplit": False,
            "vllm.ir_op_priority": {"fused_add_rms_norm": ["musa", "native"]},
        },
    }


def test_legacy_external_only_projection_without_profile_config_still_loads():
    document = _plan()
    projection = document["variants"][0]["runtime_decisions"]
    projection.pop("profile_config_id")
    projection.pop("profile_config_fingerprint")
    document.pop("fingerprint")

    parsed = parse_plan_document(seal_plan_document(document))

    assert "profile_config_id" not in parsed.variants[0]["runtime_decisions"]


def test_partial_profile_config_identity_is_rejected() -> None:
    document = _plan()
    document.pop("fingerprint")
    document["variants"][0]["runtime_decisions"].pop("profile_config_fingerprint")

    with pytest.raises(EnginePlanError, match="must be set together"):
        seal_plan_document(document)


def test_stale_profile_config_identity_is_rejected_while_sealing() -> None:
    document = _plan()
    document.pop("fingerprint")
    document["variants"][0]["runtime_decisions"]["profile_config_fingerprint"] = (
        "sha256:" + "0" * 64
    )

    with pytest.raises(EnginePlanError, match="differs from the live source"):
        seal_plan_document(document)


def test_tampered_runtime_decision_fingerprint_is_rejected(tmp_path):
    document = _plan()
    document["variants"][0]["runtime_decisions"]["values"]["vllm.ir_op_priority"][
        "fused_add_rms_norm"
    ] = ["native"]
    path = _write_plan(tmp_path, document)

    with pytest.raises(EnginePlanError, match="fingerprint mismatch"):
        load_plan(path)


def test_resealed_provider_decision_must_match_timing_selection():
    document = _plan()
    document.pop("fingerprint")
    document["variants"][0]["runtime_decisions"]["values"]["vllm.ir_op_priority"][
        "fused_add_rms_norm"
    ] = ["native"]

    with pytest.raises(EnginePlanError, match="does not match.*selections"):
        seal_plan_document(document)


def test_plugin_applies_and_projects_one_runtime_plan(monkeypatch, tmp_path):
    document = _plan(
        decisions={"qwen3.text_generation": {"qwen3.qk_rope_kv_presplit": False}}
    )
    variant = _activate_exact_variant(monkeypatch, document)
    path = _write_plan(tmp_path, document)
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    example_plugin.register()
    example_plugin.register()
    config = _config()

    application = apply_engine_plugin_defaults(config)
    decision_receipt = resolve_engine_plugin_runtime_decisions(config)
    receipt = validate_engine_plugin_runtime(config)

    assert application.selected_variant == variant["variant_id"]
    assert config.kernel_config.ir_op_priority.fused_add_rms_norm == [
        "musa",
        "native",
    ]
    assert decision_receipt is not None
    profile_config = declarative_profile_identity("qwen3.text_generation")
    assert profile_config is not None
    assert decision_receipt.profile_config_id == profile_config[0]
    assert decision_receipt.profile_config_fingerprint == profile_config[1]
    assert dict(decision_receipt.decisions) == {
        "qwen3.qk_rope_kv_presplit": False,
        "vllm.ir_op_priority": (("fused_add_rms_norm", ("musa", "native")),),
    }
    assert receipt.plan_fingerprint == application.plan_fingerprint


def test_plugin_plan_state_is_released_after_config_collection(
    monkeypatch,
    tmp_path,
):
    document = _plan()
    _activate_exact_variant(monkeypatch, document)
    path = _write_plan(tmp_path, document)
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    example_plugin.register()
    config = _config()
    identity = id(config)
    reference = weakref.ref(config)

    apply_engine_plugin_defaults(config)
    assert identity in example_plugin._PLUGIN._plans

    del config
    gc.collect()

    assert reference() is None
    assert identity not in example_plugin._PLUGIN._plans


def test_explicit_priority_conflict_fails_closed(monkeypatch, tmp_path):
    document = _plan()
    _activate_exact_variant(monkeypatch, document)
    path = _write_plan(tmp_path, document)
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    example_plugin.register()
    config = _config()
    config.kernel_config.ir_op_priority.fused_add_rms_norm = ["native"]

    with pytest.raises(
        EnginePluginActivationError,
        match="conflicts with explicit priority",
    ):
        apply_engine_plugin_defaults(config)


def test_version_provenance_does_not_gate_activation(monkeypatch, tmp_path):
    document = _plan()
    _activate_exact_variant(monkeypatch, document)
    path = _write_plan(tmp_path, document)
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    example_plugin.register()

    assert apply_engine_plugin_defaults(_config()) is not None


@pytest.mark.parametrize(
    "prefixes",
    [[], ["0.24", 24], ["0.24", "0.24"]],
)
def test_version_prefix_allowlist_is_strict(prefixes):
    document = _plan()
    document.pop("fingerprint")
    document["compatibility"]["framework_version_prefixes"] = prefixes

    with pytest.raises(EnginePlanError, match="framework_version_prefixes"):
        seal_plan_document(document)


def test_plan_change_between_phases_fails_closed(monkeypatch, tmp_path):
    first = _plan(plan_id="first")
    _activate_exact_variant(monkeypatch, first)
    path = _write_plan(tmp_path, first)
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    example_plugin.register()
    config = _config()
    apply_engine_plugin_defaults(config)

    _write_plan(tmp_path, _plan(plan_id="changed-plan"))

    with pytest.raises(
        EnginePluginActivationError,
        match="does not match the parent-pinned",
    ):
        validate_engine_plugin_runtime(config)


def test_default_mode_self_pins_loaded_plan(monkeypatch, tmp_path):
    document = _plan()
    path = _write_plan(tmp_path, document)
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))

    loaded = example_plugin.JsonPlanRuntimePlugin()._load_selected_plan()

    assert loaded.fingerprint == document["fingerprint"]
    assert os.environ[ENGINE_PLAN_FINGERPRINT_ENV] == document["fingerprint"]


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_strict_mode_requires_external_pin_before_plan_load(
    monkeypatch,
    value,
):
    monkeypatch.setenv(ENGINE_PLAN_ENV, "/path/must/not/be-read.json")
    monkeypatch.setenv(ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV, value)

    with pytest.raises(
        EnginePlanError,
        match="requires MUSA_ENGINE_PLAN_FINGERPRINT to be explicitly set",
    ):
        example_plugin.JsonPlanRuntimePlugin()._load_selected_plan()

    assert ENGINE_PLAN_FINGERPRINT_ENV not in os.environ


def test_strict_mode_rejects_empty_external_pin_before_plan_load(monkeypatch):
    monkeypatch.setenv(ENGINE_PLAN_ENV, "/path/must/not/be-read.json")
    monkeypatch.setenv(ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV, "true")
    monkeypatch.setenv(ENGINE_PLAN_FINGERPRINT_ENV, "")

    with pytest.raises(
        EnginePlanError,
        match="requires MUSA_ENGINE_PLAN_FINGERPRINT to be explicitly set",
    ):
        example_plugin.JsonPlanRuntimePlugin()._load_selected_plan()


@pytest.mark.parametrize("value", ["0", "false", "FALSE", " no ", "off"])
def test_explicit_false_strict_mode_preserves_development_self_pin(
    monkeypatch,
    tmp_path,
    value,
):
    document = _plan()
    path = _write_plan(tmp_path, document)
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    monkeypatch.setenv(ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV, value)

    loaded = example_plugin.JsonPlanRuntimePlugin()._load_selected_plan()

    assert loaded.fingerprint == document["fingerprint"]
    assert os.environ[ENGINE_PLAN_FINGERPRINT_ENV] == document["fingerprint"]


@pytest.mark.parametrize("value", ["", "2", "enabled", "truthy"])
def test_invalid_strict_mode_boolean_fails_closed(monkeypatch, value):
    monkeypatch.setenv(ENGINE_PLAN_ENV, "/path/must/not/be-read.json")
    monkeypatch.setenv(ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV, value)

    with pytest.raises(
        EnginePlanError,
        match="must be a boolean value",
    ):
        example_plugin.JsonPlanRuntimePlugin()._load_selected_plan()

    assert ENGINE_PLAN_FINGERPRINT_ENV not in os.environ


def test_strict_mode_accepts_matching_external_pin(monkeypatch, tmp_path):
    document = _plan()
    path = _write_plan(tmp_path, document)
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    monkeypatch.setenv(ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV, "true")
    monkeypatch.setenv(
        ENGINE_PLAN_FINGERPRINT_ENV,
        document["fingerprint"],
    )

    loaded = example_plugin.JsonPlanRuntimePlugin()._load_selected_plan()

    assert loaded.fingerprint == document["fingerprint"]


def test_strict_mode_rejects_mismatched_external_pin(monkeypatch, tmp_path):
    path = _write_plan(tmp_path, _plan())
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    monkeypatch.setenv(ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV, "true")
    monkeypatch.setenv(ENGINE_PLAN_FINGERPRINT_ENV, "sha256:not-the-plan")

    with pytest.raises(EnginePlanError, match="does not match the parent-pinned"):
        example_plugin.JsonPlanRuntimePlugin()._load_selected_plan()


def test_spawned_worker_rejects_plan_rotated_after_parent_pin(
    monkeypatch,
    tmp_path,
):
    first = _plan(plan_id="parent-plan")
    _activate_exact_variant(monkeypatch, first)
    path = _write_plan(tmp_path, first)
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    example_plugin.register()

    assert apply_engine_plugin_defaults(_config()) is not None
    assert os.environ[ENGINE_PLAN_FINGERPRINT_ENV] == first["fingerprint"]

    write_json_object_file(path, _plan(plan_id="rotated-before-spawn"))
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_plan_load_result, args=(queue,))
    process.start()
    process.join(timeout=15)

    assert process.exitcode == 0
    status, detail = queue.get(timeout=2)
    assert status == "error"
    assert "does not match the parent-pinned" in detail


def test_spawned_worker_honors_strict_external_pin(monkeypatch, tmp_path):
    document = _plan(plan_id="strict-parent-plan")
    path = _write_plan(tmp_path, document)
    monkeypatch.setenv(ENGINE_PLAN_ENV, str(path))
    monkeypatch.setenv(ENGINE_PLAN_REQUIRE_PINNED_FINGERPRINT_ENV, "true")
    monkeypatch.setenv(
        ENGINE_PLAN_FINGERPRINT_ENV,
        document["fingerprint"],
    )

    parent_plan = example_plugin.JsonPlanRuntimePlugin()._load_selected_plan()
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_plan_load_result, args=(queue,))
    process.start()
    process.join(timeout=15)

    assert process.exitcode == 0
    assert parent_plan.fingerprint == document["fingerprint"]
    assert queue.get(timeout=2) == ("loaded", document["fingerprint"])
