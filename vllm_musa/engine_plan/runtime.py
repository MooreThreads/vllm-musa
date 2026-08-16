# SPDX-License-Identifier: Apache-2.0

"""Runtime-context projection and typed engine-plan variant resolution.

Imports of vLLM, torch, and MTML stay inside functions so offline plan commands
remain usable in a dependency-light environment.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .artifact_io import ArtifactFileError, load_json_object_file
from .artifacts import (
    REQUIRED_SOFTWARE_VERSION_KEYS,
    REQUIRED_SOURCE_REVISION_KEYS,
    PlanTarget,
    TacticKind,
    TimingCache,
    diff_runtime_targets,
    static_topology_differences,
    static_workload_scope_differences,
)
from .core import SUPPORTED_PLAN_SCHEMA_VERSIONS, EnginePlan, EnginePlanError
from .tactic_fingerprints import runtime_decision_implementation

ENGINE_BUILD_MANIFEST_ENV = "MUSA_ENGINE_BUILD_MANIFEST"
DEFAULT_ENGINE_BUILD_MANIFEST = "/etc/vllm-musa/build-manifest.json"
ENGINE_BUILD_MANIFEST_SCHEMA = "musa.engine_build.v1"
_MUSA_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16


@dataclass(frozen=True, slots=True)
class RuntimeVariantDecision:
    variant: dict[str, Any] | None
    runtime_target: dict[str, Any]
    reason: str
    differences: tuple[str, ...]

    @property
    def is_match(self) -> bool:
        return self.variant is not None


def _enum_name(value: Any, default: str) -> str:
    if value is None:
        return default
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    rendered = str(value)
    return rendered.rsplit(".", maxsplit=1)[-1] if rendered else default


def _dtype_name(value: Any) -> str:
    rendered = str(value)
    return rendered.removeprefix("torch.") or "unknown"


def _architecture(model_config: Any) -> str:
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None)
    if isinstance(architectures, (list, tuple)) and architectures:
        return str(architectures[0])
    architectures = getattr(model_config, "architectures", None)
    if isinstance(architectures, (list, tuple)) and architectures:
        return str(architectures[0])
    return "unknown"


def _hidden_size(model_config: Any) -> int:
    hf_config = getattr(model_config, "hf_config", None)
    owners = [
        getattr(model_config, "hf_text_config", None),
        getattr(hf_config, "text_config", None),
        hf_config,
        model_config,
    ]
    get_text_config = getattr(hf_config, "get_text_config", None)
    if callable(get_text_config):
        try:
            owners.insert(0, get_text_config())
        except Exception:
            pass
    for owner in owners:
        value = getattr(owner, "hidden_size", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


def _quantization(model_config: Any, vllm_config: Any) -> str:
    quant_config = getattr(vllm_config, "quant_config", None)
    get_name = getattr(quant_config, "get_name", None)
    if callable(get_name):
        try:
            return str(get_name())
        except Exception:
            pass
    for owner in (model_config, getattr(model_config, "hf_config", None)):
        value = getattr(owner, "quantization", None)
        if value:
            return str(value)
    return "none"


def _distribution_version(name: str) -> str:
    aliases = {
        "vllm_musa": "vllm-musa",
        "torch_musa": "torch-musa",
        "flash_attn_3": "flash-attn-3",
    }
    distribution = aliases.get(name, name)
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


@cache
def _runtime_source_revisions() -> dict[str, str]:
    path = os.environ.get(
        ENGINE_BUILD_MANIFEST_ENV,
        DEFAULT_ENGINE_BUILD_MANIFEST,
    )
    try:
        document = load_json_object_file(path)
    except ArtifactFileError:
        return {}
    if set(document) != {"schema_version", "source_revisions"}:
        return {}
    if document["schema_version"] != ENGINE_BUILD_MANIFEST_SCHEMA:
        return {}
    source_revisions = document["source_revisions"]
    if not isinstance(source_revisions, dict):
        return {}
    if set(source_revisions) != {"vllm", "vllm-musa"}:
        return {}
    if any(
        not isinstance(value, str) or not value for value in source_revisions.values()
    ):
        return {}
    return dict(sorted(source_revisions.items()))


def _software_versions_for_names(
    names: set[str],
    *,
    driver_version: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    torch_module = None
    if "torch" in names or "musa" in names:
        try:
            import torch

            torch_module = torch
        except Exception:
            torch_module = None
    for name in names:
        if name == "driver":
            result[name] = driver_version
        elif name == "torch" and torch_module is not None:
            result[name] = str(torch_module.__version__)
        elif name == "musa" and torch_module is not None:
            result[name] = str(getattr(torch_module.version, "musa", "unknown"))
        else:
            result[name] = _distribution_version(name)
    return result


def _software_versions(
    expected: TimingCache,
    *,
    driver_version: str,
) -> dict[str, str]:
    return _software_versions_for_names(
        set(dict(expected.target.software.versions)),
        driver_version=driver_version,
    )


def _physical_device_index() -> int:
    visible = os.environ.get("MUSA_VISIBLE_DEVICES", "")
    first = visible.split(",", maxsplit=1)[0].strip()
    try:
        return int(first) if first else 0
    except ValueError:
        return 0


def _decode_mtml_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _active_musa_multiprocessor_count() -> int:
    """Query the active device's MP count without creating a MUSA context.

    The MUSA Driver API ordinal is logical: device 0 is the first device made
    visible to this process.  MTML's GPU-core query has different semantics on
    S5000 and must not be used as a fallback for this performance-bin key.
    """

    try:
        driver = ctypes.CDLL("libmusa.so")
        mu_init = driver.muInit
        mu_device_get = driver.muDeviceGet
        mu_device_get_attribute = driver.muDeviceGetAttribute
        mu_init.restype = ctypes.c_int
        mu_init.argtypes = [ctypes.c_uint]
        mu_device_get.restype = ctypes.c_int
        mu_device_get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        mu_device_get_attribute.restype = ctypes.c_int
        mu_device_get_attribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]

        if mu_init(0) != 0:
            return -1
        device = ctypes.c_int()
        if mu_device_get(ctypes.byref(device), 0) != 0:
            return -1
        multiprocessor_count = ctypes.c_int()
        if (
            mu_device_get_attribute(
                ctypes.byref(multiprocessor_count),
                _MUSA_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device,
            )
            != 0
        ):
            return -1
        return multiprocessor_count.value if multiprocessor_count.value > 0 else -1
    except Exception:
        return -1


@cache
def _hardware_identity() -> tuple[str, str, int, str, str]:
    """Collect stable hardware facts without intentionally creating a MUSA context."""

    multiprocessor_count = _active_musa_multiprocessor_count()
    physical_index = _physical_device_index()
    try:
        import pymtml as pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        name = _decode_mtml_text(pynvml.nvmlDeviceGetName(handle))
        device_uuid = _decode_mtml_text(pynvml.nvmlDeviceGetUUID(handle))
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        driver = _decode_mtml_text(pynvml.nvmlSystemGetDriverVersion())
        return (
            name,
            f"{int(major)}.{int(minor)}",
            int(multiprocessor_count),
            driver,
            device_uuid,
        )
    except Exception:
        return "unknown", "unknown", multiprocessor_count, "unknown", "unknown"


def collect_environment_identity(
    *,
    device_count: int,
    software_names: set[str] | None = None,
) -> dict[str, Any]:
    """Collect the auto-probed hardware/software portion of a plan target."""

    if (
        not isinstance(device_count, int)
        or isinstance(device_count, bool)
        or device_count <= 0
    ):
        raise EnginePlanError("device_count must be a positive integer")
    device_name, _, _, driver, device_uuid = _hardware_identity()
    # UUID is provenance only. Containers and partitioned hosts may not expose
    # it even when the architecture/capability facts needed for applicability
    # are available.
    try:
        import torch

        properties = torch.musa.get_device_properties(0)
        device_name = str(getattr(properties, "name", device_name))
        capability = f"{int(properties.major)}.{int(properties.minor)}"
        multiprocessor_count = int(properties.multi_processor_count)
    except Exception as exc:
        raise EnginePlanError(
            "Unable to collect MUSA capability and active multiprocessor count"
        ) from exc
    if multiprocessor_count <= 0:
        raise EnginePlanError("Unable to collect a positive MUSA multiprocessor count")
    return {
        "hardware": {
            "platform": "musa",
            "device_name": device_name,
            "device_uuid": device_uuid,
            "device_capability": capability,
            "multiprocessor_count": multiprocessor_count,
            "device_count": device_count,
        },
        "software_versions": _software_versions_for_names(
            (
                set(REQUIRED_SOFTWARE_VERSION_KEYS)
                if software_names is None
                else set(software_names)
            ),
            driver_version=driver,
        ),
        "source_revisions": _runtime_source_revisions(),
    }


def collect_environment_target(expected: PlanTarget) -> dict[str, Any]:
    """Project the live tuning environment into an existing target envelope.

    Model and workload fields are intentionally copied from the already parsed
    target: this probe validates the live hardware/software/source identity
    before offline collection, not a vLLM serving configuration.  Runtime-key
    comparison below continues to exclude evidence-only fields such as device
    UUID, device count, and image digest.
    """

    expected_source_names = set(dict(expected.software.source_revisions))
    if expected_source_names != set(REQUIRED_SOURCE_REVISION_KEYS):
        raise EnginePlanError(
            "AutoTuner live source verification supports exactly "
            f"{sorted(REQUIRED_SOURCE_REVISION_KEYS)!r}; target contains "
            f"unverifiable source revision keys "
            f"{sorted(expected_source_names - REQUIRED_SOURCE_REVISION_KEYS)!r}"
        )
    environment = collect_environment_identity(
        device_count=expected.hardware.device_count,
        software_names=set(dict(expected.software.versions)),
    )
    source_revisions = environment.get("source_revisions", {})
    if not isinstance(source_revisions, dict) or set(source_revisions) != set(
        REQUIRED_SOURCE_REVISION_KEYS
    ):
        manifest_path = os.environ.get(
            ENGINE_BUILD_MANIFEST_ENV,
            DEFAULT_ENGINE_BUILD_MANIFEST,
        )
        raise EnginePlanError(
            "AutoTuner cannot verify live source revisions. Provide a readable "
            f"build manifest via {ENGINE_BUILD_MANIFEST_ENV} (current path: "
            f"{manifest_path!r}; default: {DEFAULT_ENGINE_BUILD_MANIFEST!r}) "
            "with exactly schema_version='musa.engine_build.v1' and "
            "source_revisions={'vllm': '<sha>', 'vllm-musa': '<sha>'}. "
            "Target-provided source revisions are not trusted as live identity."
        )
    actual = expected.to_document()
    actual["hardware"] = dict(environment.get("hardware", {}))
    actual["software"] = {
        "versions": dict(environment.get("software_versions", {})),
        "source_revisions": dict(source_revisions),
        # The image digest is measurement provenance, not an applicability key.
        "image_digest": "unknown",
    }
    return actual


def diff_environment_identity(expected: PlanTarget) -> tuple[str, ...]:
    """Return live collection-environment drift using runtime applicability keys."""

    return diff_runtime_targets(
        expected,
        collect_environment_target(expected),
        final=False,
    )


def _tactic_registry_differences(variant: dict[str, Any]) -> tuple[str, ...]:
    """Verify selected and fallback tactics against their live implementations."""

    timing_cache = TimingCache.from_document(
        variant["timing_cache"],
        require_fingerprint=True,
    )
    definitions = {item.tactic_id: item for item in timing_cache.catalog}
    tactic_ids: set[str] = set()
    for selection in variant["selections"]:
        fallback = selection.get("fallback")
        if isinstance(fallback, str):
            tactic_ids.add(fallback)
        winner = selection.get("winner")
        if isinstance(winner, str):
            tactic_ids.add(winner)
        for context in selection.get("contexts", ()):
            context_winner = context.get("winner")
            if isinstance(context_winner, str):
                tactic_ids.add(context_winner)
    checked_tactic_ids = sorted(tactic_ids)
    if not checked_tactic_ids:
        return ()
    try:
        from vllm_musa.engine_plugins import find_engine_ir_provider
    except Exception as exc:
        return (f"tactic_registry_unavailable:{type(exc).__name__}:{exc}",)
    differences: list[str] = []
    resolved_providers: dict[tuple[str, str], Any] = {}
    for tactic_id in checked_tactic_ids:
        definition = definitions[tactic_id]
        implementation = None
        if definition.kind is TacticKind.VLLM_IR_PROVIDER:
            provider_key = definition.operation, str(definition.choice)
        elif definition.kind is TacticKind.RUNTIME_DECISION:
            implementation = runtime_decision_implementation(definition.operation)
            if implementation is None:
                differences.append(
                    f"tactic.{tactic_id}:runtime_decision_implementation_not_registered"
                )
                continue
            provider_key = implementation.provider_key
        else:  # pragma: no cover - TacticKind parsing rejects unknown values
            differences.append(f"tactic.{tactic_id}:unsupported_tactic_kind")
            continue
        if implementation is not None and provider_key is None:
            try:
                actual_fingerprint = implementation.fingerprint(
                    choice=str(definition.choice)
                )
            except ValueError as exc:
                differences.append(
                    f"tactic.{tactic_id}:runtime_decision_choice_unavailable:{exc}"
                )
                continue
            if actual_fingerprint != definition.implementation_fingerprint:
                differences.append(
                    f"tactic.{tactic_id}: expected implementation="
                    f"{definition.implementation_fingerprint!r}, "
                    f"actual={actual_fingerprint!r}"
                )
            continue
        assert provider_key is not None
        try:
            if provider_key not in resolved_providers:
                resolved_providers[provider_key] = find_engine_ir_provider(
                    *provider_key
                )
            provider = resolved_providers[provider_key]
        except Exception as exc:
            differences.append(
                f"tactic.{tactic_id}:registry_unavailable:"
                f"{type(exc).__name__}:{exc}"
            )
            continue
        if provider is None:
            if implementation is None:
                differences.append(f"tactic.{tactic_id}:provider_not_available")
            else:
                differences.append(
                    f"tactic.{tactic_id}:provider_not_available:"
                    f"{provider_key[0]}:{provider_key[1]}"
                )
            continue
        actual_fingerprint = (
            implementation.fingerprint(provider.implementation_fingerprint)
            if implementation is not None
            else provider.implementation_fingerprint
        )
        if actual_fingerprint != definition.implementation_fingerprint:
            differences.append(
                f"tactic.{tactic_id}: expected implementation="
                f"{definition.implementation_fingerprint!r}, "
                f"actual={actual_fingerprint!r}"
            )
    return tuple(differences)


def _topology_differences(timing_cache: TimingCache) -> tuple[str, ...]:
    return static_topology_differences(
        timing_cache.target.model,
        operations=tuple(tactic.operation for tactic in timing_cache.catalog),
    )


def collect_runtime_target(
    vllm_config: Any,
    expected: TimingCache,
    *,
    final: bool,
) -> dict[str, Any]:
    """Project a live vLLM config into the immutable plan target schema."""

    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    tensor_parallel_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
    pipeline_parallel_size = int(
        getattr(parallel_config, "pipeline_parallel_size", 1) or 1
    )
    (
        device_name,
        device_capability,
        multiprocessor_count,
        driver_version,
        device_uuid,
    ) = _hardware_identity()

    expected_target = expected.target.to_document()
    if device_uuid == expected.target.hardware.device_uuid:
        # MTML exposes the stable UUID/driver without creating a MUSA context,
        # but its compatibility query may return 0.0. When provenance matches,
        # reuse the separately probed capability without initializing MUSA
        # during platform defaults; UUID itself is not an applicability key.
        # The MP performance bin always comes from the live Driver API probe.
        device_capability = expected.target.hardware.device_capability
    actual = {
        "model": {
            # Profile is validated by the immutable runtime-plan projection.
            "profile": expected.target.model.profile,
            "architecture": _architecture(model_config),
            "model_id": str(getattr(model_config, "model", "unknown")),
            "hidden_size": _hidden_size(model_config),
            "dtype": _dtype_name(getattr(model_config, "dtype", "unknown")),
            "quantization": _quantization(model_config, vllm_config),
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
        },
        "hardware": {
            "platform": "musa",
            "device_name": device_name,
            "device_uuid": device_uuid,
            "device_capability": device_capability,
            "multiprocessor_count": multiprocessor_count,
            "device_count": tensor_parallel_size * pipeline_parallel_size,
        },
        "software": {
            "versions": _software_versions(
                expected,
                driver_version=driver_version,
            ),
            "source_revisions": _runtime_source_revisions(),
            "image_digest": "unknown",
        },
        "workload": {
            # Request-shape buckets are evidence scope for a config-time tactic.
            "phase": expected_target["workload"]["phase"],
            "batch_size": expected_target["workload"]["batch_size"],
            "tokens": expected_target["workload"]["tokens"],
            "max_model_len": int(getattr(model_config, "max_model_len", 0) or 0),
            "max_num_batched_tokens": int(
                getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
            ),
            "max_num_seqs": int(getattr(scheduler_config, "max_num_seqs", 0) or 0),
            "compile_mode": _enum_name(
                getattr(compilation_config, "mode", None),
                "unknown" if final else expected.target.workload.compile_mode,
            ),
            "graph_mode": _enum_name(
                getattr(compilation_config, "cudagraph_mode", None),
                "unknown" if final else expected.target.workload.graph_mode,
            ),
        },
    }
    if "cudagraph_capture_sizes" in expected_target["workload"]:
        capture_sizes = getattr(
            compilation_config,
            "cudagraph_capture_sizes",
            None,
        )
        actual["workload"]["cudagraph_capture_sizes"] = list(capture_sizes or ())
    return actual


def select_runtime_variant(
    plan: EnginePlan,
    vllm_config: Any,
    *,
    final: bool = False,
) -> RuntimeVariantDecision:
    if plan.schema_version not in SUPPORTED_PLAN_SCHEMA_VERSIONS:
        raise EnginePlanError(
            "Typed runtime selection requires a supported plan schema; "
            f"got {plan.schema_version}, supported={SUPPORTED_PLAN_SCHEMA_VERSIONS}"
        )
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    all_differences: list[str] = []
    last_target: dict[str, Any] = {}
    for raw_variant in plan.variants:
        variant = dict(raw_variant)
        timing_cache = TimingCache.from_document(
            variant["timing_cache"],
            require_fingerprint=True,
        )
        runtime_target = collect_runtime_target(
            vllm_config,
            timing_cache,
            final=final,
        )
        last_target = runtime_target
        differences = (
            *_topology_differences(timing_cache),
            *static_workload_scope_differences(timing_cache.target.workload),
            *diff_runtime_targets(
                timing_cache.target,
                runtime_target,
                final=final,
            ),
        )
        if not differences:
            differences = _tactic_registry_differences(variant)
        if not differences:
            matches.append((variant, runtime_target))
        else:
            all_differences.extend(
                f"{variant['variant_id']}:{difference}" for difference in differences
            )
    if len(matches) == 1:
        variant, runtime_target = matches[0]
        return RuntimeVariantDecision(
            variant=variant,
            runtime_target=runtime_target,
            reason=(
                "runtime_compatibility_key_final"
                if final
                else "runtime_compatibility_key_early"
            ),
            differences=(),
        )
    if not matches:
        return RuntimeVariantDecision(
            variant=None,
            runtime_target=last_target,
            reason="no_matching_variant",
            differences=tuple(all_differences),
        )
    return RuntimeVariantDecision(
        variant=None,
        runtime_target=matches[0][1],
        reason="ambiguous_matching_variants",
        differences=tuple(item[0]["variant_id"] for item in matches),
    )


def validate_runtime_variant(
    variant: dict[str, Any],
    vllm_config: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    timing_cache = TimingCache.from_document(
        variant["timing_cache"],
        require_fingerprint=True,
    )
    runtime_target = collect_runtime_target(
        vllm_config,
        timing_cache,
        final=True,
    )
    return runtime_target, (
        *static_workload_scope_differences(timing_cache.target.workload),
        *_topology_differences(timing_cache),
        *diff_runtime_targets(
            timing_cache.target,
            runtime_target,
            final=True,
        ),
        *_tactic_registry_differences(variant),
    )
