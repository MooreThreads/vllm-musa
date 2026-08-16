from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any

from .providers import BUILTIN_RUNTIME_PLAN_PROVIDERS
from .types import (
    ExecutionSignature,
    ModelFamily,
    ModelRole,
    ModelSignature,
    RuntimeDecision,
    RuntimeDecisionError,
    RuntimeDecisionResolution,
    RuntimePlan,
    canonical_fingerprint,
)

RUNTIME_PLAN_TRANSPORT_KEY = "vllm_musa.runtime_plan_transport"
_RUNTIME_PLAN_TRANSPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class _RuntimePlanTransport:
    """Validated worker transport decoded from ``VllmConfig``."""

    decision_projection_fingerprint: str
    model_signature_fingerprint: str
    resolution: RuntimeDecisionResolution


@dataclass(frozen=True, slots=True)
class _PlanCacheKey:
    """All immutable inputs that can change a config's materialized plan."""

    model: ModelSignature
    execution: ExecutionSignature
    is_pooling_model: bool
    attention_backend_hint: str | None
    decision_context: tuple[object, ...] | None


_PLAN_CACHE_LOCK = RLock()
_PLAN_CACHE: dict[
    int,
    tuple[weakref.ReferenceType[Any], dict[_PlanCacheKey, RuntimePlan]],
] = {}


def _drop_plan_cache(
    config_id: int,
    reference: weakref.ReferenceType[Any],
) -> None:
    with _PLAN_CACHE_LOCK:
        cached = _PLAN_CACHE.get(config_id)
        if cached is not None and cached[0] is reference:
            _PLAN_CACHE.pop(config_id, None)


def _plan_cache_for(
    vllm_config: Any,
) -> dict[_PlanCacheKey, RuntimePlan] | None:
    """Return weakly-owned cache storage, or ``None`` for non-weakrefables."""

    config_id = id(vllm_config)
    with _PLAN_CACHE_LOCK:
        cached = _PLAN_CACHE.get(config_id)
        if cached is not None:
            if cached[0]() is vllm_config:
                return cached[1]
            _PLAN_CACHE.pop(config_id, None)
        try:
            reference = weakref.ref(
                vllm_config,
                lambda released, identity=config_id: _drop_plan_cache(
                    identity, released
                ),
            )
        except TypeError:
            # Some lightweight test/config facades use ``__slots__``.  They
            # still receive a correct uncached plan; retaining a strong global
            # reference would violate the per-config lifetime contract.
            return None
        storage: dict[_PlanCacheKey, RuntimePlan] = {}
        _PLAN_CACHE[config_id] = (reference, storage)
        return storage


def _engine_application_key(vllm_config: Any) -> tuple[object, ...] | None:
    from vllm_musa.engine_plugins import get_engine_plugin_application

    application = get_engine_plugin_application(vllm_config)
    if application is None:
        return None
    return (
        "engine_application",
        application.plugin_name,
        application.plan_id,
        application.plan_fingerprint,
        application.selected_variant,
        tuple(application.selected_tactics),
        application.context_fingerprint,
        application.fallback_reason,
    )


def _require_exact_keys(
    value: object,
    *,
    field: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeDecisionError(f"{field} must be a JSON object")
    keys = set(value)
    if not all(isinstance(key, str) for key in keys):
        raise RuntimeDecisionError(f"{field} keys must be strings")
    missing = sorted(required - keys)
    unexpected = sorted(keys - required - optional)
    if missing or unexpected:
        raise RuntimeDecisionError(
            f"{field} has invalid schema: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return value


def _immutable_transport_value(value: object, *, field: str) -> object:
    """Normalize JSON list hops back to the immutable plan value ABI."""

    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) or not key for key in value):
            raise RuntimeDecisionError(f"{field} object keys must be non-empty strings")
        return tuple(
            (
                key,
                _immutable_transport_value(value[key], field=f"{field}.{key}"),
            )
            for key in sorted(value)
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _immutable_transport_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    raise RuntimeDecisionError(
        f"{field} must be an immutable JSON-like runtime decision value; "
        f"got {type(value).__name__}"
    )


def _parse_transport_resolution(value: object) -> RuntimeDecisionResolution:
    required = frozenset(
        {
            "source",
            "plugin_name",
            "plugin_version",
            "plan_id",
            "fingerprint",
            "profile",
            "decisions",
        }
    )
    optional = frozenset({"profile_config_id", "profile_config_fingerprint"})
    document = _require_exact_keys(
        value,
        field=f"{RUNTIME_PLAN_TRANSPORT_KEY}.resolution",
        required=required,
        optional=optional,
    )
    if document["source"] != "engine_plugin":
        raise RuntimeDecisionError(
            f"{RUNTIME_PLAN_TRANSPORT_KEY}.resolution.source must be " "'engine_plugin'"
        )
    for field_name in ("plugin_name", "plugin_version"):
        field_value = document[field_name]
        if not isinstance(field_value, str) or not field_value:
            raise RuntimeDecisionError(
                f"{RUNTIME_PLAN_TRANSPORT_KEY}.resolution.{field_name} "
                "must be a non-empty string"
            )
    resolution_fingerprint = document["fingerprint"]
    if not isinstance(resolution_fingerprint, str) or not (
        resolution_fingerprint.startswith("sha256:")
    ):
        raise RuntimeDecisionError(
            f"{RUNTIME_PLAN_TRANSPORT_KEY}.resolution.fingerprint must use "
            "the sha256: prefix"
        )
    raw_decisions = document["decisions"]
    if isinstance(raw_decisions, dict):
        decision_items = list(raw_decisions.items())
    elif isinstance(raw_decisions, (list, tuple)):
        decision_items = []
        for index, raw_decision in enumerate(raw_decisions):
            if not isinstance(raw_decision, (list, tuple)) or len(raw_decision) != 2:
                raise RuntimeDecisionError(
                    f"{RUNTIME_PLAN_TRANSPORT_KEY}.resolution.decisions[{index}] "
                    "must be a [key, value] pair"
                )
            decision_items.append((raw_decision[0], raw_decision[1]))
    else:
        raise RuntimeDecisionError(
            f"{RUNTIME_PLAN_TRANSPORT_KEY}.resolution.decisions must be a "
            "non-empty JSON object or array"
        )
    if not decision_items:
        raise RuntimeDecisionError(
            f"{RUNTIME_PLAN_TRANSPORT_KEY}.resolution.decisions must be non-empty"
        )
    decisions: list[tuple[str, object]] = []
    for index, (key, raw_value) in enumerate(decision_items):
        if not isinstance(key, str) or not key:
            raise RuntimeDecisionError(
                f"{RUNTIME_PLAN_TRANSPORT_KEY}.resolution.decisions[{index}].key "
                "must be a non-empty string"
            )
        decisions.append(
            (
                key,
                _immutable_transport_value(
                    raw_value,
                    field=(
                        f"{RUNTIME_PLAN_TRANSPORT_KEY}.resolution."
                        f"decisions[{index}].value"
                    ),
                ),
            )
        )
    try:
        return RuntimeDecisionResolution(
            source=document["source"],
            plugin_name=document["plugin_name"],
            plugin_version=document["plugin_version"],
            plan_id=document["plan_id"],
            fingerprint=resolution_fingerprint,
            profile=document["profile"],
            decisions=tuple(decisions),
            profile_config_id=document.get("profile_config_id"),
            profile_config_fingerprint=document.get("profile_config_fingerprint"),
        )
    except Exception as exc:
        raise RuntimeDecisionError(
            f"Invalid {RUNTIME_PLAN_TRANSPORT_KEY}.resolution: {exc}"
        ) from exc


def _transport_resolution_document(
    resolution: RuntimeDecisionResolution,
) -> dict[str, object]:
    document: dict[str, object] = {
        "source": resolution.source,
        "plugin_name": resolution.plugin_name,
        "plugin_version": resolution.plugin_version,
        "plan_id": resolution.plan_id,
        "fingerprint": resolution.fingerprint,
        "profile": resolution.profile,
        "decisions": [
            [key, _json_transport_value(value)]
            for key, value in resolution.decisions
        ],
    }
    if resolution.profile_config_id is not None:
        document["profile_config_id"] = resolution.profile_config_id
        document["profile_config_fingerprint"] = (
            resolution.profile_config_fingerprint
        )
    return document


def _decision_projection_fingerprint(
    resolution: RuntimeDecisionResolution,
) -> str:
    """Bind transported decisions without binding mutable vLLM defaults."""

    payload = json.dumps(
        _transport_resolution_document(resolution),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _runtime_plan_transport(vllm_config: Any | None) -> _RuntimePlanTransport | None:
    if vllm_config is None:
        return None
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return None
    if RUNTIME_PLAN_TRANSPORT_KEY not in additional_config:
        return None
    document = _require_exact_keys(
        additional_config[RUNTIME_PLAN_TRANSPORT_KEY],
        field=RUNTIME_PLAN_TRANSPORT_KEY,
        required=frozenset(
            {
                "schema_version",
                "decision_projection_fingerprint",
                "model_signature_fingerprint",
                "resolution",
            }
        ),
    )
    schema_version = document["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != _RUNTIME_PLAN_TRANSPORT_SCHEMA_VERSION
    ):
        raise RuntimeDecisionError(
            f"{RUNTIME_PLAN_TRANSPORT_KEY}.schema_version must be "
            f"{_RUNTIME_PLAN_TRANSPORT_SCHEMA_VERSION}"
        )
    projection_fingerprint = document["decision_projection_fingerprint"]
    if not isinstance(projection_fingerprint, str) or not (
        projection_fingerprint.startswith("sha256:")
    ):
        raise RuntimeDecisionError(
            f"{RUNTIME_PLAN_TRANSPORT_KEY}.decision_projection_fingerprint "
            "must use the sha256: prefix"
        )
    resolution = _parse_transport_resolution(document["resolution"])
    expected_fingerprint = _decision_projection_fingerprint(resolution)
    if projection_fingerprint != expected_fingerprint:
        raise RuntimeDecisionError(
            "Serialized RuntimePlan transport decision projection fingerprint "
            "does not match its resolution"
        )
    model_signature_fingerprint = document["model_signature_fingerprint"]
    if not isinstance(model_signature_fingerprint, str) or not (
        model_signature_fingerprint.startswith("sha256:")
    ):
        raise RuntimeDecisionError(
            f"{RUNTIME_PLAN_TRANSPORT_KEY}.model_signature_fingerprint must "
            "use the sha256: prefix"
        )
    return _RuntimePlanTransport(
        decision_projection_fingerprint=projection_fingerprint,
        model_signature_fingerprint=model_signature_fingerprint,
        resolution=resolution,
    )


def _decision_context_key(
    vllm_config: Any,
    transport: _RuntimePlanTransport | None,
) -> tuple[object, ...] | None:
    if transport is not None:
        return (
            "runtime_plan_transport",
            transport.decision_projection_fingerprint,
            transport.model_signature_fingerprint,
            transport.resolution,
        )
    return _engine_application_key(vllm_config)


def _normalize_dtype(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).lower().removeprefix("torch.")


def _normalize_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).lower().split(".")[-1]


def _int_tuple(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    return tuple(value)


def _int_attr(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _float_attr(config: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _str_attr(config: Any, *names: str) -> str | None:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return str(value).lower()
    return None


def _text_config(model_config: Any) -> Any:
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is not None:
        return text_config
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "text_config", hf_config)


def _architectures(model_config: Any, text_config: Any) -> tuple[str, ...]:
    values = getattr(model_config, "architectures", None) or None
    if not values:
        hf_config = getattr(model_config, "hf_config", None)
        values = getattr(hf_config, "architectures", None) or None
    if not values:
        values = getattr(text_config, "architectures", None) or ()
    return tuple(str(value) for value in values or ())


def _outer_model_type(model_config: Any, text_config: Any) -> str | None:
    value = getattr(model_config, "model_type", None)
    if value is None:
        hf_config = getattr(model_config, "hf_config", None)
        value = getattr(hf_config, "model_type", None)
    if value is None:
        value = getattr(text_config, "model_type", None)
    return str(value) if value is not None else None


def _text_architectures(text_config: Any) -> tuple[str, ...]:
    values = getattr(text_config, "architectures", None) or ()
    return tuple(str(value) for value in values or ())


def _quantization(model_config: Any, text_config: Any) -> str | None:
    value = getattr(model_config, "quantization", None)
    if value is not None:
        return str(value).lower()
    quantization_config = getattr(text_config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        value = quantization_config.get("quant_method")
        return str(value).lower() if value is not None else None
    return None


def _has_routed_experts(model_config: Any, text_config: Any) -> bool | None:
    is_model_moe = getattr(model_config, "is_model_moe", None)
    if callable(is_model_moe):
        try:
            return bool(is_model_moe())
        except (AttributeError, TypeError, ValueError):
            return None
    if bool(getattr(model_config, "is_moe", False)):
        return True
    expert_values = [
        getattr(text_config, name, None)
        for name in (
            "num_experts",
            "moe_num_experts",
            "n_routed_experts",
            "num_local_experts",
        )
    ]
    if any(value is not None for value in expert_values):
        return any(bool(value) for value in expert_values)
    return False


def _gdn_conv_signature(text_config: Any) -> tuple[int | None, int | None]:
    width = _int_attr(text_config, "linear_conv_kernel_dim")
    key_heads = _int_attr(text_config, "linear_num_key_heads")
    value_heads = _int_attr(text_config, "linear_num_value_heads")
    key_dim = _int_attr(text_config, "linear_key_head_dim")
    value_dim = _int_attr(text_config, "linear_value_head_dim")
    if None in (key_heads, value_heads, key_dim, value_dim):
        return width, None
    return width, 2 * key_heads * key_dim + value_heads * value_dim


def _model_signature(model_config: Any, vllm_config: Any | None) -> ModelSignature:
    text_config = _text_config(model_config)
    gdn_width, gdn_dim = _gdn_conv_signature(text_config)
    architectures = _architectures(model_config, text_config)
    quant_config = getattr(vllm_config, "quant_config", None)
    quantization_config = getattr(text_config, "quantization_config", None)
    quant_block_shape = _int_tuple(getattr(quant_config, "weight_block_size", None))
    if quant_block_shape is None and isinstance(quantization_config, dict):
        quant_block_shape = _int_tuple(
            quantization_config.get("weight_block_size")
            or quantization_config.get("weight_block_shape")
        )
    uses_mla = getattr(model_config, "use_mla", None)
    if not isinstance(uses_mla, bool):
        uses_mla = getattr(text_config, "use_mla", None)
    if not isinstance(uses_mla, bool):
        uses_mla = getattr(text_config, "kv_lora_rank", None) is not None
    is_hybrid = getattr(model_config, "is_hybrid", None)
    if callable(is_hybrid):
        try:
            is_hybrid = bool(is_hybrid())
        except (AttributeError, TypeError, ValueError):
            is_hybrid = None
    if not isinstance(is_hybrid, bool) and gdn_dim is not None:
        is_hybrid = True
    return ModelSignature(
        family=ModelFamily.UNKNOWN,
        role=ModelRole.UNKNOWN,
        architectures=architectures,
        model_type=getattr(text_config, "model_type", None),
        dtype=_normalize_dtype(getattr(model_config, "dtype", None)),
        quantization=_quantization(model_config, text_config),
        hidden_size=_int_attr(text_config, "hidden_size"),
        intermediate_size=_int_attr(text_config, "intermediate_size"),
        num_hidden_layers=_int_attr(text_config, "num_hidden_layers"),
        num_attention_heads=_int_attr(text_config, "num_attention_heads"),
        num_key_value_heads=_int_attr(text_config, "num_key_value_heads"),
        head_dim=_int_attr(text_config, "head_dim"),
        vocab_size=_int_attr(text_config, "vocab_size"),
        num_experts=_int_attr(
            text_config,
            "num_experts",
            "moe_num_experts",
            "n_routed_experts",
            "num_local_experts",
        ),
        num_experts_per_tok=_int_attr(
            text_config,
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k",
        ),
        num_shared_experts=_int_attr(
            text_config,
            "num_shared_experts",
            "n_shared_experts",
        ),
        moe_intermediate_size=_int_attr(text_config, "moe_intermediate_size"),
        expert_dtype=_str_attr(text_config, "expert_dtype"),
        hidden_act=_str_attr(text_config, "hidden_act"),
        swiglu_limit=_float_attr(text_config, "swiglu_limit"),
        gdn_conv_width=gdn_width,
        gdn_conv_dim=gdn_dim,
        has_routed_experts=_has_routed_experts(model_config, text_config),
        enforce_eager=bool(getattr(model_config, "enforce_eager", False)),
        outer_architectures=architectures,
        text_architectures=_text_architectures(text_config),
        outer_model_type=_outer_model_type(model_config, text_config),
        uses_mla=uses_mla,
        index_topk=_int_attr(text_config, "index_topk"),
        quant_block_shape=quant_block_shape,
        is_hybrid=is_hybrid if isinstance(is_hybrid, bool) else None,
    )


def _execution_signature(
    vllm_config: Any | None,
    *,
    is_pooling_model: bool,
) -> ExecutionSignature:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    attention_config = getattr(vllm_config, "attention_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)

    def size(name: str) -> int:
        value = getattr(parallel_config, name, 1)
        return value if isinstance(value, int) and value > 0 else 1

    block_size = getattr(cache_config, "block_size", None)
    try:
        import vllm.envs as vllm_envs

        batch_invariant_enabled = bool(vllm_envs.VLLM_BATCH_INVARIANT)
    except (AttributeError, ImportError):
        batch_invariant_enabled = False

    return ExecutionSignature(
        tensor_parallel_size=size("tensor_parallel_size"),
        pipeline_parallel_size=size("pipeline_parallel_size"),
        data_parallel_size=size("data_parallel_size"),
        decode_context_parallel_size=size("decode_context_parallel_size"),
        has_speculative_config=(
            getattr(vllm_config, "speculative_config", None) is not None
        ),
        has_quant_config=getattr(vllm_config, "quant_config", None) is not None,
        is_pooling_model=is_pooling_model,
        has_parallel_config=parallel_config is not None,
        cache_dtype=_normalize_dtype(getattr(cache_config, "cache_dtype", "auto")),
        cache_block_size=(
            block_size
            if isinstance(block_size, int) and not isinstance(block_size, bool)
            else None
        ),
        max_num_seqs=_int_attr(scheduler_config, "max_num_seqs"),
        attention_backend=_normalize_name(getattr(attention_config, "backend", None)),
        compilation_mode=_normalize_name(getattr(compilation_config, "mode", None)),
        cudagraph_mode=_normalize_name(
            getattr(compilation_config, "cudagraph_mode", None)
        ),
        batch_invariant_enabled=batch_invariant_enabled,
        speculative_method=_normalize_name(getattr(speculative_config, "method", None)),
        async_scheduling=bool(getattr(scheduler_config, "async_scheduling", False)),
    )


def resolve_runtime_plan(
    vllm_config: Any | None = None,
    *,
    model_config: Any | None = None,
    is_pooling_model: bool = False,
    attention_backend_hint: str | None = None,
) -> RuntimePlan:
    transport = _runtime_plan_transport(vllm_config)
    # External plans are allowed to apply only their declared, immutable
    # defaults, but those defaults must be visible while the host context is
    # collected.  Resolving signatures first would fingerprint stale compile,
    # cache, or scheduler settings and could select the wrong built-in
    # variant.
    # A finalized config carries the exact validated projection needed by
    # spawned workers.  Do not rediscover a plugin or reopen its artifact from
    # inherited environment/path state once that transport is present.
    if vllm_config is not None and transport is None:
        from vllm_musa.engine_plugins import apply_engine_plugin_defaults

        apply_engine_plugin_defaults(vllm_config)

    if model_config is None:
        model_config = getattr(vllm_config, "model_config", None)
    execution = _execution_signature(
        vllm_config,
        is_pooling_model=is_pooling_model,
    )
    if execution.attention_backend is None and attention_backend_hint is not None:
        execution = replace(
            execution,
            attention_backend=_normalize_name(attention_backend_hint),
        )
    if model_config is None:
        model = ModelSignature(
            family=ModelFamily.UNKNOWN,
            role=ModelRole.UNKNOWN,
            architectures=(),
            model_type=None,
            dtype=None,
            quantization=None,
            hidden_size=None,
            intermediate_size=None,
            num_hidden_layers=None,
            num_attention_heads=None,
            num_key_value_heads=None,
            head_dim=None,
            vocab_size=None,
            num_experts=None,
            num_experts_per_tok=None,
            num_shared_experts=None,
            moe_intermediate_size=None,
            expert_dtype=None,
            hidden_act=None,
            swiglu_limit=None,
            gdn_conv_width=None,
            gdn_conv_dim=None,
            has_routed_experts=False,
            enforce_eager=False,
            outer_architectures=(),
            text_architectures=(),
            outer_model_type=None,
            uses_mla=None,
            index_topk=None,
            quant_block_shape=None,
            is_hybrid=None,
        )
    else:
        model = _model_signature(model_config, vllm_config)

    cache: dict[_PlanCacheKey, RuntimePlan] | None = None
    cache_key: _PlanCacheKey | None = None
    if vllm_config is not None:
        decision_context = _decision_context_key(vllm_config, transport)
        # A no-plugin plan is cheap and must not be retained across a later
        # general-plugin discovery/reset.  Once an immutable external
        # application exists, weak per-config caching guarantees that all
        # consumers observe the same materialized decision object.
        cache = _plan_cache_for(vllm_config) if decision_context is not None else None
        cache_key = _PlanCacheKey(
            model=model,
            execution=execution,
            is_pooling_model=is_pooling_model,
            attention_backend_hint=_normalize_name(attention_backend_hint),
            decision_context=decision_context,
        )
        if cache is not None:
            cached_plan = cache.get(cache_key)
            if cached_plan is not None:
                return cached_plan

    for provider in BUILTIN_RUNTIME_PLAN_PROVIDERS:
        plan = provider(model, execution)
        if plan is not None:
            break
    else:
        plan = RuntimePlan(
            model=model,
            execution=execution,
            profile="unknown",
            supported_decisions=frozenset(),
            decision_values=(),
            reason="no built-in runtime plan matched the model profile",
        )

    if model.is_hybrid is True:
        decision = RuntimeDecision.HYBRID_KV_CACHE_POOL_LAYOUT
        values = dict(plan.decision_values)
        values[decision] = "separate"
        plan = replace(
            plan,
            supported_decisions=plan.supported_decisions | {decision},
            decision_values=tuple(
                sorted(values.items(), key=lambda item: item[0].value)
            ),
        )
    resolved = _apply_engine_runtime_decisions(
        vllm_config,
        plan,
        transport=transport,
    )
    if cache is not None and cache_key is not None:
        with _PLAN_CACHE_LOCK:
            # Do not overwrite a concurrent resolution with a plan carrying a
            # different receipt.  Both plans are valid for their immutable
            # cache key; the first one is the canonical object returned later.
            cache.setdefault(cache_key, resolved)
            return cache[cache_key]
    return resolved


def _runtime_decision_key(value: str) -> RuntimeDecision:
    try:
        return RuntimeDecision(value)
    except ValueError as exc:
        raise RuntimeDecisionError(
            f"Engine plan names unknown runtime decision {value!r}"
        ) from exc


def _apply_engine_runtime_decisions(
    vllm_config: Any | None,
    plan: RuntimePlan,
    *,
    transport: _RuntimePlanTransport | None = None,
) -> RuntimePlan:
    if vllm_config is None:
        return plan

    resolution: RuntimeDecisionResolution | None
    if transport is not None:
        resolution = transport.resolution
        # The parent may resolve again after finalization.  If its local plugin
        # state still exists, require it to agree exactly with the published
        # projection.  Spawned workers intentionally have no such state and do
        # not probe the registry or inherited plan path.
        from vllm_musa.engine_plugins import get_engine_plugin_application

        if get_engine_plugin_application(vllm_config) is not None:
            local_resolution = _engine_runtime_decision_resolution(vllm_config)
            if local_resolution != resolution:
                raise RuntimeDecisionError(
                    "Serialized RuntimePlan transport conflicts with the "
                    "process-local engine plugin projection"
                )
    else:
        resolution = _engine_runtime_decision_resolution(vllm_config)
        if resolution is None:
            return plan

    if (
        transport is not None
        and canonical_fingerprint(plan.model)
        != transport.model_signature_fingerprint
    ):
        raise RuntimeDecisionError(
            "Serialized RuntimePlan transport does not match the live model "
            "signature"
        )
    return _apply_runtime_decision_resolution(
        plan,
        resolution,
        strict_profile_config=transport is not None,
    )


def _engine_runtime_decision_resolution(
    vllm_config: Any,
) -> RuntimeDecisionResolution | None:
    from vllm_musa.engine_plugins import resolve_engine_plugin_runtime_decisions

    receipt = resolve_engine_plugin_runtime_decisions(vllm_config)
    if receipt is None:
        return None
    try:
        return RuntimeDecisionResolution(
            source="engine_plugin",
            plugin_name=receipt.plugin_name,
            plugin_version=receipt.plugin_version,
            plan_id=receipt.plan_id,
            fingerprint=receipt.plan_fingerprint,
            profile=receipt.profile,
            decisions=tuple(receipt.decisions),
            profile_config_id=receipt.profile_config_id,
            profile_config_fingerprint=receipt.profile_config_fingerprint,
        )
    except Exception as exc:
        raise RuntimeDecisionError(
            "Engine plugin returned malformed runtime decision resolution: " f"{exc}"
        ) from exc


def _apply_runtime_decision_resolution(
    plan: RuntimePlan,
    resolution: RuntimeDecisionResolution,
    *,
    strict_profile_config: bool = False,
) -> RuntimePlan:
    if resolution.profile != plan.profile:
        raise RuntimeDecisionError(
            f"Engine plan decisions target profile {resolution.profile!r}, "
            f"but the live runtime plan resolved {plan.profile!r}"
        )
    receipt_has_profile_config = resolution.profile_config_id is not None or (
        resolution.profile_config_fingerprint is not None
    )
    if (strict_profile_config or receipt_has_profile_config) and (
        resolution.profile_config_id != plan.profile_config_id
        or resolution.profile_config_fingerprint != plan.profile_config_fingerprint
    ):
        raise RuntimeDecisionError(
            "Engine plan profile-config identity does not match the live "
            f"RuntimePlan: artifact=({resolution.profile_config_id!r}, "
            f"{resolution.profile_config_fingerprint!r}), "
            f"live=({plan.profile_config_id!r}, "
            f"{plan.profile_config_fingerprint!r})"
        )

    from .catalog import (
        RuntimeDecisionTunability,
        runtime_decision_spec,
        validate_runtime_decision_value,
    )

    effective = dict(plan.decision_values)
    supported = set(plan.supported_decisions)
    for raw_key, value in resolution.decisions:
        decision = _runtime_decision_key(raw_key)
        spec = runtime_decision_spec(decision)
        if not spec.external_only and not receipt_has_profile_config:
            raise RuntimeDecisionError(
                f"Engine plan decision {decision.value!r} requires an exact "
                "profile-config identity"
            )
        try:
            value = validate_runtime_decision_value(decision, value)
        except (TypeError, ValueError) as exc:
            raise RuntimeDecisionError(
                f"Engine plan supplied invalid value for {decision.value!r}: {exc}"
            ) from exc
        if not spec.supports_profile(plan.profile):
            if value == spec.fallback:
                continue
            raise RuntimeDecisionError(
                f"Engine plan decision {decision.value!r} is not supported "
                f"by profile {plan.profile!r}"
            )
        if decision not in supported:
            if spec.external_only:
                supported.add(decision)
            elif value == spec.fallback:
                # A safe fallback may target a path absent from an early or
                # conservative profile. It remains visible in the receipt but
                # does not manufacture a model/kernel capability.
                continue
            else:
                raise RuntimeDecisionError(
                    f"Engine plan decision {decision.value!r} is not supported "
                    f"by profile {plan.profile!r}"
                )
        if spec.tunability is RuntimeDecisionTunability.FIXED and value != plan.value(
            decision
        ):
            raise RuntimeDecisionError(
                f"Engine plan cannot override fixed decision {decision.value!r}: "
                f"profile={plan.value(decision)!r}, requested={value!r}"
            )
        effective[decision] = value

    return replace(
        plan,
        supported_decisions=frozenset(supported),
        decision_values=tuple(
            sorted(effective.items(), key=lambda item: item[0].value)
        ),
        decision_resolution=resolution,
    )


def _json_transport_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_transport_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_transport_value(item) for item in value]
    return value


def _runtime_plan_transport_document(plan: RuntimePlan) -> dict[str, object]:
    resolution = plan.decision_resolution
    if resolution is None:
        raise RuntimeDecisionError(
            "Cannot publish RuntimePlan transport without an engine-plan "
            "decision resolution"
        )
    if resolution.source != "engine_plugin":
        raise RuntimeDecisionError(
            "Cannot publish RuntimePlan transport from resolution source "
            f"{resolution.source!r}"
        )
    if not resolution.plugin_name or not resolution.plugin_version:
        raise RuntimeDecisionError(
            "Cannot publish RuntimePlan transport without exact plugin identity"
        )
    resolution_document = _transport_resolution_document(resolution)
    projection_fingerprint = _decision_projection_fingerprint(resolution)
    return {
        "schema_version": _RUNTIME_PLAN_TRANSPORT_SCHEMA_VERSION,
        "decision_projection_fingerprint": projection_fingerprint,
        "model_signature_fingerprint": canonical_fingerprint(plan.model),
        "resolution": resolution_document,
    }


def publish_runtime_plan_transport(
    vllm_config: Any,
    plan: RuntimePlan | None = None,
) -> dict[str, object] | None:
    """Publish one validated projection into the spawn-safe config channel.

    This function is intentionally idempotent.  The transported decision
    projection is immutable; process-local execution defaults are deliberately
    re-derived from each worker's serialized ``VllmConfig``.
    """

    if plan is None:
        plan = resolve_runtime_plan(vllm_config)
    if plan.decision_resolution is None:
        return None
    document = _runtime_plan_transport_document(plan)
    expected = _RuntimePlanTransport(
        decision_projection_fingerprint=_decision_projection_fingerprint(
            plan.decision_resolution
        ),
        model_signature_fingerprint=canonical_fingerprint(plan.model),
        resolution=plan.decision_resolution,
    )
    additional_config = getattr(vllm_config, "additional_config", None)
    if additional_config is None:
        additional_config = {}
        try:
            setattr(vllm_config, "additional_config", additional_config)
        except (AttributeError, TypeError) as exc:
            raise RuntimeDecisionError(
                "VllmConfig cannot carry the reserved RuntimePlan transport"
            ) from exc
    if not isinstance(additional_config, dict):
        raise RuntimeDecisionError(
            "VllmConfig.additional_config must be a dict when an engine plan "
            "publishes worker RuntimePlan transport"
        )
    if RUNTIME_PLAN_TRANSPORT_KEY in additional_config:
        existing = _runtime_plan_transport(vllm_config)
        if existing is None:
            raise RuntimeDecisionError(
                "Refusing to replace malformed serialized RuntimePlan transport"
            )
        if existing != expected:
            raise RuntimeDecisionError(
                "Refusing to overwrite conflicting serialized RuntimePlan identity"
            )
        return additional_config[RUNTIME_PLAN_TRANSPORT_KEY]
    additional_config[RUNTIME_PLAN_TRANSPORT_KEY] = document
    return document


def runtime_plan_enabled(owner: Any, decision: RuntimeDecision) -> bool:
    plan = getattr(owner, "_musa_runtime_plan", None)
    return plan is not None and plan.enabled(decision)


def bind_runtime_plan(
    owner: Any,
    vllm_config: Any | None = None,
    *,
    model_config: Any | None = None,
    is_pooling_model: bool = False,
) -> RuntimePlan:
    """Resolve once and bind the immutable plan to a runtime owner."""

    existing = getattr(owner, "_musa_runtime_plan", None)
    if existing is not None:
        if not isinstance(existing, RuntimePlan):
            raise RuntimeDecisionError(
                "Runtime owner already contains an invalid _musa_runtime_plan"
            )
        return existing

    plan = resolve_runtime_plan(
        vllm_config,
        model_config=model_config,
        is_pooling_model=is_pooling_model,
    )
    try:
        setattr(owner, "_musa_runtime_plan", plan)
    except (AttributeError, TypeError) as exc:
        raise RuntimeDecisionError(
            f"Runtime owner {type(owner).__name__} cannot bind _musa_runtime_plan"
        ) from exc
    return plan
