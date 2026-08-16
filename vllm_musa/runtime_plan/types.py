from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias


def _canonical_value(value: object) -> object:
    """Convert an immutable plan value into deterministic JSON data.

    ``RuntimePlan`` is imported by platform and model code during startup.  It
    therefore must not depend on the optional engine-plugin implementation just
    to calculate a fingerprint.  Keeping this tiny serializer local also makes
    the fingerprint independent of ``repr``/module paths for dataclass values.
    """

    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if dataclasses.is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        items = (
            (_canonical_value(key), _canonical_value(item))
            for key, item in value.items()
        )
        # JSON object keys are strings in the serialized representation.  Plan
        # mappings use enum/string keys, so normalize them explicitly and sort
        # to make equivalent mappings hash identically.
        return {
            str(key): item for key, item in sorted(items, key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Runtime-plan fingerprints cannot contain non-finite floats"
            )
        return value
    raise TypeError(
        "Runtime-plan fingerprint received unsupported value type "
        f"{type(value).__name__}"
    )


def canonical_fingerprint(value: object) -> str:
    """Return the stable SHA-256 fingerprint used by runtime-plan receipts."""

    payload = json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class ModelFamily(str, Enum):
    UNKNOWN = "unknown"
    DEEPSEEK_V4 = "deepseek_v4"
    QWEN2 = "qwen2"
    QWEN3 = "qwen3"
    QWEN35_36 = "qwen3.5_3.6"


class ModelRole(str, Enum):
    UNKNOWN = "unknown"
    TEXT = "text_generation"
    MTP_DRAFT = "mtp_draft"
    COSYVOICE_TALKER = "cosyvoice_talker"


class RuntimeDecision(str, Enum):
    """Stable keys consumed through :class:`RuntimePlan` only.

    The first migration keeps the historical boolean keys stable.  The plan
    value model is intentionally typed so mutually exclusive and numeric
    choices can replace those booleans without introducing another resolver.
    """

    DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8 = "deepseek_v4.shared_mlp_clamp_fp8"
    DEEPSEEK_V4_NATIVE_SPARSE_INDEXER = "deepseek_v4.native_sparse_indexer"
    DEEPSEEK_V4_MATERIALIZED_PREFILL_INDEXER = (
        "deepseek_v4.materialized_prefill_indexer"
    )
    DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE = "deepseek_v4.flashmla_sparse_page_size"
    DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT = "deepseek_v4.tp8_mtp_sparse_direct_out"
    DEEPSEEK_V4_TP8_MTP_SPARSE_PREFILL_HEADROOM = (
        "deepseek_v4.tp8_mtp_sparse_prefill_headroom"
    )
    DEEPSEEK_V4_TP8_MTP_ASYNC_PREFILL_QUEUE_FENCE = (
        "deepseek_v4.tp8_mtp_async_prefill_queue_fence"
    )
    DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256 = (
        "deepseek_v4.tp8_fused_add_rmsnorm_block256"
    )
    DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD = (
        "deepseek_v4.car_graph_input_capture_guard"
    )
    DEEPSEEK_V4_MTP_CAR_GRAPH_REGISTERED_INPUTS = (
        "deepseek_v4.mtp_car_graph_registered_inputs"
    )
    DEEPSEEK_V4_MTP_CAR_GRAPH_STAGING_ARENA = "deepseek_v4.mtp_car_graph_staging_arena"
    QWEN_V2_SAMPLING = "qwen.v2_sampling"
    QWEN_LEGACY_SAMPLING = "qwen.legacy_sampling"
    QWEN_FA3_SCHEDULER = "qwen.fa3_scheduler"
    QWEN_FA3_SINGLE_REQUEST_METADATA = "qwen.fa3_single_request_metadata"
    QWEN2_ROPE_KV_PRESPLIT = "qwen2.rope_kv_presplit"
    QWEN3_QK_ROPE_KV_PRESPLIT = "qwen3.qk_rope_kv_presplit"
    QWEN3_DENSE_FP8_POST_GRAD_FUSIONS = "qwen3.dense_fp8_post_grad_fusions"
    QWEN35_GDN_WIDTH4_PREFILL = "qwen3.5_3.6.gdn_width4_prefill"
    QWEN35_MOE_BF16_PREFILL = "qwen3.5_3.6.moe_bf16_prefill"
    QWEN_UNIFORM_DECODE_VIEWS = "qwen.uniform_decode_views"
    QWEN_UNIFORM_SAMPLE_COUNTS = "qwen.uniform_sample_counts"
    QWEN_SAMPLE_INPUT_VIEWS = "qwen.sample_input_views"
    QWEN_LEGACY_GUMBEL = "qwen.legacy_gumbel"
    QWEN_V2_GUMBEL = "qwen.v2_gumbel"
    QWEN_TP_LOGITS_IPC_GATHER = "qwen.tp_logits_ipc_gather"
    QWEN_TP4_SHARDED_GUMBEL = "qwen.tp4_sharded_gumbel"
    QWEN35_SHARED_EXPERT_FOLD = "qwen3.5_3.6.shared_expert_fold"
    QWEN35_INTERLEAVED_MROPE_QK = "qwen3.5_3.6.interleaved_mrope_qk"
    HYBRID_KV_CACHE_POOL_LAYOUT = "hybrid.kv_cache_pool_layout"
    MUSA_FUSED_ADD_RMSNORM_MIN_ROWS = "musa.fused_add_rms_norm.min_rows"
    MUSA_FUSED_MOE_DISPATCH_POLICY = "musa.fused_moe.dispatch_policy"
    # The host IR provider order is a typed runtime decision too.  Keeping it
    # in the same keyspace lets an external plan project it through the exact
    # same receipt/fingerprint path as model-specific choices.
    VLLM_IR_OP_PRIORITY = "vllm.ir_op_priority"


RuntimeDecisionScalar: TypeAlias = bool | int | float | str
# Values are recursively immutable JSON-like tuples.  The public alias is
# intentionally broad because a plan may carry a structured resource choice;
# ``_validate_decision_value`` below enforces the actual runtime boundary.
RuntimeDecisionValue: TypeAlias = RuntimeDecisionScalar | tuple[object, ...]


def _validate_decision_value(value: object, field: str) -> None:
    if isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_decision_value(item, f"{field}[{index}]")
        return
    raise TypeError(
        f"{field} must be an immutable bool/int/float/string or tuple value; "
        f"got {type(value).__name__}"
    )


def _decision_key(value: RuntimeDecision | str, field: str) -> RuntimeDecision:
    if isinstance(value, RuntimeDecision):
        return value
    if isinstance(value, str):
        try:
            return RuntimeDecision(value)
        except ValueError as exc:
            raise ValueError(
                f"{field} names unknown runtime decision {value!r}"
            ) from exc
    raise TypeError(f"{field} must be a RuntimeDecision or string")


@dataclass(frozen=True, slots=True)
class RuntimeDecisionResolution:
    source: str
    plugin_name: str | None
    plugin_version: str | None
    plan_id: str
    fingerprint: str
    profile: str
    decisions: tuple[tuple[str, RuntimeDecisionValue], ...]
    profile_config_id: str | None = None
    profile_config_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Runtime-decision resolution source must be non-empty")
        for field_name in ("plugin_name", "plugin_version"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(
                    f"Runtime-decision resolution {field_name} must be a "
                    "non-empty string or None"
                )
        if not isinstance(self.plan_id, str) or not self.plan_id:
            raise ValueError("Runtime-decision resolution plan_id must be non-empty")
        if not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise ValueError(
                "Runtime-decision resolution fingerprint must be non-empty"
            )
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("Runtime-decision resolution profile must be non-empty")
        if (self.profile_config_id is None) != (
            self.profile_config_fingerprint is None
        ):
            raise ValueError(
                "Runtime-decision resolution profile-config identity fields "
                "must be set together"
            )
        if self.profile_config_id is not None and (
            not isinstance(self.profile_config_id, str) or not self.profile_config_id
        ):
            raise ValueError(
                "Runtime-decision resolution profile_config_id must be non-empty"
            )
        if self.profile_config_fingerprint is not None and (
            not isinstance(self.profile_config_fingerprint, str)
            or not self.profile_config_fingerprint.startswith("sha256:")
        ):
            raise ValueError(
                "Runtime-decision resolution profile_config_fingerprint must use "
                "the sha256: prefix"
            )
        normalized: list[tuple[str, RuntimeDecisionValue]] = []
        seen: set[str] = set()
        for item in self.decisions:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("Runtime-decision entries must be (key, value) tuples")
            key, value = item
            if not isinstance(key, str) or not key:
                raise ValueError("Runtime-decision keys must be non-empty strings")
            if key in seen:
                raise ValueError(f"Duplicate runtime decision: {key}")
            _validate_decision_value(value, f"decisions.{key}")
            decision = _decision_key(key, f"decisions.{key}")
            from .catalog import validate_runtime_decision_value

            validate_runtime_decision_value(decision, value)
            seen.add(key)
            normalized.append((key, value))
        object.__setattr__(
            self,
            "decisions",
            tuple(sorted(normalized, key=lambda item: item[0])),
        )

    @property
    def plan_fingerprint(self) -> str:
        """Alias matching the engine-plugin receipt terminology."""

        return self.fingerprint

    @property
    def enabled_decisions(self) -> frozenset[RuntimeDecision]:
        return frozenset(
            RuntimeDecision(key)
            for key, value in self.decisions
            if value is True and key in RuntimeDecision._value2member_map_
        )

    @property
    def disabled_decisions(self) -> frozenset[RuntimeDecision]:
        return frozenset(
            RuntimeDecision(key)
            for key, value in self.decisions
            if value is False and key in RuntimeDecision._value2member_map_
        )

    def as_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "source": self.source,
            "plugin_name": self.plugin_name,
            "plugin_version": self.plugin_version,
            "plan_id": self.plan_id,
            "fingerprint": self.fingerprint,
            "profile": self.profile,
            "decisions": dict(self.decisions),
        }
        if self.profile_config_id is not None:
            document["profile_config_id"] = self.profile_config_id
            document["profile_config_fingerprint"] = self.profile_config_fingerprint
        return document


class RuntimeDecisionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelSignature:
    family: ModelFamily
    role: ModelRole
    architectures: tuple[str, ...]
    model_type: str | None
    dtype: str | None
    quantization: str | None
    hidden_size: int | None
    intermediate_size: int | None
    num_hidden_layers: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    vocab_size: int | None
    num_experts: int | None
    num_experts_per_tok: int | None
    num_shared_experts: int | None
    moe_intermediate_size: int | None
    expert_dtype: str | None
    hidden_act: str | None
    swiglu_limit: float | None
    gdn_conv_width: int | None
    gdn_conv_dim: int | None
    has_routed_experts: bool | None
    enforce_eager: bool
    index_topk: int | None
    outer_architectures: tuple[str, ...] = ()
    text_architectures: tuple[str, ...] = ()
    outer_model_type: str | None = None
    uses_mla: bool | None = None
    quant_block_shape: tuple[int, ...] | None = None
    is_hybrid: bool | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "architectures",
            "outer_architectures",
            "text_architectures",
            "quant_block_shape",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, tuple):
                try:
                    value = tuple(value)
                except TypeError as exc:
                    raise TypeError(
                        f"Model signature {field_name} must be tuple-like"
                    ) from exc
                object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class ExecutionSignature:
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    decode_context_parallel_size: int
    has_speculative_config: bool
    has_quant_config: bool
    is_pooling_model: bool
    has_parallel_config: bool
    cache_dtype: str | None
    cache_block_size: int | None
    max_num_seqs: int | None
    attention_backend: str | None = None
    compilation_mode: str | None = None
    cudagraph_mode: str | None = None
    batch_invariant_enabled: bool = False
    speculative_method: str | None = None
    async_scheduling: bool = False


def boolean_decisions(
    enabled: set[RuntimeDecision] | frozenset[RuntimeDecision],
) -> tuple[tuple[RuntimeDecision, RuntimeDecisionValue], ...]:
    return tuple((key, True) for key in sorted(enabled, key=lambda item: item.value))


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    """Immutable, single-source runtime decision application."""

    model: ModelSignature
    execution: ExecutionSignature
    profile: str
    supported_decisions: frozenset[RuntimeDecision]
    decision_values: tuple[tuple[RuntimeDecision, RuntimeDecisionValue], ...]
    reason: str = ""
    decision_resolution: RuntimeDecisionResolution | None = None
    profile_config_id: str | None = None
    profile_config_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelSignature):
            raise TypeError("Runtime-plan model must be a ModelSignature")
        if not isinstance(self.execution, ExecutionSignature):
            raise TypeError("Runtime-plan execution must be an ExecutionSignature")
        supported = frozenset(
            _decision_key(key, "supported_decisions")
            for key in self.supported_decisions
        )
        normalized: list[tuple[RuntimeDecision, RuntimeDecisionValue]] = []
        seen: set[RuntimeDecision] = set()
        for item in self.decision_values:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "Runtime-plan decisions must be (RuntimeDecision, value) tuples"
                )
            raw_key, value = item
            key = _decision_key(raw_key, "decision")
            if key in seen:
                raise ValueError(f"Duplicate runtime decision: {key.value}")
            seen.add(key)
            _validate_decision_value(value, f"decision {key.value}")
            from .catalog import validate_runtime_decision_value

            validate_runtime_decision_value(key, value)
            if key not in supported:
                raise ValueError(
                    f"Runtime decision is not supported by profile "
                    f"{self.profile!r}: {key.value}"
                )
            normalized.append((key, value))
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("Runtime-plan profile must be non-empty")
        if not isinstance(self.reason, str):
            raise TypeError("Runtime-plan reason must be a string")
        if self.decision_resolution is not None and not isinstance(
            self.decision_resolution, RuntimeDecisionResolution
        ):
            raise TypeError(
                "Runtime-plan decision_resolution must be a "
                "RuntimeDecisionResolution or None"
            )
        if self.profile_config_id is not None and (
            not isinstance(self.profile_config_id, str) or not self.profile_config_id
        ):
            raise TypeError("profile_config_id must be a non-empty string or None")
        if (self.profile_config_id is None) != (
            self.profile_config_fingerprint is None
        ):
            raise ValueError(
                "profile_config_id and profile_config_fingerprint must be set together"
            )
        if self.profile_config_fingerprint is not None:
            if not isinstance(
                self.profile_config_fingerprint, str
            ) or not self.profile_config_fingerprint.startswith("sha256:"):
                raise TypeError(
                    "profile_config_fingerprint must use the sha256: prefix or be None"
                )
        object.__setattr__(self, "supported_decisions", supported)
        object.__setattr__(
            self,
            "decision_values",
            tuple(sorted(normalized, key=lambda item: item[0].value)),
        )

    @property
    def decisions(self) -> MappingProxyType:
        return MappingProxyType(dict(self.decision_values))

    @property
    def enabled_decisions(self) -> frozenset[RuntimeDecision]:
        return frozenset(
            key
            for key, value in self.decision_values
            if isinstance(value, bool) and value
        )

    @property
    def disabled_decisions(self) -> frozenset[RuntimeDecision]:
        from .catalog import RuntimeDecisionKind, runtime_decision_spec

        return frozenset(
            key
            for key in self.supported_decisions
            if runtime_decision_spec(key).kind is RuntimeDecisionKind.BOOLEAN
            and self.value(key) is False
        )

    def supports(self, decision: RuntimeDecision) -> bool:
        return decision in self.supported_decisions

    def value(
        self,
        decision: RuntimeDecision,
        default: RuntimeDecisionValue | None = None,
    ) -> RuntimeDecisionValue | None:
        if decision not in self.supported_decisions:
            return default
        from .catalog import runtime_decision_spec

        return self.decisions.get(decision, runtime_decision_spec(decision).fallback)

    def selected(
        self,
        decision: RuntimeDecision,
        choice: RuntimeDecisionValue,
    ) -> bool:
        """Return whether a typed decision selected one exact choice."""

        return self.value(decision) == choice

    def enabled(self, decision: RuntimeDecision) -> bool:
        from .catalog import RuntimeDecisionKind, runtime_decision_spec

        if runtime_decision_spec(decision).kind is not RuntimeDecisionKind.BOOLEAN:
            raise TypeError(
                f"Runtime decision {decision.value!r} is not boolean; "
                "use value() or selected()"
            )
        return self.value(decision, False) is True

    def decision_source(self, decision: RuntimeDecision) -> str:
        """Return the immutable source of one effective decision value."""

        if (
            decision in self.supported_decisions
            and self.decision_resolution is not None
            and any(
                raw_key == decision.value
                for raw_key, _ in self.decision_resolution.decisions
            )
        ):
            return "engine_plan"
        if decision in dict(self.decision_values):
            return (
                "profile_default"
                if self.profile_config_id is not None
                else "provider_default"
            )
        return "catalog_fallback"

    @property
    def decision_sources(self) -> tuple[tuple[RuntimeDecision, str], ...]:
        return tuple(
            (decision, self.decision_source(decision))
            for decision in sorted(
                self.supported_decisions,
                key=lambda item: item.value,
            )
        )

    @property
    def fingerprint(self) -> str:
        document: dict[str, object] = {
            "model": self.model,
            "execution": self.execution,
            "profile": self.profile,
            "supported_decisions": sorted(
                decision.value for decision in self.supported_decisions
            ),
            "decisions": {
                decision.value: value for decision, value in self.decision_values
            },
            "reason": self.reason,
            "decision_resolution": (
                self.decision_resolution.as_dict()
                if self.decision_resolution is not None
                else None
            ),
        }
        if self.profile_config_id is not None:
            document["profile_config_id"] = self.profile_config_id
            document["profile_config_fingerprint"] = self.profile_config_fingerprint
        return canonical_fingerprint(document)

    @classmethod
    def unknown(cls) -> RuntimePlan:
        return cls(
            model=ModelSignature(
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
                has_routed_experts=None,
                enforce_eager=False,
                index_topk=None,
            ),
            execution=ExecutionSignature(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                decode_context_parallel_size=1,
                has_speculative_config=False,
                has_quant_config=False,
                is_pooling_model=False,
                has_parallel_config=False,
                cache_dtype=None,
                cache_block_size=None,
                max_num_seqs=None,
                attention_backend=None,
                compilation_mode=None,
                cudagraph_mode=None,
                batch_invariant_enabled=False,
                speculative_method=None,
                async_scheduling=False,
            ),
            profile="unknown",
            supported_decisions=frozenset(),
            decision_values=(),
            reason="model configuration was not available",
        )
