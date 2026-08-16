from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from .types import RuntimeDecision, RuntimeDecisionValue


class RuntimePlanPhase(str, Enum):
    """Latest lifecycle phase at which a decision may be materialized."""

    CONFIG_DEFAULTS = "config_defaults"
    CACHE_LAYOUT = "cache_layout"
    MODEL_INIT = "model_init"
    COMPILE = "compile"
    GRAPH_CAPTURE = "graph_capture"
    REQUEST = "request"


class RuntimeDecisionKind(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    ENUM = "enum"
    STRUCTURED = "structured"


class RuntimeDecisionTunability(str, Enum):
    """How a registered decision is governed after support is established."""

    FIXED = "fixed"
    PROFILE = "profile"
    AUTOTUNE = "autotune"


@dataclass(frozen=True, slots=True)
class RuntimeDecisionSpec:
    """Host-owned schema for one plan decision.

    A plan artifact may select only registered keys and values accepted here.
    Legality of an individual kernel invocation remains in the implementation's
    own support predicate; this catalog owns configuration/plan semantics.
    """

    key: RuntimeDecision
    kind: RuntimeDecisionKind
    phase: RuntimePlanPhase
    fallback: RuntimeDecisionValue
    choices: tuple[RuntimeDecisionValue, ...] = ()
    external_only: bool = False
    profile_families: tuple[str, ...] = ()
    tunability: RuntimeDecisionTunability = RuntimeDecisionTunability.PROFILE

    def validate(self, value: object) -> RuntimeDecisionValue:
        field = self.key.value
        if self.kind is RuntimeDecisionKind.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"Runtime decision {field!r} must be boolean")
        elif self.kind is RuntimeDecisionKind.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Runtime decision {field!r} must be an integer")
            if (
                self.key is RuntimeDecision.MUSA_FUSED_ADD_RMSNORM_MIN_ROWS
                and value <= 0
            ):
                raise ValueError(
                    f"Runtime decision {field!r} must be a positive integer"
                )
        elif self.kind is RuntimeDecisionKind.ENUM:
            if not isinstance(value, str):
                raise ValueError(f"Runtime decision {field!r} must be a string choice")
        elif self.kind is RuntimeDecisionKind.STRUCTURED:
            if self.key is RuntimeDecision.VLLM_IR_OP_PRIORITY:
                _validate_ir_op_priority(value)
            elif self.key is RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY:
                _validate_fused_moe_dispatch_policy(value)
            else:  # pragma: no cover - every structured key needs an explicit codec
                raise ValueError(
                    f"Runtime decision {field!r} has no structured value codec"
                )
        if self.choices and value not in self.choices:
            raise ValueError(
                f"Runtime decision {field!r} must be one of {self.choices!r}; "
                f"got {value!r}"
            )
        return value  # type: ignore[return-value]

    def supports_profile(self, profile: str) -> bool:
        return (
            not self.profile_families
            or _profile_family(profile) in self.profile_families
        )


def _validate_ir_op_priority(value: object) -> None:
    """Validate the immutable tuple projection of ``{op: [providers...]}``."""

    if not isinstance(value, tuple) or not value:
        raise ValueError("vllm.ir_op_priority must be a non-empty tuple mapping")
    operations: set[str] = set()
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "vllm.ir_op_priority entries must be (operation, providers) tuples"
            )
        operation, providers = item
        if not isinstance(operation, str) or not operation:
            raise ValueError("vllm.ir_op_priority operation names must be non-empty")
        if operation in operations:
            raise ValueError(
                f"vllm.ir_op_priority contains duplicate operation {operation!r}"
            )
        operations.add(operation)
        if not isinstance(providers, tuple) or not providers:
            raise ValueError(
                f"vllm.ir_op_priority providers for {operation!r} must be a tuple"
            )
        if any(not isinstance(provider, str) or not provider for provider in providers):
            raise ValueError(
                f"vllm.ir_op_priority providers for {operation!r} must be non-empty"
            )
        if len(providers) != len(set(providers)):
            raise ValueError(
                f"vllm.ir_op_priority providers for {operation!r} contain duplicates"
            )
        if providers[-1] != "native":
            raise ValueError(
                f"vllm.ir_op_priority for {operation!r} must retain native fallback"
            )


_FUSED_MOE_POLICY_SCHEMA = "musa.fused_moe.dispatch_policy.v1"
_FUSED_MOE_BACKENDS = frozenset({"gemv", "grouped_gemm", "upstream"})
_FUSED_MOE_SHAPE_FIELDS = frozenset(
    {
        "device_capability",
        "multiprocessor_count",
        "local_experts",
        "w1_output_size",
        "w2_input_size",
        "hidden_size",
        "top_k",
        "block_n",
        "block_k",
        "activation",
        "expert_parallel",
        "hidden_dtype",
        "weight_dtype",
        "scale_dtype",
        "w1_scale_shape",
        "w2_scale_shape",
        "gemv_block",
        "graph_mode",
    }
)


def _tuple_mapping(
    value: object,
    *,
    field: str,
    exact_keys: frozenset[str],
) -> dict[str, object]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field} must be an immutable tuple mapping")
    result: dict[str, object] = {}
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
        ):
            raise ValueError(f"{field} entries must be (key, value) tuples")
        key, item_value = item
        if key in result:
            raise ValueError(f"{field} contains duplicate key {key!r}")
        result[key] = item_value
    if result.keys() != exact_keys:
        missing = sorted(exact_keys - result.keys())
        extra = sorted(result.keys() - exact_keys)
        raise ValueError(
            f"{field} must contain exactly {sorted(exact_keys)!r}; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return result


def _positive_integer(value: object, field: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_fused_moe_shape(value: object, field: str) -> dict[str, object]:
    shape = _tuple_mapping(
        value,
        field=field,
        exact_keys=_FUSED_MOE_SHAPE_FIELDS,
    )
    capability = shape["device_capability"]
    if (
        not isinstance(capability, tuple)
        or len(capability) != 2
        or any(
            not isinstance(component, int)
            or isinstance(component, bool)
            or component < 0
            for component in capability
        )
    ):
        raise ValueError(f"{field}.device_capability must be two non-negative ints")
    for name in (
        "multiprocessor_count",
        "local_experts",
        "w1_output_size",
        "w2_input_size",
        "hidden_size",
        "top_k",
    ):
        _positive_integer(shape[name], f"{field}.{name}")
    for name in ("block_n", "block_k"):
        _positive_integer(shape[name], f"{field}.{name}", allow_zero=True)
    for name in (
        "activation",
        "hidden_dtype",
        "weight_dtype",
        "scale_dtype",
        "gemv_block",
    ):
        _non_empty_string(shape[name], f"{field}.{name}")
    if not isinstance(shape["expert_parallel"], bool):
        raise ValueError(f"{field}.expert_parallel must be boolean")
    if shape["graph_mode"] not in ("eager", "capture"):
        raise ValueError(f"{field}.graph_mode must be 'eager' or 'capture'")
    for name in ("w1_scale_shape", "w2_scale_shape"):
        dimensions = shape[name]
        if not isinstance(dimensions, tuple) or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            for dimension in dimensions
        ):
            raise ValueError(f"{field}.{name} must be a tuple of positive ints")
    return shape


def _validate_fused_moe_ranges(
    value: object,
    *,
    field: str,
    graph_mode: object,
) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field} must be a non-empty ordered tuple")
    expected_min = 1
    range_fields = frozenset({"min_tokens", "max_tokens", "backend"})
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        token_range = _tuple_mapping(
            item,
            field=item_field,
            exact_keys=range_fields,
        )
        min_tokens = _positive_integer(
            token_range["min_tokens"], f"{item_field}.min_tokens"
        )
        max_tokens = _positive_integer(
            token_range["max_tokens"], f"{item_field}.max_tokens"
        )
        if min_tokens != expected_min:
            if index == 0:
                raise ValueError(f"{field} must start at one token")
            raise ValueError(f"{field} must be continuous and non-overlapping")
        if max_tokens < min_tokens:
            raise ValueError(f"{item_field}.max_tokens must be >= min_tokens")
        backend = token_range["backend"]
        if backend not in _FUSED_MOE_BACKENDS:
            raise ValueError(
                f"{item_field}.backend must be one of "
                f"{sorted(_FUSED_MOE_BACKENDS)!r}"
            )
        if graph_mode == "capture" and backend == "grouped_gemm":
            raise ValueError(
                f"{item_field}.backend grouped_gemm is ineligible during capture"
            )
        expected_min = max_tokens + 1


def _validate_fused_moe_dispatch_policy(value: object) -> None:
    """Validate the immutable projection of the fused-MoE policy v1 schema."""

    if value == ():
        return
    policy = _tuple_mapping(
        value,
        field=RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY.value,
        exact_keys=frozenset({"schema", "entries"}),
    )
    if policy["schema"] != _FUSED_MOE_POLICY_SCHEMA:
        raise ValueError(
            "musa.fused_moe.dispatch_policy.schema must be "
            f"{_FUSED_MOE_POLICY_SCHEMA!r}"
        )
    entries = policy["entries"]
    if not isinstance(entries, tuple) or not entries:
        raise ValueError("musa.fused_moe.dispatch_policy.entries must be non-empty")
    seen_shapes: set[tuple[tuple[str, object], ...]] = set()
    for index, item in enumerate(entries):
        field = f"musa.fused_moe.dispatch_policy.entries[{index}]"
        entry = _tuple_mapping(
            item,
            field=field,
            exact_keys=frozenset({"shape", "ranges"}),
        )
        shape = _validate_fused_moe_shape(entry["shape"], f"{field}.shape")
        shape_key = tuple(sorted(shape.items()))
        if shape_key in seen_shapes:
            raise ValueError(f"{field}.shape duplicates an earlier exact shape")
        seen_shapes.add(shape_key)
        _validate_fused_moe_ranges(
            entry["ranges"],
            field=f"{field}.ranges",
            graph_mode=shape["graph_mode"],
        )


def decode_fused_moe_dispatch_policy(
    value: object,
) -> tuple[
    tuple[dict[str, object], tuple[dict[str, object], ...]],
    ...,
]:
    """Return validated immutable shape/range mappings for runtime materialization."""

    if value == ():
        return ()
    _validate_fused_moe_dispatch_policy(value)
    policy = _tuple_mapping(
        value,
        field=RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY.value,
        exact_keys=frozenset({"schema", "entries"}),
    )
    decoded: list[tuple[dict[str, object], tuple[dict[str, object], ...]]] = []
    for index, item in enumerate(policy["entries"]):
        field = f"musa.fused_moe.dispatch_policy.entries[{index}]"
        entry = _tuple_mapping(
            item,
            field=field,
            exact_keys=frozenset({"shape", "ranges"}),
        )
        shape = _tuple_mapping(
            entry["shape"],
            field=f"{field}.shape",
            exact_keys=_FUSED_MOE_SHAPE_FIELDS,
        )
        ranges = tuple(
            _tuple_mapping(
                raw_range,
                field=f"{field}.ranges[{range_index}]",
                exact_keys=frozenset({"min_tokens", "max_tokens", "backend"}),
            )
            for range_index, raw_range in enumerate(entry["ranges"])
        )
        decoded.append((shape, ranges))
    return tuple(decoded)


_COMPILE_DECISIONS = {
    RuntimeDecision.QWEN2_ROPE_KV_PRESPLIT,
    RuntimeDecision.QWEN3_QK_ROPE_KV_PRESPLIT,
    RuntimeDecision.QWEN3_DENSE_FP8_POST_GRAD_FUSIONS,
}
_GRAPH_DECISIONS = {
    RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD,
    RuntimeDecision.DEEPSEEK_V4_MTP_CAR_GRAPH_REGISTERED_INPUTS,
    RuntimeDecision.DEEPSEEK_V4_MTP_CAR_GRAPH_STAGING_ARENA,
}
_REQUEST_DECISIONS = {
    RuntimeDecision.DEEPSEEK_V4_TP8_MTP_ASYNC_PREFILL_QUEUE_FENCE,
    RuntimeDecision.QWEN_FA3_SCHEDULER,
    RuntimeDecision.QWEN_UNIFORM_DECODE_VIEWS,
    RuntimeDecision.QWEN_UNIFORM_SAMPLE_COUNTS,
    RuntimeDecision.QWEN_SAMPLE_INPUT_VIEWS,
    RuntimeDecision.QWEN_LEGACY_GUMBEL,
    RuntimeDecision.QWEN_V2_GUMBEL,
    RuntimeDecision.QWEN_TP_LOGITS_IPC_GATHER,
    RuntimeDecision.QWEN_TP4_SHARDED_GUMBEL,
}


def _boolean_phase(key: RuntimeDecision) -> RuntimePlanPhase:
    if key in _COMPILE_DECISIONS:
        return RuntimePlanPhase.COMPILE
    if key in _GRAPH_DECISIONS:
        return RuntimePlanPhase.GRAPH_CAPTURE
    if key in _REQUEST_DECISIONS:
        return RuntimePlanPhase.REQUEST
    return RuntimePlanPhase.MODEL_INIT


def _profile_family(profile: str) -> str:
    for family in ("qwen3.5_3.6", "deepseek_v4", "qwen3", "qwen2"):
        if profile == family or profile.startswith(f"{family}."):
            return family
    return profile.split(".", 1)[0]


def _profile_families(key: RuntimeDecision) -> tuple[str, ...]:
    value = key.value
    if value.startswith("deepseek_v4."):
        return ("deepseek_v4",)
    if value.startswith("qwen3.5_3.6."):
        return ("qwen3.5_3.6",)
    if value.startswith("qwen3."):
        return ("qwen3",)
    if value.startswith("qwen2."):
        return ("qwen2",)
    if value.startswith("qwen."):
        return ("qwen2", "qwen3", "qwen3.5_3.6")
    return ()


def _boolean_tunability(key: RuntimeDecision) -> RuntimeDecisionTunability:
    if key is RuntimeDecision.DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD:
        return RuntimeDecisionTunability.FIXED
    return RuntimeDecisionTunability.PROFILE


_TYPED_KEYS = {
    RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE,
    RuntimeDecision.HYBRID_KV_CACHE_POOL_LAYOUT,
    RuntimeDecision.MUSA_FUSED_ADD_RMSNORM_MIN_ROWS,
    RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY,
    RuntimeDecision.VLLM_IR_OP_PRIORITY,
}

_SPECS = {
    key: RuntimeDecisionSpec(
        key=key,
        kind=RuntimeDecisionKind.BOOLEAN,
        phase=_boolean_phase(key),
        fallback=False,
        choices=(False, True),
        profile_families=_profile_families(key),
        tunability=_boolean_tunability(key),
    )
    for key in RuntimeDecision
    if key not in _TYPED_KEYS
}
_SPECS.update(
    {
        RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE: RuntimeDecisionSpec(
            key=RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE,
            kind=RuntimeDecisionKind.INTEGER,
            phase=RuntimePlanPhase.CACHE_LAYOUT,
            fallback=256,
            choices=(256,),
            profile_families=("deepseek_v4",),
            tunability=RuntimeDecisionTunability.FIXED,
        ),
        RuntimeDecision.HYBRID_KV_CACHE_POOL_LAYOUT: RuntimeDecisionSpec(
            key=RuntimeDecision.HYBRID_KV_CACHE_POOL_LAYOUT,
            kind=RuntimeDecisionKind.ENUM,
            phase=RuntimePlanPhase.CACHE_LAYOUT,
            fallback="shared",
            choices=("shared", "separate"),
            tunability=RuntimeDecisionTunability.FIXED,
        ),
        RuntimeDecision.MUSA_FUSED_ADD_RMSNORM_MIN_ROWS: RuntimeDecisionSpec(
            key=RuntimeDecision.MUSA_FUSED_ADD_RMSNORM_MIN_ROWS,
            kind=RuntimeDecisionKind.INTEGER,
            phase=RuntimePlanPhase.COMPILE,
            fallback=64,
            external_only=True,
            tunability=RuntimeDecisionTunability.AUTOTUNE,
        ),
        RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY: RuntimeDecisionSpec(
            key=RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY,
            kind=RuntimeDecisionKind.STRUCTURED,
            phase=RuntimePlanPhase.COMPILE,
            fallback=(),
            external_only=True,
            tunability=RuntimeDecisionTunability.AUTOTUNE,
        ),
        RuntimeDecision.VLLM_IR_OP_PRIORITY: RuntimeDecisionSpec(
            key=RuntimeDecision.VLLM_IR_OP_PRIORITY,
            kind=RuntimeDecisionKind.STRUCTURED,
            phase=RuntimePlanPhase.CONFIG_DEFAULTS,
            fallback=(),
            external_only=True,
            tunability=RuntimeDecisionTunability.AUTOTUNE,
        ),
    }
)
RUNTIME_DECISION_SPECS = MappingProxyType(_SPECS)


def runtime_decision_spec(key: RuntimeDecision) -> RuntimeDecisionSpec:
    try:
        return RUNTIME_DECISION_SPECS[key]
    except KeyError as exc:  # pragma: no cover - enum/spec coverage is unit tested
        raise ValueError(f"Runtime decision {key.value!r} is not registered") from exc


def validate_runtime_decision_value(
    key: RuntimeDecision,
    value: object,
) -> RuntimeDecisionValue:
    return runtime_decision_spec(key).validate(value)


def list_runtime_decision_specs() -> tuple[RuntimeDecisionSpec, ...]:
    return tuple(RUNTIME_DECISION_SPECS[key] for key in RuntimeDecision)
