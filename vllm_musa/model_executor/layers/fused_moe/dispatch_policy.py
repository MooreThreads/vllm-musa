# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-agnostic MUSA fused-MoE backend selection.

The hot path must not benchmark, synchronize device data to the host, or key
off a model architecture name.  Offline S5000 sweeps populate exact shape
entries below; unknown shapes keep the established upstream backend.
"""

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final

from .dispatch_types import (
    MusaFusedMoeBackend,
    MusaFusedMoeDispatchSelection,
    MusaFusedMoeRuntimePolicyReceipt,
    MusaFusedMoeShape,
    MusaFusedMoeThresholds,
    MusaFusedMoeTokenRange,
)

_LOGGER = logging.getLogger(__name__)
_BINDING_RECEIPT_KEYS: set[tuple[object, ...]] = set()


def _log_once(level: str, message: str, *args: object) -> None:
    """Use vLLM's once logger when available, with a test-safe fallback."""

    try:
        from vllm.logger import logger as vllm_logger

        method = getattr(vllm_logger, f"{level}_once", None)
        if method is not None:
            method(message, *args)
            return
    except (ImportError, AttributeError):
        pass
    getattr(_LOGGER, level)(message, *args)


def _log(level: str, message: str, *args: object) -> None:
    """Route worker-visible receipts through vLLM's configured logger."""

    try:
        from vllm.logger import logger as vllm_logger

        getattr(vllm_logger, level)(message, *args)
        return
    except (ImportError, AttributeError):
        pass
    getattr(_LOGGER, level)(message, *args)

MUSA_FUSED_MOE_DISPATCH_ENV: Final = "VLLM_MUSA_FUSED_MOE_DISPATCH"
FUSED_MOE_DISPATCH_POLICY_SCHEMA: Final = "musa.fused_moe.dispatch_policy.v1"
_FUSED_MOE_INTERNAL_POLICY_ENV: Final = "VLLM_MUSA_INTERNAL_FUSED_MOE_DISPATCH_POLICY"
_FUSED_MOE_INTERNAL_PLAN_ID_ENV: Final = "VLLM_MUSA_INTERNAL_FUSED_MOE_PLAN_ID"
_FUSED_MOE_INTERNAL_FINGERPRINT_ENV: Final = (
    "VLLM_MUSA_INTERNAL_FUSED_MOE_PLAN_FINGERPRINT"
)
_FUSED_MOE_INTERNAL_PROFILE_ENV: Final = "VLLM_MUSA_INTERNAL_FUSED_MOE_PROFILE"


@dataclass(frozen=True, slots=True)
class _MusaFusedMoeRuntimePolicyEntry:
    shape: MusaFusedMoeShape
    ranges: tuple[MusaFusedMoeTokenRange, ...]


_ACTIVE_RUNTIME_POLICY: tuple[_MusaFusedMoeRuntimePolicyEntry, ...] = ()
_ACTIVE_RUNTIME_POLICY_BY_SHAPE: MappingProxyType = MappingProxyType({})
_ACTIVE_RUNTIME_POLICY_RECEIPT = MusaFusedMoeRuntimePolicyReceipt(
    plan_id="",
    plan_fingerprint="",
    profile="",
    entry_count=0,
)
_ACTIVE_RUNTIME_POLICY_DIMENSIONS: frozenset[tuple[int, int, int, int, int]] = (
    frozenset()
)


def _freeze_runtime_value(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_runtime_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, list):
        return tuple(_freeze_runtime_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_runtime_value(item) for item in value)
    return value


def _json_runtime_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_runtime_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_runtime_value(item) for key, item in value.items()}
    return value


def _runtime_policy_entries(
    value: object,
) -> tuple[_MusaFusedMoeRuntimePolicyEntry, ...]:
    value = _freeze_runtime_value(value)
    if value == ():
        return ()
    from vllm_musa.runtime_plan.catalog import decode_fused_moe_dispatch_policy

    entries = decode_fused_moe_dispatch_policy(value)
    result: list[_MusaFusedMoeRuntimePolicyEntry] = []
    seen: set[MusaFusedMoeShape] = set()
    for shape_mapping, range_mappings in entries:
        shape = MusaFusedMoeShape(**shape_mapping)
        if shape in seen:
            raise ValueError("fused-MoE policy contains duplicate exact shapes")
        seen.add(shape)
        ranges = tuple(
            MusaFusedMoeTokenRange(
                min_tokens=int(raw_range["min_tokens"]),
                max_tokens=int(raw_range["max_tokens"]),
                backend=MusaFusedMoeBackend(raw_range["backend"]),
            )
            for raw_range in range_mappings
        )
        result.append(_MusaFusedMoeRuntimePolicyEntry(shape=shape, ranges=ranges))
    return tuple(result)


def configure_fused_moe_runtime_policy(
    value: object,
    *,
    plan_id: str = "",
    plan_fingerprint: str = "",
    profile: str = "",
) -> None:
    """Materialize a validated policy before compile/capture.

    This is intentionally the only decoder.  ``resolve_fused_moe_backend`` reads
    immutable tuples and performs no JSON parsing, allocation, or locking.
    """

    global _ACTIVE_RUNTIME_POLICY
    global _ACTIVE_RUNTIME_POLICY_BY_SHAPE
    global _ACTIVE_RUNTIME_POLICY_DIMENSIONS
    global _ACTIVE_RUNTIME_POLICY_RECEIPT
    _BINDING_RECEIPT_KEYS.clear()
    try:
        entries = _runtime_policy_entries(value)
    except (TypeError, ValueError):
        _ACTIVE_RUNTIME_POLICY = ()
        _ACTIVE_RUNTIME_POLICY_BY_SHAPE = MappingProxyType({})
        _ACTIVE_RUNTIME_POLICY_DIMENSIONS = frozenset()
        _ACTIVE_RUNTIME_POLICY_RECEIPT = MusaFusedMoeRuntimePolicyReceipt(
            plan_id="",
            plan_fingerprint="",
            profile="",
            entry_count=0,
        )
        for name in (
            _FUSED_MOE_INTERNAL_POLICY_ENV,
            _FUSED_MOE_INTERNAL_PLAN_ID_ENV,
            _FUSED_MOE_INTERNAL_FINGERPRINT_ENV,
            _FUSED_MOE_INTERNAL_PROFILE_ENV,
        ):
            os.environ.pop(name, None)
        raise
    if any(not isinstance(item, str) for item in (plan_id, plan_fingerprint, profile)):
        _ACTIVE_RUNTIME_POLICY = ()
        _ACTIVE_RUNTIME_POLICY_BY_SHAPE = MappingProxyType({})
        _ACTIVE_RUNTIME_POLICY_DIMENSIONS = frozenset()
        _ACTIVE_RUNTIME_POLICY_RECEIPT = MusaFusedMoeRuntimePolicyReceipt(
            plan_id="",
            plan_fingerprint="",
            profile="",
            entry_count=0,
        )
        for name in (
            _FUSED_MOE_INTERNAL_POLICY_ENV,
            _FUSED_MOE_INTERNAL_PLAN_ID_ENV,
            _FUSED_MOE_INTERNAL_FINGERPRINT_ENV,
            _FUSED_MOE_INTERNAL_PROFILE_ENV,
        ):
            os.environ.pop(name, None)
        raise TypeError("fused-MoE runtime-plan receipt fields must be strings")
    _ACTIVE_RUNTIME_POLICY = entries
    _ACTIVE_RUNTIME_POLICY_BY_SHAPE = MappingProxyType(
        {entry.shape: entry for entry in entries}
    )
    _ACTIVE_RUNTIME_POLICY_DIMENSIONS = frozenset(
        (
            entry.shape.local_experts,
            entry.shape.w1_output_size,
            entry.shape.w2_input_size,
            entry.shape.hidden_size,
            entry.shape.top_k,
        )
        for entry in entries
    )
    _ACTIVE_RUNTIME_POLICY_RECEIPT = MusaFusedMoeRuntimePolicyReceipt(
        plan_id=plan_id,
        plan_fingerprint=plan_fingerprint,
        profile=profile,
        entry_count=len(entries),
    )
    if entries:
        os.environ[_FUSED_MOE_INTERNAL_POLICY_ENV] = json.dumps(
            _json_runtime_value(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        os.environ[_FUSED_MOE_INTERNAL_PLAN_ID_ENV] = plan_id
        os.environ[_FUSED_MOE_INTERNAL_FINGERPRINT_ENV] = plan_fingerprint
        os.environ[_FUSED_MOE_INTERNAL_PROFILE_ENV] = profile
    else:
        for name in (
            _FUSED_MOE_INTERNAL_POLICY_ENV,
            _FUSED_MOE_INTERNAL_PLAN_ID_ENV,
            _FUSED_MOE_INTERNAL_FINGERPRINT_ENV,
            _FUSED_MOE_INTERNAL_PROFILE_ENV,
        ):
            os.environ.pop(name, None)


def active_fused_moe_runtime_policy_receipt() -> MusaFusedMoeRuntimePolicyReceipt:
    """Return immutable plan provenance for startup/diagnostic receipts."""

    return _ACTIVE_RUNTIME_POLICY_RECEIPT


def runtime_plan_bindings_for_dimensions(
    *,
    local_experts: int,
    w1_output_size: int,
    w2_input_size: int,
    hidden_size: int,
    top_k: int,
    device_capability: tuple[int, int] | None = None,
    multiprocessor_count: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    activation: str | None = None,
    expert_parallel: bool | None = None,
    hidden_dtype: str | None = None,
    weight_dtype: str | None = None,
    scale_dtype: str | None = None,
    w1_scale_shape: tuple[int, ...] | None = None,
    w2_scale_shape: tuple[int, ...] | None = None,
) -> tuple[
    tuple[MusaFusedMoeShape, tuple[MusaFusedMoeTokenRange, ...]],
    ...,
]:
    """Return every sealed entry matching a modular kernel's static shape.

    Kernel construction happens before a request exists, so this deliberately
    does not infer ``num_tokens`` or graph-capture state. Returning all eager
    and capture entries keeps a binding receipt a truthful description of the
    plan that the later runtime resolver will consult.
    """

    expected = {
        "device_capability": (
            None
            if device_capability is None
            else tuple(int(value) for value in device_capability)
        ),
        "multiprocessor_count": multiprocessor_count,
        "local_experts": int(local_experts),
        "w1_output_size": int(w1_output_size),
        "w2_input_size": int(w2_input_size),
        "hidden_size": int(hidden_size),
        "top_k": int(top_k),
        "block_n": block_n,
        "block_k": block_k,
        "activation": activation,
        "expert_parallel": expert_parallel,
        "hidden_dtype": hidden_dtype,
        "weight_dtype": weight_dtype,
        "scale_dtype": scale_dtype,
        "w1_scale_shape": (
            None
            if w1_scale_shape is None
            else tuple(int(value) for value in w1_scale_shape)
        ),
        "w2_scale_shape": (
            None
            if w2_scale_shape is None
            else tuple(int(value) for value in w2_scale_shape)
        ),
    }
    return tuple(
        (entry.shape, entry.ranges)
        for entry in _ACTIVE_RUNTIME_POLICY
        if all(
            value is None or getattr(entry.shape, field) == value
            for field, value in expected.items()
        )
    )


def record_modular_fused_moe_runtime_plan_binding(
    *,
    routed_experts: object,
    moe_kernel: object,
    layer_name: str = "",
) -> None:
    """Emit a one-time, pre-compile receipt for a modular MoE kernel.

    This function is called from vLLM's kernel-initialization boundary, not
    from the compiled forward. It only reads tensor metadata and the already
    materialized immutable policy; it never synchronizes or launches a GPU
    operation. The receipt therefore proves static plan binding, while the
    normal resolver remains responsible for per-request token-range choice.
    """

    receipt = active_fused_moe_runtime_policy_receipt()
    if (
        not receipt.plan_fingerprint
        or receipt.entry_count <= 0
        or not _ACTIVE_RUNTIME_POLICY
    ):
        # Baseline/no-plan runs should not pay for metadata inspection or emit
        # a warning for every modular layer during model construction.
        return

    implementation_object = getattr(moe_kernel, "fused_experts", moe_kernel)
    implementation = type(implementation_object).__name__
    try:
        w1 = getattr(routed_experts, "w13_weight")
        w2 = getattr(routed_experts, "w2_weight")
        moe_config = getattr(routed_experts, "moe_config")
        top_k = getattr(moe_config, "experts_per_token")
        w1_shape = tuple(int(value) for value in getattr(w1, "shape"))
        w2_shape = tuple(int(value) for value in getattr(w2, "shape"))
        if len(w1_shape) != 3 or len(w2_shape) != 3:
            raise ValueError("modular expert weights must be rank-3 tensors")
        if w1_shape[0] != w2_shape[0] or w1_shape[2] != w2_shape[1]:
            raise ValueError(
                "modular expert weights have incompatible expert dimensions"
            )
        is_act_and_mul = getattr(moe_config, "is_act_and_mul", None)
        if is_act_and_mul is not None and w1_shape[1] != w2_shape[2] * (
            2 if is_act_and_mul else 1
        ):
            raise ValueError("modular gated weights have incompatible N dimensions")

        def _dtype_name(value: object) -> str | None:
            return None if value is None else str(value)

        def _shape_of(value: object) -> tuple[int, ...] | None:
            try:
                return tuple(int(item) for item in getattr(value, "shape"))
            except (AttributeError, TypeError, ValueError):
                return None

        activation_value = getattr(moe_config, "activation", None)
        activation_value = getattr(activation_value, "value", activation_value)
        parallel_config = getattr(moe_config, "moe_parallel_config", None)
        block_size = getattr(routed_experts, "weight_block_size", None)
        block_n = block_k = None
        if block_size is not None and len(block_size) == 2:
            block_n, block_k = (int(block_size[0]), int(block_size[1]))

        w1_scale = w2_scale = None
        for w1_name, w2_name in (
            ("w13_weight_scale", "w2_weight_scale"),
            ("w13_weight_scale_inv", "w2_weight_scale_inv"),
        ):
            if hasattr(routed_experts, w1_name) and hasattr(routed_experts, w2_name):
                w1_scale = getattr(routed_experts, w1_name)
                w2_scale = getattr(routed_experts, w2_name)
                break

        device_capability = None
        multiprocessor_count = None
        try:
            import torch

            device = getattr(w1, "device", None)
            device_capability = tuple(
                int(value) for value in torch.musa.get_device_capability(device)
            )
            multiprocessor_count = int(
                torch.musa.get_device_properties(device).multi_processor_count
            )
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
            pass
        dimensions = {
            "device_capability": device_capability,
            "multiprocessor_count": multiprocessor_count,
            "local_experts": w1_shape[0],
            "w1_output_size": w1_shape[1],
            "w2_input_size": w2_shape[2],
            "hidden_size": w1_shape[2],
            "top_k": int(top_k),
            "block_n": block_n,
            "block_k": block_k,
            "activation": _dtype_name(activation_value),
            "expert_parallel": (
                None
                if parallel_config is None
                else bool(getattr(parallel_config, "use_ep", False))
            ),
            "hidden_dtype": _dtype_name(getattr(moe_config, "in_dtype", None)),
            "weight_dtype": _dtype_name(getattr(w1, "dtype", None)),
            "scale_dtype": _dtype_name(getattr(w1_scale, "dtype", None)),
            "w1_scale_shape": _shape_of(w1_scale),
            "w2_scale_shape": _shape_of(w2_scale),
        }
    except (AttributeError, TypeError, ValueError):
        _log_once(
            "warning",
            "MUSA RuntimePlan modular MoE binding skipped: invalid weight "
            "metadata implementation=%s layer=%s",
            implementation,
            layer_name,
        )
        return

    bindings = runtime_plan_bindings_for_dimensions(**dimensions)
    if not bindings:
        _log_once(
            "warning",
            "MUSA RuntimePlan modular MoE binding miss: dims=%s active_dims=%s "
            "implementation=%s layer=%s",
            tuple(dimensions.values()),
            active_runtime_policy_dimensions(),
            implementation,
            layer_name,
        )
        return

    metadata_complete = all(
        dimensions[name] is not None
        for name in (
            "device_capability",
            "multiprocessor_count",
            "block_n",
            "block_k",
            "activation",
            "expert_parallel",
            "hidden_dtype",
            "weight_dtype",
            "scale_dtype",
            "w1_scale_shape",
            "w2_scale_shape",
        )
    )

    def _env_int(name: str, default: int = -1) -> int:
        try:
            return int(os.environ[name])
        except (KeyError, ValueError):
            return default

    resolved_rank = -1
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            resolved_rank = int(dist.get_rank())
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    if resolved_rank < 0:
        resolved_rank = _env_int("RANK")
    tp_rank = _env_int("LOCAL_RANK")
    try:
        tp_rank = int(getattr(moe_config, "tp_rank"))
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        ep_rank = int(getattr(moe_config, "ep_rank"))
    except (AttributeError, TypeError, ValueError):
        ep_rank = -1

    for shape, ranges in bindings:
        for token_range in ranges:
            key = (
                receipt.plan_fingerprint,
                resolved_rank,
                implementation,
                shape,
                token_range,
            )
            if key in _BINDING_RECEIPT_KEYS:
                continue
            _BINDING_RECEIPT_KEYS.add(key)
            _log(
                "info",
                "MUSA fused-MoE plan binding receipt: phase=bind "
                "planned_backend=%s plan_source=sealed_runtime_plan "
                "implementation=%s policy=static_binding plan_id=%s "
                "plan_fingerprint=%s execution_backend=uncontrolled "
                "shape=(E=%d,N=%d,K=%d,topk=%d,graph=%s) "
                "metadata_complete=%s "
                "actual_tokens=unbound range=%d-%d rank=%d tp_rank=%d "
                "ep_rank=%d layer=%s",
                token_range.backend.value,
                implementation,
                receipt.plan_id,
                receipt.plan_fingerprint,
                shape.local_experts,
                shape.w1_output_size,
                shape.hidden_size,
                shape.top_k,
                shape.graph_mode,
                metadata_complete,
                token_range.min_tokens,
                token_range.max_tokens,
                resolved_rank,
                tp_rank,
                ep_rank,
                layer_name,
            )


def resolve_runtime_plan_for_dimensions(
    *,
    local_experts: int,
    w1_output_size: int,
    w2_input_size: int,
    hidden_size: int,
    top_k: int,
    num_tokens: int,
    stream_is_capturing: bool = False,
) -> MusaFusedMoeDispatchSelection | None:
    """Resolve an active plan for a modular expert implementation.

    Modular vLLM expert classes do not expose the complete MUSA static shape
    object.  Their stable dimensions are nevertheless sufficient to find the
    one exact transported entry; the remaining static fields come from that
    sealed entry itself.  Native tactic eligibility is intentionally false here
    because this helper only records the plan decision made for the modular
    implementation.
    """

    dimensions = (
        local_experts,
        w1_output_size,
        w2_input_size,
        hidden_size,
        top_k,
    )
    graph_mode = "capture" if stream_is_capturing else "eager"
    for entry in _ACTIVE_RUNTIME_POLICY:
        shape = entry.shape
        if (
            shape.local_experts,
            shape.w1_output_size,
            shape.w2_input_size,
            shape.hidden_size,
            shape.top_k,
        ) != dimensions:
            continue
        if shape.graph_mode != graph_mode:
            continue
        return _runtime_policy_selection(
            shape=shape,
            num_tokens=num_tokens,
            can_use_gemv=False,
            can_use_grouped_gemm=False,
            stream_is_capturing=stream_is_capturing,
        )
    return None


def fused_moe_runtime_policy_token_boundaries() -> tuple[int, ...]:
    """Return compile-range split points for all active token buckets."""

    return tuple(
        sorted(
            {
                token_range.max_tokens
                for entry in _ACTIVE_RUNTIME_POLICY
                for token_range in entry.ranges
            }
        )
    )


def _runtime_policy_selection(
    *,
    shape: MusaFusedMoeShape,
    num_tokens: int,
    can_use_gemv: bool,
    can_use_grouped_gemm: bool,
    stream_is_capturing: bool,
) -> MusaFusedMoeDispatchSelection | None:
    entry = _ACTIVE_RUNTIME_POLICY_BY_SHAPE.get(shape)
    if entry is None or shape.graph_mode != (
        "capture" if stream_is_capturing else "eager"
    ):
        return None
    for token_range in entry.ranges:
        if token_range.min_tokens <= num_tokens <= token_range.max_tokens:
            eligible = (
                token_range.backend is MusaFusedMoeBackend.UPSTREAM
                or token_range.backend is MusaFusedMoeBackend.GEMV
                and can_use_gemv
                or token_range.backend is MusaFusedMoeBackend.GROUPED_GEMM
                and can_use_grouped_gemm
                and not stream_is_capturing
            )
            receipt = _ACTIVE_RUNTIME_POLICY_RECEIPT
            return MusaFusedMoeDispatchSelection(
                backend=(
                    token_range.backend if eligible else MusaFusedMoeBackend.UPSTREAM
                ),
                source="runtime_plan" if eligible else "runtime_plan_ineligible",
                policy_identity=FUSED_MOE_DISPATCH_POLICY_SCHEMA,
                plan_id=receipt.plan_id,
                plan_fingerprint=receipt.plan_fingerprint,
                min_tokens=token_range.min_tokens,
                max_tokens=token_range.max_tokens,
            )
    return None


# Unknown shapes stay on the established upstream path until an exact S5000
# sweep is recorded.
_DEFAULT_THRESHOLDS: Final = MusaFusedMoeThresholds(
    gemv_max_tokens=None,
    grouped_gemm_min_tokens=None,
    source="uncalibrated-shape",
)


def _s5000_fp8_shape(
    *,
    local_experts: int,
    w1_output_size: int,
    w2_input_size: int,
    hidden_size: int,
    top_k: int,
    w1_scale_shape: tuple[int, ...],
    w2_scale_shape: tuple[int, ...],
    gemv_block: str,
    graph_mode: str,
) -> MusaFusedMoeShape:
    return MusaFusedMoeShape(
        device_capability=(3, 1),
        multiprocessor_count=60,
        local_experts=local_experts,
        w1_output_size=w1_output_size,
        w2_input_size=w2_input_size,
        hidden_size=hidden_size,
        top_k=top_k,
        block_n=128,
        block_k=128,
        activation="silu",
        expert_parallel=False,
        hidden_dtype="torch.bfloat16",
        weight_dtype="torch.float8_e4m3fn",
        scale_dtype="torch.float32",
        w1_scale_shape=w1_scale_shape,
        w2_scale_shape=w2_scale_shape,
        gemv_block=gemv_block,
        graph_mode=graph_mode,
    )


def _s5000_qwen35_bf16_decode_shape(
    *, graph_mode: str, folded_shared_expert: bool
) -> MusaFusedMoeShape:
    """TP4-local Qwen3.5/3.6 BF16 decode shape."""

    experts = 257 if folded_shared_expert else 256
    top_k = 9 if folded_shared_expert else 8

    return MusaFusedMoeShape(
        device_capability=(3, 1),
        multiprocessor_count=60,
        local_experts=experts,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=top_k,
        block_n=0,
        block_k=0,
        activation="silu",
        expert_parallel=False,
        hidden_dtype="torch.bfloat16",
        weight_dtype="torch.bfloat16",
        scale_dtype="none",
        w1_scale_shape=(),
        w2_scale_shape=(),
        gemv_block="auto",
        graph_mode=graph_mode,
    )


def _s5000_bf16_shape(
    *,
    local_experts: int,
    w1_output_size: int,
    w2_input_size: int,
    hidden_size: int,
    top_k: int,
    graph_mode: str,
) -> MusaFusedMoeShape:
    """Generic S5000 BF16 decode shape for independently calibrated models."""

    return MusaFusedMoeShape(
        device_capability=(3, 1),
        multiprocessor_count=60,
        local_experts=local_experts,
        w1_output_size=w1_output_size,
        w2_input_size=w2_input_size,
        hidden_size=hidden_size,
        top_k=top_k,
        block_n=0,
        block_k=0,
        activation="silu",
        expert_parallel=False,
        hidden_dtype="torch.bfloat16",
        weight_dtype="torch.bfloat16",
        scale_dtype="none",
        w1_scale_shape=(),
        w2_scale_shape=(),
        gemv_block="auto",
        graph_mode=graph_mode,
    )


def _thresholds(
    gemv_max_tokens: int | None,
    grouped_gemm_min_tokens: int | None,
    source: str,
) -> MusaFusedMoeThresholds:
    return MusaFusedMoeThresholds(
        gemv_max_tokens=gemv_max_tokens,
        grouped_gemm_min_tokens=grouped_gemm_min_tokens,
        source=source,
    )


# Exact entries are keyed by the actual per-rank kernel shape. These S5000
# entries use the worst boundary across balanced, unique-random, and hot routes
# with three independent seeds. Capture entries additionally passed eight
# bitwise-equal CUDAGraph replays. Unknown shapes remain on the established
# base path, which may itself select the large-M DeepGEMM prefill backend.
_CALIBRATED_THRESHOLDS: Final[dict[MusaFusedMoeShape, MusaFusedMoeThresholds]] = {
    _s5000_fp8_shape(
        local_experts=256,
        w1_output_size=256,
        w2_input_size=128,
        hidden_size=2048,
        top_k=8,
        w1_scale_shape=(256, 2, 16),
        w2_scale_shape=(256, 16, 1),
        gemv_block="auto",
        graph_mode=graph_mode,
    ): _thresholds(
        gemv_max_tokens=13,
        grouped_gemm_min_tokens=None,
        source=f"s5000-mp60-20260721-e256-n256-k2048-{graph_mode}-dense-v5",
    )
    for graph_mode in ("eager", "capture")
}
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_fp8_shape(
            local_experts=256,
            w1_output_size=512,
            w2_input_size=256,
            hidden_size=4096,
            top_k=6,
            w1_scale_shape=(256, 4, 32),
            w2_scale_shape=(256, 32, 2),
            gemv_block="32x8",
            graph_mode=graph_mode,
        ): _thresholds(
            gemv_max_tokens=5,
            grouped_gemm_min_tokens=None,
            source=f"s5000-mp60-20260721-e256-n512-k4096-{graph_mode}-block32-dense-v5",
        )
        for graph_mode in ("eager", "capture")
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_fp8_shape(
            local_experts=256,
            w1_output_size=512,
            w2_input_size=256,
            hidden_size=4096,
            top_k=6,
            w1_scale_shape=(256, 4, 32),
            w2_scale_shape=(256, 32, 2),
            gemv_block="16x8",
            graph_mode=graph_mode,
        ): _thresholds(
            gemv_max_tokens=5,
            grouped_gemm_min_tokens=None,
            source=f"s5000-mp60-20260721-e256-n512-k4096-{graph_mode}-block16-dense-v5",
        )
        for graph_mode in ("eager", "capture")
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_fp8_shape(
            local_experts=64,
            w1_output_size=2816,
            w2_input_size=1408,
            hidden_size=2048,
            top_k=6,
            w1_scale_shape=(64, 22, 16),
            w2_scale_shape=(64, 16, 11),
            gemv_block="auto",
            graph_mode=graph_mode,
        ): _thresholds(
            # Capture-mode A/B on S5000 shows a GEMV win at M=1, but a
            # regression at M=2/3.  Keep the wider eager boundary and use
            # GEMV only for the single-token decode case under capture.
            gemv_max_tokens=3 if graph_mode == "eager" else 1,
            # No production-safe grouped crossover is currently established
            # for this shape; retain upstream for the large-token regime.
            grouped_gemm_min_tokens=None,
            source=(
                f"s5000-mp60-20260721-e64-n2816-k2048-{graph_mode}-"
                f"{'m1-' if graph_mode == 'capture' else ''}serving-gated"
            ),
        )
        for graph_mode in ("eager", "capture")
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_qwen35_bf16_decode_shape(
            graph_mode=graph_mode,
            folded_shared_expert=folded_shared_expert,
        ): _thresholds(
            # Exact TP4-local Qwen crossover: M=1/2/4/8/12 are faster on
            # native GEMV, while M>=16 is not a stable production win.
            gemv_max_tokens=12,
            grouped_gemm_min_tokens=None,
            source=(
                f"s5000-mp60-20260806-qwen35-36-bf16-e"
                f"{257 if folded_shared_expert else 256}-"
                f"n256-k2048-v128-{graph_mode}-crossover-v1"
            ),
        )
        for graph_mode in ("eager", "capture")
        for folded_shared_expert in (False, True)
    }
)
_CALIBRATED_THRESHOLDS.update(
    {
        _s5000_bf16_shape(
            local_experts=257,
            w1_output_size=256,
            w2_input_size=128,
            hidden_size=3072,
            top_k=9,
            graph_mode=graph_mode,
        ): _thresholds(
            gemv_max_tokens=10,
            grouped_gemm_min_tokens=None,
            source=(
                f"s5000-mp60-20260806-qwen35-bf16-folded-"
                f"e257-n256-k3072-{graph_mode}"
            ),
        )
        for graph_mode in ("eager", "capture")
    }
)

_CALIBRATED_DIMENSIONS: Final = frozenset(
    (
        shape.local_experts,
        shape.w1_output_size,
        shape.w2_input_size,
        shape.hidden_size,
        shape.top_k,
    )
    for shape in _CALIBRATED_THRESHOLDS
)


def parse_dispatch_backend(value: str | None = None) -> MusaFusedMoeBackend:
    """Parse the generic force/rollback override; default is ``auto``."""

    raw_value = os.environ.get(MUSA_FUSED_MOE_DISPATCH_ENV, "auto")
    if value is not None:
        raw_value = value
    normalized = raw_value.strip().lower().replace("-", "_")
    aliases = {
        "gemm": MusaFusedMoeBackend.GROUPED_GEMM,
        "grouped": MusaFusedMoeBackend.GROUPED_GEMM,
        "native_gemv": MusaFusedMoeBackend.GEMV,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return MusaFusedMoeBackend(normalized)
    except ValueError as exc:
        choices = ", ".join(backend.value for backend in MusaFusedMoeBackend)
        raise ValueError(
            f"Invalid {MUSA_FUSED_MOE_DISPATCH_ENV}={raw_value!r}; "
            f"expected one of: {choices}"
        ) from exc


def thresholds_for_shape(shape: MusaFusedMoeShape) -> MusaFusedMoeThresholds:
    return _CALIBRATED_THRESHOLDS.get(shape, _DEFAULT_THRESHOLDS)


def has_calibrated_dimensions(
    *,
    local_experts: int,
    w1_output_size: int,
    w2_input_size: int,
    hidden_size: int,
    top_k: int,
) -> bool:
    """Cheap hot-path rejection before capability and capture checks."""

    dimensions = (
        local_experts,
        w1_output_size,
        w2_input_size,
        hidden_size,
        top_k,
    )
    return (
        dimensions in _CALIBRATED_DIMENSIONS
        or dimensions in _ACTIVE_RUNTIME_POLICY_DIMENSIONS
    )


def has_runtime_policy_dimensions(
    *,
    local_experts: int,
    w1_output_size: int,
    w2_input_size: int,
    hidden_size: int,
    top_k: int,
) -> bool:
    """Cheap check used to reach an exact plan entry before capability gates."""

    return (
        local_experts,
        w1_output_size,
        w2_input_size,
        hidden_size,
        top_k,
    ) in _ACTIVE_RUNTIME_POLICY_DIMENSIONS


def active_runtime_policy_dimensions() -> tuple[tuple[int, int, int, int, int], ...]:
    """Return exact static dimensions currently covered by the sealed policy."""

    return tuple(sorted(_ACTIVE_RUNTIME_POLICY_DIMENSIONS))


def select_fused_moe_backend(
    *,
    shape: MusaFusedMoeShape,
    num_tokens: int,
    can_use_gemv: bool,
    can_use_grouped_gemm: bool,
    stream_is_capturing: bool,
    requested: MusaFusedMoeBackend = MusaFusedMoeBackend.AUTO,
    thresholds: MusaFusedMoeThresholds | None = None,
) -> MusaFusedMoeBackend:
    """Choose one already-compiled backend without device synchronization.

    Forced modes are diagnostic controls, not correctness bypasses: if a
    requested backend is ineligible the established upstream path is used.
    """

    if requested == MusaFusedMoeBackend.GEMV:
        # Forced GEMV remains useful for eager diagnostics, but graph capture
        # must stay inside an explicitly calibrated capture entry and token
        # range.  Otherwise an unknown shape could be baked into a graph even
        # though the override is documented as preserving capture safety.
        if thresholds is None:
            thresholds = thresholds_for_shape(shape)
        capture_is_calibrated = bool(
            not stream_is_capturing
            or (
                thresholds.gemv_max_tokens is not None
                and num_tokens <= thresholds.gemv_max_tokens
            )
        )
        return (
            MusaFusedMoeBackend.GEMV
            if can_use_gemv and capture_is_calibrated
            else MusaFusedMoeBackend.UPSTREAM
        )
    if requested == MusaFusedMoeBackend.GROUPED_GEMM:
        return (
            MusaFusedMoeBackend.GROUPED_GEMM
            if can_use_grouped_gemm and not stream_is_capturing
            else MusaFusedMoeBackend.UPSTREAM
        )
    if requested == MusaFusedMoeBackend.UPSTREAM:
        return MusaFusedMoeBackend.UPSTREAM

    if thresholds is None:
        thresholds = thresholds_for_shape(shape)
    if (
        can_use_gemv
        and thresholds.gemv_max_tokens is not None
        and num_tokens <= thresholds.gemv_max_tokens
    ):
        return MusaFusedMoeBackend.GEMV
    if (
        can_use_grouped_gemm
        and not stream_is_capturing
        and thresholds.grouped_gemm_min_tokens is not None
        and num_tokens >= thresholds.grouped_gemm_min_tokens
    ):
        return MusaFusedMoeBackend.GROUPED_GEMM
    return MusaFusedMoeBackend.UPSTREAM


def resolve_fused_moe_backend(
    *,
    shape: MusaFusedMoeShape,
    num_tokens: int,
    can_use_gemv: bool,
    can_use_grouped_gemm: bool,
    stream_is_capturing: bool,
    requested: MusaFusedMoeBackend = MusaFusedMoeBackend.AUTO,
    thresholds: MusaFusedMoeThresholds | None = None,
) -> MusaFusedMoeDispatchSelection:
    """Resolve diagnostic, RuntimePlan, legacy, then upstream in that order."""

    if requested is not MusaFusedMoeBackend.AUTO:
        backend = select_fused_moe_backend(
            shape=shape,
            num_tokens=num_tokens,
            can_use_gemv=can_use_gemv,
            can_use_grouped_gemm=can_use_grouped_gemm,
            stream_is_capturing=stream_is_capturing,
            requested=requested,
            thresholds=thresholds,
        )
        return MusaFusedMoeDispatchSelection(
            backend=backend,
            source="diagnostic_override",
            policy_identity="diagnostic_override",
        )

    runtime_selection = _runtime_policy_selection(
        shape=shape,
        num_tokens=num_tokens,
        can_use_gemv=can_use_gemv,
        can_use_grouped_gemm=can_use_grouped_gemm,
        stream_is_capturing=stream_is_capturing,
    )
    if runtime_selection is not None:
        return runtime_selection
    legacy_backend = select_fused_moe_backend(
        shape=shape,
        num_tokens=num_tokens,
        can_use_gemv=can_use_gemv,
        can_use_grouped_gemm=can_use_grouped_gemm,
        stream_is_capturing=stream_is_capturing,
        requested=MusaFusedMoeBackend.AUTO,
        thresholds=thresholds,
    )
    legacy_policy = (
        thresholds if thresholds is not None else thresholds_for_shape(shape)
    )
    source = (
        "legacy" if legacy_policy.source != _DEFAULT_THRESHOLDS.source else "upstream"
    )
    return MusaFusedMoeDispatchSelection(
        backend=legacy_backend,
        source=source,
        policy_identity=legacy_policy.source,
        min_tokens=(1 if legacy_backend is MusaFusedMoeBackend.GEMV else None),
        max_tokens=(
            legacy_policy.gemv_max_tokens
            if legacy_backend is MusaFusedMoeBackend.GEMV
            else None
        ),
    )


def _restore_fused_moe_runtime_policy_from_environment() -> None:
    raw_policy = os.environ.get(_FUSED_MOE_INTERNAL_POLICY_ENV)
    if raw_policy is None:
        return
    try:
        value = json.loads(raw_policy)
        configure_fused_moe_runtime_policy(
            value,
            plan_id=os.environ.get(_FUSED_MOE_INTERNAL_PLAN_ID_ENV, ""),
            plan_fingerprint=os.environ.get(_FUSED_MOE_INTERNAL_FINGERPRINT_ENV, ""),
            profile=os.environ.get(_FUSED_MOE_INTERNAL_PROFILE_ENV, ""),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Invalid inherited MUSA fused-MoE RuntimePlan policy; refusing "
            "to run with an ambiguous child-process dispatch state"
        ) from exc


_restore_fused_moe_runtime_policy_from_environment()


__all__ = [
    "FUSED_MOE_DISPATCH_POLICY_SCHEMA",
    "MUSA_FUSED_MOE_DISPATCH_ENV",
    "MusaFusedMoeBackend",
    "MusaFusedMoeDispatchSelection",
    "MusaFusedMoeShape",
    "MusaFusedMoeThresholds",
    "MusaFusedMoeRuntimePolicyReceipt",
    "MusaFusedMoeTokenRange",
    "active_fused_moe_runtime_policy_receipt",
    "configure_fused_moe_runtime_policy",
    "fused_moe_runtime_policy_token_boundaries",
    "has_calibrated_dimensions",
    "has_runtime_policy_dimensions",
    "active_runtime_policy_dimensions",
    "parse_dispatch_backend",
    "resolve_fused_moe_backend",
    "select_fused_moe_backend",
    "thresholds_for_shape",
]
