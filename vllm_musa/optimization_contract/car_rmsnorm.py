# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project.
"""Shared compile-time contract for MUSA CAR-RMSNorm fusion.

The fusion pass, platform range setup, provider, and direct communicator must
make the same decision.  A target signature is fused only when its model
family, quantization state, TP size, dtype, and row range are known. Native rows
and fused-range bounds are enforced by the same predicate; ranges outside the
contract use the native CAR + RMSNorm graph. Unknown metadata is always native.
"""

from __future__ import annotations

from typing import Any

import torch

FUSED_ALLREDUCE_RMSNORM_POLICY_VERSION = "car-rmsnorm-operator-gate-v3"

FUSED_ALLREDUCE_RMSNORM_TARGET_HIDDEN_SIZE = 5120
FUSED_ALLREDUCE_RMSNORM_TP4_HIDDEN_SIZE = 2048

# Normalize the accepted short and resolver model-family spellings.
FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY = "qwen3.5_3.6"

# ``native_rows`` routes exact shapes to native CAR. ``fused_compile_max_rows``
# bounds the Inductor bucket that may use fusion. The platform partitions
# compile ranges at native-row boundaries so every caller makes the same choice.
CAR_RMSNORM_POLICY_TABLE: tuple[dict[str, Any], ...] = (
    {
        "tp_size": 2,
        "hidden_size": FUSED_ALLREDUCE_RMSNORM_TARGET_HIDDEN_SIZE,
        "quantized": False,
        "family": FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
        "native_rows": frozenset((16,)),
    },
    {
        "tp_size": 2,
        "hidden_size": FUSED_ALLREDUCE_RMSNORM_TARGET_HIDDEN_SIZE,
        "quantized": True,
        "family": FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
        "native_rows": frozenset((4, 16)),
    },
    {
        "tp_size": 4,
        "hidden_size": FUSED_ALLREDUCE_RMSNORM_TP4_HIDDEN_SIZE,
        "quantized": False,
        "family": FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
        "native_rows": frozenset((16, 64)),
        "generic_graph_registered_input": False,
        "fused_compile_max_rows": 15,
    },
    {
        "tp_size": 4,
        "hidden_size": FUSED_ALLREDUCE_RMSNORM_TP4_HIDDEN_SIZE,
        "quantized": True,
        "family": FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
        "native_rows": frozenset((64,)),
    },
)

_VALID_MODEL_FAMILIES = frozenset({"qwen3.5", "qwen3.5_3.6"})
_VALID_PHASES = frozenset({"decode", "prefill", "mixed"})
_VALID_PATHS = frozenset({"raw", "no_raw", "registered", "staging"})
_UNSET = object()


def _concrete_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _compile_range_bounds(compile_range: Any) -> tuple[int, int] | None:
    """Read a vLLM ``Range`` without importing vLLM at module load time."""
    if compile_range is None:
        return None
    start = getattr(compile_range, "start", None)
    end = getattr(compile_range, "end", None)
    if not _concrete_int(start) or not _concrete_int(end):
        return None
    if start < 1 or end < start:
        return None
    return int(start), int(end)


def fused_allreduce_rmsnorm_compile_endpoints(
    *, tp_size: int | None, hidden_size: int | None
) -> tuple[int, ...]:
    """Return inclusive compile-range endpoints needed by the policy table.

    The endpoint list is the union across BF16 and FP8-weight native rows for a
    target TP/hidden pair. One partition serves both quantization states.
    Non-target signatures retain vLLM's defaults.
    """
    if tp_size == 2 and hidden_size == FUSED_ALLREDUCE_RMSNORM_TARGET_HIDDEN_SIZE:
        # 3/4 and 15/16 isolate native rows; 63/64 isolates the upper boundary.
        return (3, 4, 15, 16, 63)
    if tp_size == 4 and hidden_size == FUSED_ALLREDUCE_RMSNORM_TP4_HIDDEN_SIZE:
        # Union of TP4 native rows {16, 64}.
        return (15, 16, 63, 64)
    return ()


def _canonical_model_family(model_family: str | None) -> str | None:
    if model_family is None:
        return None
    normalized = str(model_family).lower()
    if normalized not in _VALID_MODEL_FAMILIES:
        return None
    return FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY


def infer_car_rmsnorm_model_family(vllm_config: Any) -> str | None:
    """Resolve the family using the optimization-contract resolver.

    This helper is lazy so importing the provider remains cheap and does not
    create a resolver cycle. Returning ``None`` keeps unknown and non-Qwen
    configurations fail-closed.
    """
    try:
        from .resolver import resolve_optimization_contract

        family = resolve_optimization_contract(vllm_config).model.family.value
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return None
    return _canonical_model_family(family)


def current_car_rmsnorm_compile_range() -> Any | None:
    """Return the active Inductor compile range, or ``None`` for eager dispatch."""
    try:
        from vllm.compilation.passes.inductor_pass import get_pass_context

        return get_pass_context().compile_range
    except (AssertionError, AttributeError, ImportError, RuntimeError):
        return None


def current_car_rmsnorm_metadata() -> tuple[str | None, bool | None, int | None] | None:
    """Family, quantization and hidden size for the active serving config.

    These are not on a model signature: at dispatch time the config is only
    reachable through the global accessor, and a plugin cannot assume one is
    bound. ``None`` means "no active config", not "not quantized", so each
    caller keeps its own fail-closed direction instead of inheriting one.
    """
    try:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    except (AssertionError, ImportError, RuntimeError):
        return None
    if vllm_config is None:
        return None
    quant_config = getattr(vllm_config, "quant_config", _UNSET)
    quantized = None if quant_config is _UNSET else quant_config is not None
    model_config = getattr(vllm_config, "model_config", None)
    get_hidden_size = getattr(model_config, "get_hidden_size", None)
    try:
        hidden_size = int(get_hidden_size()) if callable(get_hidden_size) else None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        hidden_size = None
    return infer_car_rmsnorm_model_family(vllm_config), quantized, hidden_size


def _policy_rule(
    *, tp_size: int, hidden_size: int, quantized: bool
) -> dict[str, Any] | None:
    for rule in CAR_RMSNORM_POLICY_TABLE:
        if (
            rule["tp_size"] == tp_size
            and rule["hidden_size"] == hidden_size
            and rule["quantized"] == quantized
        ):
            return rule
    return None


def can_use_registered_graph_input_for_generic_car(
    *,
    tp_size: int | None,
    hidden_size: int | None,
    model_family: str | None,
    quantized: bool | None,
) -> bool:
    """Return the generic CAR graph-input transport for a known policy cell.

    This transport decision is independent of whether a fused operator matches
    a particular graph. Unknown and unrelated cells retain the base
    registered-input behavior for generic custom all-reduce.
    """
    if not _concrete_int(tp_size) or not _concrete_int(hidden_size):
        return True
    if not isinstance(quantized, bool):
        return True
    canonical_family = _canonical_model_family(model_family)
    if canonical_family is None:
        return True
    rule = _policy_rule(
        tp_size=int(tp_size),
        hidden_size=int(hidden_size),
        quantized=quantized,
    )
    if rule is None or canonical_family != rule["family"]:
        return True
    return bool(rule.get("generic_graph_registered_input", True))


def fused_allreduce_rmsnorm_config_reject_reason(
    *,
    tp_size: int | None,
    pp_size: int | None,
    dtype: torch.dtype | None,
    hidden_size: int | None,
    model_family: str | None,
) -> str | None:
    """Return why the base CAR-RMSNorm capability is outside this contract.

    The capability predicate controls default pass enablement; the row
    predicate routes compile ranges to the fused or native implementation.
    Keeping both decisions here prevents the policies from drifting apart.
    """
    if not _concrete_int(tp_size) or tp_size <= 1:
        return f"unsupported or unknown tensor parallel size: {tp_size}"
    if not _concrete_int(pp_size) or pp_size != 1:
        return f"unsupported or unknown pipeline parallel size: {pp_size}"
    if dtype not in (torch.float16, torch.bfloat16):
        return f"unsupported activation dtype: {dtype}"
    if not _concrete_int(hidden_size) or hidden_size <= 0 or hidden_size % 8 != 0:
        return f"unsupported hidden size: {hidden_size}"
    if hidden_size > 16384:
        return f"unsupported hidden size: {hidden_size}"
    if _canonical_model_family(model_family) is None:
        return "model family is unknown or outside the Qwen3.5/3.6 contract"
    if _policy_rule(
        tp_size=int(tp_size),
        hidden_size=int(hidden_size),
        quantized=False,
    ) is None and _policy_rule(
        tp_size=int(tp_size),
        hidden_size=int(hidden_size),
        quantized=True,
    ) is None:
        return (
            "unsupported CAR-RMSNorm capability cell: "
            f"tp={tp_size} hidden={hidden_size}"
        )
    return None


def can_enable_fused_allreduce_rmsnorm(**kwargs: Any) -> bool:
    """Return whether the contract permits default CAR-RMSNorm enablement."""
    return fused_allreduce_rmsnorm_config_reject_reason(**kwargs) is None


def fused_allreduce_rmsnorm_compile_reject_reason(
    *,
    tp_size: int | None,
    hidden_size: int | None,
    dtype: torch.dtype | None,
    rows: int | None = None,
    compile_range: Any | None = None,
    raw_needed: bool | None = None,
    registered: bool | None = None,
    model_family: str | None = None,
    quantized: bool | None = None,
    phase: str | None = None,
    path: str | None = None,
) -> str | None:
    """Return why a fused CAR-RMSNorm operator must use native fallback.

    For target signatures, a compile range is accepted only when it excludes
    native rows and stays within the rule's fused-range bound. This keeps the
    provider choice within one native/fused region. ``raw_needed``,
    ``registered``, ``phase``, and ``path`` are neutral once known.

    Hidden sizes outside the table return ``None`` so the caller's capability
    checks continue to govern those paths.
    """
    if raw_needed is not None and not isinstance(raw_needed, bool):
        return "raw_needed must be a bool or None"
    if registered is not None and not isinstance(registered, bool):
        return "registered must be a bool or None"
    if phase is not None and phase not in _VALID_PHASES:
        return f"unsupported execution phase: {phase}"
    if path is not None and path not in _VALID_PATHS:
        return f"unsupported operator path: {path}"

    target_hidden = hidden_size in (
        FUSED_ALLREDUCE_RMSNORM_TARGET_HIDDEN_SIZE,
        FUSED_ALLREDUCE_RMSNORM_TP4_HIDDEN_SIZE,
    )
    if not target_hidden:
        return None
    if dtype not in (torch.float16, torch.bfloat16):
        return f"unsupported activation dtype: {dtype}"
    # TP1 never enters the CAR pass.  Keep the broad IR provider available for
    # ordinary fused-add RMSNorm callers; the CAR pass itself already rejects
    # tp_size <= 1 before this helper is reached.
    if not _concrete_int(tp_size):
        return "tensor-parallel size is unknown"
    if tp_size <= 1:
        return None
    if not isinstance(quantized, bool):
        return "quantization state is unknown"
    canonical_family = _canonical_model_family(model_family)
    if canonical_family is None:
        return "model family is unknown or outside the Qwen3.5/3.6 contract"

    rule = _policy_rule(
        tp_size=int(tp_size), hidden_size=int(hidden_size), quantized=quantized
    )
    if rule is None:
        return f"unsupported CAR-RMSNorm policy cell: tp={tp_size} hidden={hidden_size}"
    if canonical_family != rule["family"]:
        return f"model family outside policy cell: {model_family}"

    # A concrete row takes precedence over a symbolic compile range.
    if _concrete_int(rows) and rows >= 1:
        concrete_row = int(rows)
        if concrete_row in rule["native_rows"]:
            return (
                "native row: "
                f"tp={tp_size} hidden={hidden_size} rows={concrete_row}"
            )
        max_fused_rows = rule.get("fused_compile_max_rows")
        if max_fused_rows is not None and concrete_row > max_fused_rows:
            return (
                "row exceeds fused range: "
                f"tp={tp_size} hidden={hidden_size} rows={concrete_row} "
                f"max={max_fused_rows}"
            )

    bounds = _compile_range_bounds(compile_range)
    if compile_range is not None and bounds is None:
        return "invalid compile range"
    if bounds is not None:
        start, end = bounds
        native_rows = rule["native_rows"]
        crossed = sorted(row for row in native_rows if start <= row <= end)
        if crossed:
            return (
                "compile range intersects native rows: "
                f"range=({start}, {end}) rows={tuple(crossed)}"
            )
        max_fused_rows = rule.get("fused_compile_max_rows")
        if max_fused_rows is not None and end > max_fused_rows:
            return (
                "compile range exceeds fused range: "
                f"range=({start}, {end}) max={max_fused_rows}"
            )
        # A range inside the fused bucket that excludes native rows may fuse.
        return None
    else:
        if not _concrete_int(rows) or rows < 1:
            return "rows/compile range are unknown"
    # The signature remains bounded by the TP/hidden/dtype/family checks.
    return None


def can_use_fused_allreduce_rmsnorm(**kwargs: Any) -> bool:
    """Return whether the shared contract allows the fused operator."""
    return fused_allreduce_rmsnorm_compile_reject_reason(**kwargs) is None


def car_rmsnorm_default_on(vllm_config: Any) -> bool:
    """Return whether the contract enables CAR-RMSNorm by default.

    Single definition of the default-on rule; the platform writes the result
    into ``pass_config`` and every later reader consumes that settled value.
    Family and optimization level both come from one contract resolution,
    rather than from separate raw attribute reads.

    The level gate is load-bearing. Upstream derives the same pass value from
    ``OPTIMIZATION_LEVEL_TO_CONFIG`` -- ``False`` on O1/O0, and on O2 a
    CUDA-only predicate that answers ``False`` on MUSA. The platform hook runs
    first and wins only while the field is ``None``, so without this gate
    CAR-RMSNorm would stay enabled at ``-O0``/``-O1``.
    """
    from .resolver import resolve_optimization_contract

    contract = resolve_optimization_contract(vllm_config)
    if (contract.execution.optimization_level or 0) < 2:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    get_hidden_size = getattr(model_config, "get_hidden_size", None)
    hidden_size = get_hidden_size() if callable(get_hidden_size) else None
    return can_enable_fused_allreduce_rmsnorm(
        tp_size=getattr(parallel_config, "tensor_parallel_size", None),
        pp_size=getattr(parallel_config, "pipeline_parallel_size", None),
        dtype=getattr(model_config, "dtype", None),
        hidden_size=hidden_size,
        model_family=contract.model.family.value,
    )


def resolve_car_rmsnorm_enabled(vllm_config: Any) -> bool:
    """Return the effective CAR-RMSNorm pass value for ``vllm_config``.

    An explicit ``pass_config.fuse_allreduce_rms`` setting stays authoritative;
    only an unset value falls back to the contract default.
    """
    compilation_config = getattr(vllm_config, "compilation_config", None)
    pass_config = getattr(compilation_config, "pass_config", None)
    pass_value = getattr(pass_config, "fuse_allreduce_rms", None)
    if pass_value is not None:
        return pass_value is True
    return car_rmsnorm_default_on(vllm_config)
