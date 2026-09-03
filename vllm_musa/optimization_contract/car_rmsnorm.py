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
FUSED_ALLREDUCE_RMSNORM_TP2_MIN_ROWS = 64
FUSED_ALLREDUCE_RMSNORM_TP4_HIDDEN_SIZE = 2048
FUSED_ALLREDUCE_RMSNORM_TP4_SINGLETON_ROWS = 64
FUSED_ALLREDUCE_RMSNORM_TP4_POLICY = "h2048-row64-table-v3"

# Normalize the accepted short and resolver model-family spellings.
FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY = "qwen3.5_3.6"
_MODEL_FAMILY_ALIASES = frozenset({"qwen3.5", "qwen3.5_3.6"})

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
