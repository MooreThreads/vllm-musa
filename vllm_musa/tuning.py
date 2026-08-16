# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-local kernel policy materialized before compilation.

The values in this module are conservative defaults.  An external engine plan
may replace them while vLLM is resolving platform defaults, before Dynamo or a
graph capture observes the corresponding custom-op implementation.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock

DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS = 64
FUSED_ADD_RMSNORM_TUNED_HIDDEN_SIZES = frozenset({4096, 5120})

# Backward-compatible name for code and tests that need the built-in fallback,
# not the effective value selected by a RuntimePlan.
FUSED_ADD_RMSNORM_MIN_ROWS = DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS

# The first AutoTuner vertical slice searches power-of-two crossovers.  A value
# above a workload's maximum disables the JIT runner for that envelope without
# adding a separate boolean decision.
FUSED_ADD_RMSNORM_THRESHOLD_CHOICES = tuple(1 << power for power in range(15))

_TUNING_STATE_LOCK = RLock()
_RESOLVED_FUSED_ADD_RMSNORM_MIN_ROWS_ENV = (
    "VLLM_MUSA_INTERNAL_FUSED_ADD_RMSNORM_MIN_ROWS"
)


def is_fused_add_rmsnorm_tuned_hidden_size(hidden_size):
    """Return a bool/SymBool without hashing or coercing symbolic dimensions."""

    candidates = sorted(FUSED_ADD_RMSNORM_TUNED_HIDDEN_SIZES)
    matched = hidden_size == candidates[0]
    for candidate in candidates[1:]:
        matched = matched | (hidden_size == candidate)
    return matched


def _validate_positive_rows(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("fused-add RMSNorm minimum rows must be a positive integer")
    return value


def _initial_fused_add_rmsnorm_min_rows() -> int:
    raw = os.getenv(_RESOLVED_FUSED_ADD_RMSNORM_MIN_ROWS_ENV)
    if raw is None:
        return DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS
    try:
        return _validate_positive_rows(int(raw))
    except ValueError as exc:
        raise ValueError(
            f"invalid internal fused-add RMSNorm threshold {raw!r}"
        ) from exc


_fused_add_rmsnorm_min_rows = _initial_fused_add_rmsnorm_min_rows()


def configure_fused_add_rmsnorm_min_rows(value: int) -> int:
    """Materialize the immutable plan choice used by subsequent compilation."""

    value = _validate_positive_rows(value)
    global _fused_add_rmsnorm_min_rows
    with _TUNING_STATE_LOCK:
        _fused_add_rmsnorm_min_rows = value
        # vLLM workers use spawn.  The RuntimePlan is resolved before workers
        # start, so use a private, write-only transport to materialize the same
        # immutable decision in child imports.  This is not a user tuning knob.
        os.environ[_RESOLVED_FUSED_ADD_RMSNORM_MIN_ROWS_ENV] = str(value)
    return value


def get_fused_add_rmsnorm_min_rows() -> int:
    """Return the effective threshold for the current worker process."""

    with _TUNING_STATE_LOCK:
        return _fused_add_rmsnorm_min_rows


@contextmanager
def override_fused_add_rmsnorm_min_rows(value: int) -> Iterator[None]:
    """Temporarily select a threshold for an explicit offline tuning trial."""

    previous = get_fused_add_rmsnorm_min_rows()
    configure_fused_add_rmsnorm_min_rows(value)
    try:
        yield
    finally:
        configure_fused_add_rmsnorm_min_rows(previous)
