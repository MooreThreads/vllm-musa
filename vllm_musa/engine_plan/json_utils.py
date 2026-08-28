# SPDX-License-Identifier: Apache-2.0

"""Strict JSON helpers shared by the offline SDK and its CLI."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not supported")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate members."""

    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def ensure_finite(value: Any, *, field: str = "JSON value") -> None:
    """Reject NaN/Infinity, including overflowed JSON exponents."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} must contain only finite JSON numbers")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            ensure_finite(key, field=f"{field} key")
            ensure_finite(nested, field=f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            ensure_finite(nested, field=f"{field}[{index}]")


def loads(text: str, *, source: str) -> Any:
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
        ensure_finite(value, field=source)
        return value
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Unable to parse JSON from {source}: {exc}") from exc


def dumps(value: Any, **kwargs: Any) -> str:
    ensure_finite(value)
    return json.dumps(value, allow_nan=False, **kwargs)
