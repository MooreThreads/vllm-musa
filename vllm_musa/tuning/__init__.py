# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared empirically measured kernel-selection thresholds.

Per-kernel tables live next to this module as ``<kernel>.json``:
``{"schema": 1, "kernel": ..., "source": ..., "entries": [...]}``, where every
entry value is a positive integer or a non-empty object of them. ``load``
checks that envelope; each kernel checks its own entry keys.
"""

from __future__ import annotations

import json
from importlib import resources

# Minimum rows where the MUSA JIT fused-add RMSNorm provider was measured to
# outperform the fallback for contiguous BF16 H5120 workloads on S5000.
FUSED_ADD_RMSNORM_MIN_ROWS = 64

_SCHEMA = 1
_KEYS = {"schema", "kernel", "source", "entries"}


def load(kernel: str) -> list[dict]:
    """The entries of the measured table ``<kernel>.json``, validated."""
    name = f"{kernel}.json"
    return _parse(name, kernel, resources.files(__name__).joinpath(name).read_text())


def _parse(name: str, kernel: str, text: str) -> list[dict]:
    table = json.loads(text)
    if not isinstance(table, dict) or set(table) != _KEYS:
        raise ValueError(f"{name}: top-level keys must be {sorted(_KEYS)}")
    if table["schema"] != _SCHEMA:
        raise ValueError(f"{name}: schema must be {_SCHEMA}, got {table['schema']!r}")
    if table["kernel"] != kernel:
        raise ValueError(f"{name}: kernel must be {kernel!r}, got {table['kernel']!r}")
    if not isinstance(table["source"], str) or not table["source"].strip():
        raise ValueError(f"{name}: source must name where the numbers were measured")
    entries = table["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{name}: entries must be a non-empty list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{name} entry {index}: must be an object")
        _check(name, index, entry, "entry")
    return entries


def _check(name: str, index: int, value: object, path: str) -> None:
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{name} entry {index}: {path} is empty")
        for key, item in value.items():
            _check(name, index, item, f"{path}.{key}" if path != "entry" else key)
    elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{name} entry {index}: {path} must be a positive integer, got {value!r}"
        )
