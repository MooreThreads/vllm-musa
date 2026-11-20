# SPDX-License-Identifier: Apache-2.0
"""Restore the v0.34 KV-cache block-table helper on the v0.32 pin."""

from __future__ import annotations

import math
from types import ModuleType

ADAPTED = "__vllm_musa_get_block_table_width_bridge__"


def _cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _get_block_table_width(
    max_num_blocks: int,
    block_size: int,
    *,
    token_alignment: int | None = 128,
) -> int:
    if token_alignment is not None:
        if token_alignment <= 0:
            raise ValueError("token_alignment must be positive")
        block_alignment = token_alignment // math.gcd(token_alignment, block_size)
        max_num_blocks = _cdiv(max_num_blocks, block_alignment) * block_alignment
    return max_num_blocks


def is_applied(module: ModuleType) -> list[str]:
    bridge = getattr(module, ADAPTED, None)
    return ["get_block_table_width"] if bridge is getattr(module, "get_block_table_width", None) else []


def is_satisfied(module: ModuleType) -> bool:
    return callable(getattr(module, "get_block_table_width", None))


def apply(module: ModuleType) -> list[str]:
    if hasattr(module, "get_block_table_width"):
        return []
    module.get_block_table_width = _get_block_table_width
    setattr(module, ADAPTED, _get_block_table_width)
    return ["get_block_table_width"]
