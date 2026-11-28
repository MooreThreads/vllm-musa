# SPDX-License-Identifier: Apache-2.0
"""KV-cache warmup block helpers for a pin without ``extensible_kv_cache``.

vLLM 0.34 introduced :mod:`vllm.v1.worker.extensible_kv_cache` and moved warmup
block-budget queries onto ``num_committable_kv_blocks`` and
``ensure_kv_cache_blocks``. ``vllm-omni``'s Triton attention prefill warmup
imports both helpers. On the fixed-allocation 0.32 pin ``runner`` has no
``extensible_kv_cache``, so every allocated KV block is already committed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_kv_cache_blocks", "num_committable_kv_blocks"]


def num_committable_kv_blocks(runner: Any) -> int:
    """Blocks warmup may address on the runner."""
    kv_cache = getattr(runner, "extensible_kv_cache", None)
    if kv_cache is not None:
        return int(kv_cache.warmup_committable_blocks)
    return int(runner.kv_cache_config.num_blocks)


def ensure_kv_cache_blocks(runner: Any, num_blocks: int) -> None:
    """Commit ``num_blocks`` on an extensible KV cache; no-op on 0.32."""
    kv_cache = getattr(runner, "extensible_kv_cache", None)
    if kv_cache is not None:
        kv_cache.commit(num_blocks)
