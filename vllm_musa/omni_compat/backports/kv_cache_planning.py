# SPDX-License-Identifier: Apache-2.0
"""``vllm.v1.core.kv_cache_planning`` for a pin that only has ``kv_cache_utils``.

vLLM moved the planning logic out of ``vllm.v1.core.kv_cache_utils`` into
``kv_cache_planning`` in the v0.32.1 line, then in 0.33 split the global layer
grouping out into its own public ``get_global_kv_cache_groups`` and gave
``get_kv_cache_configs`` a fourth parameter carrying those groups. The pin
predates both.
vLLM-Omni main calls the 0.33 shape:

    global_kv_cache_groups = get_global_kv_cache_groups(vllm_config, worker_specs)
    vllm_config.adjust_dcp_kv_cache_interleave_size(global_kv_cache_groups)
    worker_configs = get_kv_cache_configs(
        vllm_config, worker_specs, available_memory, global_kv_cache_groups
    )

Both functions here are built from the pinned implementation rather than
reimplemented: 0.32's ``get_kv_cache_configs`` computes the same global groups
internally, so :func:`get_global_kv_cache_groups` reproduces that prefix
statement for statement against the pinned helpers, and
:func:`get_kv_cache_configs` drops the argument the pin does not take and lets it
recompute the identical grouping. The grouping is a pure function of
``(vllm_config, merged specs)``, so computing it twice costs one extra pass over
the layer dict at startup and cannot diverge.

Not covered, by construction: ``VllmConfig.adjust_dcp_kv_cache_interleave_size``
between the two calls above means something different on the pin (it wants a
``KVCacheConfig``, not the group list), so decode context parallelism with a
NixlConnector is unsupported here. That is recorded as the ``CAPABILITY`` entry
in :mod:`vllm_musa.omni_compat.table`, which the audit reports; it is deliberately
not papered over, because the pinned value would be silently wrong rather than
missing.

Delete this module when ``third_party/PINS`` reaches a vLLM that ships
``kv_cache_planning``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import KVCacheConfig
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec

__all__ = ["get_global_kv_cache_groups", "get_kv_cache_configs"]


def get_global_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
) -> list[KVCacheGroupSpec]:
    """Group the layers of the whole model (all workers' specs merged).

    Mirrors the prefix of the pinned ``get_kv_cache_configs``: merge every
    worker's specs, assert the registry knows them, apply the prefill-lookahead
    retention to sliding-window layers, then group.
    """
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    from vllm.v1.kv_cache_interface import SlidingWindowSpec
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)

    extra_retained_tokens = max(0, vllm_config.num_prefill_lookahead_tokens - 1)
    for layer_name, layer_spec in merged_kv_cache_specs.items():
        if isinstance(layer_spec, SlidingWindowSpec):
            merged_kv_cache_specs[layer_name] = replace(
                layer_spec, extra_retained_tokens=extra_retained_tokens
            )

    return get_kv_cache_groups(vllm_config, merged_kv_cache_specs)


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
    global_kv_cache_groups: list[KVCacheGroupSpec] | None = None,
) -> list[KVCacheConfig]:
    """The pinned planner, accepting 0.33's fourth argument and recomputing it.

    ``global_kv_cache_groups`` is accepted for signature compatibility and
    deliberately not forwarded: the pinned planner derives the same groups from
    ``(vllm_config, kv_cache_specs)`` itself, and passing a caller-supplied list
    in would require reaching into the pin's private planning steps.
    """
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs as _pinned

    del global_kv_cache_groups  # recomputed by the pinned planner
    return _pinned(vllm_config, kv_cache_specs, available_memory)
