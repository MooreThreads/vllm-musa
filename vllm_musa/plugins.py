# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM general plugins of vLLM-MUSA.

Each ``register_*`` function registers one part of the MUSA integration and runs
at most once per process; the parts that touch vLLM apply the runtime object
patches first. ``register_all`` registers every part in order. It is published
under the platform plugin's entry-point name ``musa``, so any ``VLLM_PLUGINS``
value that enables the MUSA platform also registers every part.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

PARTS = ("kv_connector", "patches", "custom_ops", "attn_backends", "runtime")

_registered: set[str] = set()
_registering: set[str] = set()


def _register_once(part: str, register: Callable[[], None]) -> None:
    # A part re-entered while it registers is skipped; a part that raised is
    # registered again on the next call.
    if part in _registered or part in _registering:
        return
    _registering.add(part)
    try:
        register()
    finally:
        _registering.discard(part)
    _registered.add(part)


def _configure_kv_connector() -> None:
    # Standard library only, so it runs before vLLM constructs a Mooncake worker.
    from .distributed.mooncake_compat import configure_legacy_device_filter

    configure_legacy_device_filter()


def _apply_patches() -> None:
    try:
        from .patches import apply_object_patches

        apply_object_patches()
    except Exception as e:
        logger.error("Failed to apply object patches: %s", e)


def _register_custom_ops() -> None:
    import vllm_musa.model_executor  # noqa: F401

    from .custom_op_modules import preload_custom_op_modules
    from .jit_kernel.csrc import moe

    preload_custom_op_modules()
    moe.prewarm()


def _register_attn_backends() -> None:
    import vllm_musa.v1.attention.backends.mla.common  # noqa: F401
    import vllm_musa.v1.attention.backends.mla.flashmla  # noqa: F401
    import vllm_musa.v1.attention.backends.mla.flashmla_sparse  # noqa: F401

    from .v1.attention.registry import register_attention_backends

    register_attention_backends()


def _register_runtime() -> None:
    import vllm_musa.v1.executor  # noqa: F401
    import vllm_musa.v1.sample  # noqa: F401
    import vllm_musa.v1.spec_decode.utils  # noqa: F401
    import vllm_musa.v1.worker  # noqa: F401


def register_kv_connector() -> None:
    """Prepare KV-transfer connectors: map the legacy Mooncake device filter."""
    _register_once("kv_connector", _configure_kv_connector)


def register_patches() -> None:
    """Apply the runtime object patches (best effort; failures are logged)."""
    _register_once("patches", _apply_patches)


def register_custom_ops() -> None:
    """Register OOT custom ops, preload kernel modules and prewarm the MoE JIT."""
    register_patches()
    _register_once("custom_ops", _register_custom_ops)


def register_attn_backends() -> None:
    """Register the MUSA attention backends and their MLA overrides."""
    register_patches()
    _register_once("attn_backends", _register_attn_backends)


def register_runtime() -> None:
    """Install the executor, sampler, spec-decode and worker overrides."""
    register_patches()
    _register_once("runtime", _register_runtime)


def register_all() -> None:
    """Register every MUSA part in order."""
    if _registered.issuperset(PARTS):
        return
    register_kv_connector()
    register_patches()
    register_custom_ops()
    register_attn_backends()
    register_runtime()
    logger.info("MUSA plugins registered: %s", ", ".join(PARTS))
