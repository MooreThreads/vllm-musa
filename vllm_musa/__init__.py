# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM MUSA Platform Plugin

This plugin enables vLLM to run on Moore Threads MUSA GPUs.
It provides a MUSAPlatform implementation that integrates with vLLM's
platform abstraction layer.

Usage:
    Install this package alongside vLLM, and the MUSA platform will be
    automatically detected and used when running on Moore Threads hardware.
"""

import logging

__all__ = [
    "MUSAPlatform",
    "musa_platform_plugin",
    "register_custom_ops",
    "collect_env",
]
__version__ = "0.1.1"

logger = logging.getLogger(__name__)

# Import torchada early to ensure torch.device patching happens before
# any torch.device("cuda:X") calls in vLLM. This is critical for MUSA
# to work correctly - it patches torch.cuda to redirect to MUSA.
try:
    import torchada  # noqa: F401
    # XXX(MUSA): Remove once #JIRA MTAI-2652 fixed
    import torch
    torch.accelerator = torch.musa

    _torchada_available = True
except ImportError:
    _torchada_available = False

# Track whether patches have been applied in this process
_patches_applied = False


########### platform plugin ###########


def musa_platform_plugin() -> str | None:
    """Register the MUSA platform.

    vLLM platform plugin entry point. Called by vLLM to check if the MUSA
    platform is available. Returns the qualified class name if available.

    Note: We intentionally do NOT apply patches here because this function
    is called during vLLM module initialization which can cause circular
    import issues. Patches are applied via the general plugin mechanism.
    """
    # Check if torchada detected MUSA platform
    if _torchada_available:
        import torchada

        if torchada.is_musa_platform():
            return "vllm_musa.platform.MUSAPlatform"

    # Fallback: check if torch_musa is available
    try:
        import torch_musa  # noqa: F401

        return "vllm_musa.platform.MUSAPlatform"
    except ImportError:
        pass

    return None


########### general plugins ###########


def _apply_vllm_patches() -> None:
    """Apply vLLM source patches for MUSA compatibility.

    This function is idempotent - it only applies patches once per process.
    """
    global _patches_applied
    if _patches_applied:
        return

    try:
        from .patches import apply_patches

        apply_patches()
    except Exception as e:
        logger.error(f"Failed to apply vLLM patches: {e}")

    _patches_applied = True


def _register_patches() -> None:
    """Apply vLLM source patches for MUSA compatibility."""
    _apply_vllm_patches()


def _register_ops() -> None:
    """Register OOT custom ops (activation, layernorm, fused_moe, etc.)."""
    import vllm_musa.model_executor  # noqa: F401


def _register_modules() -> None:
    """Register distributed connectors, utils, and v1 attention backends."""
    import vllm_musa.distributed  # noqa: F401
    import vllm_musa.utils  # noqa: F401
    import vllm_musa.v1  # noqa: F401


def register_custom_ops() -> None:
    """
    vLLM general plugin entry point for MUSA customizations.

    This function is called by vLLM's general plugin mechanism after the
    platform is initialized, which avoids circular import issues.
    It applies vLLM source patches and registers all MUSA-specific ops,
    distributed connectors, and attention backends.
    """
    _register_patches()
    _register_ops()
    _register_modules()
    logger.info("MUSA patches and custom ops registered")


########### console scripts ###########


def collect_env() -> None:
    """Entry point for vllm_collect_env console script."""
    from .collect_env import main

    main()


########### lazy imports ###########


def __getattr__(name: str):
    """Lazy import module components."""
    if name == "MUSAPlatform":
        from .platform import MUSAPlatform

        return MUSAPlatform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
