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
from contextlib import nullcontext
from functools import wraps
from typing import Any

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
    # isort: off
    import torchada  # noqa: F401
    import torch

    # isort: on
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


# MUSA-0087: register Inductor template heuristics for device_type='musa'
# so compiled mm/bmm/addmm/baddbmm/scaled_mm ops use Triton autotune
# instead of falling through to the empty fallback heuristic.
# Default-OFF (Eagle3 TP=8 crash on M2.5); opt in for non-Eagle3 workloads
# via VLLM_MUSA_ENABLE_INDUCTOR_HEURISTICS=1. Also silently no-ops on
# old torch versions.
try:
    from vllm_musa._inductor import maybe_register_musa_template_heuristics

    maybe_register_musa_template_heuristics()
except Exception as _exc:  # pragma: no cover
    logger.warning(
        "MUSA-0087: failed to register Inductor template heuristics for "
        "MUSA (%s); falling back to ATen path for compiled `mm` ops.",
        _exc,
    )


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


def _patch_vllm_backend_call_options() -> None:
    """Accept torch.compile backend keyword options on this vLLM snapshot."""
    try:
        from vllm.compilation.backends import VllmBackend
    except Exception as e:
        logger.debug("Skipping VllmBackend options patch: %s", e)
        return

    original_call = VllmBackend.__call__
    if getattr(original_call, "_musa_accepts_backend_options", False):
        return

    @wraps(original_call)
    def call_with_ignored_options(self, graph, example_inputs, **kwargs):
        return original_call(self, graph, example_inputs)

    call_with_ignored_options._musa_accepts_backend_options = True
    VllmBackend.__call__ = call_with_ignored_options


def _filter_existing_config(
    config: dict[str, Any], config_module: Any
) -> dict[str, Any]:
    """Drop config keys that are absent in the installed Torch."""
    return {key: value for key, value in config.items() if hasattr(config_module, key)}


def _make_config_patch_filter(original_patch: Any, config_module: Any) -> Any:
    @wraps(original_patch)
    def patch_existing_config(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], dict):
            config = _filter_existing_config(args[0], config_module)
            if not config and not kwargs:
                return nullcontext()
            args = (config, *args[1:])
        elif args and isinstance(args[0], str):
            if not hasattr(config_module, args[0]):
                return nullcontext()

        if kwargs:
            kwargs = _filter_existing_config(kwargs, config_module)
            if not args and not kwargs:
                return nullcontext()

        return original_patch(*args, **kwargs)

    patch_existing_config._musa_filters_missing_config_keys = True
    return patch_existing_config


def _patch_functorch_config_patch() -> None:
    """Ignore missing functorch config keys in vLLM compile contexts."""
    try:
        from torch._functorch import config as functorch_config
    except Exception as e:
        logger.debug("Skipping functorch config.patch patch: %s", e)
        return

    original_patch = functorch_config.__dict__.get("patch", functorch_config.patch)
    if getattr(original_patch, "_musa_filters_missing_config_keys", False):
        return

    functorch_config.__dict__["patch"] = _make_config_patch_filter(
        original_patch, functorch_config
    )


def _patch_inductor_config_patch() -> None:
    """Ignore missing inductor config keys in vLLM compile contexts."""
    try:
        from torch._inductor import config as inductor_config
    except Exception as e:
        logger.debug("Skipping inductor config.patch patch: %s", e)
        return

    original_patch = inductor_config.__dict__.get("patch", inductor_config.patch)
    if getattr(original_patch, "_musa_filters_missing_config_keys", False):
        return

    inductor_config.__dict__["patch"] = _make_config_patch_filter(
        original_patch, inductor_config
    )


def _patch_vllm_functorch_config() -> None:
    """Filter vLLM functorch config overrides for this Torch version."""
    try:
        import importlib

        compiler_interface = importlib.import_module(
            "vllm.compilation.compiler_interface"
        )
        from torch._functorch import config as functorch_config
    except Exception as e:
        logger.debug("Skipping functorch config patch: %s", e)
        return

    original_get_config = compiler_interface._get_vllm_functorch_config
    if getattr(original_get_config, "_musa_filters_functorch_config", False):
        return

    @wraps(original_get_config)
    def get_existing_functorch_config() -> dict[str, Any]:
        return _filter_existing_config(original_get_config(), functorch_config)

    get_existing_functorch_config._musa_filters_functorch_config = True
    compiler_interface._get_vllm_functorch_config = get_existing_functorch_config


def _register_patches() -> None:
    """Apply vLLM source patches for MUSA compatibility."""
    _apply_vllm_patches()
    _patch_functorch_config_patch()
    _patch_inductor_config_patch()
    _patch_vllm_backend_call_options()
    _patch_vllm_functorch_config()


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
