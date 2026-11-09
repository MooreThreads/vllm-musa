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

from .plugins import (
    register_all,
    register_attn_backends,
    register_custom_ops,
    register_kv_connector,
    register_runtime,
)

__all__ = [
    "MUSAPlatform",
    "musa_platform_plugin",
    "register_all",
    "register_attn_backends",
    "register_custom_ops",
    "register_kv_connector",
    "register_runtime",
    "collect_env",
]
__version__ = "0.1.32"

logger = logging.getLogger(__name__)

# Import torchada early to ensure torch.device patching happens before
# any torch.device("cuda:X") calls in vLLM. This is critical for MUSA
# to work correctly - it patches torch.cuda to redirect to MUSA.
try:
    # isort: off
    import torchada  # noqa: F401
    import torch  # noqa: F401

    # isort: on
    _torchada_available = True
except ImportError:
    _torchada_available = False

# vLLM source patches are applied at BUILD time to the cloned vLLM
# (setup.py -> build_apply.py -> vllm_musa/patches/series/ diff series). There is
# no runtime source-patching and no fallback; at runtime only the object patches
# under vllm_musa/patches/ run, applied by vllm_musa.plugins.register_patches.


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


# Optional, default-off registration of Inductor GEMM template heuristics
# for device_type='musa'. The default ATen lowering is the fast path; set
# VLLM_MUSA_ENABLE_INDUCTOR_HEURISTICS=1 to experiment with Triton GEMM
# autotuning. No-ops quietly on old torch versions.
try:
    from vllm_musa._inductor import maybe_register_musa_template_heuristics

    maybe_register_musa_template_heuristics()
except Exception as _exc:  # pragma: no cover
    logger.warning(
        "failed to register Inductor template heuristics for "
        "MUSA (%s); falling back to ATen path for compiled `mm` ops.",
        _exc,
    )


########### general plugins ###########
# The vllm.general_plugins entry points are defined in vllm_musa.plugins.


def patch_report() -> list[dict]:
    """Status of the vLLM-MUSA runtime object patches, read-only.

    Public entry point; delegates to ``vllm_musa.patches.patch_report``. Useful
    for ``vllm_collect_env`` and for debugging the object patches on a given vLLM
    version. (Source edits are applied at build time and are not audited here.)
    Returns a list of per-patch dicts; see that function.
    """
    from .patches import patch_report as _patch_report

    return _patch_report()


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
