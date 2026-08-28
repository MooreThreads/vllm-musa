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
# Canonical package version.  ``pyproject.toml`` reads this literal through
# ``tool.setuptools.dynamic`` so source checkouts (including uninstalled
# overlays) and installed distribution metadata expose the same identity.
__version__ = "0.1.28"

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
# no runtime source-patching and no fallback; only the object patches
# (_apply_object_patches) run at import time. The torch/vLLM config compat
# shims live as cat-6 object patches under vllm_musa/patches/ (applied by
# apply_object_patches alongside the spec-decode kernel prime and draft-TP=1
# wiring).


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


def _apply_object_patches() -> None:
    """Apply explicit in-process object/monkey patches for MUSA.

    These are the runtime patches: the spec-decode kernel prime, the draft-TP=1
    wiring, and the torch/vLLM config compat shims and dflash
    _custom_ops fallbacks — live-object edits with no source-diff form. (vLLM
    source edits are applied at build time; see the note above.) They
    are explicit, ordered, and idempotent ``apply()`` functions called by
    ``vllm_musa.patches.apply_object_patches``. Best-effort: a failure is logged,
    not raised.
    """
    try:
        from .patches import apply_object_patches

        apply_object_patches()
    except Exception as e:
        logger.error(f"Failed to apply object patches: {e}")


def patch_report() -> list[dict]:
    """Status of the vLLM-MUSA runtime object patches, read-only.

    Public entry point; delegates to ``vllm_musa.patches.patch_report``. Useful
    for ``vllm_collect_env`` and for debugging the object patches on a given vLLM
    version. (Source edits are applied at build time and are not audited here.)
    Returns a list of per-patch dicts; see that function.
    """
    from .patches import patch_report as _patch_report

    return _patch_report()


def _register_patches() -> None:
    """Register MUSA runtime patches.

    vLLM **source** edits are applied at BUILD time (setup.py +
    ``vllm_musa/patches/series/``), not here. the torch/vLLM config
    compat shims are now cat-6 object patches too, so ``_register_patches``
    installs only the runtime **object** monkey-patches (live-object edits that
    have no source-diff form), all via :func:`_apply_object_patches`.
    """
    _apply_object_patches()


def _register_ops() -> None:
    """Register OOT custom ops (activation, layernorm, fused_moe, etc.)."""
    # Preserve registration order: generic model-executor ops must exist before
    # the fused-MoE module performs its import-time registration and prewarm.
    # isort: off
    import vllm_musa.model_executor  # noqa: F401
    import vllm_musa.jit_kernel.csrc.moe as _moe  # noqa: F401

    # isort: on

    _moe.prewarm()


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
    It applies the runtime object patches and registers all MUSA-specific ops,
    distributed connectors, and attention backends.
    """
    # Must run before any Mooncake worker is constructed.  Keep this helper
    # stdlib-only so it does not pull in torch or replace upstream connector
    # objects during plugin discovery.
    from .distributed.mooncake_compat import configure_legacy_device_filter

    configure_legacy_device_filter()
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
