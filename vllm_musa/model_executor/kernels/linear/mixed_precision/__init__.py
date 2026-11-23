# SPDX-License-Identifier: Apache-2.0
"""MUSA ``MPLinearKernel`` implementations.

Importing this package must stay free of heavy imports: it is reached from a ``vllm.general_plugins``
entry point during startup, and MATE is an optional dependency. ``register`` therefore imports the
kernel module lazily and returns quietly when MATE is absent, so a MUSA image without MATE keeps the
shipped kernels instead of failing to start.
"""

from __future__ import annotations


def register() -> None:
    """Register the MATE-backed W4A16 kernel for the out-of-tree MUSA platform."""
    try:
        from .mate_w4a16 import MateW4A16LinearKernel
    except ImportError:
        # MATE (and therefore its dense int4 x fp16 kernel) is not installed in this image.
        return

    from vllm.model_executor.kernels.linear import register_linear_kernel
    from vllm.platforms.interface import PlatformEnum

    # "mp" is the name vLLM's MUSA platform reports for a mixed-precision layer, and the last
    # registrant wins, so this replaces the upstream W4A16 entry rather than competing with it.
    register_linear_kernel(MateW4A16LinearKernel, PlatformEnum.OOT, "mp")
