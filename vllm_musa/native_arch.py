# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check a device against the MUSA archs the native extensions were built for."""

from __future__ import annotations

import torch
from torchada import arch as musa_arch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _names(archs) -> str:
    return "[{}]".format(", ".join(a.name for a in archs))


def built_archs() -> tuple[musa_arch.MusaArch, ...] | None:
    """The archs setup.py recorded for the native build, or ``None`` without a record."""
    try:
        from vllm_musa._build_info import MUSA_ARCHS
    except ImportError:
        return None
    return tuple(musa_arch.MusaArch(major, minor) for major, minor in MUSA_ARCHS)


def check_device_arch(device: torch.device) -> None:
    """Raise when ``device``'s arch is not one the native extensions were built for.

    A kernel built for another arch does not launch on the device
    (``musaErrorInvalidDeviceFunction``) and leaves its output unwritten.
    """
    archs = built_archs()
    if archs is None:
        logger.debug("No MUSA build arch record; skipping the native arch check.")
        return
    device_arch = musa_arch.MusaArch(*torch.musa.get_device_capability(device))
    if device_arch not in archs:
        if device_arch in musa_arch.ARCHS:
            remedy = f"Rebuild vllm-musa with MTGPU_TARGET={device_arch.name}."
        else:
            remedy = f"vllm-musa builds only for {_names(musa_arch.ARCHS)}."
        raise RuntimeError(
            f"vllm-musa's native extensions were built for {_names(archs)}, but "
            f"{device} is {device_arch.name}; their kernels do not launch there "
            f"(musaErrorInvalidDeviceFunction). {remedy}"
        )
