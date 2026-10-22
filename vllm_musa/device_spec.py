# SPDX-License-Identifier: Apache-2.0
"""Per-architecture hardware facts for MUSA devices.

Why this module exists
----------------------
MUSA architectures are **not** ordered the way NVIDIA compute capabilities are.
The axes that change kernel selection are independent:

* warp width is 32 on ``mp_31`` and 128 elsewhere;
* native FP8 is an ``mp_31``-only build feature (``setup.py`` appends
  ``-DENABLE_FP8`` only for that target);
* ``__syncthreads_lm`` and the ``ST.B64 ... slc=new`` cache hint exist only on
  ``mp_31``;
* the multiprocessor count differs per bin (48/52/56/60) *within* one arch.

Comparing a capability tuple against a literal therefore encodes one machine's
properties as if they were an ordering.  ``(3, 1)`` is an identity, not a
threshold, and an arch we have never validated must never silently inherit the
S5000 tuning.

Two rules this module enforces:

1. Use the capability tuple as an **identity** (which arch is this), never as a
   magnitude (is this at least as new as X).
2. Expose each hardware *capability* as an explicit boolean backed by an
   explicit set, so an unknown arch answers ``False`` instead of falling into
   whichever side of a numeric comparison it happens to land on.

The field set mirrors ``torch_musa.get_device_properties()`` (verified on an
S5000 with torch_musa 2.11.0.post2): every field below is reported by the
driver, so nothing here needs a hand-maintained per-arch table.

Note on ``project`` vs. runtime arch: ``torch_musa.get_arch_list()`` reports the
arches the **wheel was built for** (``['31']`` on the 5.2.0 wheel), not the
arch of the device in front of you.  Always take the device arch from
``get_device_properties().major/minor``.
"""

from __future__ import annotations

import importlib

import functools
import enum
from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


__all__ = [
    "MusaFeature",
    "MusaDeviceSpec",
    "VALIDATED_MUSA_ARCHS",
    "get_device_spec",
    "current_device_spec",
]


# Arches on which the S5000-gated code paths have actually been validated.
# Adding an arch here is a hardware decision: every kernel behind
# ``is_validated`` must have been run on that arch first.  An arch that is
# absent gets the generic path and a one-time warning, never the tuned one.
VALIDATED_MUSA_ARCHS: frozenset[int] = frozenset({310})

# Native FP8 (``-DENABLE_FP8``) is a per-target build feature, not a property of
# "newer" arches: ``setup.py`` adds the define only for ``mp_31``.  Keep this an
# explicit set.
NATIVE_FP8_ARCHS: frozenset[int] = frozenset({310})

# Arches whose warp is 32 lanes wide.  Kept as a set for the same reason as
# above; the width is also read from the device at runtime, and this set only
# serves as a consistency check.
_WARP32_ARCHS: frozenset[int] = frozenset({310})


class MusaFeature(enum.Enum):
    """Hardware capabilities that are selected by explicit sets, not ordering."""

    #: The build carries native FP8 kernels for this arch.
    NATIVE_FP8 = enum.auto()
    #: Kernel/scheduling choices gated on the S5000 (mp_31) validation.
    VALIDATED_TARGET = enum.auto()


@dataclass(frozen=True)
class MusaDeviceSpec:
    """Immutable hardware facts for one MUSA device.

    Construct with :func:`get_device_spec`; the raw dataclass is frozen so a
    cached instance cannot be mutated into disagreeing with the device.
    """

    #: ``(major, minor)`` as reported by the driver, e.g. ``(3, 1)``.
    device_capability: tuple[int, int]
    #: Integer arch code, ``major * 100 + minor * 10`` (``310`` for ``(3, 1)``).
    #: This is the value compared against ``__MUSA_ARCH__`` in device code and
    #: against ``-D__MUSA_ARCH_LIST__`` in TileLang templates.
    arch_code: int
    #: Compiler/CMake target name for this arch, e.g. ``"mp_31"``.
    arch_name: str
    #: Runtime multiprocessor count for this bin (48/52/56/60 on S5000-class
    #: parts).  Never derive the tuning from the arch alone.
    mp_count: int
    #: Lanes per warp. 32 on mp_31, 128 on other MUSA arches.
    warp_size: int
    total_memory: int
    l2_cache_size: int
    shared_memory_per_block: int
    shared_memory_per_multiprocessor: int
    regs_per_multiprocessor: int
    max_threads_per_block: int
    max_threads_per_multiprocessor: int
    device_name: str
    uuid: Optional[str] = None

    # ---- identity helpers -------------------------------------------------

    @property
    def is_validated(self) -> bool:
        """Whether S5000-gated code paths may run on this arch."""
        return self.arch_code in VALIDATED_MUSA_ARCHS

    @property
    def supports_native_fp8(self) -> bool:
        return self.arch_code in NATIVE_FP8_ARCHS

    def supports(self, feature: MusaFeature) -> bool:
        """Answer a capability question from the explicit per-arch sets."""
        if feature is MusaFeature.NATIVE_FP8:
            return self.supports_native_fp8
        if feature is MusaFeature.VALIDATED_TARGET:
            return self.is_validated
        raise ValueError(f"unknown MUSA feature: {feature!r}")

    # ---- cache-key helpers ----------------------------------------------

    @property
    def cache_tag(self) -> str:
        """Stable tag for compile-cache paths.

        Includes the multiprocessor count because two bins of the same arch
        carry different ideal tilings, and a cached artifact built for one bin
        must not be silently reused by the other.
        """
        return f"{self.arch_name}_mp{self.mp_count}"

    def __str__(self) -> str:  # pragma: no cover - logging aid
        return (
            f"{self.device_name} ({self.arch_name}, {self.device_capability}, "
            f"mp={self.mp_count}, warp={self.warp_size})"
        )


def _arch_code(major: int, minor: int) -> int:
    # Matches the compiler's arch numbering: (3, 1) -> 310, (4, 2) -> 420.
    return int(major) * 100 + int(minor) * 10


def _arch_name(major: int, minor: int) -> str:
    return f"mp_{int(major)}{int(minor)}"


def _int_attr(obj: Any, *names: str) -> int:
    """First present attribute among ``names`` as int, else 0.

    torch_musa has renamed a couple of property fields between releases
    (``max_threads_per_multi_processor`` vs ``max_threads_per_multiprocessor``),
    so read them tolerantly instead of pinning one spelling.
    """
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _get_properties(device_id: int) -> Any:
    """Return a ``_MusaDeviceProperties`` for ``device_id``.

    ``torch.musa`` is the torchada-provided namespace and is what the rest of
    the tree uses; ``torch_musa`` is the direct module.  Either is fine, but
    both must be reached lazily so importing this module stays cheap.
    """
    musa = getattr(torch, "musa", None)
    if musa is not None and hasattr(musa, "get_device_properties"):
        return musa.get_device_properties(device_id)
    import torch_musa  # noqa: PLC0415 - deliberately lazy

    return torch_musa.get_device_properties(device_id)


@functools.lru_cache(maxsize=None)
def get_device_spec(device_id: int = 0) -> Optional[MusaDeviceSpec]:
    """Return the :class:`MusaDeviceSpec` for ``device_id``, or ``None``.

    ``None`` means "could not determine", which every caller must treat as
    "not validated": falling back to the generic path is safe, whereas guessing
    an arch is not.  Runtime failures are swallowed because this is called from
    platform probes that run before the device is necessarily usable.
    """
    try:
        props = _get_properties(device_id)
        major = int(getattr(props, "major"))
        minor = int(getattr(props, "minor"))
    except Exception as exc:  # noqa: BLE001 - probe path, must not raise
        logger.debug("MUSA device properties unavailable for device %s: %s",
                     device_id, exc)
        return None

    arch_code = _arch_code(major, minor)
    spec = MusaDeviceSpec(
        device_capability=(major, minor),
        arch_code=arch_code,
        arch_name=_arch_name(major, minor),
        mp_count=_int_attr(props, "multi_processor_count"),
        warp_size=_int_attr(props, "warp_size"),
        total_memory=_int_attr(props, "total_memory"),
        l2_cache_size=_int_attr(props, "L2_cache_size", "l2_cache_size"),
        shared_memory_per_block=_int_attr(props, "shared_memory_per_block"),
        shared_memory_per_multiprocessor=_int_attr(
            props,
            "shared_memory_per_multiprocessor",
            "shared_memory_per_multi_processor",
        ),
        regs_per_multiprocessor=_int_attr(
            props, "regs_per_multiprocessor", "regs_per_multi_processor"
        ),
        max_threads_per_block=_int_attr(props, "max_threads_per_block"),
        max_threads_per_multiprocessor=_int_attr(
            props,
            "max_threads_per_multiprocessor",
            "max_threads_per_multi_processor",
        ),
        device_name=str(getattr(props, "name", "unknown")),
        uuid=getattr(props, "uuid", None),
    )

    if arch_code not in VALIDATED_MUSA_ARCHS:
        logger.warning_once(
            "MUSA arch %s (%s, mp=%d) has not been validated; S5000-specific "
            "kernel and tuning paths stay disabled and the generic paths are "
            "used instead. See vllm_musa/device_spec.py.",
            spec.arch_name,
            spec.device_capability,
            spec.mp_count,
        )

    _assert_build_targets_device(spec)

    if (spec.warp_size == 32) != (arch_code in _WARP32_ARCHS):
        # The device is authoritative; this only catches a wrong assumption in
        # code that hardcodes a warp width for an arch.
        logger.warning_once(
            "MUSA arch %s reports warp_size=%d, which disagrees with the "
            "recorded narrow-warp arch set; device properties win.",
            spec.arch_name,
            spec.warp_size,
        )

    return spec


def is_validated_device(device_id: Optional[int] = None) -> bool:
    """Whether S5000-calibrated code paths may run on this device.

    The single call-site helper for "is this arch one we have validated?":
    asked as an identity check on the arch code, never as an ordering, because
    MUSA capability tuples are not monotonic.  A device whose spec cannot be
    read is not validated -- the safe answer for a gating question.
    """
    try:
        spec = current_device_spec() if device_id is None else get_device_spec(device_id)
    except Exception:  # noqa: BLE001 - a failed probe must not enable anything
        return False
    return spec is not None and spec.is_validated


def current_device_spec() -> Optional[MusaDeviceSpec]:
    """Spec for the current device, or ``None`` when it cannot be determined."""
    try:
        device_id = torch.musa.current_device()
    except Exception:  # noqa: BLE001
        device_id = 0
    return get_device_spec(device_id)

# --------------------------------------------------------------------------
# Build-time arch record
# --------------------------------------------------------------------------
#: Written by ``setup.py`` at build time into both the source tree and
#: ``build_lib``, so an editable install and a wheel both carry it.  Only a
#: checkout that was never built has none, which is how "unknown" stays
#: distinguishable from "built for the wrong arch".
_BUILD_STAMP_MODULE = "vllm_musa._build_stamp"


@functools.cache
def read_build_stamp() -> Optional[dict]:
    """The arch record of the compiled extension, or ``None`` if unknown.

    Cached for the process lifetime: the stamp is written by ``setup.py``, so a
    rebuild needs a restart to be seen.  Caching it deliberately (rather than
    letting ``import_module`` cache it accidentally) keeps "absent stamp" a
    stable answer instead of a per-call race with a build running alongside.
    """
    try:
        module = importlib.import_module(_BUILD_STAMP_MODULE)
    except Exception:  # noqa: BLE001 - a checkout that was never built
        return None
    return {
        "targets": tuple(getattr(module, "MTGPU_TARGETS", ())),
        "fp8": bool(getattr(module, "ENABLE_FP8", False)),
        "mcc_version": getattr(module, "MCC_VERSION", "unknown"),
        "source": getattr(module, "TARGET_SOURCE", "unknown"),
    }


def _assert_build_targets_device(spec: MusaDeviceSpec) -> None:
    """Fail closed when the extension was built for a different arch.

    A cubin for the wrong arch does not fault on MUSA: ``musaDeviceSynchronize``
    returns success and the output buffer keeps whatever it had.  Without this
    check the only symptom is a wrong answer, so a build/device mismatch has to
    be an error at the point where the device is first described.

    An absent stamp is *unknown*, not a mismatch: editable and source installs
    have never recorded one, so it warns once instead of refusing to run.
    """
    stamp = read_build_stamp()
    if stamp is None:
        logger.warning_once(
            "No MUSA build stamp found (%s); the extension's target arch cannot "
            "be checked against device %s. Set MTGPU_TARGET and rebuild to "
            "enable the check.",
            _BUILD_STAMP_MODULE,
            spec.arch_name,
        )
        return
    targets = stamp["targets"]
    if not targets:
        return
    if spec.arch_name not in targets:
        raise RuntimeError(
            f"vllm_musa was built for MUSA arch {', '.join(targets)} but this "
            f"device is {spec.arch_name} ({spec.device_name}). A kernel binary "
            "for another arch loads and runs here without reporting an error and "
            "produces wrong results, so this is refused. Rebuild with "
            f"MTGPU_TARGET={spec.arch_name} or install a wheel built for it."
        )
    if spec.supports(MusaFeature.NATIVE_FP8) and not stamp["fp8"]:
        # Not fatal: the FP8 kernels are simply not in this build, and every FP8
        # gate already falls back.  Worth saying because it looks like a
        # performance cliff rather than a build choice.
        logger.warning_once(
            "This vllm_musa build has no FP8 support (%s) although %s has native "
            "FP8; FP8 quantisation will fall back to the non-FP8 paths. mcc %s.",
            spec.arch_name,
            spec.arch_name,
            stamp["mcc_version"],
        )
