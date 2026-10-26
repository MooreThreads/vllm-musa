# SPDX-License-Identifier: Apache-2.0
"""Per-architecture MUSA hardware facts, provided by :mod:`torchada.musa`.

This module used to carry its own device probe, capability sets, build-stamp
reader and arch-mismatch check.  The same facts decide kernel selection in
vLLM-MUSA, SGLang and MATE, the three copies had drifted, and a wrong-arch cubin
does not fault on MUSA — it returns wrong numbers — so the probing half now lives
in ``torchada.musa`` (0.1.104 and later).

What stays here is the wiring this package needs on top of it:

* the name of *its* build-stamp module, which ``setup.py`` writes;
* the build/device check *inside* the probe, so a wrong-arch install fails at the
  first question asked about the device rather than at an arbitrary later call;
* the policy that an absent stamp warns, because the comparison it would have
  performed is the only thing that catches a wrong-arch wheel.

The two rules the shared module enforces, restated because they are load-bearing
here — see ``torchada/src/torchada/musa/device.py`` for the full reasoning:

1. An arch is an **identity**, never a magnitude: warp width (32 only on
   ``mp_31``), native FP8 (a per-target build define) and the multiprocessor
   count (48/52/56/60, varying *within* one arch) move independently, so
   ``(3, 1)`` must never be compared as "at least as new as".
2. A capability question is answered from an explicit set, so an arch nobody has
   validated answers ``False`` rather than landing on a side of a comparison.

Note on ``project`` vs. runtime arch: ``torch_musa.get_arch_list()`` reports the
arches the **wheel was built for** (``['31']`` on the 5.2.0 wheel), not the arch
of the device in front of you.  The device arch always comes from
``get_device_properties().major/minor``; the wheel's answer is what
:func:`read_build_stamp` returns.
"""

from __future__ import annotations

import functools
from typing import Optional

from vllm.logger import init_logger

from torchada.musa import build as _torchada_build
from torchada.musa.build import MusaBuildTargetMismatch
from torchada.musa.device import (  # noqa: F401 - re-exported for call sites
    NATIVE_FP8_ARCHS,
    VALIDATED_MUSA_ARCHS,
    MusaDeviceSpec,
    MusaFeature,
    arch_code,
    arch_name,
    clear_device_spec_cache,
)
from torchada.musa.device import current_device_spec as _probed_current_device_spec
from torchada.musa.device import get_device_spec as _probed_device_spec

__all__ = [
    "MusaBuildTargetMismatch",
    "MusaDeviceSpec",
    "MusaFeature",
    "NATIVE_FP8_ARCHS",
    "VALIDATED_MUSA_ARCHS",
    "arch_code",
    "arch_name",
    "clear_device_spec_cache",
    "current_device_spec",
    "get_device_spec",
    "is_validated_device",
    "read_build_stamp",
]

# ``init_logger`` (not stdlib) because its ``warning_once`` is what keeps the
# missing-stamp notice to one line per process; an editable install would
# otherwise repeat it on every device query.
logger = init_logger(__name__)

#: The stamp module ``setup.py`` writes into this package.
_BUILD_STAMP_MODULE = "vllm_musa._build_stamp"


@functools.cache
def read_build_stamp() -> Optional[_torchada_build.BuildStamp]:
    """The arch record of the compiled extension, or ``None`` if unknown.

    Cached for the process lifetime: the stamp is written by ``setup.py``, so a
    rebuild needs a restart to be seen.  ``import_module`` caches a successful
    import but not a failed one, and "absent stamp" has to be a stable answer
    rather than a per-call race with a build running alongside.
    """
    return _torchada_build.read_build_stamp(_BUILD_STAMP_MODULE)


def _assert_build_targets_device(spec: MusaDeviceSpec) -> None:
    """Fail closed when the extension was built for a different arch.

    The comparison is ``torchada.musa.build``'s; what stays here is the warning
    for the case that module deliberately treats as *unknown*.  An editable or
    source install has no stamp, and staying silent would hide the fact that the
    check is simply off for this process.
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
    _torchada_build.assert_build_targets_device(spec, stamp)


def get_device_spec(device_id: Optional[int] = None) -> Optional[MusaDeviceSpec]:
    """Spec for ``device_id`` (default: the current device), or ``None``.

    The build/device check runs here rather than at each caller: a wheel built for
    another arch is a property of the install, so every question about the device
    is the wrong question to answer "normally" until it has been raised.
    """
    spec = _probed_device_spec(device_id)
    if spec is not None:
        _assert_build_targets_device(spec)
    return spec


def current_device_spec() -> Optional[MusaDeviceSpec]:
    """Spec for the current device, or ``None`` when it cannot be determined."""
    return get_device_spec(None)


def is_validated_device(device_id: Optional[int] = None) -> bool:
    """Whether the arch of ``device_id`` is one the tuned paths were validated on.

    A build/device mismatch is raised, not folded into ``False``: gating an
    optimized kernel off would leave a wrong-arch binary running the generic path,
    which is still the wrong binary.
    """
    try:
        spec = current_device_spec() if device_id is None else get_device_spec(device_id)
    except MusaBuildTargetMismatch:
        raise
    except Exception:  # noqa: BLE001 - a failed probe must not enable anything
        return False
    return spec is not None and spec.is_validated
