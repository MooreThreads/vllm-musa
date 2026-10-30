# SPDX-License-Identifier: Apache-2.0
"""Per-architecture MUSA hardware facts, provided by :mod:`torchada.musa`.

The probing half — device probe, capability sets, build-stamp reader and the
arch-mismatch check — lives in ``torchada.musa`` (0.1.104 and later), because the
same facts decide kernel selection in vLLM-MUSA, SGLang and MATE, and a
wrong-arch mubin does not fault on MUSA: it returns wrong numbers.

What stays here is the wiring this package needs on top of it:

* the name of *its* build-stamp module, which ``setup.py`` writes;
* the build/device check *inside* the probe, so a wrong-arch install fails at the
  first question asked about the device rather than at an arbitrary later call;
* the policy that an absent stamp warns, because the comparison it would have
  performed is the only thing that catches a wrong-arch wheel;
* the list of arches *this project's* tuned kernels have been validated on.  That is
  not a hardware fact: a shared table cannot say which kernel, version or input
  contract was validated, so the answer lives here and names the run that
  justifies each entry.

The two rules the shared module enforces, restated because they are load-bearing
here — see ``torchada/src/torchada/musa/device.py`` for the full reasoning:

1. An arch is an **identity**, never a magnitude: warp width (32 only on
   ``mp_31``), native FP8 (a per-target build define) and the multiprocessor
   count (48/52/56/60, varying *within* one arch) move independently, so
   ``(3, 1)`` must never be compared as "at least as new as".
2. A capability question is answered from an explicit set or an exactly named
   feature, so an arch nobody has validated answers ``False`` rather than landing on
   a side of a comparison.

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

try:
    from torchada.musa import build as _torchada_build
except ImportError as exc:  # pragma: no cover - a pre-release install
    raise RuntimeError(
        "vllm-musa needs torchada>=0.1.104: this module re-exports the device facts "
        "that release moved into torchada.musa. Install that release, or torchada "
        "from its branch with "
        "`pip install -e <torchada checkout> --no-deps --no-build-isolation`."
    ) from None
from torchada.musa.build import MusaBuildTargetMismatch
from torchada.musa.build import (
    assert_build_matches_device as _assert_build_matches_device,
)
from torchada.musa.device import (  # noqa: F401 - re-exported for call sites
    NATIVE_FP4_ARCHS,
    NATIVE_FP8_ARCHS,
    MusaDeviceSpec,
    arch_code,
    arch_name,
    clear_device_spec_cache,
)
from torchada.musa.device import get_device_spec as _probed_device_spec

__all__ = [
    "MusaBuildTargetMismatch",
    "MusaDeviceSpec",
    "NATIVE_FP4_ARCHS",
    "NATIVE_FP8_ARCHS",
    "VALIDATED_MUSA_ARCHS",
    "VALIDATED_ARCHES",
    "arch_code",
    "arch_name",
    "clear_device_spec_cache",
    "current_device_spec",
    "get_device_spec",
    "is_validated_arch",
    "read_build_stamp",
]

#: Arch codes on which this project's tuned kernels have been run and validated, with
#: the run that justifies each entry.  Keyed by ``arch_code`` (``310``), not by name, so
#: a card whose ``major.minor`` maps to a listed arch matches without parsing strings.
#:
#: torchada answers what the hardware is; this answers what *we* have validated.  The
#: two are separate on purpose: the FP8 paths, the chunked-min-p sampler and the tuning
#: tables here are this project's code, and an arch added for another project's kernels
#: must not enable them.
VALIDATED_ARCHES = {
    310: "S5000 (mp_31), PH1 kernels and sampler, validated on a leased S5000",
}

#: The arch codes behind :data:`VALIDATED_ARCHES`, as the set call sites compare with.
VALIDATED_MUSA_ARCHS = frozenset(VALIDATED_ARCHES)

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

    One call to the shared module: it reads this package's stamp, compares it and
    names this package in the message.  ``warn_when_unstamped`` is set because an
    editable install has no stamp, and staying silent would hide the fact that the
    only check that catches a wrong-arch wheel is off for this process.
    """
    _assert_build_matches_device(
        spec,
        module_name=_BUILD_STAMP_MODULE,
        name="vllm_musa",
        warn_when_unstamped=True,
    )


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


def is_validated_arch(device_id: Optional[int] = None) -> bool:
    """Whether this project's tuned kernels have been validated on this device's arch.

    Two things this is not.  It is not a torchada answer -- ``torchada.musa`` reports
    the hardware, and the validation policy for *these* kernels is
    :data:`VALIDATED_ARCHES` above.  And it does not route around the build/device
    check: the probe below is this module's, so a wrong-arch wheel raises instead of
    quietly disabling every tuned path and running the generic kernels of the same
    wrong binary.
    """
    try:
        spec = (
            current_device_spec() if device_id is None else get_device_spec(device_id)
        )
    except MusaBuildTargetMismatch:
        raise
    except Exception:  # noqa: BLE001 - a failed probe must not enable anything
        return False
    return spec is not None and spec.arch_code in VALIDATED_ARCHES
