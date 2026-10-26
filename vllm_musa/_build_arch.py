# SPDX-License-Identifier: Apache-2.0
"""Which arch(es) a build targets -- the ``setup.py`` half of the arch contract.

``setup.py`` runs before the package it builds exists, so this module imports only
the standard library plus :mod:`torchada.musa.build` (itself stdlib-only) -- it must
never import ``vllm_musa``.  The runtime half is ``vllm_musa.device_spec``.

Two spellings reach the compiler and they must not disagree:

* ``MTGPU_TARGET`` -- what this package stamps into the wheel and uses to decide
  ``-DENABLE_FP8``;
* ``TORCH_MUSA_ARCH_LIST`` -- what ``torch_musa``'s extension builder turns into
  ``--offload-arch=mp_<code>`` flags.

When they drift, the wheel claims one arch and the object code is built for
another.  On MUSA that does not fault: the kernel runs, returns success and
leaves the output untouched, so the mismatch is silent.  The Dockerfile already
asserts agreement for a single arch; :func:`resolve` makes it a rule of the build
itself, which also covers a bare ``pip install .``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Optional

from torchada.musa import build as ta_build

#: Environment variable naming the arch(es) this package is built for.
TARGET_ENV = "MTGPU_TARGET"
#: Environment variable ``torch_musa`` reads to emit ``--offload-arch`` flags.
COMPILER_LIST_ENV = "TORCH_MUSA_ARCH_LIST"

DEFAULT_TARGET = "mp_31"


class ArchBuildError(ValueError):
    """The requested arch selection cannot produce a self-consistent build."""


@dataclass(frozen=True)
class ArchSelection:
    """The arches one build targets, and what they allow."""

    targets: tuple[str, ...]
    #: ``"environment"``, ``"device"`` or ``"default"`` -- where ``targets`` came
    #: from, recorded in the build stamp because "nobody asked" is not "detected".
    source: str
    #: Whether the build carries native FP8 (one flag for the whole build).
    fp8: bool

    @property
    def display(self) -> str:
        return ",".join(self.targets)

    @property
    def is_fat(self) -> bool:
        return len(self.targets) > 1

    @property
    def compiler_arch_list(self) -> str:
        """The value ``torch_musa`` needs: two-digit codes, ``;``-separated."""
        return ";".join(target[3:] for target in self.targets)


def parse_targets(raw: str) -> tuple[str, ...]:
    """``"mp_31,mp_22"`` (or ``"mp_31 mp_22"``, ``"31;22"``, ``"310 220"``) -> names.

    Normalisation is :func:`torchada.musa.build.arch_codes`, so the three existing
    spellings of one arch agree here exactly as they do in the stamp.
    """
    parts = [part for part in re.split(r"[;,\s]+", raw.strip()) if part]
    if not parts:
        raise ArchBuildError(f"{TARGET_ENV} is empty")
    try:
        codes = ta_build.arch_codes(parts)
    except RuntimeError as exc:  # unparsable arch
        raise ArchBuildError(str(exc)) from exc
    return tuple(f"mp_{code}" for code in codes)


def resolve(
    raw: Optional[str],
    *,
    device_arch: Optional[str],
    capabilities: Mapping[str, Mapping[str, bool]],
    default: str = DEFAULT_TARGET,
) -> ArchSelection:
    """Decide the target set from the environment, then the device, then ``default``.

    ``capabilities`` is ``setup.py``'s ``ARCH_BUILD_CAPABILITIES``: an explicit
    per-arch table, never a comparison, because ``mp_32`` must be able to build
    *with* FP8 while ``mp_22`` builds without it.
    """
    if raw and raw.strip():
        targets = parse_targets(raw)
        source = "environment"
    elif device_arch:
        targets = (device_arch,)
        source = "device"
    else:
        targets = (default,)
        source = "default"

    unknown = [target for target in targets if target not in capabilities]
    if unknown:
        raise ArchBuildError(
            f"unsupported MUSA arch {', '.join(unknown)}; "
            f"expected one of {sorted(capabilities)}"
        )

    fp8 = {bool(capabilities[target].get("native_fp8")) for target in targets}
    if len(fp8) > 1:
        nosupport = sorted(t for t in targets if not capabilities[t].get("native_fp8"))
        raise ArchBuildError(
            "-DENABLE_FP8 is one flag for the whole build, but "
            f"{', '.join(targets)} disagree on native FP8 ({', '.join(nosupport)} "
            "does not have it). Build one arch per wheel, or move the FP8 sources "
            "behind per-arch __MUSA_ARCH__ guards; a global flag would compile FP8 "
            "code for an arch that cannot execute it."
        )

    return ArchSelection(targets=targets, source=source, fp8=fp8.pop())


def resolve_compiler_arch_list(declared: Optional[str], selection: ArchSelection) -> str:
    """The ``TORCH_MUSA_ARCH_LIST`` to build with, refusing a disagreement.

    A caller that set the variable keeps control of it, but only if it names the
    same arches -- otherwise the wheel and its object code would describe
    different hardware.
    """
    wanted = selection.compiler_arch_list
    if not declared or not declared.strip():
        return wanted

    codes = [part for part in re.split(r"[;,\s]+", declared.strip()) if part]
    normalised = {code[3:] if code.startswith("mp_") else code for code in codes}
    if normalised != set(wanted.split(";")):
        raise ArchBuildError(
            f"{COMPILER_LIST_ENV}={declared!r} disagrees with "
            f"{TARGET_ENV}={selection.display!r}: the wheel would be stamped for one "
            f"arch while --offload-arch builds another. Set both, or neither."
        )
    return ";".join(sorted(normalised))
