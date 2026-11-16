# SPDX-License-Identifier: Apache-2.0
"""Check that the image's MUSA arch variables name one arch from torchada's table.

Runs in the image build before vllm-musa is built, and on any host as
``python docker/check_musa_arch_env.py``. Each variable must be spelled for its
reader, and all of them must name the same arch:

    MTGPU_TARGET          setup.py, through torchada.arch        mp_31
    TORCH_MUSA_ARCH_LIST  torch_musa MUSAExtension, ";"-joined    31
    MATE_MUSA_ARCH_LIST   MATE JIT, space-joined X.Y              3.1

VLLM_MUSA_ARCH_LIST overrides MATE_MUSA_ARCH_LIST for the vllm-musa JIT, so it
must be unset or name the same arch.
"""

from __future__ import annotations

import os
import sys

from torchada import arch as musa_arch

REQUIRED = ("MTGPU_TARGET", "TORCH_MUSA_ARCH_LIST", "MATE_MUSA_ARCH_LIST")


def _names(archs) -> str:
    return "[{}]".format(", ".join(a.name for a in archs))


def mate_arch_list(archs) -> str:
    """The MATE_MUSA_ARCH_LIST spelling of ``archs``, e.g. ``3.1``."""
    return " ".join(f"{a.major}.{a.minor}" for a in archs)


def check() -> tuple[list[str], tuple]:
    """Return the problems found and the archs MTGPU_TARGET names."""
    missing = [name for name in REQUIRED if not os.environ.get(name, "").strip()]
    if missing:
        return [f"{name} is not set" for name in missing], ()
    archs = {}
    problems = []
    for name in REQUIRED + ("VLLM_MUSA_ARCH_LIST",):
        # Unset or empty means unset, as for torchada; whitespace is a value it rejects.
        if not os.environ.get(name):
            continue
        try:
            archs[name] = musa_arch.target_archs(env=(name,))
        except ValueError as e:
            problems.append(str(e))
    if problems:
        return problems, ()
    target = archs["MTGPU_TARGET"]
    for name, found in archs.items():
        if found != target:
            problems.append(
                f"{name}={os.environ[name]!r} names {_names(found)}, "
                f"MTGPU_TARGET={os.environ['MTGPU_TARGET']!r} names {_names(target)}"
            )
    spellings = {
        "TORCH_MUSA_ARCH_LIST": musa_arch.torch_arch_list(target),
        "MATE_MUSA_ARCH_LIST": mate_arch_list(target),
    }
    for name, spelling in spellings.items():
        if archs[name] == target and os.environ[name] != spelling:
            problems.append(f"{name}={os.environ[name]!r} must be spelled {spelling!r}")
    if len(target) != 1:
        problems.append(
            f"vllm-musa builds one arch per image; MTGPU_TARGET names {_names(target)}"
        )
    unsupported = [a for a in target if a not in musa_arch.ARCHS]
    if unsupported:
        problems.append(
            f"{_names(unsupported)} is not in torchada's arch table {_names(musa_arch.ARCHS)}"
        )
    return problems, target


def main() -> int:
    problems, target = check()
    if problems:
        for problem in problems:
            print(f"MUSA arch variables: {problem}", file=sys.stderr)
        return 1
    values = " ".join(f"{name}={os.environ[name]}" for name in REQUIRED)
    print(f"MUSA arch variables agree on {_names(target)}: {values}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
