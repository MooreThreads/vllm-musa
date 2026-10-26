# SPDX-License-Identifier: Apache-2.0
"""Build-time arch selection: ``MTGPU_TARGET`` and ``TORCH_MUSA_ARCH_LIST``.

These run in ``setup.py`` before anything is built, so they get their own tests:
a wheel stamped for one arch while the object code was compiled for another is
the failure mode they exist to prevent, and it is invisible at runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "vllm_musa" / "_build_arch.py"
_spec = importlib.util.spec_from_file_location("vllm_musa_build_arch_test", _SRC)
build_arch = importlib.util.module_from_spec(_spec)
# dataclasses resolve annotations through sys.modules[cls.__module__] at class
# creation, so the module has to be registered before it is executed.
sys.modules[_spec.name] = build_arch
_spec.loader.exec_module(build_arch)

#: Two arches that agree on FP8, and one that does not -- synthetic on purpose:
#: the table is ``setup.py``'s claim about hardware and the rules are what is
#: under test here.
AGREEING = {
    "mp_31": {"native_fp8": True},
    "mp_32": {"native_fp8": True},
}
MIXED = {
    "mp_22": {"native_fp8": False},
    "mp_31": {"native_fp8": True},
}


def test_one_arch_from_the_environment():
    selection = build_arch.resolve("mp_31", device_arch="mp_22", capabilities=MIXED)

    assert selection.targets == ("mp_31",)
    assert selection.source == "environment"
    assert selection.fp8 is True
    assert selection.is_fat is False


def test_the_device_decides_when_nothing_is_requested():
    selection = build_arch.resolve(None, device_arch="mp_22", capabilities=MIXED)

    assert selection.targets == ("mp_22",)
    assert selection.source == "device"


def test_the_default_is_recorded_as_a_default():
    """``mp_31`` is the fallback, and the stamp says so rather than "detected"."""
    selection = build_arch.resolve(None, device_arch=None, capabilities=MIXED)

    assert selection.targets == ("mp_31",)
    assert selection.source == "default"


@pytest.mark.parametrize(
    "raw",
    ["mp_31,mp_32", "mp_31;mp_32", "mp_31 mp_32", "310,320", "31;32"],
)
def test_every_spelling_of_a_fat_selection_agrees(raw):
    selection = build_arch.resolve(raw, device_arch=None, capabilities=AGREEING)

    assert selection.targets == ("mp_31", "mp_32")
    assert selection.is_fat is True
    assert selection.fp8 is True


def test_a_repeated_arch_is_one_target():
    assert build_arch.parse_targets("mp_31,310,31") == ("mp_31",)


def test_a_fat_build_of_arches_that_disagree_on_fp8_is_refused():
    """``-DENABLE_FP8`` is one flag: it cannot be set for one pass and not another."""
    with pytest.raises(build_arch.ArchBuildError, match="one flag for the whole build"):
        build_arch.resolve("mp_31,mp_22", device_arch=None, capabilities=MIXED)


def test_an_unknown_arch_is_refused():
    with pytest.raises(build_arch.ArchBuildError, match="unsupported MUSA arch mp_99"):
        build_arch.resolve("mp_99", device_arch=None, capabilities=MIXED)


def test_the_compiler_list_is_derived_from_the_targets():
    selection = build_arch.resolve("mp_31,mp_32", device_arch=None, capabilities=AGREEING)

    assert build_arch.resolve_compiler_arch_list(None, selection) == "31;32"
    assert build_arch.resolve_compiler_arch_list("", selection) == "31;32"


def test_an_agreeing_compiler_list_is_kept():
    selection = build_arch.resolve("mp_31,mp_32", device_arch=None, capabilities=AGREEING)

    assert build_arch.resolve_compiler_arch_list("32;31", selection) == "31;32"


def test_a_disagreeing_compiler_list_is_refused():
    """The stamp and ``--offload-arch`` must describe the same hardware."""
    selection = build_arch.resolve("mp_31,mp_32", device_arch=None, capabilities=AGREEING)

    with pytest.raises(build_arch.ArchBuildError, match="disagrees with MTGPU_TARGET"):
        build_arch.resolve_compiler_arch_list("31", selection)
