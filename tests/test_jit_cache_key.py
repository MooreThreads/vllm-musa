# SPDX-License-Identifier: Apache-2.0
"""JIT cache-key and arch-selection contracts (no device required)."""

from __future__ import annotations

import pytest

from torchada.musa import build as ta_build
from vllm_musa.jit_kernel.csrc import jit


@pytest.fixture(autouse=True)
def _pin_compiler(monkeypatch):
    """The cache tag includes the compiler, so pin it rather than shell out to mcc."""
    monkeypatch.setattr(ta_build, "mcc_version", lambda *a, **k: "5.2.0")
    yield


def test_explicit_arch_list_wins(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "31")
    assert jit._musa_arch_flags() == ["--offload-arch=mp_31"]
    assert jit.musa_arch_codes() == ["31"]


def test_fat_arch_list_keeps_every_arch(monkeypatch) -> None:
    """Two arches, one flag set: the fat path keeps both, in the given order."""
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "21 22")
    assert jit._musa_arch_flags() == ["--offload-arch=mp_21", "--offload-arch=mp_22"]
    assert jit.musa_arch_codes() == ["21", "22"]


def test_fat_build_is_refused_for_single_arch_templates(monkeypatch) -> None:
    """tl_templates compare __MUSA_ARCH_LIST__ with one value; fail loud."""
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "21 22")
    with pytest.raises(RuntimeError, match="single __MUSA_ARCH__"):
        jit.musa_arch_define()


def test_single_arch_define_matches_the_device(monkeypatch) -> None:
    # An arch other than the one this host has, so a device-defaulted answer
    # could not pass by accident; ``22`` because only supported arches are
    # selectable and ``32`` is not one of them.
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "22")
    assert jit.musa_arch_define() == "-D__MUSA_ARCH_LIST__=220"


def test_no_silent_mp31_fallback(monkeypatch) -> None:
    """A device query failure must not quietly target mp_31.

    A mubin for the wrong arch loads and runs on MUSA without an error and
    returns an unmodified output buffer, so the old fallback turned a loud
    build failure into a silent wrong answer.
    """
    monkeypatch.delenv("VLLM_MUSA_ARCH_LIST", raising=False)
    monkeypatch.delenv("MATE_MUSA_ARCH_LIST", raising=False)

    import sys
    import types

    def boom(*_args, **_kwargs):
        raise RuntimeError("no device")

    monkeypatch.setitem(
        sys.modules, "torch_musa", types.SimpleNamespace(get_device_properties=boom)
    )

    with pytest.raises(RuntimeError, match="cannot determine the MUSA arch"):
        jit._musa_arch_flags()


def test_build_tag_carries_arch_and_compiler(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "31")
    monkeypatch.setattr(ta_build, "mcc_version", lambda *a, **k: "5.2.0")

    tag = jit.build_tag()

    assert tag == "mp_31__mcc5.2.0"


def test_build_tag_distinguishes_arches_and_compilers(monkeypatch) -> None:
    """The old key was the kernel name alone, so arch/compiler changes reused it."""
    monkeypatch.setattr(ta_build, "mcc_version", lambda *a, **k: "5.2.0")
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "31")
    a = jit.build_tag()
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "22")
    b = jit.build_tag()
    monkeypatch.setattr(ta_build, "mcc_version", lambda *a, **k: "5.3.0")
    c = jit.build_tag()

    assert len({a, b, c}) == 3


@pytest.mark.parametrize("raw", ["31", "mp_31", "MP31", "310"])
def test_the_usual_arch_spellings_agree(raw, monkeypatch) -> None:
    """Normalisation is torchada's, so a JIT flag and a wheel's stamp agree."""
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", raw)

    assert jit._musa_arch_flags() == ["--offload-arch=mp_31"]


def test_an_unparsable_arch_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "bogus")

    with pytest.raises(ta_build.MusaTargetSelectionError):
        jit._musa_arch_flags()
