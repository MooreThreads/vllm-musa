# SPDX-License-Identifier: Apache-2.0
"""The image build's MUSA arch variable check."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torchada")

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "docker/check_musa_arch_env.py"
IMAGE_ENV = {
    "MTGPU_TARGET": "mp_31",
    "TORCH_MUSA_ARCH_LIST": "31",
    "MATE_MUSA_ARCH_LIST": "3.1",
}


def _load_script():
    spec = importlib.util.spec_from_file_location("check_musa_arch_env", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run(monkeypatch, capsys):
    script = _load_script()

    def run(**overrides):
        env = {**IMAGE_ENV, "VLLM_MUSA_ARCH_LIST": None, **overrides}
        for name, value in env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        rc = script.main()
        out = capsys.readouterr()
        return rc, out.out + out.err

    return run


def test_image_env_passes(run):
    rc, out = run()
    assert rc == 0
    assert "agree on [mp_31]" in out


def test_vllm_musa_arch_list_may_agree(run):
    rc, _ = run(VLLM_MUSA_ARCH_LIST="mp_31")
    assert rc == 0


def test_mtgpu_target_takes_any_torchada_spelling(run):
    assert run(MTGPU_TARGET="3.1")[0] == 0


@pytest.mark.parametrize(
    "name", ["MTGPU_TARGET", "TORCH_MUSA_ARCH_LIST", "MATE_MUSA_ARCH_LIST"]
)
def test_unset_variable_fails(run, name):
    rc, out = run(**{name: None})
    assert rc == 1
    assert f"{name} is not set" in out


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("MATE_MUSA_ARCH_LIST", "2.2", "names [mp_22]"),
        ("TORCH_MUSA_ARCH_LIST", "22", "names [mp_22]"),
        ("VLLM_MUSA_ARCH_LIST", "mp_22", "names [mp_22]"),
        ("VLLM_MUSA_ARCH_LIST", " ", "names no MUSA arch"),
        # torchada reads these as mp_31, but MATE and torch_musa would refuse them.
        ("MATE_MUSA_ARCH_LIST", "31", "must be spelled '3.1'"),
        ("TORCH_MUSA_ARCH_LIST", "3.1", "must be spelled '31'"),
        ("MATE_MUSA_ARCH_LIST", "mp_31", "must be spelled '3.1'"),
        ("MTGPU_TARGET", "mp_31,mp_22", "MTGPU_TARGET='mp_31,mp_22'"),
    ],
)
def test_disagreeing_or_misspelled_variable_fails(run, name, value, message):
    rc, out = run(**{name: value})
    assert rc == 1
    assert message in out


def test_several_archs_fail(run):
    rc, out = run(
        MTGPU_TARGET="mp_22 mp_31",
        TORCH_MUSA_ARCH_LIST="22;31",
        MATE_MUSA_ARCH_LIST="2.2 3.1",
    )
    assert rc == 1
    assert "one arch per image" in out


def test_arch_outside_the_table_fails(run):
    rc, out = run(
        MTGPU_TARGET="mp_32", TORCH_MUSA_ARCH_LIST="32", MATE_MUSA_ARCH_LIST="3.2"
    )
    assert rc == 1
    assert "[mp_32] is not in torchada's arch table" in out
