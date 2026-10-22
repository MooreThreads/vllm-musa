# SPDX-License-Identifier: Apache-2.0
"""Build-time arch record: setup.py's table and the runtime check must agree."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from vllm_musa import device_spec

_SETUP_PY = Path(__file__).resolve().parent.parent / "setup.py"


def _setup_arch_table() -> dict[str, dict[str, bool]]:
    """Read ``ARCH_BUILD_CAPABILITIES`` out of setup.py without executing it."""
    tree = ast.parse(_SETUP_PY.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and getattr(node.targets[0], "id", None) == "ARCH_BUILD_CAPABILITIES"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("ARCH_BUILD_CAPABILITIES not found in setup.py")


class _FakeProps:
    major = 3
    minor = 1
    multi_processor_count = 56
    warp_size = 32
    total_memory = 85804969984
    L2_cache_size = 60 * 1024 * 1024
    shared_memory_per_block = 196608
    shared_memory_per_multiprocessor = 196608
    regs_per_multiprocessor = 65536
    max_threads_per_block = 1024
    max_threads_per_multiprocessor = 2048
    name = "MTT S5000"
    uuid = "fake-uuid"


@pytest.fixture
def stamp(monkeypatch):
    """Install a fake build stamp; the real one only exists after a build."""

    def install(targets=("mp_31",), fp8=True):
        module = types.ModuleType("vllm_musa._build_stamp")
        module.MTGPPU_TARGETS_UNUSED = None
        module.MTGPU_TARGETS = tuple(targets)
        module.ENABLE_FP8 = fp8
        module.MCC_VERSION = "5.2.0"
        module.TARGET_SOURCE = "environment"
        monkeypatch.setitem(sys.modules, "vllm_musa._build_stamp", module)
        device_spec.read_build_stamp.cache_clear()
        return module

    yield install
    sys.modules.pop("vllm_musa._build_stamp", None)
    device_spec.read_build_stamp.cache_clear()
    device_spec.get_device_spec.cache_clear()


@pytest.fixture
def spec(monkeypatch):
    monkeypatch.setattr(device_spec, "_get_properties", lambda device_id: _FakeProps())
    device_spec.get_device_spec.cache_clear()
    yield device_spec.get_device_spec(0)
    device_spec.get_device_spec.cache_clear()


def test_setup_fp8_table_matches_runtime_arch_set() -> None:
    """setup.py and device_spec must not disagree about which arch has FP8.

    setup.py cannot import the package it is building, so the two tables are
    duplicated by necessity; this is the thing that keeps them equal.
    """
    table = _setup_arch_table()
    built_with_fp8 = {
        arch for arch, caps in table.items() if caps.get("native_fp8")
    }
    runtime = {f"mp_{code // 10}" for code in device_spec.NATIVE_FP8_ARCHS}

    assert built_with_fp8 == runtime, (
        "setup.py ARCH_BUILD_CAPABILITIES and device_spec.NATIVE_FP8_ARCHS "
        f"disagree: build={sorted(built_with_fp8)} runtime={sorted(runtime)}"
    )


def test_setup_table_covers_every_validated_arch() -> None:
    table = _setup_arch_table()
    validated = {f"mp_{code // 10}" for code in device_spec.VALIDATED_MUSA_ARCHS}

    assert validated <= set(table)


def test_matching_build_is_accepted(stamp, spec) -> None:
    stamp(("mp_31",))

    assert spec.arch_name == "mp_31"
    assert device_spec.read_build_stamp()["targets"] == ("mp_31",)


def test_wrong_arch_build_is_refused(stamp, monkeypatch) -> None:
    """A binary for another arch must be an error, not a wrong number."""
    stamp(("mp_22",))
    monkeypatch.setattr(device_spec, "_get_properties", lambda device_id: _FakeProps())
    device_spec.get_device_spec.cache_clear()

    with pytest.raises(RuntimeError, match="built for MUSA arch mp_22"):
        device_spec.get_device_spec(0)


def test_fat_build_accepts_either_arch(stamp, monkeypatch) -> None:
    stamp(("mp_21", "mp_31"))
    monkeypatch.setattr(device_spec, "_get_properties", lambda device_id: _FakeProps())
    device_spec.get_device_spec.cache_clear()

    assert device_spec.get_device_spec(0).arch_name == "mp_31"


def test_missing_stamp_is_unknown_not_a_mismatch(monkeypatch, spec) -> None:
    """Editable installs never wrote a stamp; warn, do not refuse to run."""
    monkeypatch.setitem(sys.modules, "vllm_musa._build_stamp", None)
    device_spec.read_build_stamp.cache_clear()

    assert device_spec.read_build_stamp() is None
    assert spec.is_validated  # the spec was still produced


def test_stamp_without_fp8_warns_but_does_not_fail(stamp, monkeypatch) -> None:
    stamp(("mp_31",), fp8=False)
    monkeypatch.setattr(device_spec, "_get_properties", lambda device_id: _FakeProps())
    device_spec.get_device_spec.cache_clear()

    spec = device_spec.get_device_spec(0)

    assert spec.supports(device_spec.MusaFeature.NATIVE_FP8)
