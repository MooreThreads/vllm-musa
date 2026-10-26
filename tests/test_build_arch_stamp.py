# SPDX-License-Identifier: Apache-2.0
"""Build-time arch record: setup.py's table and the runtime check must agree."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from torchada.musa import build as ta_build
from torchada.musa import device as ta_device

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
    device_spec.clear_device_spec_cache()


@pytest.fixture
def spec(monkeypatch):
    monkeypatch.setattr(ta_device, "_get_properties", lambda device_id: _FakeProps())
    device_spec.clear_device_spec_cache()
    yield device_spec.get_device_spec(0)
    device_spec.clear_device_spec_cache()


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
    assert device_spec.read_build_stamp().targets == ("mp_31",)


def test_wrong_arch_build_is_refused(stamp, monkeypatch) -> None:
    """A binary for another arch must be an error, not a wrong number."""
    stamp(("mp_22",))
    monkeypatch.setattr(ta_device, "_get_properties", lambda device_id: _FakeProps())
    device_spec.clear_device_spec_cache()

    with pytest.raises(RuntimeError, match="built for MUSA arch mp_22"):
        device_spec.get_device_spec(0)


def test_fat_build_accepts_either_arch(stamp, monkeypatch) -> None:
    stamp(("mp_21", "mp_31"))
    monkeypatch.setattr(ta_device, "_get_properties", lambda device_id: _FakeProps())
    device_spec.clear_device_spec_cache()

    assert device_spec.get_device_spec(0).arch_name == "mp_31"


def test_missing_stamp_is_unknown_not_a_mismatch(monkeypatch, spec) -> None:
    """Editable installs never wrote a stamp; warn, do not refuse to run."""
    monkeypatch.setitem(sys.modules, "vllm_musa._build_stamp", None)
    device_spec.read_build_stamp.cache_clear()

    assert device_spec.read_build_stamp() is None
    assert spec.is_validated  # the spec was still produced


def test_stamp_without_fp8_warns_but_does_not_fail(stamp, monkeypatch) -> None:
    stamp(("mp_31",), fp8=False)
    monkeypatch.setattr(ta_device, "_get_properties", lambda device_id: _FakeProps())
    device_spec.clear_device_spec_cache()

    spec = device_spec.get_device_spec(0)

    assert spec.supports(device_spec.MusaFeature.NATIVE_FP8)


def _stamp_writer(tmp_path):
    """Extract ``_write_build_stamp`` from setup.py and give it its globals.

    setup.py cannot be imported - it calls ``setup()`` at module scope and needs
    the MUSA toolchain - but the file it writes is the contract the runtime
    check reads, so the real function is extracted and exercised.
    """
    tree = ast.parse(_SETUP_PY.read_text())
    node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_write_build_stamp"),
        None,
    )
    assert node is not None, "_write_build_stamp not found in setup.py"

    (tmp_path / "vllm_musa").mkdir()
    namespace = {
        "Path": Path,
        "ta_build": ta_build,
        "MTGPU_TARGET": "mp_31",
        "MTGPU_TARGETS": ["mp_31"],
        "ENABLE_FP8": True,
        "MTGPU_TARGET_SOURCE": "device",
        "_mcc_version_string": lambda: "5.2.0",
        "__file__": str(tmp_path / "setup.py"),
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(_SETUP_PY), "exec"),
        namespace,
    )
    return namespace["_write_build_stamp"], namespace


def test_setup_writes_a_stamp_the_runtime_can_read(tmp_path) -> None:
    """The writer's output must be exactly what read_build_stamp() expects."""
    write_stamp, _ = _stamp_writer(tmp_path)

    write_stamp(tmp_path / "vllm_musa")

    produced = tmp_path / "vllm_musa" / "_build_stamp.py"
    rendered: dict = {}
    exec(compile(produced.read_text(), str(produced), "exec"), rendered)

    assert rendered["MTGPU_TARGETS"] == ("mp_31",)
    assert rendered["ENABLE_FP8"] is True
    assert rendered["MCC_VERSION"] == "5.2.0"
    assert rendered["TARGET_SOURCE"] == "device"


def test_stamp_writer_records_a_non_fp8_arch(tmp_path) -> None:
    write_stamp, namespace = _stamp_writer(tmp_path)
    namespace["MTGPU_TARGET"] = "mp_22"
    namespace["MTGPU_TARGETS"] = ["mp_22"]
    namespace["ENABLE_FP8"] = False

    write_stamp(tmp_path / "vllm_musa")

    produced = tmp_path / "vllm_musa" / "_build_stamp.py"
    rendered: dict = {}
    exec(compile(produced.read_text(), str(produced), "exec"), rendered)

    assert rendered["MTGPU_TARGETS"] == ("mp_22",)
    assert rendered["ENABLE_FP8"] is False


def test_stamp_writer_records_every_arch_of_a_fat_build(tmp_path) -> None:
    """A fat wheel names its set, so the runtime accepts either arch and no other."""
    write_stamp, namespace = _stamp_writer(tmp_path)
    namespace["MTGPU_TARGET"] = "mp_21,mp_22"
    namespace["MTGPU_TARGETS"] = ["mp_21", "mp_22"]
    namespace["ENABLE_FP8"] = False

    write_stamp(tmp_path / "vllm_musa")

    produced = tmp_path / "vllm_musa" / "_build_stamp.py"
    rendered: dict = {}
    exec(compile(produced.read_text(), str(produced), "exec"), rendered)

    assert rendered["MTGPU_TARGETS"] == ("mp_21", "mp_22")
    assert rendered["ENABLE_FP8"] is False


def test_stamp_reaches_build_lib_not_only_the_source_tree() -> None:
    """`build_py` runs before `build_ext`, so a wheel packs what is in build_lib.

    A stamp written only to the source tree is never in the artifact, and the
    runtime check then silently degrades to its warning path on every install.
    """
    tree = ast.parse(_SETUP_PY.read_text())
    build_ext = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "build_extensions"
    )
    destinations = [
        ast.unparse(call.args[0])
        for call in ast.walk(build_ext)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "_write_build_stamp"
    ]
    assert len(destinations) == 2, (
        "build_extensions must write the stamp to both the source tree and "
        f"build_lib; it writes to {destinations}"
    )
    assert any("build_lib" in d for d in destinations), destinations
    assert any("__file__" in d for d in destinations), destinations


def test_the_stamp_writer_takes_its_destination_as_an_argument() -> None:
    """The destination must be a parameter, not the module's own directory."""
    tree = ast.parse(_SETUP_PY.read_text())
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_write_build_stamp"
    )
    assert [a.arg for a in node.args.args] == ["dest"], ast.dump(node.args)
