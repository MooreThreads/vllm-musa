# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard DeepSelect registration and indexer dispatch without GPU hardware."""

from __future__ import annotations

import ast
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"
BUILD_PATCH = SERIES / "0211-MUSA-build-DeepSelect-through-vLLM-extension-target.patch"
PLUGIN = ROOT / "vllm_musa" / "__init__.py"
PINS = ROOT / "third_party" / "PINS"
DOC = ROOT / "docs" / "deepselect.md"
INDEXER_PATCH = (
    SERIES / "0210-MUSA-route-explicit-DeepSelect-through-DSV4-indexer.patch"
)


def _deep_select_patches() -> list[Path]:
    patches = sorted(SERIES.glob("*-MUSA-*DeepSelect*.patch"))
    assert patches, "no DeepSelect series patch found"
    return patches


def _added_lines(text: str) -> str:
    return "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _deepselect_source(relative_path: str) -> Path:
    root = Path(os.environ.get("DEEPSELECT_SRC_DIR", ROOT.parent / "DeepSelect"))
    source = root / relative_path
    if not source.is_file():
        pytest.skip(
            f"set DEEPSELECT_SRC_DIR to an updated MUSA checkout: missing {source}"
        )
    return source


def _bindings_source() -> str:
    return _deepselect_source("csrc/musa_api.cpp").read_text(encoding="utf-8")


def test_series_patch_retains_native_extension_import_detection() -> None:
    source = (
        SERIES / "0209-MUSA-register-an-out-of-tree-DeepSelect-top-k-backen.patch"
    ).read_text(encoding="utf-8")
    assert "elif not _HAS_DEEP_SELECT_C:" in source
    assert "vllm._deepselect_C could not be imported" in source
    assert "_deep_select_op_available" not in source


def test_series_patch_keeps_the_cuda_capability_check() -> None:
    added = "\n".join(
        _added_lines(p.read_text(encoding="utf-8")) for p in _deep_select_patches()
    )
    # SM100-only stays the rule for CUDA; only the "op exists" half became
    # platform-neutral.
    assert "current_platform.is_device_capability_family(100)" in added
    assert (
        "self._is_cuda and not current_platform.is_device_capability_family(100)"
        in added
    )
    assert "self._is_musa = current_platform.is_musa()" in added
    assert "if not self._is_cuda and not self._is_musa:" in added


def _deep_select_validation_helpers(
    family: int, extension_available: bool, supported: bool
) -> dict:
    patch = SERIES / "0209-MUSA-register-an-out-of-tree-DeepSelect-top-k-backen.patch"
    source = "\n".join(
        line[1:]
        for line in patch.read_text(encoding="utf-8").splitlines()
        if line.startswith((" ", "+")) and not line.startswith("+++")
    )
    branch = source[source.index('        elif self._backend == "deep_select":') :]
    branch = branch.split('        elif self._backend == "aiter":', 1)[0]
    validation = (
        "def validate(self, logits, topk_tokens):\n    failures = []\n"
        + textwrap.indent(textwrap.dedent(branch).replace("elif ", "if ", 1), "    ")
        + "\n    return failures\n"
    )
    calls = []

    def is_family(requested: int) -> bool:
        calls.append("platform")
        return family // 10 == requested // 10

    def supports_input(*args) -> bool:
        calls.append("input")
        return supported

    namespace = {
        "_HAS_DEEP_SELECT_C": extension_available,
        "current_platform": SimpleNamespace(is_device_capability_family=is_family),
        "is_deep_select_supported": supports_input,
    }
    exec(compile(validation, str(patch), "exec"), namespace)
    namespace["calls"] = calls
    return namespace


@pytest.mark.parametrize(
    ("platform", "family", "extension_available", "supported", "error"),
    [
        ("musa", 0, True, True, None),
        ("cuda", 100, True, True, None),
        ("cuda", 103, True, True, None),
        ("cuda", 90, True, True, "SM100a/SM103a"),
        ("cpu", 0, True, True, "CUDA or MUSA"),
        ("rocm", 0, True, True, "CUDA or MUSA"),
        ("musa", 0, False, True, "could not be imported"),
        ("musa", 0, True, False, "inputs violate"),
    ],
)
def test_deep_select_validation_separates_platform_extension_and_input(
    platform: str,
    family: int,
    extension_available: bool,
    supported: bool,
    error: str | None,
) -> None:
    namespace = _deep_select_validation_helpers(family, extension_available, supported)
    selector = SimpleNamespace(
        _is_cuda=platform == "cuda",
        _is_musa=platform == "musa",
        _backend="deep_select",
    )
    logits = SimpleNamespace(dtype="float32", stride=lambda: (1024, 1))
    failures = namespace["validate"](selector, logits, 512)
    if error is None:
        assert failures == []
    else:
        assert len(failures) == 1 and error in failures[0]
    assert ("platform" in namespace["calls"]) == (platform == "cuda")
    assert ("input" in namespace["calls"]) == (
        error is None or error == "inputs violate"
    )


def test_series_patch_routes_supported_windows_to_explicit_backend() -> None:
    source = INDEXER_PATCH.read_text(encoding="utf-8")
    added = _added_lines(source)
    assert added.count('topk_backend == "deep_select"') >= 3
    assert "_musa_try_fill_decode_topk_from_paged_logits" in added
    assert "deep_select_topk(" in added
    assert "requires zero-based indexer windows" not in added


def _indexer_selection_helpers(
    capturing: bool = False, supported: bool = True
) -> tuple[dict, list]:
    staged_patch = (
        SERIES / "0192-MUSA-select-indexer-top-k-with-the-staged-radix-kern.patch"
    )
    source = _added_lines(staged_patch.read_text(encoding="utf-8").split("@@")[2])
    for hunk in INDEXER_PATCH.read_text(encoding="utf-8").split("@@")[2::2]:
        lines = hunk.splitlines()
        before = "\n".join(line[1:] for line in lines if line.startswith((" ", "-")))
        after = "\n".join(line[1:] for line in lines if line.startswith((" ", "+")))
        if before and before in source:
            source = source.replace(before, after, 1)
    calls = []

    def record(name: str):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))

        return call

    namespace = {
        "deep_select_topk": record("deep_select"),
        "is_deep_select_supported": lambda logits, topk: supported,
        "_musa_sparse_indexer_is_current_stream_capturing": lambda: capturing,
        "_musa_custom_ops": SimpleNamespace(
            deepseek_v4_sparse_indexer_topk_decode=record("staged"),
            sparse_indexer_topk_decode=record("decode"),
            sparse_indexer_topk=record("windows"),
        ),
    }
    exec(
        compile(
            "from __future__ import annotations\n" + source, str(INDEXER_PATCH), "exec"
        ),
        namespace,
    )
    return namespace, calls


@pytest.mark.parametrize(
    ("backend", "starts_at_zero", "topk", "expected"),
    [
        ("deep_select", True, 512, "deep_select"),
        ("deep_select", False, 512, "windows"),
        ("deep_select", True, 1024, "deep_select"),
        ("deep_select", False, 1024, "windows"),
        ("auto", True, 512, "staged"),
        ("auto", True, 1024, "decode"),
        ("auto", False, 512, "windows"),
        ("per_row", False, 512, "windows"),
    ],
)
def test_window_dispatch_preserves_legacy_fallback(
    backend: str, starts_at_zero: bool, topk: int, expected: str
) -> None:
    namespace, calls = _indexer_selection_helpers()
    logits, starts, ends, output = (object() for _ in range(4))
    namespace["_musa_select_indexer_topk_windows"](
        logits, starts, ends, output, topk, starts_at_zero, backend
    )
    if expected == "deep_select":
        assert calls == [
            (expected, (logits, topk), {"end": ends, "output_idx": output})
        ]
    elif expected == "windows":
        assert calls == [(expected, (logits, starts, ends, output, topk), {})]
    else:
        assert calls == [(expected, (logits, ends, output, topk), {})]


@pytest.mark.parametrize("topk, expected", [(512, "staged"), (1024, "decode")])
def test_unsupported_deep_select_layout_uses_native_topk(
    topk: int, expected: str
) -> None:
    namespace, calls = _indexer_selection_helpers(capturing=True, supported=False)
    logits, starts, ends, output = (object() for _ in range(4))
    namespace["_musa_select_indexer_topk_windows"](
        logits, starts, ends, output, topk, True, "deep_select"
    )
    assert calls == [(expected, (logits, ends, output, topk), {})]


@pytest.mark.parametrize(
    ("backend", "topk", "nonzero", "capturing", "expected"),
    [
        ("deep_select", 512, False, False, True),
        ("deep_select", 1024, False, False, True),
        ("deep_select", 512, True, False, False),
        ("deep_select", 512, False, True, False),
        ("auto", 512, False, False, True),
        ("auto", 1024, False, False, False),
        ("auto", 512, True, False, False),
        ("auto", 512, False, True, False),
    ],
)
def test_window_support_keeps_legacy_gates_without_capture_sync(
    backend: str, topk: int, nonzero: bool, capturing: bool, expected: bool
) -> None:
    namespace, _ = _indexer_selection_helpers(capturing)

    def any_start():
        assert not capturing, "capture must not read a GPU scalar"
        return nonzero

    starts = SimpleNamespace(any=any_start)
    assert (
        namespace["_musa_indexer_windows_start_at_zero"](starts, topk, backend)
        is expected
    )


def test_prefill_deep_select_guards_leave_nonzero_windows_on_legacy_paths() -> None:
    added = _added_lines(INDEXER_PATCH.read_text(encoding="utf-8"))
    guards = [
        "_musa_indexer_windows_start_at_zero(\n            starts, topk, topk_backend\n",
        "_musa_indexer_windows_start_at_zero(\n            lo.squeeze(1), width, topk_backend\n",
        "_musa_indexer_windows_start_at_zero(\n        chunk.cu_seqlen_ks, topk_tokens, topk_backend\n",
    ]
    for guard in guards:
        assert guard in added
    assert "if bool(starts.any()):" not in added
    assert "if bool(lo.any()):" not in added


def test_plugin_does_not_register_deep_select() -> None:
    source = PLUGIN.read_text(encoding="utf-8")
    assert "_register_deep_select" not in source
    assert "from .deep_select" not in source
    assert not (ROOT / "vllm_musa/deep_select/registry.py").exists()


def test_microbenchmark_uses_vllm_extension_without_standalone_package() -> None:
    source = (ROOT / "benchmarks/benchmark_deepselect_topk.py").read_text()
    assert "import vllm._deepselect_C as deepselect_extension" in source
    assert "vllm_musa.deep_select" not in source
    assert "deep_select.deep_select_musa_cpp" not in source
    assert '"deep-select"' not in source


def test_extension_defines_upstream_schema_and_native_musa_meta_ops() -> None:
    source = _bindings_source()
    assert "TORCH_LIBRARY(deep_select, library)" in source
    assert "get_alignment_requirement() -> (int, int)" in source
    assert "Tensor(a!)? output_value, Tensor(b!) output_index" in source
    assert 'library.impl("topk", torch::kMUSA, &deepselect_topk)' in source
    assert "TORCH_LIBRARY_IMPL(deep_select, Meta, library)" in source
    assert "CompositeExplicitAutograd" in source
    assert '#include "musa_kernels/topk_bf16.mu"' in source
    assert "torch::kCPU" not in source


def test_build_provides_vllm_extension_from_deepselect_source() -> None:
    source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'name="vllm._deepselect_C"' not in source
    assert "_DEEPSELECT_REPO" not in source
    assert "_DEEPSELECT_SRC_DIR" not in source
    assert "deep_select.deep_select_musa_cpp" not in source
    build = BUILD_PATCH.read_text(encoding="utf-8")
    assert 'CMakeExtension(name="vllm._deepselect_C")' in build
    assert "deepselect_extension.py_limited_api = False" in build
    assert 'include("${deepselect_SOURCE_DIR}/cmake/vllm_musa.cmake")' in build
    assert 'if(VLLM_TARGET_DEVICE STREQUAL "empty")' in build
    assert (
        'include("${deepselect_SOURCE_DIR}/cmake/vllm_musa_toolchain.cmake")' in build
    )
    cmake = _deepselect_source("cmake/vllm_musa.cmake").read_text(encoding="utf-8")
    assert "define_extension_target(\n    _deepselect_C" in cmake
    assert 'SOURCES "${deepselect_SOURCE_DIR}/csrc/musa_api.cpp"' in cmake
    assert (
        "include cmake/vllm_musa_toolchain.cmake"
        in _deepselect_source("MANIFEST.in").read_text()
    )
    assert not (ROOT / "csrc/deep_select/torch_bindings.mu").exists()


@pytest.mark.parametrize("skip", [False, True])
def test_native_build_delegates_deepselect_to_vllm(skip: bool) -> None:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    build_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_CustomBuildExt"
    )
    build_class.body = [
        node
        for node in build_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    ]
    module = ast.Module(body=[build_class], type_ignores=[])
    events = []
    clones = []

    class BuildExtension:
        def run(self) -> None:
            events.append("build")

    environment = {"SKIP_THIRD_PARTY": "1" if skip else "0"}
    namespace = {
        "BuildExtension": BuildExtension,
        "os": SimpleNamespace(environ=environment),
    }
    for name in ("VLLM", "FLASHINFER"):
        namespace[f"_{name}_REPO"] = SimpleNamespace(
            source_dir=name,
            git_repository="repository",
            git_tag="pinned-commit",
            git_shallow=False,
        )
    exec(compile(module, "setup.py", "exec"), namespace)
    build = namespace["_CustomBuildExt"]()
    build._clone_and_checkout = lambda source, *arguments: clones.append(source)
    build._apply_musa_patch_series = lambda source: events.append("patch")
    build._install_vllm = lambda source: events.append("install")
    build.run()
    assert events == ["patch", "install", "build"]
    expected_clones = [] if skip else ["VLLM", "FLASHINFER"]
    assert clones == expected_clones


def test_native_dispatch_retains_kernel_policy_and_capacity_handling() -> None:
    source = _bindings_source()
    assert "output_index.narrow(0, 0, input.size(0)).narrow(1, 0, topk)" in source
    assert "input.size(0) <= 8 && input.size(1) > 65536" in source
    assert "input.size(1) >= 1048576 ? 48 : 32" in source
    assert "launch_topk_bf16_multi" in source
    assert "launch_topk_bf16(input, indices" in source
    assert "launch_topk_fp32(input, indices" in source
    assert "static thread_local" in source
    assert "getCurrentMUSAStream" in source
    assert "input.size(0) == 0 || input.is_meta()" in source


def test_installation_documentation_exists() -> None:
    assert DOC.is_file()
    source = DOC.read_text(encoding="utf-8")
    assert "DEEPSELECT_SRC_DIR" in source
    assert "vllm._deepselect_C" in source
    assert "sparse_indexer_topk_backend" in source


def test_deepselect_source_selection_stays_upstream() -> None:
    assert "DEEPSELECT_" not in PINS.read_text(encoding="utf-8")
    added = _added_lines(BUILD_PATCH.read_text())
    assert "GIT_REPOSITORY https://github.com/vllm-project/DeepSelect.git" in added
    assert "GIT_TAG d96d33afe1fab0d6066da49cdc91e64c2bee65ea" in added
    assert "DEEPSELECT_REVISION" not in added


@pytest.mark.parametrize(
    "configured",
    [{}, {"VLLM_TARGET_DEVICE": "empty", "MTGPU_TARGET": "mp_22"}],
)
def test_install_vllm_keeps_empty_mode_without_injecting_hardware(
    configured: dict[str, str],
) -> None:
    tree = ast.parse((ROOT / "setup.py").read_text())
    build_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_CustomBuildExt"
    )
    method = next(
        node
        for node in build_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_install_vllm"
    )
    prefix = method.body[
        : next(
            index
            for index, node in enumerate(method.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(name, ast.Name) and name.id == "vllm_install_cmd"
                for name in node.targets
            )
        )
    ]
    function = ast.FunctionDef(
        name="environment",
        args=method.args,
        body=[*prefix, ast.Return(value=ast.Name(id="env", ctx=ast.Load()))],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    inherited = {"DEEPSELECT_SRC_DIR": "/prepared/DeepSelect", **configured}
    namespace = {
        "Path": Path,
        "os": SimpleNamespace(environ=inherited),
        "_PINS": {"VLLM_TAG": "v0.32.0"},
    }
    exec(compile(module, "setup.py", "exec"), namespace)
    environment = namespace["environment"]("/prepared/vllm")
    assert environment == {
        **inherited,
        "VLLM_TARGET_DEVICE": "empty",
        "VLLM_VERSION_OVERRIDE": "0.32.0",
    }


def test_vllm_build_does_not_add_musa_platform_or_compiler_detection() -> None:
    added = _added_lines(BUILD_PATCH.read_text())
    for forbidden in (
        "torch.version.musa",
        'getattr(torch.version, "musa"',
        "_is_musa()",
        'VLLM_TARGET_DEVICE = "musa"',
        'VLLM_TARGET_DEVICE STREQUAL "musa"',
        "CMAKE_CXX_COMPILER=",
        "MTGPU_TARGET",
    ):
        assert forbidden not in added


@pytest.mark.parametrize("target", [None, "mp_31", "mp_22"])
@pytest.mark.parametrize("empty", [False, True])
def test_vllm_empty_mode_builds_only_deepselect_without_architecture_gating(
    target: str | None,
    empty: bool,
) -> None:
    setup_patch = BUILD_PATCH.read_text().split("diff --git a/setup.py", 1)[1]
    source = "if _no_device():\n" + _added_lines(setup_patch)
    extensions = [SimpleNamespace(name="existing")]
    environment = {} if target is None else {"MTGPU_TARGET": target}
    namespace = {
        "_no_device": lambda: empty,
        "os": SimpleNamespace(environ=environment),
        "CMakeExtension": lambda **kwargs: SimpleNamespace(**kwargs),
        "ext_modules": extensions,
    }
    exec(compile(source, str(BUILD_PATCH), "exec"), namespace)
    result = namespace["ext_modules"]
    if empty:
        assert [extension.name for extension in result] == ["vllm._deepselect_C"]
        assert result[0].py_limited_api is False
    else:
        assert result is extensions


@pytest.mark.parametrize("missing_toolchain", [False, True])
def test_deepselect_selects_compiler_before_vllm_project(
    tmp_path: Path, missing_toolchain: bool
) -> None:
    patch = BUILD_PATCH.read_text()
    root_patch = patch.split("diff --git a/CMakeLists.txt", 1)[1]
    configure = _added_lines(root_patch.split("@@")[2])
    source_patch = patch.split(
        "diff --git a/cmake/external_projects/deepselect_source.cmake", 1
    )[1].split("diff --git", 1)[0]
    acquisition = tmp_path / "cmake/external_projects/deepselect_source.cmake"
    acquisition.parent.mkdir(parents=True)
    acquisition.write_text(_added_lines(source_patch) + "\n")
    backend = tmp_path / "DeepSelect source"
    (backend / "cmake").mkdir(parents=True)
    (backend / "CMakeLists.txt").write_text(
        'message(FATAL_ERROR "DeepSelect must not configure before vLLM")\n'
    )
    if not missing_toolchain:
        shutil.copy2(
            _deepselect_source("cmake/vllm_musa_toolchain.cmake"),
            backend / "cmake/vllm_musa_toolchain.cmake",
        )
    compiler = tmp_path / "toolkit/bin/mcc"
    compiler.parent.mkdir(parents=True)
    host_compiler = shutil.which("c++")
    assert host_compiler is not None
    compiler.write_text(f'#!/bin/sh\nexec {shlex.quote(host_compiler)} "$@"\n')
    compiler.chmod(0o755)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        'set(VLLM_TARGET_DEVICE "empty")\n'
        + configure
        + "\nproject(deepselect_toolchain LANGUAGES CXX)\n"
        + 'file(WRITE "${CMAKE_BINARY_DIR}/compiler" "${CMAKE_CXX_COMPILER}")\n'
    )
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
            "-DDEEPSELECT_SRC_DIR=/must-not-download",
        ],
        env={
            **os.environ,
            "DEEPSELECT_SRC_DIR": str(backend),
            "MUSA_HOME": str(compiler.parent.parent),
        },
        capture_output=True,
        text=True,
    )
    if missing_toolchain:
        assert result.returncode != 0
        assert "DeepSelect build integration missing" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "build/compiler").read_text() == str(compiler)


@pytest.mark.parametrize("stale_only", [False, True])
def test_editable_copy_does_not_overwrite_vllm_owned_deepselect(
    tmp_path: Path, stale_only: bool
) -> None:
    tree = ast.parse((ROOT / "setup.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "develop_dynamic_library"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"Path": Path, "shutil": shutil, "PackageNotFoundError": RuntimeError}
    exec(compile(module, "setup.py", "exec"), namespace)
    source = tmp_path / "plugin" / "vllm"
    target = tmp_path / "upstream" / "vllm"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    (source / "_deepselect_C.so").write_text("stale plugin binary")
    (target / "_deepselect_C.so").write_text("new vllm binary")
    binary_source = source
    if stale_only:
        binary_source = source.parent / "build/lib.test/vllm"
        binary_source.mkdir(parents=True)
    (binary_source / "_C.so").write_text("current musa binary")
    if stale_only:
        os.utime(binary_source, (1, 1))
    namespace["develop_dynamic_library"]("vllm", source.parent, target)
    assert (target / "_deepselect_C.so").read_text() == "new vllm binary"
    assert (target / "_C.so").read_text() == "current musa binary"


@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake unavailable")
@pytest.mark.parametrize(
    ("target", "missing", "success"),
    [
        ("mp_31", False, True),
        ("mp_22", False, False),
        ("", False, False),
        ("mp_22;mp_31", False, False),
        ("mp_31", True, False),
    ],
)
def test_musa_cmake_target_configuration(
    tmp_path: Path, target: str, missing: bool, success: bool
) -> None:
    kernel = tmp_path / "DeepSelect/csrc/musa_kernels/topk_bf16.mu"
    kernel.parent.mkdir(parents=True)
    if not missing:
        kernel.write_text("int source_marker;")
    bindings = kernel.parent.parent / "musa_api.cpp"
    bindings.write_text("int binding_marker;")
    integration = _deepselect_source("cmake/vllm_musa.cmake")
    project = f'''cmake_minimum_required(VERSION 3.18)
project(deepselect_configuration LANGUAGES CXX)
set(deepselect_SOURCE_DIR "{tmp_path}/DeepSelect")
function(run_python output)
  if(output STREQUAL "DEEPSELECT_MUSA_ARCHS")
    set(${{output}} "{target}" PARENT_SCOPE)
  elseif(output STREQUAL "DEEPSELECT_MUSA_LIBRARIES")
    set(${{output}} "c10;torch;torch_cpu;torch_python;musa_python" PARENT_SCOPE)
  else()
    set(${{output}} "0" PARENT_SCOPE)
  endif()
endfunction()
function(define_extension_target name)
  cmake_parse_arguments(ARG "WITH_SOABI" "DESTINATION;LANGUAGE"
    "SOURCES;INCLUDE_DIRECTORIES;COMPILE_FLAGS;LIBRARIES" ${{ARGN}})
  if(NOT name STREQUAL "_deepselect_C" OR NOT ARG_DESTINATION STREQUAL "vllm"
      OR NOT ARG_LANGUAGE STREQUAL "CXX" OR NOT ARG_WITH_SOABI)
    message(FATAL_ERROR "DeepSelect must use the vLLM extension target")
  endif()
  if(NOT "musa_python" IN_LIST ARG_LIBRARIES OR "torch_musa" IN_LIST ARG_LIBRARIES)
    message(FATAL_ERROR "DeepSelect must use torch_musa's extension libraries")
  endif()
  add_library(${{name}} MODULE ${{ARG_SOURCES}})
  target_compile_options(${{name}} PRIVATE ${{ARG_COMPILE_FLAGS}})
endfunction()
include("{integration}")
'''
    (tmp_path / "CMakeLists.txt").write_text(project)
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ],
        env={key: value for key, value in os.environ.items() if key != "MTGPU_TARGET"},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    if success:
        commands = json.loads((tmp_path / "build/compile_commands.json").read_text())
        arguments = shlex.split(commands[0]["command"])
        for option in (
            "-mtgpu-opt-level=1",
            "-mtgpu-load-store-opt=1",
            "-mtgpu-fold-global-ldst=1",
        ):
            assert arguments[arguments.index(option) - 1] == "-mllvm"
    if not success:
        expected = "set DEEPSELECT_SRC_DIR" if missing else "requires visible MP31 GPUs"
        assert expected in result.stderr


@pytest.mark.parametrize(
    "capabilities", [[], [(3, 1)], [(2, 2)], [(3, 1), (3, 1)], [(3, 1), (2, 2)]]
)
def test_deepselect_probes_architecture_from_its_own_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    capabilities: list,
) -> None:
    integration = _deepselect_source("cmake/vllm_musa.cmake").read_text()
    assert "MTGPU_TARGET" not in integration
    probe = integration.split('run_python(DEEPSELECT_MUSA_ARCHS\n    "', 1)[1]
    probe = probe.split('"\n', 1)[0]
    runtime = SimpleNamespace(
        device_count=lambda: len(capabilities),
        get_device_capability=lambda device_index: capabilities[device_index],
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(musa=runtime))
    monkeypatch.setitem(sys.modules, "torch_musa", SimpleNamespace())
    exec(compile(probe, "DeepSelect architecture probe", "exec"), {})
    expected = ";".join(sorted({f"mp_{major}{minor}" for major, minor in capabilities}))
    assert capsys.readouterr().out.strip() == expected
