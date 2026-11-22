from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD_PATCH = ROOT / (
    "vllm_musa/patches/series/"
    "0218-MUSA-build-DeepSelect-through-vLLM-extension-target.patch"
)


def _musa_cmake() -> str:
    source = (
        BUILD_PATCH.read_text()
        .split("diff --git a/cmake/external_projects/deepselect.cmake", 1)[1]
        .split("diff --git", 1)[0]
    )
    added = "\n".join(
        line[1:]
        for line in source.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    start = (
        'if(VLLM_TARGET_DEVICE STREQUAL "empty")\n'
        '  set(DeepSelect_MUSA_SOURCE "${deepselect_SOURCE_DIR}/csrc/musa_api.mu")'
    )
    return added[added.index(start) :] + "\n"


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("c++") is None,
    reason="CMake and a host C++ compiler are required",
)
@pytest.mark.parametrize("target", ("empty", "cuda", "rocm"))
@pytest.mark.parametrize("missing", (None, "kernel", "mutlass"))
@pytest.mark.parametrize("generator", ("Unix Makefiles", "Ninja"))
def test_musa_target_build_is_isolated(
    tmp_path: Path, target: str, missing: str | None, generator: str
) -> None:
    if generator == "Ninja" and shutil.which("ninja") is None:
        pytest.skip("Ninja is not installed")
    source = tmp_path / "DeepSelect public source"
    kernel = source / "csrc/musa_api.mu"
    mutlass = source / "csrc/3rdparty/mutlass/include/mute/tensor.hpp"
    for path in (kernel, mutlass):
        path.parent.mkdir(parents=True, exist_ok=True)
    if missing != "kernel":
        kernel.write_text("int deepselect_build_probe() { return 0; }\n")
    if missing != "mutlass":
        mutlass.write_text("")
    before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    compiler = shutil.which("c++")
    mcc = tmp_path / "mcc"
    command_log = tmp_path / "mcc-arguments.json"
    mcc.write_text(
        f"#!{sys.executable}\n"
        "import json, subprocess, sys\n"
        "from pathlib import Path\n"
        f"Path({str(command_log)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "arguments = []\n"
        "position = 1\n"
        "while position < len(sys.argv):\n"
        "    argument = sys.argv[position]\n"
        "    if argument == '-mllvm':\n"
        "        position += 2\n"
        "        continue\n"
        "    if argument.startswith(('--offload-arch=', '-fmusa-', "
        "'-Wno-error=address-of-temporary')):\n"
        "        position += 1\n"
        "        continue\n"
        "    arguments.append('c++' if argument == 'musa' else argument)\n"
        "    position += 1\n"
        f"sys.exit(subprocess.call([{compiler!r}, *arguments]))\n"
    )
    mcc.chmod(0o755)
    native = tmp_path / "native.cmake"
    native.write_text(_musa_cmake())
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(deepselect_probe LANGUAGES CXX)\n"
        "set(CMAKE_CXX_STANDARD 20)\n"
        f'set(VLLM_TARGET_DEVICE "{target}")\n'
        f'set(deepselect_SOURCE_DIR "{source.as_posix()}")\n'
        f'set(DEEPSELECT_MCC "{mcc.as_posix()}")\n'
        "set(original_compiler ${CMAKE_CXX_COMPILER})\n"
        "function(run_python output source error)\n"
        '  if(output STREQUAL "DEEPSELECT_MUSA_LIBRARIES")\n'
        '    set(${output} "musa_python" PARENT_SCOPE)\n'
        "  else()\n"
        '    set(${output} "" PARENT_SCOPE)\n'
        "  endif()\n"
        "endfunction()\n"
        "function(define_extension_target name)\n"
        '  cmake_parse_arguments(ARG "WITH_SOABI" "DESTINATION;LANGUAGE" '
        '"SOURCES;LIBRARIES" ${ARGN})\n'
        '  if(NOT ARG_LANGUAGE STREQUAL "CXX" OR NOT ARG_WITH_SOABI)\n'
        '    message(FATAL_ERROR "Unexpected extension target configuration")\n'
        "  endif()\n"
        '  if(NOT ARG_LIBRARIES STREQUAL "musa_python")\n'
        '    message(FATAL_ERROR "Installed MUSA libraries were not used")\n'
        "  endif()\n"
        "  add_library(${name} MODULE ${ARG_SOURCES})\n"
        '  set_target_properties(${name} PROPERTIES PREFIX "")\n'
        "  install(TARGETS ${name} LIBRARY DESTINATION ${ARG_DESTINATION})\n"
        "endfunction()\n"
        f'include("{native.as_posix()}")\n'
        "if(NOT CMAKE_CXX_COMPILER STREQUAL original_compiler)\n"
        '  message(FATAL_ERROR "DeepSelect changed the host compiler")\n'
        "endif()\n"
        'if(NOT CMAKE_CXX_STANDARD STREQUAL "20")\n'
        '  message(FATAL_ERROR "DeepSelect changed the host C++ standard")\n'
        "endif()\n"
    )
    build = tmp_path / "build"
    configured = subprocess.run(
        ["cmake", "-G", generator, "-S", str(tmp_path), "-B", str(build)],
        capture_output=True,
        text=True,
    )
    if target == "empty" and missing:
        assert configured.returncode != 0
        expected_error = {
            "kernel": "DeepSelect MUSA source missing",
            "mutlass": "DeepSelect MUTLASS submodule is missing",
        }[missing]
        assert expected_error in (configured.stdout + configured.stderr)
    else:
        assert configured.returncode == 0, configured.stdout + configured.stderr
        subprocess.run(["cmake", "--build", str(build)], check=True)
        if target == "empty":
            installed = tmp_path / "installed"
            subprocess.run(
                ["cmake", "--install", str(build), "--prefix", str(installed)],
                check=True,
            )
            assert (installed / "vllm/_deepselect_C.so").is_file()
            arguments = json.loads(command_log.read_text())
            assert str(kernel) in arguments
            assert "-DTORCH_EXTENSION_NAME=_deepselect_C" in arguments
            assert "-DUSE_MUSA" in arguments
            assert "-DTORCH_API_INCLUDE_EXTENSION_H" in arguments
            assert "-MMD" in arguments
            assert "-std=c++17" in arguments
            assert "-std=c++20" not in arguments
            command_log.unlink()
            time.sleep(1.1)
            native.write_text(native.read_text() + "\n")
            subprocess.run(["cmake", "--build", str(build)], check=True)
            assert command_log.exists(), "CMake changes must rebuild the MUSA object"
        else:
            assert not command_log.exists()
    after = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert after == before
