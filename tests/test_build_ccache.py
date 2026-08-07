# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm-musa ccache build wiring."""

import os
import subprocess
from pathlib import Path

import pytest

from build_utils.ccache import configure_compiler_cache

_CONFIGURE_ENV_KEYS = (
    "PATH",
    "CXX",
    "PYTORCH_MCC",
    "CCACHE_DIR",
    "CCACHE_BASEDIR",
    "CCACHE_COMPILERCHECK",
    "CCACHE_NOHASHDIR",
    "CCACHE_SLOPPINESS",
    "CCACHE_MAXSIZE",
    "VLLM_MUSA_REAL_CCACHE",
    "VLLM_MUSA_REAL_MCC",
    "VLLM_MUSA_CCACHE_MUSA_COMPILER",
    "VLLM_MUSA_CCACHE_SOURCE_DIR",
)


@pytest.fixture(autouse=True)
def _restore_compiler_cache_environment():
    before = {
        name: os.environ[name]
        for name in _CONFIGURE_ENV_KEYS
        if name in os.environ
    }
    yield
    for name in _CONFIGURE_ENV_KEYS:
        if name in before:
            os.environ[name] = before[name]
        else:
            os.environ.pop(name, None)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_configure_compiler_cache_sets_torch_musa_env(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ccache = bin_dir / "ccache"
    mcc = bin_dir / "mcc"
    _write_executable(ccache, "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(mcc, "#!/usr/bin/env bash\nexit 0\n")

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("VLLM_MUSA_CCACHE", "ccache")
    monkeypatch.setenv("VLLM_MUSA_REAL_MCC", "mcc")
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("PYTORCH_MCC", raising=False)
    monkeypatch.delenv("CCACHE_DIR", raising=False)

    assert configure_compiler_cache(root) is True

    cache_dir = root / ".ccache"
    assert os.environ["CCACHE_DIR"] == str(cache_dir.resolve())
    assert os.environ["PYTORCH_MCC"] == str(cache_dir / "wrappers" / "mcc")
    assert os.environ["CXX"] == str(cache_dir / "wrappers" / "c++")
    assert os.environ["VLLM_MUSA_REAL_CCACHE"] == str(ccache)
    assert os.environ["VLLM_MUSA_REAL_MCC"] == str(mcc)
    assert os.environ["VLLM_MUSA_CCACHE_MUSA_COMPILER"] == str(
        cache_dir / "wrappers" / "mcc-musa-compiler"
    )
    assert str(cache_dir / "wrappers") in os.environ["PATH"].split(os.pathsep)


def test_host_compiler_fallback_uses_real_compiler_path(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    arg_log = tmp_path / "ccache-args.txt"
    ccache = bin_dir / "ccache"
    mcc = bin_dir / "mcc"
    _write_executable(
        ccache,
        "#!/usr/bin/env bash\n" f"printf '%s\\n' \"$@\" > {arg_log}\n" "exit 0\n",
    )
    for compiler_name in ("cc", "c++", "gcc", "g++", "mcc"):
        _write_executable(bin_dir / compiler_name, "#!/usr/bin/env bash\nexit 0\n")

    def fail_symlink(*_args, **_kwargs):
        raise OSError("symlinks disabled")

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(Path, "symlink_to", fail_symlink)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("VLLM_MUSA_CCACHE", str(ccache))
    monkeypatch.setenv("VLLM_MUSA_REAL_MCC", str(mcc))
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("PYTORCH_MCC", raising=False)
    monkeypatch.delenv("CCACHE_DIR", raising=False)

    assert configure_compiler_cache(root) is True

    cache_dir = root / ".ccache"
    gcc_wrapper = cache_dir / "wrappers" / "gcc"
    subprocess.check_call([str(gcc_wrapper), "-c", "source.cpp"])

    args = arg_log.read_text(encoding="utf-8").splitlines()
    assert args[0] == str((bin_dir / "gcc").resolve())
    assert args[0] != str(gcc_wrapper)


def test_mcc_wrapper_presents_musa_sources_as_cu_to_ccache(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    arg_log = tmp_path / "ccache-args.txt"
    ccache = bin_dir / "ccache"
    mcc = bin_dir / "mcc"
    _write_executable(
        ccache,
        "#!/usr/bin/env bash\n" f"printf '%s\\n' \"$@\" > {arg_log}\n" "exit 0\n",
    )
    _write_executable(mcc, "#!/usr/bin/env bash\nexit 0\n")

    root = tmp_path / "repo"
    root.mkdir()
    source = root / "kernel.mu"
    source.write_text("__global__ void kernel() {}\n", encoding="utf-8")

    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("VLLM_MUSA_CCACHE", str(ccache))
    monkeypatch.setenv("VLLM_MUSA_REAL_MCC", str(mcc))
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("PYTORCH_MCC", raising=False)
    monkeypatch.delenv("CCACHE_DIR", raising=False)
    monkeypatch.delenv("VLLM_MUSA_REAL_CCACHE", raising=False)
    monkeypatch.delenv("VLLM_MUSA_CCACHE_SOURCE_DIR", raising=False)

    assert configure_compiler_cache(root) is True

    subprocess.check_call(
        [
            os.environ["PYTORCH_MCC"],
            "-x",
            "musa",
            "-c",
            str(source),
            "-o",
            str(root / "kernel.o"),
        ]
    )

    args = arg_log.read_text(encoding="utf-8").splitlines()
    musa_compiler = Path(args[0])
    assert musa_compiler == Path(os.environ["VLLM_MUSA_CCACHE_MUSA_COMPILER"])
    assert str(mcc) in musa_compiler.read_text(encoding="utf-8")
    assert "-x musa" in musa_compiler.read_text(encoding="utf-8")
    assert f"-I{source.parent}" in args
    assert "-x" not in args
    assert "musa" not in args
    cu_sources = [arg for arg in args if arg.endswith(".cu")]
    assert len(cu_sources) == 1
    assert Path(cu_sources[0]).read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )


def test_mcc_wrapper_only_strips_musa_language_flag(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    arg_log = tmp_path / "ccache-args.txt"
    ccache = bin_dir / "ccache"
    mcc = bin_dir / "mcc"
    _write_executable(
        ccache,
        "#!/usr/bin/env bash\n" f"printf '%s\\n' \"$@\" > {arg_log}\n" "exit 0\n",
    )
    _write_executable(mcc, "#!/usr/bin/env bash\nexit 0\n")

    root = tmp_path / "repo"
    root.mkdir()
    source = root / "host.cpp"
    source.write_text("int f() { return 0; }\n", encoding="utf-8")

    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("VLLM_MUSA_CCACHE", str(ccache))
    monkeypatch.setenv("VLLM_MUSA_REAL_MCC", str(mcc))
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("PYTORCH_MCC", raising=False)
    monkeypatch.delenv("CCACHE_DIR", raising=False)
    monkeypatch.delenv("VLLM_MUSA_REAL_CCACHE", raising=False)
    monkeypatch.delenv("VLLM_MUSA_CCACHE_SOURCE_DIR", raising=False)

    assert configure_compiler_cache(root) is True

    subprocess.check_call(
        [
            os.environ["PYTORCH_MCC"],
            "-x",
            "c++",
            "-c",
            str(source),
            "-o",
            str(root / "host.o"),
        ]
    )

    args = arg_log.read_text(encoding="utf-8").splitlines()
    assert "-x" in args
    assert "c++" in args
    assert str(source) in args
