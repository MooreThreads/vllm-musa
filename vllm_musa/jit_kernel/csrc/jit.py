from __future__ import annotations

import functools
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import tvm_ffi
import tvm_ffi.cpp.extension as tvm_ffi_ext
from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path, find_libtvm_ffi
from tvm_ffi.utils import FileLock

_CSRC_DIR = Path(__file__).resolve().parent
_CACHE_DIR = Path(
    os.getenv("VLLM_MUSA_JIT_CACHE_DIR", Path.home() / ".cache" / "vllm_musa_jit")
)


def _parse_env_flags(name: str) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return []
    try:
        return shlex.split(value)
    except ValueError:
        return value.split()


@functools.cache
def _musa_home() -> Path:
    return Path(tvm_ffi_ext._find_musa_home())


def _mcc() -> str:
    return os.environ.get("VLLM_MUSA_MCC", str(_musa_home() / "bin" / "mcc"))


def _normalize_musa_arch(arch: str) -> str:
    arch = arch.strip().lower().replace("mp", "").replace("_", "").replace(".", "")
    if not arch.isdigit():
        raise ValueError(f"Invalid MUSA arch value: {arch!r}")
    return f"mp_{arch}"


def _musa_arch_flags() -> list[str]:
    arch_list = os.environ.get("VLLM_MUSA_ARCH_LIST") or os.environ.get(
        "MATE_MUSA_ARCH_LIST"
    )
    if arch_list:
        return [
            f"--offload-arch={_normalize_musa_arch(arch)}" for arch in arch_list.split()
        ]
    try:
        import torch_musa

        props = torch_musa.get_device_properties(torch_musa.current_device())
        arch = f"{int(props.major)}{int(props.minor)}"
        return [f"--offload-arch={_normalize_musa_arch(arch)}"]
    except Exception:
        return ["--offload-arch=mp_31"]


def _escape(path: Path) -> str:
    return path.resolve().as_posix().replace(":", "$:")


def _write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _object_path(build_dir: Path, source: Path) -> Path:
    suffix = ".musa.o" if source.suffix in {".mu", ".cu"} else ".o"
    return build_dir / f"{source.parent.name}_{source.stem}{suffix}"


def _ninja_content(
    name: str,
    sources: Sequence[Path],
    extra_cflags: Sequence[str],
    extra_musa_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
) -> str:
    include_flags = [
        f"-I{find_include_path()}",
        f"-I{find_dlpack_include_path()}",
        f"-I{_musa_home() / 'include'}",
    ]
    cflags = [
        "-std=c++17",
        "-fPIC",
        "-O3",
        *extra_cflags,
        *_parse_env_flags("VLLM_MUSA_EXTRA_CFLAGS"),
        *include_flags,
    ]
    musa_cflags = [
        "-std=c++17",
        "-fPIC",
        "-O3",
        "-ffast-math",
        *_musa_arch_flags(),
        *extra_musa_cflags,
        *_parse_env_flags("VLLM_MUSA_EXTRA_MUSAFLAGS"),
        *include_flags,
    ]
    libtvm = Path(find_libtvm_ffi())
    ldflags = [
        "-shared",
        f"-L{libtvm.parent}",
        "-ltvm_ffi",
        f"-L{_musa_home() / 'lib'}",
        "-lmusa",
        "-lmusart",
        *extra_ldflags,
        *_parse_env_flags("VLLM_MUSA_EXTRA_LDFLAGS"),
    ]

    build_dir = _CACHE_DIR / name
    lines = [
        "ninja_required_version = 1.3",
        f"cxx = {os.environ.get('CXX', 'c++')}",
        f"mcc = {_mcc()}",
        f"cflags = {' '.join(cflags)}",
        f"musa_cflags = {' '.join(musa_cflags)}",
        f"ldflags = {' '.join(ldflags)}",
        "",
        "rule compile",
        "  depfile = $out.d",
        "  deps = gcc",
        "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out",
        "",
        "rule compile_musa",
        "  depfile = $out.d",
        "  deps = gcc",
        "  command = $mcc -MD -MF $out.d $musa_cflags -c $in -o $out",
        "",
        "rule link",
        "  command = $cxx $in $ldflags -o $out",
        "",
    ]
    objects: list[str] = []
    for source in sources:
        obj = _object_path(build_dir, source)
        objects.append(_escape(obj))
        rule = "compile_musa" if source.suffix in {".mu", ".cu"} else "compile"
        lines.append(f"build {objects[-1]}: {rule} {_escape(source)}")
    output_so = _escape(build_dir / f"{name}.so")
    lines += [
        "",
        f"build {output_so}: link {' '.join(objects)}",
        f"default {output_so}",
        "",
    ]
    return "\n".join(lines)


@functools.lru_cache(maxsize=16)
def load_musa_jit(
    name: str,
    sources: tuple[str, ...],
    extra_cflags: tuple[str, ...] = (),
    extra_musa_cflags: tuple[str, ...] = (),
    extra_ldflags: tuple[str, ...] = (),
):
    resolved_sources = tuple((_CSRC_DIR / source).resolve() for source in sources)
    build_dir = _CACHE_DIR / name
    ninja_path = build_dir / "build.ninja"
    so_path = build_dir / f"{name}.so"
    lock_path = _CACHE_DIR / "tmp" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        build_dir.mkdir(parents=True, exist_ok=True)
        _write_if_different(
            ninja_path,
            _ninja_content(
                name,
                resolved_sources,
                extra_cflags,
                extra_musa_cflags,
                extra_ldflags,
            ),
        )
        command = ["ninja", "-C", str(build_dir), "-f", str(ninja_path)]
        if os.environ.get("VLLM_MUSA_JIT_VERBOSE") == "1":
            command.insert(1, "-v")
        sys.stdout.flush()
        sys.stderr.flush()
        subprocess.run(command, check=True)
    return tvm_ffi.load_module(str(so_path))
