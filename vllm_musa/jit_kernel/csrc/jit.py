from __future__ import annotations

import functools
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import tvm_ffi
import tvm_ffi.cpp.extension as tvm_ffi_ext

from torchada.musa import build as ta_build
from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path, find_libtvm_ffi
from tvm_ffi.utils import FileLock

_CSRC_DIR = Path(__file__).resolve().parent
_CACHE_DIR = Path(
    os.getenv("VLLM_MUSA_JIT_CACHE_DIR", Path.home() / ".cache" / "vllm_musa_jit")
)


def cache_root() -> Path:
    """Directory the JIT build caches live in (``VLLM_MUSA_JIT_CACHE_DIR`` overrides)."""
    return _CACHE_DIR


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


def _build_targets() -> ta_build.BuildTargets:
    """Which arch(es) this JIT build targets: env first, then the device.

    ``default=None`` on purpose: a JIT object has no stamp to check later, so an
    unanswerable arch must raise here rather than be cached under a guess.  The
    resolution itself lives in ``torchada.musa.build`` so this cache and a wheel's
    stamp cannot disagree about what "the target arch" means.
    """
    raw = next(
        (os.environ[name] for name in ta_build.JIT_ARCH_LIST_ENVS if os.environ.get(name)),
        None,
    )
    device_arch = None
    if not raw:
        try:
            import torch_musa

            props = torch_musa.get_device_properties(torch_musa.current_device())
            device_arch = f"mp_{int(props.major)}{int(props.minor)}"
        except Exception as exc:
            raise RuntimeError(
                "cannot determine the MUSA arch for a JIT build: "
                f"torch_musa device query failed ({exc}). Set VLLM_MUSA_ARCH_LIST "
                "(e.g. '31') to build without a visible device."
            ) from exc
    return ta_build.resolve_targets(raw, device_arch=device_arch, default=None)


def _musa_arch_flags() -> list[str]:
    """``--offload-arch`` flags for the arch(es) this build targets.

    The device arch is authoritative.  There is deliberately **no** mp_31
    fallback: a mubin built for the wrong arch does not fail on MUSA, it runs
    and produces a wrong result (``musaDeviceSynchronize`` returns success and
    the output buffer is left untouched), so guessing here trades a loud build
    error for a silent numerics bug.
    """
    return ta_build.arch_flags(_build_targets().names)


def musa_arch_codes() -> list[str]:
    """Arch codes (``"31"``) this build targets, in flag order."""
    return [name[3:] for name in _build_targets().names]


def musa_arch_define(name: str = "__MUSA_ARCH_LIST__") -> str:
    """The single-arch ``-D<name>=<code>`` flag the TileLang templates branch on.

    The template headers compare this macro against a single ``__MUSA_ARCH__``
    value (``== 310``, ``>= 220``), so it cannot express a fat build; the refusal
    for a multi-arch selection lives in ``torchada.musa.build.arch_define``.
    """
    return ta_build.arch_define(name, targets=_build_targets().names)


def build_tag() -> str:
    """Cache-directory tag for the current target: arch plus compiler.

    A cached object is valid only for the arch and compiler that produced it, so
    both are in the tag and switching either one must not resolve to an existing
    directory.
    """
    return ta_build.build_tag(_build_targets().names, compiler=ta_build.mcc_version(_mcc()))


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

    build_dir = _CACHE_DIR / f"{name}__{build_tag()}"
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
    build_dir = _CACHE_DIR / f"{name}__{build_tag()}"
    ninja_path = build_dir / "build.ninja"
    so_path = build_dir / f"{name}.so"
    lock_path = _CACHE_DIR / "tmp" / f"{name}__{build_tag()}.lock"
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
