# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ccache integration for vllm-musa native extension builds."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
from pathlib import Path

FALSE_VALUES = {"0", "false", "off", "no"}
WRAPPER_SCRIPT = (
    r"""#!/usr/bin/env bash
set -euo pipefail

ccache_bin="${VLLM_MUSA_REAL_CCACHE:-ccache}"
real_mcc="${VLLM_MUSA_REAL_MCC:-mcc}"
musa_compiler="${VLLM_MUSA_CCACHE_MUSA_COMPILER:-}"
source_dir="${VLLM_MUSA_CCACHE_SOURCE_DIR:-${CCACHE_DIR:-/tmp}/vllm-musa-sources}"
mkdir -p "${source_dir}"

raw_args=("$@")
force_musa=0
for ((i = 0; i < ${#raw_args[@]}; i++)); do
  arg="${raw_args[$i]}"
  if [[ "${arg}" == "-x" ]]; then
    next_arg=""
    if ((i + 1 < ${#raw_args[@]})); then
      next_arg="${raw_args[$((i + 1))]}"
    fi
    if [[ "${next_arg}" == "musa" ]]; then
      force_musa=1
    fi
  elif [[ "${arg}" == "-xmusa" || "${arg}" == "-x=musa" ]]; then
    force_musa=1
  elif [[ "${arg}" == *.mu || "${arg}" == *.muh ]]; then
    force_musa=1
  fi
done

args=()
source_include_args=()
for ((i = 0; i < ${#raw_args[@]}; i++)); do
  arg="${raw_args[$i]}"
  if [[ "${arg}" == "-x" ]]; then
    next_arg=""
    if ((i + 1 < ${#raw_args[@]})); then
      next_arg="${raw_args[$((i + 1))]}"
    fi
    if [[ "${next_arg}" == "musa" ]]; then
      i=$((i + 1))
      continue
    fi
  elif [[ "${arg}" == "-xmusa" || "${arg}" == "-x=musa" ]]; then
    continue
  fi

  case "${arg}" in
    *.mu|*.muh|*.cu|*.cuh)
      source_path="$(realpath -m "${arg}")"
      source_include_args+=("-I$(dirname "${source_path}")")
      source_hash="$(printf '%s' "${source_path}" | sha256sum | awk '{print $1}')"
      source_base="$(basename "${source_path}")"
      source_stem="${source_base%.*}"
      ccache_source="${source_dir}/${source_hash}_${source_stem}.cu"
      if [[ ! -e "${ccache_source}" ]] """
    r"""|| ! cmp -s "${source_path}" "${ccache_source}"; then
        tmp_source="${ccache_source}.$$"
        cp "${source_path}" "${tmp_source}"
        mv "${tmp_source}" "${ccache_source}"
      fi
      args+=("${ccache_source}")
      ;;
    *.cc|*.cpp|*.cxx)
      if [[ "${force_musa}" == "1" ]]; then
        source_path="$(realpath -m "${arg}")"
        source_include_args+=("-I$(dirname "${source_path}")")
        source_hash="$(printf '%s' "${source_path}" | sha256sum | awk '{print $1}')"
        source_base="$(basename "${source_path}")"
        source_stem="${source_base%.*}"
        ccache_source="${source_dir}/${source_hash}_${source_stem}.cu"
        if [[ ! -e "${ccache_source}" ]] """
    r"""|| ! cmp -s "${source_path}" "${ccache_source}"; then
          tmp_source="${ccache_source}.$$"
          cp "${source_path}" "${tmp_source}"
          mv "${tmp_source}" "${ccache_source}"
        fi
        args+=("${ccache_source}")
      else
        args+=("${arg}")
      fi
      ;;
    *)
      args+=("${arg}")
      ;;
  esac
done

compiler="${real_mcc}"
if [[ "${force_musa}" == "1" ]]; then
  if [[ -z "${musa_compiler}" ]]; then
    echo "vllm-musa: missing VLLM_MUSA_CCACHE_MUSA_COMPILER for MUSA ccache compile" >&2
    exit 1
  fi
  compiler="${musa_compiler}"
fi

exec "${ccache_bin}" "${compiler}" "${source_include_args[@]}" "${args[@]}"
"""
)


def _compiler_digest(path: str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in FALSE_VALUES


def _find_real_compiler(name: str) -> str | None:
    if name == "mcc":
        musa_home = os.environ.get("MUSA_HOME") or os.environ.get("MUSA_PATH")
        if musa_home:
            candidate = Path(musa_home) / "bin" / "mcc"
            if candidate.exists():
                return str(candidate)

        default_mcc = Path("/usr/local/musa/bin/mcc")
        if default_mcc.exists():
            return str(default_mcc)

    return shutil.which(name)


def _resolve_path(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _resolve_executable(
    executable: str,
    *,
    follow_symlinks: bool = True,
    excluded_dirs: tuple[Path, ...] = (),
) -> str:
    path = Path(executable)
    if path.parent != Path("."):
        if follow_symlinks:
            return str(path.resolve())
        return str(path.absolute())

    excluded = {_resolve_path(excluded_dir) for excluded_dir in excluded_dirs}
    path_entries = []
    for path_entry in os.environ.get("PATH", "").split(os.pathsep):
        search_dir = Path(path_entry or os.curdir)
        if _resolve_path(search_dir) not in excluded:
            path_entries.append(path_entry)

    search_path = os.pathsep.join(path_entries)
    found = shutil.which(executable, path=search_path)
    if not found:
        return executable
    if follow_symlinks:
        return str(Path(found).resolve())
    return found


def _write_mcc_wrapper(wrapper_dir: Path) -> Path:
    wrapper = wrapper_dir / "mcc"
    wrapper.write_text(WRAPPER_SCRIPT, encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def _write_musa_compiler_wrapper(wrapper_dir: Path, real_mcc: str) -> Path:
    wrapper = wrapper_dir / "mcc-musa-compiler"
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"# real_mcc_sha256: {_compiler_digest(real_mcc)}\n"
        f"real_mcc={shlex.quote(real_mcc)}\n"
        'exec "${real_mcc}" -x musa "$@"\n'
    )
    wrapper.write_text(script, encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def _link_ccache_wrapper(
    wrapper_dir: Path, compiler_name: str, ccache_bin: str
) -> None:
    wrapper = wrapper_dir / compiler_name
    try:
        if wrapper.exists() or wrapper.is_symlink():
            wrapper.unlink()
        wrapper.symlink_to(ccache_bin)
    except OSError:
        # Some filesystems disallow symlinks; a tiny exec wrapper is good enough.
        real_compiler = _resolve_executable(compiler_name, excluded_dirs=(wrapper_dir,))
        wrapper.write_text(
            "#!/usr/bin/env bash\n"
            f'exec {shlex.quote(ccache_bin)} {shlex.quote(real_compiler)} "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)


def configure_compiler_cache(root: Path) -> bool:
    """Configure ccache for torch_musa/torch extension compilation.

    torch_musa honors ``PYTORCH_MCC`` for MUSA compilation and PyTorch honors
    ``CXX`` for host C++ compilation. The MUSA compiler emits/consumes `.mu`
    files and sometimes uses `-x musa`; ccache 4.x treats those as unsupported
    inputs. The generated `mcc` wrapper creates stable `.cu` copies of those
    sources before invoking ccache, so the real compile becomes cacheable while
    preserving the original source files and mcc invocation semantics.
    """

    if not _env_enabled("VLLM_MUSA_USE_CCACHE", "1"):
        print("vllm-musa: ccache disabled by VLLM_MUSA_USE_CCACHE")
        return False

    ccache_bin = os.environ.get("VLLM_MUSA_CCACHE") or shutil.which("ccache")
    if not ccache_bin:
        print(
            "vllm-musa: ccache not found; native extension builds will not use ccache"
        )
        return False
    ccache_bin = _resolve_executable(ccache_bin)

    real_mcc = os.environ.get("VLLM_MUSA_REAL_MCC") or _find_real_compiler("mcc")
    if not real_mcc:
        print("vllm-musa: mcc not found; cannot configure ccache for MUSA sources")
        return False
    real_mcc = _resolve_executable(real_mcc, follow_symlinks=False)

    cache_dir = Path(
        os.environ.get("VLLM_MUSA_CCACHE_DIR")
        or os.environ.get("CCACHE_DIR")
        or root / ".ccache"
    ).resolve()
    wrapper_dir = cache_dir / "wrappers"
    source_dir = cache_dir / "musa-sources"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    musa_compiler_wrapper = _write_musa_compiler_wrapper(wrapper_dir, real_mcc)
    mcc_wrapper = _write_mcc_wrapper(wrapper_dir)
    for compiler_name in ("cc", "c++", "gcc", "g++"):
        _link_ccache_wrapper(wrapper_dir, compiler_name, ccache_bin)

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(wrapper_dir) not in path_entries:
        os.environ["PATH"] = str(wrapper_dir) + os.pathsep + os.environ.get("PATH", "")

    os.environ.setdefault("CCACHE_DIR", str(cache_dir))
    os.environ.setdefault("CCACHE_BASEDIR", str(root))
    os.environ.setdefault("CCACHE_COMPILERCHECK", "content")
    os.environ.setdefault("CCACHE_NOHASHDIR", "true")
    os.environ.setdefault(
        "CCACHE_SLOPPINESS", "include_file_ctime,include_file_mtime,time_macros"
    )
    os.environ["VLLM_MUSA_REAL_CCACHE"] = ccache_bin
    os.environ["VLLM_MUSA_REAL_MCC"] = real_mcc
    os.environ["VLLM_MUSA_CCACHE_MUSA_COMPILER"] = str(musa_compiler_wrapper)
    os.environ.setdefault("VLLM_MUSA_CCACHE_SOURCE_DIR", str(source_dir))
    os.environ.setdefault("PYTORCH_MCC", str(mcc_wrapper))

    if "CXX" not in os.environ:
        os.environ["CXX"] = str(wrapper_dir / "c++")

    max_size = os.environ.get("VLLM_MUSA_CCACHE_MAXSIZE")
    if max_size:
        os.environ.setdefault("CCACHE_MAXSIZE", max_size)

    print(
        "vllm-musa: ccache enabled for native builds "
        f"(cache_dir={cache_dir}, mcc={mcc_wrapper}, cxx={os.environ.get('CXX')})"
    )
    return True
