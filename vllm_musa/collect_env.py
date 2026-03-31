# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa
# code borrowed from https://github.com/pytorch/pytorch/blob/main/torch/utils/collect_env.py

import datetime
import locale
import os
import re
import subprocess
import sys
from collections import namedtuple

try:
    import vllm

    VLLM_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    VLLM_AVAILABLE = False

try:
    import vllm_musa

    VLLM_MUSA_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    VLLM_MUSA_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

# System Environment Information
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "torch_version",
        "is_debug_build",
        "musa_compiled_version",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "os",
        "libc_version",
        "python_version",
        "python_platform",
        "is_musa_available",
        "musa_runtime_version",
        "musa_sdk_version",
        "musa_driver_version",
        "gpu_models",
        "pip_version",
        "pip_packages",
        "is_xnnpack_available",
        "cpu_info",
        "vllm_version",
        "vllm_musa_version",
        "vllm_build_flags",
        "gpu_topo",
        "env_vars",
    ],
)

DEFAULT_PIP_PATTERNS = {
    "musa",
    "mthreads",
    "torch",
    "vllm",
    "numpy",
    "mypy",
    "flake8",
    "triton",
    "optree",
    "onnx",
    "mccl",
    "transformers",
    "zmq",
    "torchada",
    "mate",
}


def run(command):
    """Return (return-code, stdout, stderr)."""
    shell = True if type(command) is str else False
    try:
        p = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell
        )
        raw_output, raw_err = p.communicate()
        rc = p.returncode
        if get_platform() == "win32":
            enc = "oem"
        else:
            enc = locale.getpreferredencoding()
        output = raw_output.decode(enc)
        if command == "mthreads-gmi topo -m":
            output = output.rstrip()
        else:
            output = output.strip()
        err = raw_err.decode(enc)
        return rc, output, err.strip()

    except FileNotFoundError:
        cmd_str = command if isinstance(command, str) else command[0]
        return 127, "", f"Command not found: {cmd_str}"


def run_and_read_all(run_lambda, command):
    """Run command using run_lambda; reads and returns entire output if rc is 0."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    """Run command using run_lambda, returns the first regex match if it exists."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def run_and_return_first_line(run_lambda, command):
    """Run command using run_lambda and returns first line if output is not empty."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out.split("\n")[0]


def get_gcc_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")


def get_clang_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "clang --version", r"clang version (.*)"
    )


def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")


def get_musa_driver_version(run_lambda):
    """Get MUSA driver version using mthreads-gmi."""
    rc, out, _ = run_lambda("mthreads-gmi -q")
    if rc != 0:
        return None
    for line in out.splitlines():
        if "Driver Version" in line:
            return line.split(":", 1)[1].strip()
    return None


def get_gpu_info(run_lambda):
    """Get GPU info using mthreads-gmi."""
    rc, out, _ = run_lambda("mthreads-gmi -L")
    if rc != 0:
        return None
    # mthreads-gmi uses "UUID : xxx" (with spaces around colon)
    uuid_regex = re.compile(r"\s*\(UUID\s*:\s*.+?\)")
    return re.sub(uuid_regex, "", out)


def get_musa_runtime_version(run_lambda):
    """Get MUSA toolkit version via mcc."""
    return run_and_parse_first_match(
        run_lambda, "mcc --version", r"mcc version ([^\s]+)"
    )


def get_musa_sdk_version(run_lambda):
    """Get MUSA SDK version info from version.json, formatted with aligned labels."""
    import json

    label_width = 29  # matches "MUSA driver version          "

    musa_home = os.environ.get("MUSA_HOME", "/usr/local/musa")
    version_file = os.path.join(musa_home, "version.json")
    if os.path.isfile(version_file):
        try:
            with open(version_file) as f:
                data = json.load(f)
            lines = []
            for component, info in data.items():
                version = info.get("version", "unknown")
                label = f"{component} version"
                lines.append(f"{label:<{label_width}}: {version}")
            return "\n".join(lines)
        except Exception:
            pass


def get_vllm_version():
    if not VLLM_AVAILABLE:
        return "N/A"

    from vllm import __version__, __version_tuple__

    if __version__ == "dev":
        return "N/A (dev)"
    version_str = __version_tuple__[-1]
    if isinstance(version_str, str) and version_str.startswith("g"):
        if "." in version_str:
            git_sha = version_str.split(".")[0][1:]
            date = version_str.split(".")[-1][1:]
            return f"{__version__} (git sha: {git_sha}, date: {date})"
        else:
            git_sha = version_str[1:]
            return f"{__version__} (git sha: {git_sha})"
    return __version__


def get_vllm_musa_version():
    if not VLLM_MUSA_AVAILABLE:
        return "N/A"
    return vllm_musa.__version__


def summarize_vllm_build_flags():
    return "MUSA Archs: {}".format(
        os.environ.get("MTGPU_TARGET", "Not Set"),
    )


def get_gpu_topo(run_lambda):
    output = None
    if get_platform() == "linux":
        output = run_and_read_all(run_lambda, "mthreads-gmi topo -m")
    return output


def get_cpu_info(run_lambda):
    rc, out, err = 0, "", ""
    if get_platform() == "linux":
        rc, out, err = run_lambda("lscpu")
    elif get_platform() == "win32":
        rc, out, err = run_lambda(
            "wmic cpu get Name,Manufacturer,Family,Architecture,ProcessorType,DeviceID, \
        CurrentClockSpeed,MaxClockSpeed,L2CacheSize,L2CacheSpeed,Revision /VALUE"
        )
    elif get_platform() == "darwin":
        rc, out, err = run_lambda("sysctl -n machdep.cpu.brand_string")
    cpu_info = "None"
    if rc == 0:
        cpu_info = out
    else:
        cpu_info = err
    return cpu_info


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("win32"):
        return "win32"
    elif sys.platform.startswith("cygwin"):
        return "cygwin"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    else:
        return sys.platform


def get_lsb_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "lsb_release -a", r"Description:\t(.*)"
    )


def check_release_file(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "cat /etc/*-release", r'PRETTY_NAME="(.*)"'
    )


def get_os(run_lambda):
    from platform import machine

    platform = get_platform()

    if platform == "win32" or platform == "cygwin":
        return None

    if platform == "darwin":
        return None

    if platform == "linux":
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return "{} ({})".format(desc, machine())

        desc = check_release_file(run_lambda)
        if desc is not None:
            return "{} ({})".format(desc, machine())

        return "{} ({})".format(platform, machine())

    return platform


def get_python_platform():
    import platform

    return platform.platform()


def get_libc_version():
    import platform

    if get_platform() != "linux":
        return "N/A"
    return "-".join(platform.libc_ver())


def get_pip_packages(run_lambda, patterns=None):
    """Return `pip list` output."""
    if patterns is None:
        patterns = DEFAULT_PIP_PATTERNS

    def run_with_pip():
        try:
            import importlib.util

            pip_spec = importlib.util.find_spec("pip")
            pip_available = pip_spec is not None
        except ImportError:
            pip_available = False

        if pip_available:
            cmd = [sys.executable, "-mpip", "list", "--format=freeze"]
        elif os.environ.get("UV") is not None:
            cmd = ["uv", "pip", "list", "--format=freeze"]
        else:
            raise RuntimeError(
                "Could not collect pip list output (pip or uv module not available)"
            )

        out = run_and_read_all(run_lambda, cmd)
        return "\n".join(
            line for line in out.splitlines() if any(name in line for name in patterns)
        )

    pip_version = "pip3" if sys.version[0] == "3" else "pip"
    out = run_with_pip()
    return pip_version, out


def is_xnnpack_available():
    if TORCH_AVAILABLE:
        import torch.backends.xnnpack

        return str(torch.backends.xnnpack.enabled)
    else:
        return "N/A"


def get_env_vars():
    try:
        from vllm.envs import environment_variables as vllm_envs
    except Exception:
        vllm_envs = {}

    env_vars = ""
    secret_terms = ("secret", "token", "api", "access", "password")
    report_prefix = (
        "TORCH",
        "MCCL",
        "PYTORCH",
        "MUBLAS",
        "MUDNN",
        "OMP_",
        "MKL_",
        "MUSA",
        "MTHREADS",
    )
    for k, v in os.environ.items():
        if any(term in k.lower() for term in secret_terms):
            continue
        if k in vllm_envs:
            env_vars = env_vars + "{}={}".format(k, v) + "\n"
        if k.startswith(report_prefix):
            env_vars = env_vars + "{}={}".format(k, v) + "\n"

    return env_vars


def _get_musa_available():
    """Check if MUSA is available via torch.musa."""
    if not TORCH_AVAILABLE:
        return "N/A"
    try:
        import torch_musa  # noqa: F401

        return str(torch.musa.is_available())
    except ImportError:
        pass
    return str(False)


def _get_musa_compiled_version():
    """Get the MUSA version used to build PyTorch (torch_musa)."""
    try:
        import torch_musa

        return getattr(torch_musa, "__version__", "N/A")
    except ImportError:
        return "N/A"


def get_env_info():
    run_lambda = run
    pip_version, pip_list_output = get_pip_packages(run_lambda)

    if TORCH_AVAILABLE:
        version_str = torch.__version__
        debug_mode_str = str(torch.version.debug)
    else:
        version_str = "N/A"
        debug_mode_str = "N/A"

    musa_available_str = _get_musa_available()
    musa_compiled_version = _get_musa_compiled_version()

    sys_version = sys.version.replace("\n", " ")

    vllm_version = get_vllm_version()
    vllm_build_flags = summarize_vllm_build_flags()
    gpu_topo = get_gpu_topo(run_lambda)

    return SystemEnv(
        torch_version=version_str,
        is_debug_build=debug_mode_str,
        python_version="{} ({}-bit runtime)".format(
            sys_version, sys.maxsize.bit_length() + 1
        ),
        python_platform=get_python_platform(),
        is_musa_available=musa_available_str,
        musa_compiled_version=musa_compiled_version,
        musa_runtime_version=get_musa_runtime_version(run_lambda),
        musa_sdk_version=get_musa_sdk_version(run_lambda),
        gpu_models=get_gpu_info(run_lambda),
        musa_driver_version=get_musa_driver_version(run_lambda),
        pip_version=pip_version,
        pip_packages=pip_list_output,
        os=get_os(run_lambda),
        libc_version=get_libc_version(),
        gcc_version=get_gcc_version(run_lambda),
        clang_version=get_clang_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
        is_xnnpack_available=is_xnnpack_available(),
        cpu_info=get_cpu_info(run_lambda),
        vllm_version=vllm_version,
        vllm_musa_version=get_vllm_musa_version(),
        vllm_build_flags=vllm_build_flags,
        gpu_topo=gpu_topo,
        env_vars=get_env_vars(),
    )


env_info_fmt = """
==============================
        System Info
==============================
OS                           : {os}
GCC version                  : {gcc_version}
Clang version                : {clang_version}
CMake version                : {cmake_version}
Libc version                 : {libc_version}

==============================
       MUSA Info
==============================
MUSA driver version          : {musa_driver_version}
{musa_sdk_version}

==============================
       PyTorch Info
==============================
PyTorch version              : {torch_version}
Is debug build               : {is_debug_build}
torch_musa version           : {musa_compiled_version}

==============================
      Python Environment
==============================
Python version               : {python_version}
Python platform              : {python_platform}

==============================
         vLLM Info
==============================
vLLM Version                 : {vllm_version}
vLLM-MUSA Version            : {vllm_musa_version}
vLLM Build Flags:
  {vllm_build_flags}
GPU Topology:
  {gpu_topo}

==============================
       MUSA / GPU Info
==============================
Is MUSA available            : {is_musa_available}
MUSA runtime version         : {musa_runtime_version}
GPU models and configuration : {gpu_models}
Is XNNPACK available         : {is_xnnpack_available}

==============================
          CPU Info
==============================
{cpu_info}

==============================
Versions of relevant libraries
==============================
{pip_packages}
""".strip()

env_info_fmt += "\n\n"

env_info_fmt += """

==============================
     Environment Variables
==============================
{env_vars}
""".strip()


def pretty_str(envinfo):
    def replace_nones(dct, replacement="Could not collect"):
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true="Yes", false="No"):
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text, tag="[prepend]"):
        lines = text.split("\n")
        updated_lines = [tag + line for line in lines]
        return "\n".join(updated_lines)

    def replace_if_empty(text, replacement="No relevant packages"):
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(string):
        if string is not None and len(string.split("\n")) > 1:
            return "\n{}\n".format(string)
        return string

    mutable_dict = envinfo._asdict()

    # If gpu_models is multiline, start on the next line
    mutable_dict["gpu_models"] = maybe_start_on_next_line(envinfo.gpu_models)

    # If the machine doesn't have MUSA, report some fields as 'No MUSA'
    dynamic_musa_fields = [
        "musa_runtime_version",
        "gpu_models",
        "musa_driver_version",
    ]
    all_musa_fields = dynamic_musa_fields
    all_dynamic_musa_fields_missing = all(
        mutable_dict[field] is None for field in dynamic_musa_fields
    )
    musa_available = False
    if TORCH_AVAILABLE:
        try:
            import torch_musa  # noqa: F401

            musa_available = torch.musa.is_available()
        except ImportError:
            pass
    if TORCH_AVAILABLE and not musa_available and all_dynamic_musa_fields_missing:
        for field in all_musa_fields:
            mutable_dict[field] = "No MUSA"
        if envinfo.musa_compiled_version is None:
            mutable_dict["musa_compiled_version"] = "None"

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    # If either of these are '', replace with 'No relevant packages'
    mutable_dict["pip_packages"] = replace_if_empty(mutable_dict["pip_packages"])

    # Tag pip packages with a prefix
    if mutable_dict["pip_packages"]:
        mutable_dict["pip_packages"] = prepend(
            mutable_dict["pip_packages"], "[{}] ".format(envinfo.pip_version)
        )
    mutable_dict["cpu_info"] = envinfo.cpu_info
    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info():
    return pretty_str(get_env_info())


def main():
    print("Collecting environment information...")
    output = get_pretty_env_info()
    print(output)

    if (
        TORCH_AVAILABLE
        and hasattr(torch, "utils")
        and hasattr(torch.utils, "_crash_handler")
    ):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == "linux" and os.path.exists(minidump_dir):
            dumps = [
                os.path.join(minidump_dir, dump) for dump in os.listdir(minidump_dir)
            ]
            latest = max(dumps, key=os.path.getctime)
            ctime = os.path.getctime(latest)
            creation_time = datetime.datetime.fromtimestamp(ctime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            msg = (
                "\n*** Detected a minidump at {} created on {}, ".format(
                    latest, creation_time
                )
                + "if this is related to your bug please include it when you file a report ***"
            )
            print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
