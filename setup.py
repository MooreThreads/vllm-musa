# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import torch


def _ensure_numpy_compatible():
    """Ensure numpy<2 is installed (MUSA/PyTorch compatibility requirement).

    This may need to be called multiple times during setup because vllm
    installation can pull in numpy>=2 as a dependency.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2", "-q"])


def _ensure_torchada_installed():
    """Ensure torchada is installed (needed for torch.cuda patching)."""
    try:
        import torchada  # noqa: F401
    except ImportError:
        print("Installing torchada...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torchada", "--upgrade", "-q"]
        )
        import torchada  # noqa: F401


# Run dependency checks at setup start
_ensure_numpy_compatible()
_ensure_torchada_installed()

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
third_party = Path("third_party")
arch = platform.machine().lower()

# Detect editable install (pip install -e .) or develop mode
_is_editable_install = (
    "develop" in sys.argv
    or "editable_wheel" in sys.argv
    or any("--editable" in arg or "-e" in arg for arg in sys.argv)
)

if _is_editable_install:
    Path("vllm").mkdir(exist_ok=True)


def develop_dynamic_library(package_name, source_dir="./"):
    try:
        dist = distribution(package_name)
        install_path = dist.locate_file(package_name)

        target_dir = Path(install_path)
        source_path = Path(source_dir) / "vllm"

        for file_path in source_path.glob("*.so"):
            shutil.copy2(file_path, target_dir)

    except PackageNotFoundError:
        print(f"vLLM is not installed '{package_name}'")


class _RepoInfo:
    """Configuration for a third-party git repository."""

    def __init__(self, name, git_repository, git_tag, git_shallow=False):
        self.name = name
        self.git_repository = git_repository
        self.git_tag = git_tag
        self.git_shallow = git_shallow
        self.source_dir = third_party / name


_VLLM_REPO = _RepoInfo(
    name="vllm",
    git_repository="https://github.com/vllm-project/vllm.git",
    git_tag="v0.17.0",
    git_shallow=False,
)

INCLUDE_DIRS = [
    root / "csrc",
    root / _VLLM_REPO.source_dir / "csrc",
]

# =============================================================================
# C/C++ Source Files for Extension Modules
# =============================================================================

VLLM_CSRC_SOURCES = [
    str(_VLLM_REPO.source_dir / "csrc/activation_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cache_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cuda_utils_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cumem_allocator.cpp"),
    str(_VLLM_REPO.source_dir / "csrc/layernorm_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/permute_cols.cu"),
    str(_VLLM_REPO.source_dir / "csrc/pos_encoding_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/sampler.cu"),
    str(_VLLM_REPO.source_dir / "csrc/attention/merge_attn_states.cu"),
    str(_VLLM_REPO.source_dir / "csrc/quantization/w8a8/fp8/per_token_group_quant.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cuda_view.cu"),
    str(_VLLM_REPO.source_dir / "csrc/quantization/w8a8/int8/per_token_group_quant.cu"),
    str(_VLLM_REPO.source_dir / "csrc/quantization/w8a8/int8/scaled_quant.cu"),
    str(_VLLM_REPO.source_dir / "csrc/mamba/mamba_ssm/selective_scan_fwd.cu"),
    str(_VLLM_REPO.source_dir / "csrc/quantization/gptq/q_gemm.cu"),
    str(_VLLM_REPO.source_dir / "csrc/quantization/gguf/gguf_kernel.cu"),
    str(_VLLM_REPO.source_dir / "csrc/layernorm_quant_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/fused_qknorm_rope_kernel.cu"),
    str(_VLLM_REPO.source_dir / "csrc/quantization/activation_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/attention/paged_attention_v1.cu"),
    str(_VLLM_REPO.source_dir / "csrc/attention/paged_attention_v2.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cache_kernels_fused.cu"),
    str(
        _VLLM_REPO.source_dir
        / "csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
    ),
    str(_VLLM_REPO.source_dir / "csrc/quantization/w8a8/fp8/common.cu"),
    str(_VLLM_REPO.source_dir / "csrc/custom_all_reduce.cu"),
    str(_VLLM_REPO.source_dir / "csrc/sparse/cutlass/sparse_scaled_mm_entry.cu"),
    str(_VLLM_REPO.source_dir / "csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu"),
    str(_VLLM_REPO.source_dir / "csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu"),
    str(_VLLM_REPO.source_dir / "csrc/quantization/awq/gemm_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/quantization/fp4/nvfp4_quant_entry.cu"),
    str(_VLLM_REPO.source_dir / "csrc/attention/vertical_slash_index.cu"),
    str(_VLLM_REPO.source_dir / "csrc/torch_bindings.cpp"),
    str(_VLLM_REPO.source_dir / "csrc/topk.cu"),
]

VLLM_MUSA_CSRC_SOURCES = [
    "csrc/musa/torch_bindings.cpp",
    "csrc/musa/fp8/scaled_mm.cpp",
    "csrc/musa/gemv.mu",
]

VLLM_MOE_CSRC_SOURCES = [
    str(_VLLM_REPO.source_dir / "csrc/moe/moe_align_sum_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/moe/topk_softmax_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/moe/torch_bindings.cpp"),
]

# =============================================================================
# Source Code Patching Configuration
# =============================================================================

# Files to completely replace with local versions from csrc/
# These are full file overrides where the local copy replaces the upstream version.
CSRC_FILE_OVERRIDES = [
    "csrc/custom_all_reduce.cu",
    "csrc/custom_all_reduce.cuh",
    "csrc/mamba/mamba_ssm/selective_scan_fwd.cu",
    "csrc/quantization/activation_kernels.cu",
    "csrc/quantization/gptq/q_gemm.cu",
    "csrc/quantization/gptq/compat.cuh",
]

# Inline text replacements to apply to upstream source files.
# Format: {file_path: [{old_text: new_text}, ...]}
# Special case: empty old_text ("") means prepend new_text to file.
CSRC_TEXT_PATCHES = {
    str(_VLLM_REPO.source_dir / "csrc/moe/torch_bindings.cpp"): [
        {"#ifndef USE_ROCM": "#ifndef USE_MUSA"}
    ],
    str(_VLLM_REPO.source_dir / "csrc/torch_bindings.cpp"): [
        {"": '#include "torch_musa/csrc/aten/musa/MUSAContext.h"'}
    ],
    str(_VLLM_REPO.source_dir / "csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh"): [
        {
            '#include "../../../../attention/attention_dtypes.h"': '#include "../../../../attention_musa/attention_dtypes.h"'
        }
    ],
    str(_VLLM_REPO.source_dir / "csrc/moe/moe_align_sum_kernels.cu"): [
        {'#include "../dispatch_utils.h"': '#include "dispatch_utils.h"'},
        {'#include "../cuda_compat.h"': '#include "cuda_compat.h"'},
    ],
    str(_VLLM_REPO.source_dir / "csrc/attention/attention_kernels.cuh"): [
        {
            '#include "../quantization/w8a8/fp8/nvidia/quant_utils.cuh"': '#include "../quantization_musa/w8a8/fp8/nvidia/quant_utils.cuh"'
        }
    ],
    str(_VLLM_REPO.source_dir / "csrc/attention/paged_attention_v1.cu"): [
        {'#include "../cuda_compat.h"': '#include "cuda_compat.h"'}
    ],
    str(_VLLM_REPO.source_dir / "csrc/attention/paged_attention_v2.cu"): [
        {'#include "../cuda_compat.h"': '#include "cuda_compat.h"'}
    ],
    str(_VLLM_REPO.source_dir / "csrc/type_convert.cuh"): [
        {"defined(USE_ROCM)": "defined(USE_MUSA)"}
    ],
    str(_VLLM_REPO.source_dir / "csrc/sparse/cutlass/sparse_scaled_mm_entry.cu"): [
        {'#include "cutlass_extensions/common.hpp"': ""},
        {"get_sm_version_num()": "31"},
    ],
    str(_VLLM_REPO.source_dir / "csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu"): [
        {'#include "cutlass_extensions/common.hpp"': ""},
        {"get_sm_version_num()": "31"},
    ],
    str(_VLLM_REPO.source_dir / "csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu"): [
        {'#include "cutlass_extensions/common.hpp"': ""},
        {"get_sm_version_num()": "31"},
    ],
    str(_VLLM_REPO.source_dir / "vllm/_custom_ops.py"): [
        {
            'if hasattr(torch.ops._C, "gptq_marlin_24_gemm"):': 'if not hasattr(torch.ops._C, "gptq_marlin_24_gemm"):'
        }
    ],
    str(_VLLM_REPO.source_dir / "csrc/activation_kernels.cu"): [
        {"CUDA_VERSION >= 12090 &&": "VLLM_CUDA_HAS_VERSION_CHECK &&"},
        {
            '#include "cuda_vec_utils.cuh"\n#include "dispatch_utils.h"': '#include "dispatch_utils.h"\n#include "cuda_vec_utils.cuh"\n\n'
            "#if defined(__CUDACC__) && defined(CUDA_VERSION)\n"
            "#define VLLM_CUDA_HAS_VERSION_CHECK (CUDA_VERSION >= 12090)\n"
            "#else\n"
            "#define VLLM_CUDA_HAS_VERSION_CHECK 0\n"
            "#endif"
        },
    ],
}

# =============================================================================
# Compiler and Linker Configuration
# =============================================================================

CXX_FLAGS = ["force_mcc"]
LINK_LIBRARIES = ["c10", "torch", "torch_python"]
EXTRA_LINK_ARGS = [
    "-Wl,-rpath,$ORIGIN/../../torch/lib",
    f"-L/usr/lib/{arch}-linux-gnu",
    "-lmublasLt",
]

# Detect MTGPU target architecture
DEFAULT_MTGPU_TARGET = "mp_31"
MTGPU_TARGET = os.environ.get("MTGPU_TARGET", DEFAULT_MTGPU_TARGET)

if torch.musa.is_available():
    try:
        device_props = torch.musa.get_device_properties(0)
        MTGPU_TARGET = f"mp_{device_props.major}{device_props.minor}"
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
else:
    print(f"Warning: torch.musa not available. Using default target: {MTGPU_TARGET}")

SUPPORTED_MTGPU_TARGETS = ["mp_22", "mp_31"]
if MTGPU_TARGET not in SUPPORTED_MTGPU_TARGETS:
    print(
        f"Warning: Unsupported GPU architecture '{MTGPU_TARGET}'. "
        f"Expected one of: {SUPPORTED_MTGPU_TARGETS}"
    )
    sys.exit(1)

MCC_FLAGS = [
    "-DNDEBUG",
    "-O3",
    "-fPIC",
    "-std=c++17",
    "-x",
    "musa",
    "-mtgpu",
    "-Od3",
    "-ffast-math",
    "-fmusa-flush-denormals-to-zero",
    "-fno-strict-aliasing",
    "-DUSE_MUSA",
]

if MTGPU_TARGET == "mp_31":
    MCC_FLAGS.append("-DENABLE_FP8")

COMPILE_ARGS = {
    "mcc": MCC_FLAGS,
    "cxx": CXX_FLAGS,
}

# =============================================================================
# Extension Modules
# =============================================================================

EXT_MODULES = [
    CUDAExtension(
        name="vllm._C",
        sources=VLLM_CSRC_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=LINK_LIBRARIES,
        extra_link_args=EXTRA_LINK_ARGS,
        py_limited_api=False,
    ),
    CUDAExtension(
        name="vllm_musa._C",
        sources=VLLM_MUSA_CSRC_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=LINK_LIBRARIES,
        extra_link_args=EXTRA_LINK_ARGS,
        py_limited_api=False,
    ),
    CUDAExtension(
        name="vllm._moe_C",
        sources=VLLM_MOE_CSRC_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=LINK_LIBRARIES,
        extra_link_args=EXTRA_LINK_ARGS,
        py_limited_api=False,
    ),
]


class _CustomBuildExt(BuildExtension):
    """Custom build extension that clones third-party repositories before building."""

    @staticmethod
    def _clone_and_checkout(repo_path, repo_url, git_tag, git_shallow):
        """Clone a git repository and checkout a specific tag/commit."""
        repo_path.parent.mkdir(exist_ok=True)
        if not repo_path.exists():
            clone_cmd = ["git", "clone"]
            if git_shallow:
                clone_cmd += ["--depth", "1"]
            clone_cmd += [repo_url, str(repo_path)]
            subprocess.check_call(clone_cmd)
            subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)
        else:
            subprocess.check_call(["git", "fetch", "--all"], cwd=repo_path)
            subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)

    @staticmethod
    def _install_vllm(repo_path):
        """install vllm at a specific tag/commit, using existing torch"""
        source_dir = Path(repo_path)

        env = os.environ.copy()
        env["VLLM_TARGET_DEVICE"] = "empty"

        steps = [
            {
                "name": "Install vllm use existing torch",
                "cmd": f"cd {source_dir} && python use_existing_torch.py",
                "shell": True,
                "env": None,
            },
            {
                "name": "Install vllm build requirements",
                "cmd": [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(source_dir / "requirements" / "build.txt"),
                ],
                "shell": False,
                "env": None,
            },
            {
                "name": "Install vllm without target device",
                "cmd": [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    str(source_dir),
                    "--no-build-isolation",
                    "-v",
                ],
                "shell": False,
                "env": env,
            },
        ]

        for step in steps:
            print(f"{step['name']}")

            if step["shell"]:
                print(f"Command: {step['cmd']}")
                subprocess.check_call(step["cmd"], shell=True, env=step["env"])
            else:
                print(f"Command: {' '.join(step['cmd'])}")
                subprocess.check_call(step["cmd"], env=step["env"])

    @staticmethod
    def _apply_file_overrides(repo_path):
        """Copy local patched files to replace upstream versions."""
        for file_path in CSRC_FILE_OVERRIDES:
            src_path = Path(root) / file_path
            dst_path = Path(repo_path) / file_path

            print(f"Applying file override: {file_path}")
            try:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            except (OSError, IOError) as e:
                print(f"Error applying file override {file_path}: {e}")

    @staticmethod
    def _apply_text_patches():
        """Apply inline text replacements to upstream source files."""
        for file_path, replacement_rules in CSRC_TEXT_PATCHES.items():
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            for rule in replacement_rules:
                for old_str, new_str in rule.items():
                    if old_str == "":
                        # Empty old_str means prepend content (if not already present)
                        if new_str not in content:
                            content = new_str + "\n" + content
                    elif old_str == "CUDA_VERSION >= 12090":
                        # Special case: only replace in lines without the macro definition
                        lines = content.split("\n")
                        new_lines = []
                        for line in lines:
                            if "VLLM_CUDA_HAS_VERSION_CHECK" in line:
                                new_lines.append(line)
                            else:
                                new_lines.append(line.replace(old_str, new_str))
                        content = "\n".join(new_lines)
                    else:
                        content = content.replace(old_str, new_str)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Applied text patches: {file_path}")
            else:
                print(f"Skipping (already patched): {file_path}")

    def run(self):
        if os.environ.get("SKIP_THIRD_PARTY", "0") == "1":
            print("Skipping third-party repositories cloning (SKIP_THIRD_PARTY=1)")
        else:
            print("Cloning third-party repositories...")
            self._clone_and_checkout(
                _VLLM_REPO.source_dir,
                _VLLM_REPO.git_repository,
                _VLLM_REPO.git_tag,
                _VLLM_REPO.git_shallow,
            )
            print("Third-party repositories ready.")

        self._install_vllm(_VLLM_REPO.source_dir)

        # Re-ensure numpy<2 after vllm installation (vllm may pull in numpy>=2)
        _ensure_numpy_compatible()

        self._apply_file_overrides(_VLLM_REPO.source_dir)
        self._apply_text_patches()

        super().run()


setup(
    ext_modules=EXT_MODULES,
    cmdclass={"build_ext": _CustomBuildExt.with_options(use_ninja=True)},
    include_package_data=False,
    # Force these dependencies even with --no-build-isolation
    # (pyproject.toml dependencies aren't processed with --no-build-isolation)
    install_requires=[
        "torchada>=0.1.49",
        "mthreads-ml-py>=2.2.11",
        "numpy<2",
        "openai>=2.24.0",
    ],
)

if _is_editable_install:
    develop_dynamic_library("vllm")
