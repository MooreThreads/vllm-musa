# SPDX-License-Identifier: Apache-2.0
"""Verify the installed MUSA runtime cohort against the repository pins.

Runs in the image build after the dependency stage installs. Every expected
version comes from the requirement files, except tilelang-musa, which MATE's
``[tilelang]`` extra declares and is therefore read back from that metadata.
"""

from __future__ import annotations

import importlib
import re
from importlib.metadata import requires, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENT_FILES = (
    ROOT / "requirements/musa_private.txt",
    ROOT / "requirements/common.txt",
)

# Importing these initializes a MUSA device, directly or through tilelang, and
# the build stage has no GPU.
SKIP_IMPORT = frozenset({"flashinfer-python", "flash_mla", "tilelang_musa"})

EXACT_VERSION_DISTS = frozenset(
    {
        "torchada",
        "torch",
        "torch_musa",
        "torchvision",
        "torchaudio",
        "deep_ep",
        "mate",
        "mate-mubin",
        "flash_attn_3",
        "flash_mla",
        "deep-gemm",
        "flashinfer-python",
        "sageattention",
        "tilelang_musa",
        "apache-tvm-ffi",
        "torch_c_dlpack_ext",
    }
)

#: Distributions imported in the image, outside the versioned cohort.
IMPORT_ONLY = ("vllm", "vllm_musa", "transformers", "huggingface_hub")


def mate_extra_pin(extra: str, dist_name: str) -> str:
    """Return the version MATE pins ``dist_name`` to through ``extra``."""
    for requirement in requires("mate") or ():
        name, separator, rest = requirement.partition("==")
        if separator and name.strip().lower() == dist_name and extra in rest:
            return rest.split(";", 1)[0].strip()
    raise RuntimeError(f"mate declares no {dist_name} pin for extra {extra}")


def requirement_prefix(dist_name: str) -> str:
    """Return the version pinned for ``dist_name`` in the requirement files."""
    pattern = re.compile(rf"^{re.escape(dist_name)}(?:\[[^\]]+\])?==(.+)$")
    for requirement_file in REQUIREMENT_FILES:
        for line in requirement_file.read_text().splitlines():
            match = pattern.match(line.strip())
            if match:
                return match.group(1).split("*", 1)[0]
    raise RuntimeError(f"missing {dist_name} pin in {REQUIREMENT_FILES}")


def expected() -> tuple[tuple[str, str, str], ...]:
    """Return ``(distribution, module, expected version)`` for the cohort."""
    return (
        ("torchada", "torchada", requirement_prefix("torchada")),
        ("numpy", "numpy", "1.26."),
        ("torch", "torch", requirement_prefix("torch")),
        ("torch_musa", "torch_musa", requirement_prefix("torch_musa")),
        ("torchvision", "torchvision", requirement_prefix("torchvision")),
        ("torchaudio", "torchaudio", requirement_prefix("torchaudio")),
        ("mate", "mate", requirement_prefix("mate")),
        ("mate-mubin", "mate_mubin", requirement_prefix("mate-mubin")),
        ("flash_attn_3", "flash_attn_3", requirement_prefix("flash_attn_3")),
        ("flash_mla", "flash_mla", requirement_prefix("flash_mla")),
        ("deep-gemm", "deep_gemm", requirement_prefix("deep-gemm")),
        (
            "flashinfer-python",
            "flashinfer",
            requirement_prefix("flashinfer-python"),
        ),
        ("sageattention", "sageattention", requirement_prefix("sageattention")),
        ("deep_ep", "deep_ep", requirement_prefix("deep_ep")),
        ("tilelang_musa", "tilelang", mate_extra_pin("tilelang", "tilelang-musa")),
        ("triton", "triton", requirement_prefix("triton")),
        ("uvloop", "uvloop", ""),
        ("pycountry", "pycountry", ""),
        ("pytest", "pytest", ""),
        ("apache-tvm-ffi", "tvm_ffi", requirement_prefix("apache-tvm-ffi")),
        (
            "torch_c_dlpack_ext",
            "torch_c_dlpack_ext",
            requirement_prefix("torch-c-dlpack-ext"),
        ),
    )


def check(dist_name: str, module_name: str, pinned: str) -> None:
    """Import one module and verify the installed version against ``pinned``."""
    skip_import = dist_name in SKIP_IMPORT
    if not skip_import:
        importlib.import_module(module_name)
    installed = version(dist_name)
    if dist_name in EXACT_VERSION_DISTS and installed != pinned:
        raise RuntimeError(f"{dist_name} expected exactly {pinned}, got {installed}")
    if dist_name not in EXACT_VERSION_DISTS and pinned and not installed.startswith(
        pinned
    ):
        raise RuntimeError(f"{dist_name} expected {pinned}, got {installed}")
    action = "skip import" if skip_import else "import"
    print(f"PASS {action} {module_name} version={installed}")


def main() -> None:
    for entry in expected():
        check(*entry)
    for module_name in IMPORT_ONLY:
        module = importlib.import_module(module_name)
        version_info = getattr(module, "__version__", "unknown")
        print(f"PASS import {module_name} version={version_info}")


if __name__ == "__main__":
    main()
