"""Tripwire on the torchada MUSA arch table that vllm-musa builds and gates against."""

import re
from pathlib import Path

from torchada import arch as musa_arch

ROOT = Path(__file__).resolve().parents[1]
MP31_ONLY_DEVICE_CODE = (
    "csrc/musa/gemv.mu",
    "csrc/musa/common.muh",
    "csrc/musa/dtype.muh",
    "vllm_musa/jit_kernel/csrc/device_utils.h",
    "vllm_musa/jit_kernel/csrc/norm/common.mu",
    "vllm_musa/jit_kernel/csrc/norm/rmsnorm.mu",
    "vllm_musa/jit_kernel/csrc/rope/rotary_embedding.mu",
    "vllm_musa/jit_kernel/csrc/distributed/custom_all_reduce.mu",
)
MP31_BRANCH = re.compile(r"(__MUSA_ARCH__|VLLM_MUSA_ARCH)\s*==\s*310")
DEVICE_SOURCE_SUFFIXES = {".mu", ".muh", ".cu", ".cuh", ".h", ".hpp", ".cc", ".cpp"}


def test_musa_arch_table_is_mp31_only():
    """A new torchada row makes setup.py build it and supports_fp8() follow it."""
    assert {a.capability for a in musa_arch.ARCHS} == {(3, 1)}, (
        f"torchada.arch.ARCHS lists {[a.name for a in musa_arch.ARCHS]}, not only "
        "mp_31. setup.py accepts every listed arch and supports_fp8() follows the "
        "table's fp8 column, so before accepting a new row review the device code "
        "that branches on __MUSA_ARCH__ == 310 for that arch: "
        + ", ".join(MP31_ONLY_DEVICE_CODE)
    )


def test_mp31_only_device_code_list_is_current():
    found = {
        path.relative_to(ROOT).as_posix()
        for top in ("csrc", "vllm_musa")
        for path in (ROOT / top).rglob("*")
        if path.suffix in DEVICE_SOURCE_SUFFIXES
        and path.is_file()
        and MP31_BRANCH.search(path.read_text(errors="replace"))
    }
    assert found == set(MP31_ONLY_DEVICE_CODE)
