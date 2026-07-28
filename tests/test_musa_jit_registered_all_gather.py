from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_WRAPPER = ROOT / "vllm_musa/jit_kernel/csrc/allreduce.py"
MUSA_SOURCE = ROOT / "vllm_musa/jit_kernel/csrc/distributed/custom_all_reduce.mu"


def _registered_body(source: str) -> str:
    start = source.index("void vllm_musa_custom_ar_launch_all_gather_registered(")
    end = source.index("void vllm_musa_custom_ar_launch_registered(", start)
    return source[start:end]


def test_registered_all_gather_wrapper_exports_ffi_entry() -> None:
    source = PYTHON_WRAPPER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def launch_all_gather_registered(" in source
    assert "vllm_musa_custom_ar_launch_all_gather_registered" in source


def test_registered_all_gather_reuses_fenced_kernel_without_staging() -> None:
    source = MUSA_SOURCE.read_text(encoding="utf-8")
    body = _registered_body(source)
    assert "musaMemcpyAsync" not in body
    assert "dispatch_all_gather_world_size" in body
    assert "TVM_FFI_ICHECK_EQ(data.ptrs[rank], inp.data_ptr())" in body
    assert "world_size == 2 || world_size == 4 || world_size == 8" in body
    assert "TVM_FFI_ICHECK_EQ(inp.device().device_id, out.device().device_id)" in body
    assert "TVM_FFI_ICHECK_GT(inp.size(0), 0)" in body
    assert "TVM_FFI_ICHECK_GT(inp.size(1), 0)" in body


def test_registered_all_gather_keeps_system_scope_fences() -> None:
    source = MUSA_SOURCE.read_text(encoding="utf-8")
    assert "cross_device_all_gather_start_barrier" in source
    assert "cross_device_all_gather_end_barrier" in source
    assert source.count("__threadfence_system();") >= 4
    assert "vllm_musa_custom_ar_launch_all_gather_registered," in source
