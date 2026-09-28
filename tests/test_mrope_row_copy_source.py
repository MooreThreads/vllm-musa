# SPDX-License-Identifier: Apache-2.0
"""Source contract for asynchronous MRoPE position uploads in pinned vLLM."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/vllm/vllm/v1/worker/gpu_model_runner.py"


def test_musa_mrope_copy_uses_contiguous_rows_without_a_gate() -> None:
    # The pinned upstream source already implements row-wise copies. The old
    # MUSA-only patch was removed; keep testing the actual upload behavior.
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and ast.unparse(node.iter) == "range(self.mrope_positions.gpu.shape[0])"
    ]
    assert len(loops) == 1
    calls = [
        node for node in ast.walk(loops[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy_"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert ast.unparse(call.func.value) == (
        "self.mrope_positions.gpu[row, :total_num_scheduled_tokens]"
    )
    assert ast.unparse(call.args[0]) == (
        "self.mrope_positions.cpu[row, :total_num_scheduled_tokens]"
    )
    assert any(
        kw.arg == "non_blocking" and isinstance(kw.value, ast.Constant)
        and kw.value.value is True for kw in call.keywords
    )
    assert "VLLM_MUSA_MROPE_ROW_COPY" not in source
