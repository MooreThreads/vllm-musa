from __future__ import annotations

import ast
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm_musa"
    / "deepseek_v4_jit"
    / "fp8_einsum.py"
)


def _module_tree() -> ast.Module:
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


def test_o_proj_uses_fixed_deepgemm_threshold() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = _module_tree()
    threshold = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_DEEPGEMM_MIN_TOKENS"
            for target in node.targets
        )
    )

    assert isinstance(threshold.value, ast.Constant)
    assert threshold.value.value == 128
    assert "VLLM_MUSA_DEEPSEEK_V4_FP8_EINSUM_IMPL" not in source
    assert "VLLM_MUSA_DEEPSEEK_V4_FP8_EINSUM_DEEPGEMM_MIN_TOKENS" not in source


def test_o_proj_dispatch_keeps_small_m_gemv_fallback() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = _module_tree()
    dispatcher = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "try_musa_deepseek_v4_fp8_einsum_gemv"
    )
    call_names = {
        node.func.id
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attribute_calls = {
        node.func.attr
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "fp8_gemm_nt" in call_names
    assert "musa_fused_gemv" in attribute_calls
    assert source.index("fp8_gemm_nt(") < source.index("musa_ops.musa_fused_gemv(")
    assert "tokens >= _DEEPGEMM_MIN_TOKENS" in source
    assert "is_deep_gemm_e8m0_used=False" in source
