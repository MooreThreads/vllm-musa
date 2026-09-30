from __future__ import annotations

import ast
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm_musa"
    / "deepseek_v4_jit"
    / "fp8_einsum.py"
)
GEMV_PATH = Path(__file__).resolve().parents[1] / "csrc" / "musa" / "gemv.mu"
CUSTOM_OPS_PATH = (
    Path(__file__).resolve().parents[1] / "vllm_musa" / "_custom_ops.py"
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
    # Multi-request decode batches (M >= 8) must not stay on the per-row GEMV.
    assert threshold.value.value == 8
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


def test_o_proj_gemv_uses_calibrated_capture_ladder_tiles() -> None:
    source = GEMV_PATH.read_text(encoding="utf-8")
    helper = source[
        source.index("bool SelectDeepSeekV4Fp8OProjTile(") : source.index(
            "void musa_fused_gemv("
        )
    ]
    generic = source[
        source.index("void musa_fused_gemv(") : source.index(
            "void musa_fused_gemv_moe("
        )
    ]

    assert "reduce_size != 1024" in helper
    assert "hidden_size != 4096" in helper
    assert "nr_n != 1024" in helper
    assert "scale_k_group_tile != 128" in helper
    assert "case 1:" in helper and "case 2:" in helper and "case 8:" in helper
    assert "BlockConfig{4, 32" in helper
    assert "case 4:" in helper and "case 32:" in helper and "case 64:" in helper
    assert "case 20:" in helper and "case 40:" in helper and "case 80:" in helper
    assert "BlockConfig{8, 16" in helper
    assert "case 16:" in helper and "BlockConfig{32, 4" in helper
    assert "bseqlen != 20" in source and "bseqlen != 40" in source
    assert "bseqlen == 80" in source

    dispatch = generic[
        generic.index("BlockConfig forced_config") : generic.index("switch (")
    ]
    assert dispatch.index("SelectDeepSeekV4Fp8OProjTile(") < dispatch.index(
        "ParseForcedBlockConfig(&forced_config)"
    )


def test_o_proj_gemv_writes_one_group_result_directly() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = _module_tree()
    dispatcher = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "try_musa_deepseek_v4_fp8_einsum_gemv"
    )
    gemv_call = next(
        node
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "musa_fused_gemv"
    )
    output_keyword = next(
        keyword for keyword in gemv_call.keywords if keyword.arg == "output"
    )

    assert isinstance(output_keyword.value, ast.Name)
    assert output_keyword.value.id == "direct_group_out"
    assert "group_out_view if group_out_view.is_contiguous() else None" in source
    assert "if direct_group_out is None:" in source
    assert "group_out_view.copy_(group_out)" in source


def test_musa_fused_gemv_accepts_caller_owned_fp8_output() -> None:
    tree = ast.parse(CUSTOM_OPS_PATH.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "musa_fused_gemv"
    )
    output_arg = next(arg for arg in wrapper.args.args if arg.arg == "output")
    assert output_arg.arg == "output"

    native_call = next(
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "musa_fused_gemv"
    )
    assert isinstance(native_call.args[2], ast.Name)
    assert native_call.args[2].id == "output"
