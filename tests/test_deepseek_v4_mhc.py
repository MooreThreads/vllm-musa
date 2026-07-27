# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-level checks for DeepSeek-V4 MHC MUSA dispatch."""

import ast
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


def _calls(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call) and _call_name(child.func) == name
        for child in ast.walk(node)
    )


def _loads_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Name)
        and child.id == name
        and isinstance(child.ctx, ast.Load)
        for child in ast.walk(node)
    )


def _stores_subscript(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Subscript)
        and isinstance(child.value, ast.Name)
        and child.value.id == name
        and isinstance(child.ctx, ast.Store)
        for child in ast.walk(node)
    )


def test_mhc_pre_defaults_to_auto_provider():
    source = _read("vllm_musa/deepseek_v4_mhc.py")

    assert "def _select_mhc_pre_auto_impl(" in source
    assert "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS" in source
    assert "hidden_size in {4096, 7168}" in source
    assert "def _mhc_pre_native_provider(" in source
    assert "deepseek_v4_mhc_pre(" in source
    assert "mhc_pre_torch_fallback(" in source
    assert source.index("def _mhc_pre_native_provider(") < source.index(
        "def _mhc_pre_tilelang_provider("
    )
    assert '"deepgemm_big_fuse"' in source
    assert "impl = _select_mhc_pre_auto_impl(residual)" in source
    assert "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL" not in source


def test_mhc_pre_custom_op_is_registered():
    source = _read("vllm_musa/_custom_ops.py")
    bindings = _read("csrc/musa/torch_bindings.cpp")
    headers = _read("csrc/musa/musa_ops.h")
    setup = _read("setup.py")

    assert "def deepseek_v4_mhc_pre(" in source
    assert "torch.ops._C_musa_ops.deepseek_v4_mhc_pre" in source
    assert "deepseek_v4_mhc_pre(Tensor residual" in bindings
    assert "&deepseek_v4_mhc_pre" in bindings
    assert "void deepseek_v4_mhc_pre(" in headers
    assert "csrc/musa/mhc/deepseek_v4_mhc_pre.mu" in setup


def test_fp8_einsum_fallback_validates_group_size_divisibility():
    source = _read("vllm_musa/deepseek_v4_jit/fp8_einsum.py")

    assert "def _validate_group_size_divisible(" in source
    assert "must be divisible by" in source
    assert '_validate_group_size_divisible("activation hidden", hidden)' in source
    assert '_validate_group_size_divisible("weight output", out_dim)' in source
    assert '_validate_group_size_divisible("weight input", in_dim)' in source


def test_mhc_pre_deepgemm_big_fuse_is_shape_selected_by_default():
    source = _read("vllm_musa/deepseek_v4_mhc.py")

    assert "def _mhc_pre_deepgemm_big_fuse_provider(" in source
    assert "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_DEEPGEMM_SPLIT_K" in source
    assert "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_BIG_FUSE_HIDDEN_BLOCK" in source
    assert "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_BIG_FUSE_PASS_CONFIG" in source
    assert "return _mhc_pre_deepgemm_big_fuse_provider(" in source
    assert 'return "deepgemm_big_fuse"' in source


def test_mhc_fused_post_pre_composes_musa_post_pre_and_norm():
    source = _read("vllm_musa/deepseek_v4_mhc.py")
    kernels = _read("vllm_musa/deepseek_v4_jit/tilelang_kernels.py")
    source_tree = ast.parse(source)
    kernels_tree = ast.parse(kernels)
    apply_norm = _function_node(source_tree, "_apply_optional_rms_norm")
    weighted_norm = _function_node(source_tree, "_try_mhc_weighted_rms_norm_musa")
    kernel_factory = _function_node(kernels_tree, "mhc_weighted_rmsnorm_kernel")
    mudnn_like_factory = _function_node(
        kernels_tree, "mhc_weighted_rmsnorm_mudnn_like_kernel"
    )
    inner_kernel = _function_node(kernel_factory, "_kernel")
    mudnn_like_inner = _function_node(mudnn_like_factory, "_kernel")

    assert "def mhc_fused_post_pre_musa(" in source
    assert "residual_cur = mhc_post_musa(" in source
    assert "post_mix_cur, comb_mix_cur, layer_input_cur = mhc_pre_musa(" in source
    assert "def _apply_optional_rms_norm(" in source
    assert "def _try_mhc_weighted_rms_norm_musa(" in source
    assert "VLLM_MUSA_DEEPSEEK_V4_MHC_WEIGHTED_RMSNORM_IMPL" not in source
    assert "def mhc_weighted_rmsnorm_kernel(" in kernels
    assert "def mhc_weighted_rmsnorm_mudnn_like_kernel(" in kernels
    assert _calls(apply_norm, "_try_mhc_weighted_rms_norm_musa")
    assert _loads_name(apply_norm, "norm_weight")
    assert _loads_name(weighted_norm, "kernel_factory")
    assert "mhc_weighted_rmsnorm_kernel" in source
    assert "mhc_weighted_rmsnorm_mudnn_like_kernel" in source
    assert _calls(kernel_factory, "T.rsqrt")
    kernel_args = [arg.arg for arg in inner_kernel.args.args]
    assert len(kernel_args) == 4
    _, weight_arg, out_arg, _ = kernel_args
    assert _loads_name(inner_kernel, weight_arg)
    assert _stores_subscript(inner_kernel, out_arg)
    assert _calls(mudnn_like_factory, "T.ieee_frsqrt")
    mudnn_like_args = [arg.arg for arg in mudnn_like_inner.args.args]
    assert len(mudnn_like_args) == 4
    _, mudnn_weight_arg, mudnn_out_arg, _ = mudnn_like_args
    assert _loads_name(mudnn_like_inner, mudnn_weight_arg)
    assert _stores_subscript(mudnn_like_inner, mudnn_out_arg)


def test_mhc_fused_post_prenorm_small_m_path_is_shape_bounded():
    source = _read("vllm_musa/deepseek_v4_mhc.py")
    kernels = _read("vllm_musa/deepseek_v4_jit/tilelang_kernels.py")

    assert "def _try_mhc_fused_post_prenorm_musa(" in source
    assert "num_tokens <= 16" in source
    assert "hidden_size == 4096" in source
    assert "tile_n = 2 if num_tokens < 8 else 3" in source
    assert "split_k = 8 if num_tokens < 8 else 4" in source
    assert "def mhc_fused_post_prenorm_kernel(" in kernels
    assert "new_residual[row] = T.cast(" in kernels
    assert "residual_out[token_id, row, hidden_idx]" in kernels
    assert "out_partial[" in kernels
    assert "sqrsum_partial[split_id, token_id]" in kernels


def test_mhc_auto_paths_fall_back_when_tilelang_is_unavailable():
    source = _read("vllm_musa/deepseek_v4_mhc.py")

    assert "except (ImportError, OSError, NotImplementedError, RuntimeError):" in source
    assert "return mhc_post_torch_fallback(" in source
    assert "return _mhc_pre_native_provider(" in source


def test_hc_head_musa_eager_path_preserves_token_dimensions():
    source = _read("vllm_musa/deepseek_v4_mhc.py")

    assert "def _reshape_hc_head_input(" in source
    assert "def hc_head_musa(" in source
    assert "hidden_states.flatten(1)" not in source
    assert "token_shape = hidden_states.shape[:-1]" in source
    assert "token_shape = hidden_states.shape[:-2]" in source
    assert "grouped = hidden_states.reshape(-1, hc_mult, hidden_size)" in source
    assert "x = grouped.reshape(grouped.shape[0], -1)" in source
    assert "x @ hc_fn.to(torch.float32).t()" in source
    assert "torch.sum(pre.unsqueeze(-1) * grouped, dim=1)" in source
    assert "return y.reshape(*token_shape, hidden_size).to(dtype)" in source


def test_mhc_pre_decode_prenorm_tilelang_selector_is_default_off():
    source = _read("vllm_musa/deepseek_v4_mhc.py")

    assert "def _select_mhc_pre_big_fuse_prenorm_impl(" in source
    assert "hc_hidden_size == 16384 and num_tokens <= 64" in source
    assert 'return "tilelang"' in source
    assert 'return "deepgemm"' in source
    assert "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_DECODE_PRENORM_IMPL" not in source


def test_mhc_pre_deepgemm_big_fuse_shape_guards_and_splitk():
    source = _read("vllm_musa/deepseek_v4_mhc.py")
    kernels = _read("vllm_musa/deepseek_v4_jit/tilelang_kernels.py")

    assert "def select_mhc_prenorm_split_k(" in source
    assert "if hc_hidden_size == 16384:" in source
    assert "if num_tokens <= 64:" in source
    assert "return 64" in source
    assert "return 16 if num_tokens <= 1024 else 8" in source
    assert "hc_mult != 4" in source
    assert "fn.shape != (hc_mult3, hc_hidden_size)" in source
    assert "hidden_size % hidden_block != 0" in source
    assert "hc_hidden_size % split_k != 0" in source
    assert "tf32_hc_prenorm_gemm" in source
    assert "mhc_pre_big_fuse_decode_split_kernel" in source
    assert "mhc_pre_big_fuse_kernel" in source
    assert "def mhc_pre_big_fuse_kernel(" in kernels
    assert "def mhc_pre_big_fuse_decode_split_kernel(" in kernels


def test_mhc_pre_decode_prenorm_tilelang_partials_path_exists():
    source = _read("vllm_musa/deepseek_v4_mhc.py")
    kernels = _read("vllm_musa/deepseek_v4_jit/tilelang_kernels.py")

    assert "def _mhc_prenorm_gemm_sqrsum_tilelang_decode_partials(" in source
    assert "num_tokens > 64 or hc_hidden_size != 16384" in source
    assert "split_size % 128 != 0" in source
    assert "mhc_prenorm_splitk_x_tme_cast_kernel" in source
    assert "fn.float().contiguous()" in source
    assert "def mhc_prenorm_splitk_x_tme_cast_kernel(" in kernels
    assert "TL_ENABLE_MUSA_TMA_PREFETCH" in kernels
    assert "T.alloc_barrier" in kernels
    assert "with T.ws(" in kernels
    assert "T.gemm(" in kernels
