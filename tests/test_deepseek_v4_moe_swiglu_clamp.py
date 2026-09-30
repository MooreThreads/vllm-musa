"""DeepSeek-V4 routed experts honour swiglu_limit on every MUSA MoE path."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FUSED_MOE = ROOT / "vllm_musa/model_executor/layers/fused_moe/fused_moe.py"
SERIES_PATCH = (
    ROOT / "vllm_musa/patches/series/"
    "0176-MUSA-pass-the-SwiGLU-clamp-through-functional-fused_.patch"
)


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(FUSED_MOE.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _keyword_values(func: ast.FunctionDef, keyword: str) -> list[str]:
    return [
        ast.unparse(kw.value)
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == keyword
    ]


def test_upstream_functional_path_forwards_clamp() -> None:
    patch = SERIES_PATCH.read_text(encoding="utf-8")
    assert "gemm1_clamp_limit=quant_config.gemm1_clamp_limit" in patch
    assert "ApplyMoEActivationConfig(clamp_limit=gemm1_clamp_limit)" in patch
    assert "gemm1_clamp_limit=gemm1_clamp_limit," in patch


def test_dispatcher_forwards_clamp_to_every_backend() -> None:
    dispatch = _function("_musa_fused_experts_impl_dispatch")
    assert "gemm1_clamp_limit" in [a.arg for a in dispatch.args.args]
    assert _keyword_values(dispatch, "swiglu_limit").count("gemm1_clamp_limit") == 2
    assert _keyword_values(dispatch, "gemm1_clamp_limit") == ["gemm1_clamp_limit"]
    source = ast.unparse(dispatch)
    assert "'_swiglu_limit': gemm1_clamp_limit" in source
    assert "gemm1_clamp_limit is None" in source


def test_native_gemv_w1_receives_clamp() -> None:
    impl = _function("fused_experts_impl")
    assert "_swiglu_limit" in [a.arg for a in impl.args.kwonlyargs]
    assert sorted(_keyword_values(impl, "swiglu_limit")) == [
        "_swiglu_limit",
        "_swiglu_limit or 0.0",
    ]


def _clamped_swiglu_ref(gate_up, limit: float):
    import torch

    d = gate_up.shape[-1] // 2
    gate = torch.clamp(gate_up[..., :d], max=limit)
    up = torch.clamp(gate_up[..., d:], min=-limit, max=limit)
    return gate * torch.sigmoid(gate) * up


@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_moe_gemv_swiglu_epilogue_matches_reference(limit: float) -> None:
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        pytest.skip("MUSA-only test")
    from vllm_musa import _custom_ops as musa_ops

    torch.manual_seed(0)
    num_experts, n2, k, topk, tokens = 8, 256, 512, 2, 3
    a = torch.randn(tokens, k, device="musa", dtype=torch.bfloat16)
    # Large weights push gate/up well past the clamp for many columns.
    w = torch.randn(num_experts, n2, k, device="musa", dtype=torch.bfloat16) * 0.5
    topk_ids = torch.randint(0, num_experts, (tokens, topk), device="musa").int()
    topk_w = torch.ones(tokens, topk, device="musa", dtype=torch.float32)
    out = torch.empty(tokens * topk, n2 // 2, device="musa", dtype=torch.bfloat16)

    musa_ops.musa_fused_gemv_moe(
        a,
        w,
        out,
        None,
        None,
        topk_w,
        topk_ids,
        False,
        topk,
        False,
        use_swigelu=True,
        swiglu_limit=limit,
    )
    gate_up = torch.einsum("tk,tjnk->tjn", a.float(), w[topk_ids.long()].float())
    gate_up = gate_up.reshape(tokens * topk, n2)
    if limit > 0:
        ref = _clamped_swiglu_ref(gate_up, limit)
        assert gate_up.abs().max() > limit
    else:
        d = n2 // 2
        ref = torch.nn.functional.silu(gate_up[:, :d]) * gate_up[:, d:]
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-1)


def test_upstream_clamped_activation_matches_reference() -> None:
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        pytest.skip("MUSA-only test")
    import vllm._custom_ops  # noqa: F401  (loads the stable-ABI activation ops)
    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig,
        MoEActivation,
        apply_moe_activation,
    )

    torch.manual_seed(0)
    x = torch.randn(40, 512, device="musa", dtype=torch.bfloat16) * 12
    out = torch.empty(40, 256, device="musa", dtype=torch.bfloat16)
    apply_moe_activation(
        MoEActivation.SILU,
        out,
        x,
        activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
    )
    torch.testing.assert_close(
        out.float(), _clamped_swiglu_ref(x.float(), 10.0), rtol=2e-2, atol=5e-2
    )
