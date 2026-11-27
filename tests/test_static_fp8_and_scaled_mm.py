# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused Triton static scaled_fp8_quant and MUSA ScaledMM kernels."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_0051_PATH = (
    REPO_ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0051-MUSA-vllm.model_executor.kernels.linear.patch"
)
CUSTOM_OPS_PATCH_PATH = (
    REPO_ROOT / "vllm_musa" / "patches" / "vllm___custom_ops.patch.py"
)


def _install_patched_custom_ops(monkeypatch: pytest.MonkeyPatch, torch_mod):
    import vllm

    from vllm_musa.patches import _load_patch_module

    delegated = object()
    vllm_ops = ModuleType("vllm._custom_ops")
    vllm_ops.scaled_fp8_quant = lambda *args, **kwargs: (delegated, args, kwargs)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", vllm_ops)
    monkeypatch.setattr(vllm, "_custom_ops", vllm_ops, raising=False)

    patch_module = _load_patch_module(CUSTOM_OPS_PATCH_PATH)
    monkeypatch.setattr(
        patch_module,
        "current_platform",
        SimpleNamespace(fp8_dtype=lambda: torch_mod.float8_e4m3fn),
    )
    patch_module.apply()
    return vllm_ops, patch_module, delegated


def test_custom_ops_patch_removes_repeat_interleave_and_uses_triton_kernel() -> None:
    source = CUSTOM_OPS_PATCH_PATH.read_text(encoding="utf-8")
    assert "repeat_interleave" not in source
    assert "def _musa_static_scaled_fp8_quant_kernel(" in source
    assert "HAS_ROW_PADDING" in source


def test_static_fp8_quant_scales_padding_and_strided_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    vllm_ops, _patch_mod, delegated = _install_patched_custom_ops(monkeypatch, torch)

    # Dynamic path still delegates to original.
    out_dyn = vllm_ops.scaled_fp8_quant(torch.ones((2, 4)), None)
    assert out_dyn[0] is delegated

    # 1. Scalar per-tensor scale with num_token_padding and preallocated output.
    x = torch.tensor([[2.0, -4.0, 8.0, -16.0], [1.0, 2.0, 3.0, 4.0]])
    s_scalar = torch.tensor(2.0)
    q_pad, s_ret = vllm_ops.scaled_fp8_quant(x, s_scalar, num_token_padding=5)
    assert s_ret is s_scalar
    assert q_pad.shape == (5, 4)
    assert q_pad.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        q_pad[:2].float(),
        torch.tensor([[1.0, -2.0, 4.0, -8.0], [0.5, 1.0, 1.5, 2.0]]),
        atol=0.1,
        rtol=0.1,
    )
    assert torch.count_nonzero(q_pad[2:].float()).item() == 0

    prealloc = torch.full((6, 4), 7.0, dtype=torch.float8_e4m3fn)
    q_pre, _ = vllm_ops.scaled_fp8_quant(
        x, s_scalar, num_token_padding=4, output=prealloc
    )
    assert q_pre is prealloc
    assert torch.count_nonzero(prealloc[2:4].float()).item() == 0

    # 2. 1D per-token (1, -1) and per-channel (-1, 1) scales.
    s_tok = torch.tensor([2.0, 0.5])
    q_tok, _ = vllm_ops.scaled_fp8_quant(x, s_tok, group_shape=(1, -1))
    torch.testing.assert_close(
        q_tok.float(),
        torch.tensor([[1.0, -2.0, 4.0, -8.0], [2.0, 4.0, 6.0, 8.0]]),
        atol=0.1,
        rtol=0.1,
    )

    # 3. 2D block/group scale without explicit group_shape (inferred) + strided view.
    big = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    strided = big[:, ::2]  # (4, 4), stride (8, 2)
    assert not strided.is_contiguous()
    s_2d = torch.tensor([[1.0, 2.0], [4.0, 8.0]])  # 2x2 groups of (2, 2)
    q_2d, _ = vllm_ops.scaled_fp8_quant(strided, s_2d)
    expected = strided / torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [4.0, 4.0, 8.0, 8.0],
            [4.0, 4.0, 8.0, 8.0],
        ]
    )
    torch.testing.assert_close(q_2d.float(), expected, atol=0.25, rtol=0.1)

    # 4. Validation errors.
    with pytest.raises(ValueError, match="explicit group_shape"):
        vllm_ops.scaled_fp8_quant(x, torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="2D input"):
        vllm_ops.scaled_fp8_quant(torch.ones((2, 2, 2)), s_scalar)


def test_patch_0051_registers_rowwise_fp8_musa_int8_and_subclass_backend_filter() -> (
    None
):
    patch_text = PATCH_0051_PATH.read_text(encoding="utf-8")
    assert "MUSARowWiseTorchFP8ScaledMMLinearKernel" in patch_text
    assert "_POSSIBLE_INT8_KERNELS" in patch_text
    assert "MUSATorchInt8ScaledMMLinearKernel" in patch_text
    rowwise_pos = patch_text.rindex("MUSARowWiseTorchFP8ScaledMMLinearKernel,")
    channel_pos = patch_text.rindex("MUSAChannelWiseTorchFP8ScaledMMLinearKernel,")
    # RowWise must precede ChannelWise in priority order so BF16 per-token x
    # per-channel layers hit fused mublasLt _scaled_mm before unfused FP32 DQ.
    assert rowwise_pos < channel_pos


def test_musa_rowwise_and_channelwise_torch_fp8_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        FP8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
        kFp8StaticTensorSym,
    )

    from vllm_musa.model_executor.kernels.linear.scaled_mm import torch_scaled_mm

    monkeypatch.setattr(torch_scaled_mm.current_platform, "is_musa", lambda: True)
    assert torch_scaled_mm.MUSARowWiseTorchFP8ScaledMMLinearKernel.is_supported() == (
        True,
        None,
    )
    assert (
        torch_scaled_mm.MUSARowWiseTorchFP8ScaledMMLinearKernel.get_output_padding(None)
        is None
    )

    cfg_rowwise_bf16 = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8DynamicTokenSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(64, 64),
    )
    assert (
        torch_scaled_mm.MUSARowWiseTorchFP8ScaledMMLinearKernel.can_implement(
            cfg_rowwise_bf16
        )[0]
        is True
    )

    cfg_mixed_static = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8StaticTensorSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(64, 64),
    )
    assert (
        torch_scaled_mm.MUSARowWiseTorchFP8ScaledMMLinearKernel.can_implement(
            cfg_mixed_static
        )[0]
        is False
    )
    assert (
        torch_scaled_mm.MUSAChannelWiseTorchFP8ScaledMMLinearKernel.can_implement(
            cfg_mixed_static
        )[0]
        is True
    )

    # Verify apply_scaled_mm shapes and cached unit_scale reuse without GPU.
    calls = []

    def fake_scaled_mm(A, B, *, out_dtype, scale_a, scale_b, bias=None):
        calls.append(
            (
                scale_a.shape,
                scale_b.shape,
                scale_a.is_contiguous(),
                scale_b.is_contiguous(),
                out_dtype,
                scale_a.data_ptr(),
            )
        )
        out = (A.float() @ B.float()) * scale_a * scale_b
        if bias is not None:
            out = out + bias
        return out.to(out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", fake_scaled_mm)
    rw = object.__new__(torch_scaled_mm.MUSARowWiseTorchFP8ScaledMMLinearKernel)
    A = torch.ones((3, 8), dtype=torch.float8_e4m3fn)
    B = torch.ones((8, 6), dtype=torch.float8_e4m3fn)
    As = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    Bs = torch.full((6, 1), 0.25, dtype=torch.float32)
    y = rw.apply_scaled_mm(
        A=A,
        B=B,
        out_dtype=torch.bfloat16,
        As=As,
        Bs=Bs,
        bias=None,
        output_shape=[3, 6],
    )
    assert y.shape == (3, 6) and y.dtype == torch.bfloat16
    assert calls[-1][:5] == ((3, 1), (1, 6), True, True, torch.bfloat16)

    cw = object.__new__(torch_scaled_mm.MUSAChannelWiseTorchFP8ScaledMMLinearKernel)
    _ = cw.apply_scaled_mm(
        A=A,
        B=B,
        out_dtype=torch.bfloat16,
        As=torch.tensor([0.5]),
        Bs=Bs,
        bias=None,
        output_shape=[3, 6],
    )
    ptr1 = calls[-1][5]
    _ = cw.apply_scaled_mm(
        A=A,
        B=B,
        out_dtype=torch.bfloat16,
        As=torch.tensor([0.5]),
        Bs=Bs,
        bias=None,
        output_shape=[3, 6],
    )
    ptr2 = calls[-1][5]
    assert ptr1 == ptr2

    # Verify MUSATorchInt8ScaledMMLinearKernel (sym + asym + FP16 non-overflow).
    def fake_scaled_int8_quant(x_in, i_s=None, i_zp=None, symmetric=True):
        if i_s is None:
            if symmetric:
                s = (x_in.float().abs().amax(dim=-1, keepdim=True) / 127.0).clamp(
                    min=1e-5
                )
                q = (x_in.float() / s).round().clamp(-128, 127).to(torch.int8)
                return q, s, None
            xmin = x_in.float().amin(dim=-1, keepdim=True)
            xmax = x_in.float().amax(dim=-1, keepdim=True)
            s = ((xmax - xmin) / 255.0).clamp(min=1e-5)
            zp = (-128.0 - xmin / s).round().to(torch.int32)
            q = (x_in.float() / s + zp.float()).round().clamp(-128, 127).to(torch.int8)
            return q, s, zp
        zp_f = 0.0 if i_zp is None else i_zp.float()
        q = (x_in.float() / i_s + zp_f).round().clamp(-128, 127).to(torch.int8)
        return q, i_s, i_zp

    monkeypatch.setattr(
        torch_scaled_mm.ops, "scaled_int8_quant", fake_scaled_int8_quant
    )
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        Int8ScaledMMLinearLayerConfig,
    )

    for sym in (True, False):
        i8_kern = torch_scaled_mm.MUSATorchInt8ScaledMMLinearKernel(
            Int8ScaledMMLinearLayerConfig(
                is_channelwise=True,
                is_static_input_scheme=False,
                input_symmetric=sym,
            ),
            layer_param_names=[
                "weight",
                "weight_scale",
                "input_scale",
                "input_zero_point",
                "azp_adj",
            ],
        )
        layer = torch.nn.Module()
        layer.logical_widths = [16]
        w_f = torch.randn((16, 64), dtype=torch.float32) * 2.0
        w_s = (w_f.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-5)
        w_q = (w_f / w_s).round().clamp(-128, 127).to(torch.int8)
        layer.weight = torch.nn.Parameter(w_q, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(w_s, requires_grad=False)
        layer.input_scale = None
        layer.input_zero_point = None
        i8_kern.process_weights_after_loading(layer)

        # Use float16 with large magnitudes so unscaled int8 dot > 65504 tests
        # BF16 carrier non-overflow!
        x_fp16 = torch.full((2, 3, 64), 10.0, dtype=torch.float16)
        bias_fp16 = torch.full((16,), 0.5, dtype=torch.float16)
        y_i8 = i8_kern.apply_weights(layer, x_fp16, bias=bias_fp16)
        y_ref = (x_fp16.float() @ w_f.t() + bias_fp16.float()).to(torch.float16)
        assert y_i8.shape == (2, 3, 16) and y_i8.dtype == torch.float16
        assert torch.isfinite(y_i8).all()
        torch.testing.assert_close(y_i8.float(), y_ref.float(), rtol=0.05, atol=0.5)
