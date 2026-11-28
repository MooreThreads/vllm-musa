# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: I001
"""Unit and MUSA hardware tests for Qwen3.5 GDN packed GEMM & dual-stream overlap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

try:
    import torchada  # noqa: F401
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - host CI without torch
    torch = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
GDN_PATH = (
    REPO_ROOT
    / "vllm_musa"
    / "model_executor"
    / "layers"
    / "mamba"
    / "gdn"
    / "qwen_gdn_linear_attn.py"
)
FUSED_PROJ_PATH = (
    REPO_ROOT / "vllm_musa" / "jit_kernel" / "tilelang" / "gdn_fused_proj.py"
)


def _has_musa() -> bool:
    return (
        torch is not None
        and getattr(torch.version, "musa", None) is not None
        and torch.musa.is_available()
    )


def test_qwen_gdn_proj_source_contract() -> None:
    gdn_src = GDN_PATH.read_text(encoding="utf-8")
    proj_src = FUSED_PROJ_PATH.read_text(encoding="utf-8")
    assert "def process_weights_after_loading" in gdn_src
    assert "_packed_qkvzba_weight" in gdn_src
    assert "fused_zba_packed" in gdn_src
    assert "musa_gdn_fp8_qkvz_bf16_ba_proj" in gdn_src
    assert "musa_fused_zba_packed" in proj_src


@pytest.mark.skipif(not _has_musa(), reason="Requires MUSA GPU")
@pytest.mark.parametrize("num_tokens", [0, 1, 4, 16, 64, 1025])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_fused_zba_packed_matches_separate_fused_zba(
    num_tokens: int,
    dtype: str,
) -> None:
    from vllm_musa.jit_kernel.tilelang.gdn_fused_proj import (
        fused_zba,
        fused_zba_packed,
    )

    torch_dtype = getattr(torch, dtype)
    num_heads_qk = 8
    num_heads_v = 16
    head_qk = 128
    head_v = 128
    hidden_size = 2048
    total_qkvz = num_heads_qk * head_qk * 2 + num_heads_v * head_v * 2
    total_ba = num_heads_v * 2

    torch.manual_seed(42 + num_tokens)
    x = torch.randn((num_tokens, hidden_size), device="musa", dtype=torch_dtype)
    w_qkvz = (
        torch.randn((total_qkvz, hidden_size), device="musa", dtype=torch_dtype) * 0.02
    )
    w_ba = torch.randn((total_ba, hidden_size), device="musa", dtype=torch_dtype) * 0.02
    w_packed = torch.cat((w_qkvz, w_ba), dim=0).contiguous()

    mixed_qkvz = F.linear(x, w_qkvz)
    mixed_ba = F.linear(x, w_ba)
    z_ref, b_ref, a_ref = fused_zba(
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
    )

    mixed_qkvzba = F.linear(x, w_packed)
    z_out, b_out, a_out = fused_zba_packed(
        mixed_qkvzba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
    )
    torch.musa.synchronize()

    qkv_size = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    torch.testing.assert_close(
        mixed_qkvzba[:, :qkv_size], mixed_qkvz[:, :qkv_size], rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(z_out, z_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(b_out, b_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(a_out, a_ref, rtol=1e-2, atol=1e-2)
    assert z_out.is_contiguous()
    assert b_out.is_contiguous()
    assert a_out.is_contiguous()


@pytest.mark.skipif(not _has_musa(), reason="Requires MUSA GPU")
def test_process_weights_after_loading_packs_and_shares_storage(
    monkeypatch: Any,
) -> None:
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        UnquantizedLinearMethod,
    )

    from vllm_musa.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    qkvz_proj = object.__new__(MergedColumnParallelLinear)
    torch.nn.Module.__init__(qkvz_proj)
    qkvz_proj.quant_method = UnquantizedLinearMethod()
    qkvz_proj.bias = None
    qkvz_proj.weight = torch.nn.Parameter(
        torch.randn((2048, 512), device="musa", dtype=torch.bfloat16),
        requires_grad=False,
    )

    ba_proj = object.__new__(MergedColumnParallelLinear)
    torch.nn.Module.__init__(ba_proj)
    ba_proj.quant_method = UnquantizedLinearMethod()
    ba_proj.bias = None
    ba_proj.weight = torch.nn.Parameter(
        torch.randn((32, 512), device="musa", dtype=torch.bfloat16),
        requires_grad=False,
    )

    orig_qkvz = qkvz_proj.weight.clone()
    orig_ba = ba_proj.weight.clone()

    layer = object.__new__(gdn.MusaQwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.gqa_interleaved_layout = False
    layer.disable_tp_for_ba_proj = False
    layer._lora_enabled = False
    layer.key_dim = 512
    layer.value_dim = 512
    layer.num_v_heads = 16
    layer.tp_size = 1
    layer.hidden_size = 512
    layer.in_proj_qkvz = qkvz_proj
    layer.in_proj_ba = ba_proj
    layer._packed_qkvzba_weight = None

    layer.process_weights_after_loading(torch.bfloat16)
    assert layer._packed_qkvzba_weight is not None
    assert layer._packed_qkvzba_weight.shape == (2080, 512)
    packed_storage = layer._packed_qkvzba_weight.untyped_storage().data_ptr()
    assert qkvz_proj.weight.untyped_storage().data_ptr() == packed_storage
    assert ba_proj.weight.untyped_storage().data_ptr() == packed_storage
    assert torch.equal(qkvz_proj.weight, orig_qkvz)
    assert torch.equal(ba_proj.weight, orig_ba)

    # Env disable leaves _packed_qkvzba_weight None
    monkeypatch.setenv("MUSA_GDN_PACKED_PROJ", "0")
    layer.process_weights_after_loading(torch.bfloat16)
    assert layer._packed_qkvzba_weight is None


@pytest.mark.skipif(not _has_musa(), reason="Requires MUSA GPU")
def test_fp8_qkvz_bf16_ba_proj_captures_dual_stream_under_compile(
    monkeypatch: Any,
) -> None:
    import vllm_musa.model_executor.kernels.linear.scaled_mm.deep_gemm as dg_mod
    from vllm_musa.jit_kernel.tilelang.gdn_fused_proj import fused_zba
    from vllm_musa.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn  # noqa: F401
    from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )

    wq_raw = (torch.randn((1536, 512), device="musa") * 0.1).to(torch.float8_e4m3fn)
    ws_raw = (
        torch.ones((1536 // 128, 512 // 128), device="musa", dtype=torch.float32) * 0.01
    )
    wq_dg, ws_dg = deepgemm_post_process_fp8_weight_block(
        wq=wq_raw, ws=ws_raw, quant_block_shape=(128, 128), use_e8m0=False
    )
    w_ba = torch.randn((8, 512), device="musa", dtype=torch.bfloat16) * 0.02

    mm_streams: list[int] = []
    orig_mm = torch.mm

    def spy_mm(
        a_t: torch.Tensor, b_t: torch.Tensor, *, out: torch.Tensor
    ) -> torch.Tensor:
        mm_streams.append(torch.musa.current_stream().musa_stream)
        return orig_mm(a_t, b_t, out=out)

    monkeypatch.setattr(torch, "mm", spy_mm)

    def fn(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkvz, ba = torch.ops.vllm.musa_gdn_fp8_qkvz_bf16_ba_proj(
            x, wq_dg, ws_dg, w_ba, 128, False
        )
        z, b, a = fused_zba(qkvz, ba, 2, 4, 128, 128)
        return qkvz[:, :1024], z, b, a

    compiled_fn = torch.compile(fn, fullgraph=True)
    x_buf = torch.randn((4, 512), device="musa", dtype=torch.bfloat16)

    # Warmup outside capture: creates alt_stream and uses F.linear (not spy_mm)
    compiled_fn(x_buf)
    torch.musa.synchronize()
    main_stream_id = torch.musa.current_stream().musa_stream
    assert len(mm_streams) == 0

    # Capture compiled graph: must run torch.mm on alt_stream != main_stream_id!
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qkv_cg, z_cg, b_cg, a_cg = compiled_fn(x_buf)
    torch.musa.synchronize()
    assert len(mm_streams) == 1
    assert mm_streams[0] != main_stream_id

    # Replay with fresh inputs and verify exact numerical parity
    for _ in range(3):
        x_new = torch.randn((4, 512), device="musa", dtype=torch.bfloat16)
        x_buf.copy_(x_new)
        graph.replay()
        torch.musa.synchronize()
        qkvz_ref = dg_mod.run_deepgemm(x_new, wq_dg, ws_dg, 128, False)
        ba_ref = F.linear(x_new, w_ba)
        z_ref, b_ref, a_ref = fused_zba(qkvz_ref, ba_ref, 2, 4, 128, 128)
        torch.testing.assert_close(qkv_cg, qkvz_ref[:, :1024], rtol=0, atol=0)
        torch.testing.assert_close(z_cg, z_ref, rtol=0, atol=0)
        torch.testing.assert_close(b_cg, b_ref, rtol=0, atol=0)
        torch.testing.assert_close(a_cg, a_ref, rtol=0, atol=0)
