# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit and MUSA hardware tests for MusaMMEncoderAttention & ViT backend routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - host CI without torch
    torch = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
MM_ATTN_PATH = (
    REPO_ROOT
    / "vllm_musa"
    / "model_executor"
    / "layers"
    / "attention"
    / "mm_encoder_attention.py"
)
PATCH_0159_PATH = (
    REPO_ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0159-MUSA-select-MUSA-owned-graph-capture-streams.patch"
)


def test_mm_encoder_attention_source_contract() -> None:
    source = MM_ATTN_PATH.read_text(encoding="utf-8")
    assert "einops" not in source
    assert "torch.arange(" not in source
    assert "with gpu_sync_allowed():" in source
    assert "def forward_oot(" in source
    assert "torch.zeros(shape, dtype=dtype, device=device)" in source


def test_patch_0159_includes_encoder_cudagraph_stream_guard() -> None:
    patch_text = PATCH_0159_PATH.read_text(encoding="utf-8")
    assert "vllm/v1/worker/encoder_cudagraph.py" in patch_text
    assert 'if current_platform.device_type == "musa"' in patch_text


def _has_musa() -> bool:
    return (
        torch is not None
        and getattr(torch.version, "musa", None) is not None
        and torch.musa.is_available()
    )


_DTYPE_CASES = (
    [torch.bfloat16, torch.float16] if torch is not None else ["bf16", "fp16"]
)


@pytest.mark.skipif(not _has_musa(), reason="Requires MUSA GPU")
@pytest.mark.parametrize("dtype", _DTYPE_CASES)
@pytest.mark.parametrize(
    ("batch_size", "seq_len", "num_heads", "num_kv_heads", "head_dim"),
    [
        (1, 576, 16, 16, 72),
        (4, 256, 16, 16, 80),
        (2, 1024, 16, 8, 128),
    ],
)
def test_musa_mm_encoder_dense_4d_matches_sdpa_and_handles_strided_qkv(
    dtype: Any,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    from vllm_musa.model_executor.layers.attention.mm_encoder_attention import (
        _musa_flash_attn_maxseqlen_wrapper,
        _musa_torch_sdpa_wrapper,
    )

    torch.manual_seed(0)
    total_heads = num_heads + 2 * num_kv_heads
    # Interleaved QKV layout from a fused Linear projection (non-contiguous Q/K/V views)
    qkv = torch.randn(
        (batch_size, seq_len, total_heads * head_dim),
        device="musa",
        dtype=dtype,
    )
    q_end = num_heads * head_dim
    k_end = q_end + num_kv_heads * head_dim
    q = qkv[..., :q_end].view(batch_size, seq_len, num_heads, head_dim)
    k = qkv[..., q_end:k_end].view(batch_size, seq_len, num_kv_heads, head_dim)
    v = qkv[..., k_end:].view(batch_size, seq_len, num_kv_heads, head_dim)
    assert not q.is_contiguous()

    scale = head_dim**-0.5
    out_fa = _musa_flash_attn_maxseqlen_wrapper(
        q, k, v, batch_size=batch_size, fa_version=3, scale=scale
    )
    out_sdpa = _musa_torch_sdpa_wrapper(
        q,
        k,
        v,
        scale=scale,
        cu_seqlens=None,
        enable_gqa=(num_heads > num_kv_heads),
    )
    ref = (
        torch.nn.functional.scaled_dot_product_attention(
            q.cpu().float().transpose(1, 2),
            k.cpu().float().transpose(1, 2),
            v.cpu().float().transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=(num_heads > num_kv_heads),
        )
        .transpose(1, 2)
        .to(device="musa", dtype=dtype)
    )

    assert out_fa.is_contiguous()
    assert out_sdpa.is_contiguous()
    torch.testing.assert_close(out_fa, ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out_sdpa, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not _has_musa(), reason="Requires MUSA GPU")
def test_musa_mm_encoder_varlen_and_padded_cudagraph_replay_zeroes_padding() -> None:
    from vllm_musa.model_executor.layers.attention.mm_encoder_attention import (
        _musa_flash_attn_maxseqlen_wrapper,
    )

    torch.manual_seed(42)
    budget_tokens = 1024
    num_heads = 16
    head_dim = 80
    scale = head_dim**-0.5

    q_buf = torch.randn(
        (1, budget_tokens, num_heads, head_dim), device="musa", dtype=torch.bfloat16
    )
    k_buf = torch.randn(
        (1, budget_tokens, num_heads, head_dim), device="musa", dtype=torch.bfloat16
    )
    v_buf = torch.randn(
        (1, budget_tokens, num_heads, head_dim), device="musa", dtype=torch.bfloat16
    )
    cu_buf = torch.tensor([0, 512, 1024, 1024], dtype=torch.int32, device="musa")
    max_seqlen = torch.tensor(budget_tokens, dtype=torch.int32, device="cpu")

    # Warmup before capture
    _ = _musa_flash_attn_maxseqlen_wrapper(
        q_buf,
        k_buf,
        v_buf,
        batch_size=1,
        fa_version=3,
        scale=scale,
        cu_seqlens=cu_buf,
        max_seqlen=max_seqlen,
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_buf = _musa_flash_attn_maxseqlen_wrapper(
            q_buf,
            k_buf,
            v_buf,
            batch_size=1,
            fa_version=3,
            scale=scale,
            cu_seqlens=cu_buf,
            max_seqlen=max_seqlen,
        )

    # Poison out_buf with NaNs (simulating CUDAGraph memory-pool reuse) and
    # zero-pad q/k/v past 384 active tokens, matching EncoderCudaGraphManager.
    out_buf.fill_(float("nan"))
    q_live = torch.randn(
        (1, 384, num_heads, head_dim), device="musa", dtype=torch.bfloat16
    )
    k_live = torch.randn(
        (1, 384, num_heads, head_dim), device="musa", dtype=torch.bfloat16
    )
    v_live = torch.randn(
        (1, 384, num_heads, head_dim), device="musa", dtype=torch.bfloat16
    )
    q_buf.zero_()
    k_buf.zero_()
    v_buf.zero_()
    q_buf[:, :384].copy_(q_live)
    k_buf[:, :384].copy_(k_live)
    v_buf[:, :384].copy_(v_live)
    cu_replay = torch.tensor([0, 128, 384, 384], dtype=torch.int32, device="musa")
    cu_buf.copy_(cu_replay)

    g.replay()
    torch.musa.synchronize()

    assert not torch.isnan(out_buf).any(), (
        "Replay output must not contain NaNs in padding!"
    )
    assert torch.count_nonzero(out_buf[:, 384:]).item() == 0
    ref_0 = (
        torch.nn.functional.scaled_dot_product_attention(
            q_live[:, :128].cpu().float().transpose(1, 2),
            k_live[:, :128].cpu().float().transpose(1, 2),
            v_live[:, :128].cpu().float().transpose(1, 2),
            scale=scale,
        )
        .transpose(1, 2)
        .to(device="musa", dtype=torch.bfloat16)
    )
    ref_1 = (
        torch.nn.functional.scaled_dot_product_attention(
            q_live[:, 128:384].cpu().float().transpose(1, 2),
            k_live[:, 128:384].cpu().float().transpose(1, 2),
            v_live[:, 128:384].cpu().float().transpose(1, 2),
            scale=scale,
        )
        .transpose(1, 2)
        .to(device="musa", dtype=torch.bfloat16)
    )
    ref_active = torch.cat([ref_0, ref_1], dim=1)
    torch.testing.assert_close(out_buf[:, :384], ref_active, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(torch is None, reason="Requires torch")
def test_vit_backend_prefers_torch_sdpa_when_flash_attn_rejects_dtype() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_musa.platform import MUSAPlatform

    with patch.object(MUSAPlatform, "get_device_capability", return_value=None):
        assert (
            MUSAPlatform.get_vit_attn_backend(head_size=80, dtype=torch.bfloat16)
            == AttentionBackendEnum.FLASH_ATTN
        )
        assert (
            MUSAPlatform.get_vit_attn_backend(head_size=80, dtype=torch.float32)
            == AttentionBackendEnum.TORCH_SDPA
        )
        assert (
            MUSAPlatform.get_vit_attn_backend(
                head_size=64,
                dtype=torch.float32,
                backend=AttentionBackendEnum.TRITON_ATTN,
            )
            == AttentionBackendEnum.TRITON_ATTN
        )
