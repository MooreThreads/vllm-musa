# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, I001

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from vllm_musa.optimization_contract.qwen import (
    matches_qwen35_moe_bf16_decode_gemv_layer,
    matches_qwen35_moe_bf16_prefill_layer,
)


class _DummyTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype


def _dummy_inputs(
    *,
    tokens: int = 8192,
    hidden_size: int = 2048,
    experts: int = 257,
    inter_size: int = 256,
    topk: int = 9,
    dtype: str = "torch.bfloat16",
    topk_tokens: int | None = None,
) -> tuple[_DummyTensor, ...]:
    t_rows = tokens if topk_tokens is None else topk_tokens
    hidden_states = _DummyTensor((tokens, hidden_size), dtype)
    w1 = _DummyTensor((experts, inter_size, hidden_size), dtype)
    w2 = _DummyTensor((experts, hidden_size, inter_size // 2), dtype)
    topk_weights = _DummyTensor((t_rows, topk), "torch.float32")
    topk_ids = _DummyTensor((t_rows, topk), "torch.int32")
    return hidden_states, w1, w2, topk_weights, topk_ids


@pytest.mark.parametrize(
    ("experts", "topk", "global_num_experts", "tokens"),
    [
        (256, 8, 256, 1024),
        (256, 8, 256, 8192),
        (257, 9, 257, 1024),
        (257, 9, 257, 8192),
    ],
)
def test_qwen_moe_bf16_prefill_matches_unfolded_and_folded_contracts(
    experts: int, topk: int, global_num_experts: int, tokens: int
) -> None:
    assert matches_qwen35_moe_bf16_prefill_layer(
        *_dummy_inputs(tokens=tokens, experts=experts, topk=topk),
        global_num_experts=global_num_experts,
        min_tokens=1024,
    )


@pytest.mark.parametrize(
    ("kwargs", "global_num_experts"),
    [
        ({"tokens": 1023, "experts": 257, "topk": 9}, 257),
        ({"tokens": 1023, "experts": 256, "topk": 8}, 256),
        ({"hidden_size": 4096, "experts": 257, "topk": 9}, 257),
        ({"experts": 128, "topk": 8}, 128),
        ({"experts": 257, "topk": 8}, 257),
        ({"experts": 256, "topk": 9}, 256),
        ({"experts": 257, "topk": 9}, 256),
        ({"experts": 257, "topk": 9, "inter_size": 512}, 257),
        ({"experts": 257, "topk": 9, "topk_tokens": 4096}, 257),
        ({"experts": 257, "topk": 9, "dtype": "torch.float16"}, 257),
    ],
)
def test_qwen_moe_bf16_prefill_shape_rejects_other_contracts(
    kwargs: dict[str, object], global_num_experts: int
) -> None:
    assert not matches_qwen35_moe_bf16_prefill_layer(
        *_dummy_inputs(**kwargs),
        global_num_experts=global_num_experts,
        min_tokens=1024,
    )


def test_qwen_moe_bf16_prefill_policy_wires_upstream_fallback() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "vllm_musa/model_executor/layers/fused_moe/fused_moe.py"
    ).read_text(encoding="utf-8")
    assert "prefer_upstream_qwen_prefill" in source
    assert "and not prefer_upstream_qwen_prefill" in source
    assert "VLLM_MUSA_QWEN_MOE_UPSTREAM_PREFILL" not in source


def test_qwen_moe_bf16_prefill_policy_via_fused_moe_module() -> None:
    pytest.importorskip("torchada")
    import torch

    import vllm_musa.model_executor.layers.fused_moe.fused_moe as fused_moe

    for experts, topk in ((256, 8), (257, 9)):
        hidden_states = torch.empty((4096, 2048), dtype=torch.bfloat16, device="meta")
        w1 = torch.empty((experts, 256, 2048), dtype=torch.bfloat16, device="meta")
        w2 = torch.empty((experts, 2048, 128), dtype=torch.bfloat16, device="meta")
        topk_weights = torch.empty((4096, topk), dtype=torch.float32, device="meta")
        topk_ids = torch.empty((4096, topk), dtype=torch.int32, device="meta")
        assert fused_moe._is_calibrated_qwen_moe_bf16_prefill_shape(
            hidden_states, w1, w2, topk_weights, topk_ids, global_num_experts=experts
        )


def test_bf16_grouped_moe_warmup_skips_upstream_routed_qwen_tp4_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm_musa.model_executor.warmup import deep_gemm_warmup

    deep_gemm_warmup._BF16_WARMUP_CACHE.clear()
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    fake_dg = types.ModuleType("deep_gemm")
    fake_dg.m_grouped_bf16_gemm_nt_contiguous = (
        lambda lhs, weight, out, expert_ids, alignment_m: calls.append(
            (tuple(lhs.shape), tuple(weight.shape))
        )
    )
    monkeypatch.setitem(sys.modules, "deep_gemm", fake_dg)
    monkeypatch.setattr(deep_gemm_warmup, "_alignment_m", lambda: 128)
    monkeypatch.delenv("VLLM_MUSA_FUSED_MOE_BACKEND", raising=False)

    class _FakeMoE(torch.nn.Module):
        def __init__(
            self,
            experts: int,
            inter_size: int,
            top_k: int,
            *,
            shared_folded: bool = False,
        ) -> None:
            super().__init__()
            self.w13_weight = torch.nn.Parameter(
                torch.empty((experts, inter_size, 2048), dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.w2_weight = torch.nn.Parameter(
                torch.empty((experts, 2048, inter_size // 2), dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.top_k = top_k
            self.activation = types.SimpleNamespace(value="silu")
            self.global_num_experts = experts - 1 if shared_folded else experts
            if shared_folded:
                self._musa_shared_folded = True
                self._musa_shared_expert_id = experts - 1

    # TP=4 folded (w1=257, module.top_k=8, module.global_num_experts=256,
    # _musa_shared_folded=True) and unfolded (256, 256, 2048) route to
    # upstream Triton prefill and must not compile dead DeepGEMM buckets.
    tp4_model = torch.nn.Sequential(
        _FakeMoE(257, 256, 8, shared_folded=True),
        _FakeMoE(256, 256, 8),
    )
    deep_gemm_warmup._warmup_bf16_grouped_moe(tp4_model, max_tokens=2048)
    assert calls == []

    # TP=2 folded (w1=257, module.top_k=8, _musa_shared_folded=True) still uses
    # DeepGEMM BF16 prefill and must warm the runtime top_k=9 bucket sizes.
    tp2_model = torch.nn.Sequential(_FakeMoE(257, 512, 8, shared_folded=True))
    deep_gemm_warmup._warmup_bf16_grouped_moe(tp2_model, max_tokens=2048)
    assert len(calls) == 4  # 2 token points (1024, 2048) x 2 weights (w1, w2)
    expected_1024_k9 = deep_gemm_warmup._aligned_grouped_tokens(1024, 9, 257, 128)
    assert calls[0][0][0] == expected_1024_k9


@pytest.mark.parametrize(
    ("tokens", "experts", "topk", "global_num_experts"),
    [(1, 257, 9, 257), (4, 257, 9, 257), (8, 256, 8, 256), (12, 256, 8, 256)],
)
def test_qwen_moe_bf16_decode_gemv_matches_calibrated_tokens(
    tokens: int, experts: int, topk: int, global_num_experts: int
) -> None:
    assert matches_qwen35_moe_bf16_decode_gemv_layer(
        *_dummy_inputs(tokens=tokens, experts=experts, topk=topk),
        global_num_experts=global_num_experts,
        max_tokens=12,
    )


@pytest.mark.parametrize(
    ("kwargs", "global_num_experts"),
    [
        ({"tokens": 16, "experts": 257, "topk": 9}, 257),
        ({"tokens": 4, "hidden_size": 4096, "experts": 257, "topk": 9}, 257),
        ({"tokens": 4, "experts": 255, "topk": 8}, 255),
        ({"tokens": 4, "experts": 257, "topk": 8}, 257),
        ({"tokens": 4, "experts": 257, "topk": 9, "dtype": "torch.float16"}, 257),
    ],
)
def test_qwen_moe_bf16_decode_gemv_rejects_other_contracts(
    kwargs: dict[str, object], global_num_experts: int
) -> None:
    assert not matches_qwen35_moe_bf16_decode_gemv_layer(
        *_dummy_inputs(**kwargs),
        global_num_experts=global_num_experts,
        max_tokens=12,
    )
