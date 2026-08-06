# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, I001

from __future__ import annotations

from pathlib import Path

import pytest

torchada = pytest.importorskip("torchada")
import torch

import vllm_musa.model_executor.layers.fused_moe.fused_moe as fused_moe


def _inputs(
    *,
    tokens: int = 8192,
    hidden_size: int = 2048,
    experts: int = 256,
    topk: int = 8,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, ...]:
    hidden_states = torch.empty((tokens, hidden_size), dtype=dtype, device="meta")
    w1 = torch.empty((experts, 256, hidden_size), dtype=dtype, device="meta")
    w2 = torch.empty((experts, hidden_size, 128), dtype=dtype, device="meta")
    topk_weights = torch.empty((tokens, topk), dtype=torch.float32, device="meta")
    topk_ids = torch.empty((tokens, topk), dtype=torch.int32, device="meta")
    return hidden_states, w1, w2, topk_weights, topk_ids


def _decode_inputs(
    *,
    tokens: int = 1,
    hidden_size: int = 2048,
    experts: int = 257,
    topk: int = 9,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, ...]:
    hidden_states = torch.empty((tokens, hidden_size), dtype=dtype, device="meta")
    w1 = torch.empty((experts, 256, hidden_size), dtype=dtype, device="meta")
    w2 = torch.empty((experts, hidden_size, 128), dtype=dtype, device="meta")
    topk_weights = torch.empty((tokens, topk), dtype=torch.float32, device="meta")
    topk_ids = torch.empty((tokens, topk), dtype=torch.int32, device="meta")
    return hidden_states, w1, w2, topk_weights, topk_ids


def test_qwen_moe_bf16_prefill_shape_is_enabled_by_default() -> None:
    assert fused_moe._is_calibrated_qwen_moe_bf16_prefill_shape(
        *_inputs(), global_num_experts=256
    )


@pytest.mark.parametrize(
    ("kwargs", "global_num_experts"),
    [
        ({"tokens": 1023}, 256),
        ({"hidden_size": 4096}, 256),
        ({"experts": 128}, 128),
        ({"topk": 4}, 256),
        ({"dtype": torch.float16}, 256),
    ],
)
def test_qwen_moe_bf16_prefill_shape_rejects_other_contracts(
    kwargs: dict[str, object], global_num_experts: int
) -> None:
    assert not fused_moe._is_calibrated_qwen_moe_bf16_prefill_shape(
        *_inputs(**kwargs), global_num_experts=global_num_experts
    )


def test_qwen_moe_bf16_prefill_policy_wires_upstream_fallback() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "vllm_musa/model_executor/layers/fused_moe/fused_moe.py"
    ).read_text(encoding="utf-8")
    assert "prefer_upstream_qwen_prefill" in source
    assert "and not prefer_upstream_qwen_prefill" in source
    assert "VLLM_MUSA_QWEN_MOE_UPSTREAM_PREFILL" not in source


@pytest.mark.parametrize(
    ("tokens", "experts", "topk", "global_num_experts"),
    [(1, 257, 9, 257), (4, 257, 9, 257), (8, 256, 8, 256), (12, 256, 8, 256)],
)
def test_qwen_moe_bf16_decode_gemv_matches_calibrated_tokens(
    tokens: int, experts: int, topk: int, global_num_experts: int
) -> None:
    assert fused_moe.matches_qwen35_moe_bf16_decode_gemv_layer(
        *_decode_inputs(tokens=tokens, experts=experts, topk=topk),
        global_num_experts=global_num_experts,
        max_tokens=12,
    )


@pytest.mark.parametrize(
    ("kwargs", "global_num_experts"),
    [
        ({"tokens": 16}, 257),
        ({"hidden_size": 4096}, 257),
        ({"experts": 255, "topk": 8}, 255),
        ({"topk": 8}, 257),
        ({"dtype": torch.float16}, 257),
    ],
)
def test_qwen_moe_bf16_decode_gemv_rejects_other_contracts(
    kwargs: dict[str, object], global_num_experts: int
) -> None:
    assert not fused_moe.matches_qwen35_moe_bf16_decode_gemv_layer(
        *_decode_inputs(**kwargs),
        global_num_experts=global_num_experts,
        max_tokens=12,
    )
