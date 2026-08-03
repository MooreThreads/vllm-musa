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
