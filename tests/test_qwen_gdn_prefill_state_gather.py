# SPDX-License-Identifier: Apache-2.0
"""Focused caller contract for fused Qwen GDN prefill state gathering."""

from types import SimpleNamespace

# isort: off
import torchada  # noqa: F401
import torch
# isort: on

from vllm_musa.jit_kernel import gdn_state_gather_mask as fused_state
from vllm_musa.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn


def _inputs() -> tuple[torch.Tensor, ...]:
    q = torch.randn(4, 2, 2)
    k = torch.randn(4, 2, 2)
    v = torch.randn(4, 2, 2)
    g = torch.randn(4, 2)
    beta = torch.randn(4, 2)
    output = torch.empty_like(v)
    return q, k, v, g, beta, output


def _layer() -> tuple[gdn.Qwen3_5GatedDeltaNet, torch.Tensor]:
    layer = object.__new__(gdn.Qwen3_5GatedDeltaNet)
    state = torch.arange(4 * 2 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2, 2)
    layer.kv_cache = [torch.empty(0), state]
    return layer, state


def _metadata() -> SimpleNamespace:
    return SimpleNamespace(
        num_prefills=2,
        num_prefill_tokens=4,
        num_decode_tokens=0,
    )


def test_prefill_uses_fused_initial_state_and_preserves_scatter(monkeypatch) -> None:
    layer, state = _layer()
    q, k, v, g, beta, output = _inputs()
    indices = torch.tensor([1, 3], dtype=torch.int32)
    has_initial = torch.tensor([True, False])
    query_start = torch.tensor([0, 2, 4], dtype=torch.int32)
    fused_initial = torch.full((2, 2, 2, 2), 7.0)
    final_state = torch.full_like(fused_initial, 11.0)
    seen: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(
        fused_state, "can_use_fused_gdn_state_gather_mask", lambda *args: True
    )
    monkeypatch.setattr(
        fused_state,
        "fused_gdn_state_gather_mask",
        lambda *args: fused_initial,
    )
    monkeypatch.setattr(gdn, "_MATE_GDN_PREFILL_HAS_OUTPUT", True)

    def fake_chunk(*args, initial_state, output, **kwargs):
        seen["initial_state"] = initial_state
        output.fill_(3.0)
        return output, final_state

    monkeypatch.setattr(gdn, "chunk_gated_delta_rule", fake_chunk)

    result = layer._try_mate_prefill(
        q,
        k,
        v,
        g,
        beta,
        output,
        _metadata(),
        query_start,
        indices,
        has_initial,
    )

    assert result is not None
    torch.testing.assert_close(seen["initial_state"], fused_initial)
    torch.testing.assert_close(result, torch.full_like(result, 3.0))
    torch.testing.assert_close(state[1], final_state[0])
    torch.testing.assert_close(state[3], final_state[1])


def test_prefill_falls_back_to_index_and_mask(monkeypatch) -> None:
    layer, state = _layer()
    q, k, v, g, beta, output = _inputs()
    indices = torch.tensor([1, 3], dtype=torch.int32)
    has_initial = torch.tensor([True, False])
    query_start = torch.tensor([0, 2, 4], dtype=torch.int32)
    expected_initial = torch.stack((state[1], torch.zeros_like(state[3])))
    seen: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(
        fused_state, "can_use_fused_gdn_state_gather_mask", lambda *args: False
    )
    monkeypatch.setattr(gdn, "_MATE_GDN_PREFILL_HAS_OUTPUT", True)

    def fake_chunk(*args, initial_state, output, **kwargs):
        seen["initial_state"] = initial_state.clone()
        return output.zero_(), torch.zeros_like(initial_state)

    monkeypatch.setattr(gdn, "chunk_gated_delta_rule", fake_chunk)

    result = layer._try_mate_prefill(
        q,
        k,
        v,
        g,
        beta,
        output,
        _metadata(),
        query_start,
        indices,
        has_initial,
    )

    assert result is not None
    torch.testing.assert_close(seen["initial_state"], expected_initial)
