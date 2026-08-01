# SPDX-License-Identifier: Apache-2.0
"""Focused caller contract for fused Qwen GDN prefill state gathering."""

# isort: off
import torchada  # noqa: F401
import torch
# isort: on

from vllm_musa.jit_kernel import fused_gdn_gating as gating
from vllm_musa.jit_kernel import gdn_state_gather_mask as fused_state
from vllm_musa.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn


def _attention() -> gdn.MusaQwenGatedDeltaNetAttention:
    attention = object.__new__(gdn.MusaQwenGatedDeltaNetAttention)
    attention.num_k_heads = 2
    attention.num_v_heads = 2
    attention.tp_size = 1
    attention.head_k_dim = 2
    attention.head_v_dim = 2
    attention.A_log = torch.ones(2)
    attention.dt_bias = torch.ones(2)
    return attention


def _inputs() -> tuple[torch.Tensor, ...]:
    mixed_qkv = torch.randn(4, 12)
    a = torch.randn(4, 2)
    b = torch.randn(4, 2)
    state = torch.arange(4 * 2 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2, 2)
    indices = torch.tensor([1, 3], dtype=torch.int32)
    query_start = torch.tensor([0, 2, 4], dtype=torch.int32)
    has_initial = torch.tensor([True, False])
    output = torch.empty(4, 2, 2)
    return mixed_qkv, a, b, state, indices, query_start, has_initial, output


def _patch_gating(monkeypatch) -> None:
    monkeypatch.setattr(
        gating,
        "fused_gdn_gating",
        lambda _a_log, a, b, _dt_bias: (torch.ones_like(a), torch.ones_like(b)),
    )
    monkeypatch.setattr(gdn, "_MATE_GDN_PREFILL_HAS_IS_LOG_SPACE", True)
    monkeypatch.setattr(gdn, "_MATE_GDN_PREFILL_HAS_OUTPUT", True)


def test_prefill_uses_fused_initial_state_and_preserves_scatter(monkeypatch) -> None:
    attention = _attention()
    mixed_qkv, a, b, state, indices, query_start, has_initial, output = _inputs()
    fused_initial = torch.full((2, 2, 2, 2), 7.0)
    final_state = torch.full_like(fused_initial, 11.0)
    seen: dict[str, torch.Tensor] = {}

    _patch_gating(monkeypatch)
    monkeypatch.setattr(
        fused_state, "can_use_fused_gdn_state_gather_mask", lambda *args: True
    )
    monkeypatch.setattr(
        fused_state,
        "fused_gdn_state_gather_mask",
        lambda *args: fused_initial,
    )

    def fake_chunk(**kwargs):
        seen["initial_state"] = kwargs["initial_state"]
        kwargs["output"].fill_(3.0)
        return kwargs["output"], final_state

    monkeypatch.setattr(gdn, "chunk_gated_delta_rule", fake_chunk)

    result = attention._try_mate_prefill(
        mixed_qkv,
        a,
        b,
        state,
        indices,
        query_start,
        has_initial,
        out=output,
    )

    assert result is not None
    torch.testing.assert_close(seen["initial_state"], fused_initial)
    torch.testing.assert_close(result, torch.full_like(result, 3.0))
    torch.testing.assert_close(state[1], final_state[0])
    torch.testing.assert_close(state[3], final_state[1])


def test_prefill_falls_back_to_index_and_mask(monkeypatch) -> None:
    attention = _attention()
    mixed_qkv, a, b, state, indices, query_start, has_initial, output = _inputs()
    expected_initial = torch.stack((state[1], torch.zeros_like(state[3])))
    seen: dict[str, torch.Tensor] = {}

    _patch_gating(monkeypatch)
    monkeypatch.setattr(
        fused_state, "can_use_fused_gdn_state_gather_mask", lambda *args: False
    )

    def fake_chunk(**kwargs):
        seen["initial_state"] = kwargs["initial_state"].clone()
        return kwargs["output"].zero_(), torch.zeros_like(kwargs["initial_state"])

    monkeypatch.setattr(gdn, "chunk_gated_delta_rule", fake_chunk)

    result = attention._try_mate_prefill(
        mixed_qkv,
        a,
        b,
        state,
        indices,
        query_start,
        has_initial,
        out=output,
    )

    assert result is not None
    torch.testing.assert_close(seen["initial_state"], expected_initial)


def test_prefill_fused_failure_keeps_mate_prefill(monkeypatch) -> None:
    attention = _attention()
    mixed_qkv, a, b, state, indices, query_start, has_initial, output = _inputs()
    expected_initial = torch.stack((state[1], torch.zeros_like(state[3])))
    seen: dict[str, torch.Tensor] = {}

    _patch_gating(monkeypatch)
    monkeypatch.setattr(
        fused_state, "can_use_fused_gdn_state_gather_mask", lambda *args: True
    )

    def fail_fused_gather(*args):
        raise RuntimeError("synthetic fused-gather failure")

    monkeypatch.setattr(fused_state, "fused_gdn_state_gather_mask", fail_fused_gather)

    def fake_chunk(**kwargs):
        seen["initial_state"] = kwargs["initial_state"].clone()
        return kwargs["output"].zero_(), torch.zeros_like(kwargs["initial_state"])

    monkeypatch.setattr(gdn, "chunk_gated_delta_rule", fake_chunk)

    result = attention._try_mate_prefill(
        mixed_qkv,
        a,
        b,
        state,
        indices,
        query_start,
        has_initial,
        out=output,
    )

    assert result is not None
    torch.testing.assert_close(seen["initial_state"], expected_initial)
