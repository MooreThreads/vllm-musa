# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001
"""Focused caller-output contract for the MUSA Qwen GDN decode path."""

from types import SimpleNamespace

# isort: off
import torchada  # noqa: F401
import torch

# isort: on

from qwen_contract_test_utils import qwen_hybrid_contract, qwen_sampler
from vllm_musa.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn


def _fake_attention(
    *, separate_pool: bool = True
) -> gdn.MusaQwenGatedDeltaNetAttention:
    attention = object.__new__(gdn.MusaQwenGatedDeltaNetAttention)
    attention.key_dim = 4
    attention.value_dim = 4
    attention.head_k_dim = 2
    attention.head_v_dim = 2
    attention.num_v_heads = 2
    attention.tp_size = 1
    attention.activation = "silu"
    attention.A_log = torch.ones(2)
    attention.dt_bias = torch.ones(2)
    attention.conv1d = SimpleNamespace(weight=torch.ones(4, 1, 1), bias=torch.zeros(4))
    attention.kv_cache = (
        torch.zeros(2, 4, 1),
        torch.zeros(4, 2, 2, 2, dtype=torch.float32),
    )
    attention._musa_optimization_contract = (
        qwen_hybrid_contract()
        if separate_pool
        else qwen_sampler(enabled=False)._musa_optimization_contract
    )
    return attention


def _patch_decode_prerequisites(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update",
        lambda x, *_args, **_kwargs: x,
    )


def _metadata(num_decode_tokens: int, state_indices: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        spec_sequence_masks=None,
        num_decodes=num_decode_tokens,
        num_decode_tokens=num_decode_tokens,
        non_spec_query_start_loc=torch.arange(num_decode_tokens + 1, dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor(state_indices, dtype=torch.int64),
    )


def test_mate_decode_writes_caller_output_and_ignores_decoy(monkeypatch) -> None:
    attention = _fake_attention()
    num_tokens = 2
    mixed_qkv = torch.randn(num_tokens, 12)
    a = torch.ones(num_tokens, 2)
    b = torch.ones(num_tokens, 2)
    core_attn_out = torch.zeros(num_tokens, 2, 2)
    metadata = _metadata(num_tokens, [0, 1])
    captured = {}

    _patch_decode_prerequisites(monkeypatch)

    def fake_decode(**kwargs):
        captured.update(kwargs)
        kwargs["output"].fill_(7)
        return torch.full_like(kwargs["output"], 99), None

    monkeypatch.setattr(gdn, "gated_delta_rule_decode", fake_decode)
    monkeypatch.setattr(gdn, "_MATE_GDN_DECODE_HAS_OUTPUT", True)
    assert attention._try_mate_decode(mixed_qkv, b, a, core_attn_out, metadata)

    assert captured["output"].shape == (num_tokens, 1, 2, 2)
    assert captured["output"].data_ptr() == core_attn_out.data_ptr()
    assert torch.equal(core_attn_out, torch.full_like(core_attn_out, 7))
    assert not torch.any(core_attn_out == 99)


def test_mate_decode_legacy_api_copies_returned_output(monkeypatch) -> None:
    attention = _fake_attention()
    num_tokens = 2
    mixed_qkv = torch.randn(num_tokens, 12)
    a = torch.ones(num_tokens, 2)
    b = torch.ones(num_tokens, 2)
    core_attn_out = torch.zeros(num_tokens, 2, 2)
    metadata = _metadata(num_tokens, [0, 1])
    captured = {}

    _patch_decode_prerequisites(monkeypatch)

    def fake_decode(**kwargs):
        captured.update(kwargs)
        return torch.full((num_tokens, 1, 2, 2), 9.0), None

    monkeypatch.setattr(gdn, "gated_delta_rule_decode", fake_decode)
    monkeypatch.setattr(gdn, "_MATE_GDN_DECODE_HAS_OUTPUT", False)
    assert attention._try_mate_decode(mixed_qkv, b, a, core_attn_out, metadata)

    assert "output" not in captured
    assert torch.equal(core_attn_out, torch.full_like(core_attn_out, 9))


def test_mate_decode_gather_scatter_reuses_output_and_updates_active_state(
    monkeypatch,
) -> None:
    attention = _fake_attention(separate_pool=False)
    num_tokens = 2
    mixed_qkv = torch.randn(num_tokens, 12)
    a = torch.ones(num_tokens, 2)
    b = torch.ones(num_tokens, 2)
    core_attn_out = torch.zeros(num_tokens, 2, 2)
    metadata = _metadata(num_tokens, [1, 3])
    original_state = attention.kv_cache[1].clone()

    _patch_decode_prerequisites(monkeypatch)

    def fake_decode(**kwargs):
        kwargs["output"].fill_(7)
        return kwargs["output"], torch.full_like(kwargs["state"], 5)

    monkeypatch.setattr(gdn, "gated_delta_rule_decode", fake_decode)
    monkeypatch.setattr(gdn, "_MATE_GDN_DECODE_HAS_OUTPUT", True)
    assert attention._try_mate_decode(mixed_qkv, b, a, core_attn_out, metadata)

    assert torch.equal(core_attn_out, torch.full_like(core_attn_out, 7))
    assert torch.equal(attention.kv_cache[1][[1, 3]], torch.full((2, 2, 2, 2), 5.0))
    assert torch.equal(attention.kv_cache[1][[0, 2]], original_state[[0, 2]])


def test_mate_decode_output_reuse_only_writes_decode_prefix(monkeypatch) -> None:
    attention = _fake_attention()
    num_tokens = 3
    num_decode_tokens = 2
    mixed_qkv = torch.randn(num_tokens, 12)
    a = torch.ones(num_tokens, 2)
    b = torch.ones(num_tokens, 2)
    core_attn_out = torch.full((num_tokens, 2, 2), -3.0)
    metadata = _metadata(num_decode_tokens, [0, 1])

    _patch_decode_prerequisites(monkeypatch)

    def fake_decode(**kwargs):
        kwargs["output"].fill_(7)
        return kwargs["output"], None

    monkeypatch.setattr(gdn, "gated_delta_rule_decode", fake_decode)
    monkeypatch.setattr(gdn, "_MATE_GDN_DECODE_HAS_OUTPUT", True)
    assert attention._try_mate_decode(mixed_qkv, b, a, core_attn_out, metadata)

    assert torch.equal(core_attn_out[:num_decode_tokens], torch.full((2, 2, 2), 7.0))
    assert torch.equal(core_attn_out[num_decode_tokens:], torch.full((1, 2, 2), -3.0))
