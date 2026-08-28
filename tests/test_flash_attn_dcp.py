# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

try:
    from vllm_musa.v1.attention.backends import flash_attn as flash_attn_module
    from vllm_musa.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
    )
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(f"requires full vllm-musa runtime: {exc}", allow_module_level=True)


class _FakeDCPGroup:
    world_size = 2
    rank_in_group = 0

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.cat((tensor, tensor + 1), dim=dim)


def _make_impl(**kwargs) -> FlashAttentionImpl:
    defaults = dict(
        num_heads=2,
        head_size=4,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    defaults.update(kwargs)
    impl = FlashAttentionImpl(**defaults)
    impl.dcp_world_size = 2
    return impl


def _make_metadata(**kwargs) -> FlashAttentionMetadata:
    defaults = dict(
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([4, 5], dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        slot_mapping=torch.arange(3, dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        decode_query_start_loc=None,
        decode_seq_lens=None,
        decode_block_table=None,
        num_prefills=2,
        num_prefill_tokens=3,
        prefill_query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        prefill_max_seq_len=2,
        cu_seqlens_k=None,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=4,
        dcp_context_kv_lens=torch.tensor([3, 3], dtype=torch.int32),
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=0,
        causal=True,
    )
    defaults.update(kwargs)
    return FlashAttentionMetadata(**defaults)


def test_forward_dispatches_to_dcp_path(monkeypatch):
    impl = _make_impl()
    metadata = _make_metadata(num_actual_tokens=1)
    called = {}

    def fake_forward_with_dcp(*args, **kwargs):
        called["kwargs"] = kwargs
        args[5].fill_(7)

    monkeypatch.setattr(impl, "_forward_with_dcp", fake_forward_with_dcp)

    layer = SimpleNamespace(
        _q_scale=torch.ones((1, 1)),
        _k_scale=torch.ones((1, 1)),
        _v_scale=torch.ones((1, 1)),
    )
    output = torch.zeros((1, 2, 4))
    result = impl.forward(
        layer,
        torch.zeros((1, 2, 4)),
        torch.zeros((1, 1, 4)),
        torch.zeros((1, 1, 4)),
        torch.zeros((2, 1, 16, 1, 4)),
        metadata,
        output,
    )

    assert result is output
    assert torch.all(output == 7)
    assert set(called["kwargs"]) == {"q_descale", "k_descale", "v_descale"}


def test_forward_with_dcp_runs_context_suffix_and_merge(monkeypatch):
    calls = []
    merged_lses = {}

    def fake_merge_attn_states(
        output,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
    ):
        assert prefix_output.shape == suffix_output.shape == output.shape
        assert prefix_lse.shape == suffix_lse.shape == (2, 3)
        merged_lses["prefix"] = prefix_lse.clone()
        merged_lses["suffix"] = suffix_lse.clone()
        output.copy_(prefix_output + suffix_output)

    def fake_flash_attn_varlen_func(**kwargs):
        calls.append(kwargs)
        q = kwargs["q"]
        num_tokens, num_heads, head_size = q.shape
        lse = torch.arange(num_heads * num_tokens, dtype=q.dtype, device=q.device).view(
            num_heads, num_tokens
        )
        if num_heads == 4:
            out = torch.ones(
                (num_tokens, num_heads, head_size), dtype=q.dtype, device=q.device
            )
            return out.reshape(num_tokens, num_heads * head_size), lse
        out = torch.full(
            (num_tokens, 1, num_heads, head_size),
            2,
            dtype=q.dtype,
            device=q.device,
        )
        return out, lse

    def fake_dcp_combine(context_out, context_lse, group, return_lse=False):
        assert return_lse is True
        assert group.world_size == 2
        assert context_out.shape == (3, 4, 4)
        assert context_lse.shape == (3, 4)
        expected_lse = torch.arange(12, dtype=context_lse.dtype).view(4, 3)
        assert torch.equal(context_lse, expected_lse.transpose(0, 1))
        return context_out[:, :2, :].contiguous(), context_lse[:, :2].contiguous()

    monkeypatch.setattr(flash_attn_module, "get_dcp_group", lambda: _FakeDCPGroup())
    monkeypatch.setattr(flash_attn_module, "merge_attn_states", fake_merge_attn_states)
    monkeypatch.setattr(
        flash_attn_module,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=False,
    )

    impl = _make_impl()
    impl.dcp_combine = fake_dcp_combine
    output = torch.empty((3, 2, 4))

    impl._forward_with_dcp(
        torch.zeros((3, 2, 4)),
        torch.zeros((3, 1, 4)),
        torch.zeros((3, 1, 4)),
        torch.zeros((2, 16, 1, 4)),
        torch.zeros((2, 16, 1, 4)),
        output,
        _make_metadata(),
    )

    assert len(calls) == 2
    assert calls[0]["causal"] is False
    assert calls[0]["seqused_k"].shape == (2,)
    assert calls[0]["max_seqlen_k"] == 4
    assert calls[0]["block_table"].shape == (2, 2)
    assert calls[1]["causal"] is True
    assert calls[1]["cu_seqlens_k"].tolist() == [0, 1, 3]
    for call in calls:
        assert "out" not in call
        assert "fa_version" not in call
        assert "alibi_slopes" not in call
        assert "s_aux" not in call

    assert torch.equal(merged_lses["prefix"], torch.arange(12.0).view(4, 3)[:2])
    assert torch.equal(merged_lses["suffix"], torch.arange(6.0).view(2, 3))
    assert torch.all(output == 3)


def test_lse_layout_helpers_accept_mate_and_tokens_heads_layouts():
    lse_heads_tokens = torch.arange(12.0).view(4, 3)
    lse_tokens_heads = lse_heads_tokens.transpose(0, 1).contiguous()

    assert torch.equal(
        FlashAttentionImpl._lse_to_tokens_heads(lse_heads_tokens, 3, 4),
        lse_tokens_heads,
    )
    assert torch.equal(
        FlashAttentionImpl._lse_to_heads_tokens(lse_tokens_heads, 3, 4),
        lse_heads_tokens,
    )
    assert torch.equal(
        FlashAttentionImpl._lse_to_tokens_heads(lse_heads_tokens.unsqueeze(-1), 3, 4),
        lse_tokens_heads,
    )


@pytest.mark.parametrize("has_device_group", [True, False])
def test_musa_dcp_reduce_scatter_uses_torch_dist(monkeypatch, has_device_group):
    class Group:
        world_size = 2
        rank_in_group = 1

    group = Group()
    if has_device_group:
        group.device_group = object()

    captured = {}

    def fake_reduce_scatter(output, chunks, group=None):
        captured["group"] = group
        captured["chunks"] = [chunk.clone() for chunk in chunks]
        output.copy_(chunks[1])

    monkeypatch.setattr(torch.distributed, "reduce_scatter", fake_reduce_scatter)

    tensor = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    result = flash_attn_module._torch_reduce_scatter_dim(tensor, group, dim=1)

    assert captured["group"] is (group.device_group if has_device_group else group)
    assert torch.equal(captured["chunks"][0], tensor[:, :2, :])
    assert torch.equal(captured["chunks"][1], tensor[:, 2:, :])
    assert torch.equal(result, tensor[:, 2:, :])


def test_a2a_backend_selects_a2a_lse_reduce(monkeypatch):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            dcp_comm_backend="a2a",
        )
    )
    monkeypatch.setattr(
        flash_attn_module, "get_current_vllm_config_or_none", lambda: config
    )

    impl = _make_impl()

    assert impl.dcp_combine is flash_attn_module.dcp_a2a_lse_reduce


@pytest.mark.parametrize(
    ("attr", "value", "message"),
    [
        ("sinks", torch.ones(2), "attention sinks"),
        ("alibi_slopes", torch.ones(2), "ALiBi"),
    ],
)
def test_forward_with_dcp_rejects_unsupported_decisions(attr, value, message):
    impl = _make_impl()
    setattr(impl, attr, value)

    with pytest.raises(NotImplementedError, match=message):
        impl._forward_with_dcp(
            torch.zeros((3, 2, 4)),
            torch.zeros((3, 1, 4)),
            torch.zeros((3, 1, 4)),
            torch.zeros((2, 16, 1, 4)),
            torch.zeros((2, 16, 1, 4)),
            torch.empty((3, 2, 4)),
            _make_metadata(),
        )
