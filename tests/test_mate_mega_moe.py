# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MATE Mega MoE experts: deployment limits and kernel-path selection."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
mega = pytest.importorskip("vllm_musa.model_executor.layers.fused_moe.mate_mega_moe")


def _reasons(**overrides):
    shape = {
        "hidden_size": 2048,
        "intermediate_size": 768,
        "num_experts": 128,
        "top_k": 8,
        "ep_size": 8,
        "max_num_tokens": 4096,
    }
    shape.update(overrides)
    return mega.mate_mega_moe_shape_reasons(**shape)


def test_qwen3_moe_shapes_are_supported():
    assert _reasons() == ()
    assert (
        _reasons(hidden_size=4096, intermediate_size=1024, num_experts=512, top_k=10)
        == ()
    )


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"hidden_size": 2112}, "hidden size 2112"),
        ({"intermediate_size": 704}, "intermediate size 704 not divisible"),
        ({"intermediate_size": 8320}, "> 8192"),
        ({"ep_size": 16}, "intranode"),
        ({"num_experts": 100}, "not divisible by 8 ranks"),
        ({"top_k": 33}, "top-k 33"),
        ({"max_num_tokens": 70000}, "tokens per rank"),
    ],
)
def test_shape_limits_name_the_violation(overrides, fragment):
    reasons = _reasons(**overrides)
    assert len(reasons) == 1
    assert fragment in reasons[0]


def test_round_up_tokens_matches_the_buffer_alignment():
    assert mega.round_up_tokens(0) == 256
    assert mega.round_up_tokens(1) == 256
    assert mega.round_up_tokens(256) == 256
    assert mega.round_up_tokens(4097) == 4352


def _fake_dp(monkeypatch, counts):
    if counts is None:
        monkeypatch.setattr(mega, "is_forward_context_available", lambda: False)
        return
    ctx = types.SimpleNamespace(
        dp_metadata=types.SimpleNamespace(
            num_tokens_across_dp_cpu=torch.tensor(counts, dtype=torch.int32)
        )
    )
    monkeypatch.setattr(mega, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mega, "get_forward_context", lambda: ctx)


@pytest.mark.parametrize(
    ("local", "counts", "decode"),
    [
        (4, None, True),
        (128, None, True),
        (129, None, False),
        (4, [4, 1, 128, 2], True),
        (4, [4, 1, 129, 2], False),
    ],
)
def test_kernel_path_follows_the_largest_rank(monkeypatch, local, counts, decode):
    _fake_dp(monkeypatch, counts)
    assert mega.use_decode_kernels(local) is decode


class _FakeMegaMoE(types.ModuleType):
    def __init__(self, split: bool = True):
        super().__init__("mate.mega_moe")
        self.calls = []
        if split:
            self._run_decode_mega_moe = self._record("decode")
            self._run_prefill_mega_moe = self._record("prefill")
        self.fp8_fp8_mega_moe = self._record("fused")

    def _record(self, name):
        def run(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return run


def _experts(monkeypatch, fake, capacity=256):
    package = types.ModuleType("mate")
    package.mega_moe = fake
    monkeypatch.setitem(sys.modules, "mate", package)
    monkeypatch.setitem(sys.modules, "mate.mega_moe", fake)
    experts = object.__new__(mega.MateMegaMoEExperts)
    experts._l1_weights = ("w1", "s1")
    experts._l2_weights = ("w2", "s2")
    experts._buffer = types.SimpleNamespace(num_max_tokens_per_rank=capacity)
    experts._activation_clamp = None
    return experts


def test_split_entry_points_take_the_dp_consistent_path(monkeypatch):
    fake = _FakeMegaMoE()
    experts = _experts(monkeypatch, fake)
    y = torch.empty(3, 8)

    _fake_dp(monkeypatch, [3, 200])
    experts._run(y, 3)
    _fake_dp(monkeypatch, [3, 1])
    experts._run(y, 3)

    assert [call[0] for call in fake.calls] == ["prefill", "decode"]
    _, args, _ = fake.calls[1]
    assert args[0] is y
    assert args[1:3] == (("w1", "s1"), ("w2", "s2"))
    assert args[4:] == (3, "swiglu", None, None, None, True)


def test_without_split_entry_points_ranks_must_agree(monkeypatch):
    fake = _FakeMegaMoE(split=False)
    experts = _experts(monkeypatch, fake)

    _fake_dp(monkeypatch, [3, 3])
    experts._run(torch.empty(3, 8), 3)
    assert [call[0] for call in fake.calls] == ["fused"]

    _fake_dp(monkeypatch, [3, 200])
    with pytest.raises(RuntimeError, match="different kernel paths"):
        experts._run(torch.empty(3, 8), 3)


def _apply(experts, num_tokens, apply_router_weight_on_input=False):
    hidden = torch.zeros(num_tokens, 8)
    experts.apply(
        output=torch.empty_like(hidden),
        hidden_states=hidden,
        w1=None,
        w2=None,
        topk_weights=torch.ones(num_tokens, 1),
        topk_ids=torch.zeros(num_tokens, 1, dtype=torch.int64),
        activation=None,
        global_num_experts=8,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )


def test_apply_rejects_more_tokens_than_the_buffer_holds(monkeypatch):
    experts = _experts(monkeypatch, _FakeMegaMoE(), capacity=256)
    with pytest.raises(ValueError, match="holds 256"):
        _apply(experts, 257)


def test_apply_rejects_router_weights_on_input(monkeypatch):
    experts = _experts(monkeypatch, _FakeMegaMoE())
    with pytest.raises(NotImplementedError):
        _apply(experts, 2, apply_router_weight_on_input=True)


def test_supported_schemes_and_topology():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
    )

    experts = mega.MateMegaMoEExperts
    assert experts._supports_quant_scheme(kFp8Static128BlockSym, kFp8Dynamic128Sym)
    assert experts._supports_quant_scheme(kFp8Static128BlockSym, None)
    assert not experts._supports_quant_scheme(None, None)
    assert {a for a in MoEActivation if experts._supports_activation(a)} == {
        MoEActivation.SILU
    }
    ep = types.SimpleNamespace(use_ep=True, use_batched_activation_format=False)
    tp = types.SimpleNamespace(use_ep=False, use_batched_activation_format=False)
    assert experts._supports_parallel_config(ep)
    assert not experts._supports_parallel_config(tp)


def test_unavailable_without_musa(monkeypatch):
    monkeypatch.setattr(torch.version, "musa", None, raising=False)
    assert not mega.mate_mega_moe_available()
