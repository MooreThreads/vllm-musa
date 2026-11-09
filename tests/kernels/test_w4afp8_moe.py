from __future__ import annotations

import pytest

try:
    __import__("torchada")
except ModuleNotFoundError:
    pass

import torch

pytestmark = pytest.mark.skipif(
    getattr(torch.version, "musa", None) is None,
    reason="requires leased MUSA hardware",
)


def _pack_int4(weight):
    return (
        (weight[..., ::2].to(torch.int16) & 15)
        | ((weight[..., 1::2].to(torch.int16) & 15) << 4)
    ).to(torch.int8)


def _dynamic_fp8(values):
    scale = values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4) / 448.0
    quantized = (values.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.float() * scale


def _reference(
    hidden, gateup, down, gateup_scale, down_scale, weights, ids, expert_map
):
    output = torch.zeros_like(hidden, dtype=torch.float32)
    gateup = gateup.float() * gateup_scale.float().repeat_interleave(128, dim=-1)
    down = down.float() * down_scale.float().repeat_interleave(128, dim=-1)
    quantized_hidden = _dynamic_fp8(hidden)
    for token in range(hidden.shape[0]):
        for selection in range(ids.shape[1]):
            expert = int(ids[token, selection])
            if expert_map is not None:
                expert = int(expert_map[expert])
            if expert < 0:
                continue
            projections = (quantized_hidden[token] @ gateup[expert].T).to(
                torch.bfloat16
            )
            gate, up = projections.float().chunk(2)
            activated = (
                torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
            ).to(torch.bfloat16)
            expert_output = (_dynamic_fp8(activated) @ down[expert].T).to(
                torch.bfloat16
            )
            output[token] += expert_output.float() * weights[token, selection]
    return output.to(torch.bfloat16)


def _make_case(tokens, routing):
    generator = torch.Generator().manual_seed(128 + tokens)
    experts, hidden, intermediate = 4, 256, 256
    states = (torch.randn(tokens, hidden, generator=generator) * 0.3).to(torch.bfloat16)
    gateup = torch.randint(
        -8,
        8,
        (experts, 2 * intermediate, hidden),
        generator=generator,
        dtype=torch.int8,
    )
    down = torch.randint(
        -8, 8, (experts, hidden, intermediate), generator=generator, dtype=torch.int8
    )
    gateup_scale = (
        torch.rand(experts, 2 * intermediate, hidden // 128, generator=generator) * 0.01
        + 0.002
    ).to(torch.bfloat16)
    down_scale = (
        torch.rand(experts, hidden, intermediate // 128, generator=generator) * 0.01
        + 0.002
    ).to(torch.bfloat16)
    mapped = routing in ("mapped", "all_remote")
    total_experts = experts * 2 if mapped else experts
    ids = torch.stack(
        [torch.randperm(total_experts, generator=generator)[:2] for _ in range(tokens)]
    ).to(torch.int32)
    if routing == "all_remote":
        ids.remainder_(experts)
    elif routing == "hot_expert":
        ids.zero_()
    weights = torch.softmax(torch.randn(tokens, 2, generator=generator), dim=-1)
    expert_map = (
        torch.cat((torch.full((experts,), -1), torch.arange(experts))).to(torch.int32)
        if mapped
        else None
    )
    expected = _reference(
        states, gateup, down, gateup_scale, down_scale, weights, ids, expert_map
    )
    arguments = [
        states,
        _pack_int4(gateup),
        _pack_int4(down),
        gateup_scale,
        down_scale,
        weights,
        ids,
    ]
    arguments = [value.to("musa") for value in arguments]
    device_map = expert_map.to("musa") if expert_map is not None else None
    return arguments, device_map, expected


@pytest.mark.parametrize("tokens", [1, 8, 16, 17, 65, 257])
@pytest.mark.parametrize("routing", ["local", "mapped", "all_remote", "hot_expert"])
def test_mate_s4fp8_moe_matches_dynamic_quantized_reference(tokens, routing):
    from vllm_musa.kernels.w4afp8_moe import w4afp8_moe

    arguments, device_map, expected = _make_case(tokens, routing)
    actual = w4afp8_moe(*arguments, device_map, swiglu_limit=10.0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.04, atol=0.008)


def test_small_masked_capacity_handles_all_six_routes_in_one_expert():
    from vllm_musa.kernels.w4afp8_moe import w4afp8_moe

    arguments, device_map, expected = _make_case(1, "hot_expert")
    arguments[-1] = arguments[-1].repeat(1, 3)
    arguments[-2] = arguments[-2].repeat(1, 3) / 3

    actual = w4afp8_moe(*arguments, device_map, swiglu_limit=10.0)

    torch.testing.assert_close(actual.cpu(), expected, rtol=0.04, atol=0.008)


@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("routing", ["mapped", "all_remote", "hot_expert"])
def test_masked_moe_ignores_nan_poisoned_padding(tokens, routing, monkeypatch):
    from vllm_musa.kernels.w4afp8_moe import w4afp8_moe

    arguments, device_map, expected = _make_case(tokens, routing)
    allocate = torch.empty

    def poisoned_empty(*args, **kwargs):
        result = allocate(*args, **kwargs)
        if result.dtype == torch.float8_e4m3fn:
            result.view(torch.uint8).fill_(0x7F)
        elif result.is_floating_point():
            result.fill_(float("nan"))
        return result

    monkeypatch.setattr(torch, "empty", poisoned_empty)
    actual = w4afp8_moe(*arguments, device_map, swiglu_limit=10.0)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.04, atol=0.008)


@pytest.mark.parametrize("tokens", [2, 17, 65])
def test_mate_s4fp8_moe_accepts_noncontiguous_hidden_states(tokens):
    from vllm_musa.kernels.w4afp8_moe import w4afp8_moe

    arguments, device_map, expected = _make_case(tokens, "mapped")
    arguments[0] = arguments[0].T.contiguous().T
    assert not arguments[0].is_contiguous()

    actual = w4afp8_moe(*arguments, device_map, swiglu_limit=10.0)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.04, atol=0.008)


@pytest.mark.parametrize("tokens", [1, 8, 16, 17, 65])
def test_mate_s4fp8_graph_replay_resets_routing_metadata(tokens):
    from vllm_musa.kernels.w4afp8_moe import w4afp8_moe

    arguments, device_map, expected = _make_case(tokens, "mapped")
    for _ in range(3):
        w4afp8_moe(*arguments, device_map, swiglu_limit=10.0)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        actual = w4afp8_moe(*arguments, device_map, swiglu_limit=10.0)
    graph.replay()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.04, atol=0.008)
    original_ids = arguments[-1].clone()
    experts = arguments[1].shape[0]
    for hot_expert in (experts, 2 * experts - 1):
        arguments[-1].fill_(hot_expert)
        eager = w4afp8_moe(*arguments, device_map, swiglu_limit=10.0)
        graph.replay()
        torch.testing.assert_close(actual.cpu(), eager.cpu(), rtol=0.04, atol=0.008)
        arguments[-1].zero_()
        graph.replay()
        torch.testing.assert_close(
            actual.cpu(), torch.zeros_like(expected), rtol=0, atol=0
        )
        arguments[-1].copy_(original_ids)
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(actual.cpu(), expected, rtol=0.04, atol=0.008)
