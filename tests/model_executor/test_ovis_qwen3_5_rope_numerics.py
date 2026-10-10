# SPDX-License-Identifier: Apache-2.0
"""Bound BF16 rounding on the packed Q/K shape used by Ovis vision."""

import math

import pytest

pytest.importorskip("torchada")

import torch  # noqa: E402

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.rotary_embedding.common import (  # noqa: E402
    ApplyRotaryEmb,
)
from vllm_musa.optimization_contract.rotary import (  # noqa: E402
    MusaVisionApplyRotaryEmb,
    MusaVisionRotaryPositions,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA device",
)


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _adapter() -> MusaVisionApplyRotaryEmb:
    return MusaVisionApplyRotaryEmb(
        is_neox_style=True,
        enable_fp32_compute=False,
        inplace=True,
        flatten=True,
        positions_cache=MusaVisionRotaryPositions(),
        required_bf16_neox_shape=(64, 32),
    )


def _inputs(
    leading: int, seq: int, cancellation: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(17)
    x = (4 * torch.randn((leading, seq, 12, 64), generator=generator)).bfloat16()
    angles = torch.rand((seq, 32), generator=generator)
    if cancellation:
        # Keep genuine sin/cos pairs and avoid division by a near-zero sine.
        angles = math.pi / 6 + angles * (math.pi / 6)
    else:
        angles = angles * (2 * math.pi)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    if cancellation:
        x[..., 32:] = (
            x[..., :32].float()
            * cos.float()[None, :, None, :]
            / sin.float()[None, :, None, :]
        ).bfloat16()
    return tuple(t.to("musa") for t in (x, cos, sin))


def _assert_bounded_rounding(x, cos, sin, candidate) -> None:
    # Use the same quantized coefficients as forward_static, then widen them.
    reference = ApplyRotaryEmb.forward_static(x.clone(), cos, sin)
    oracle = ApplyRotaryEmb.forward_static(x.float(), cos.float(), sin.float())
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos.float()[None, :, None, :], sin.float()[None, :, None, :]
    scale = torch.cat(
        (
            (x1 * c).abs() + (x2 * s).abs(),
            (x2 * c).abs() + (x1 * s).abs(),
        ),
        dim=-1,
    )
    # Both BF16 operation paths have at most two rounding stages per term.
    # gamma_2 bounds their forward error by |x*c| + |y*s|, including cancellation.
    # The FP32 addition in the oracle contributes its own unit roundoff.
    unit_roundoff = torch.finfo(torch.bfloat16).eps / 2
    gamma_2 = 2 * unit_roundoff / (1 - 2 * unit_roundoff)
    bound = (gamma_2 + torch.finfo(torch.float32).eps / 2) * scale
    assert candidate.shape == reference.shape == x.shape
    assert candidate.dtype == reference.dtype == torch.bfloat16
    assert torch.isfinite(oracle).all()
    for actual in (reference, candidate):
        assert torch.isfinite(actual).all()
        error = (actual.float() - oracle).abs()
        # No arbitrary epsilon at exact zero: a zero scale requires zero error.
        assert torch.all(error <= bound), (
            f"BF16 forward-error bound exceeded: "
            f"{torch.count_nonzero(error > bound).item()} elements"
        )


@pytest.mark.parametrize("leading,seq", [(2, 1), (2, 257), (2, 15604), (4, 257)])
@pytest.mark.parametrize("cancellation", [False, True])
@torch.inference_mode()
def test_ovis_packed_qk_matches_reference_with_bf16_roundoff(
    leading: int, seq: int, cancellation: bool
) -> None:
    x, cos, sin = _inputs(leading, seq, cancellation)
    original = x.clone()
    candidate = _adapter()(x.clone(), cos, sin)
    torch.musa.synchronize()
    assert torch.equal(x, original)
    _assert_bounded_rounding(x, cos, sin, candidate)


@pytest.mark.parametrize("quarter_turn", [False, True])
@torch.inference_mode()
def test_ovis_real_shape_cardinal_rotation_is_exact(quarter_turn: bool) -> None:
    x, cos, sin = _inputs(2, 15604)
    cos.fill_(0 if quarter_turn else 1)
    sin.fill_(1 if quarter_turn else 0)
    reference = ApplyRotaryEmb.forward_static(x.clone(), cos, sin)
    candidate = _adapter()(x.clone(), cos, sin)
    torch.musa.synchronize()
    assert torch.equal(candidate, reference)


@torch.inference_mode()
def test_ovis_graph_replay_survives_another_eager_shape() -> None:
    x, cos, sin = _inputs(2, 257)
    adapter = _adapter()
    query = x.clone()
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(3):
            query.copy_(x)
            adapter(query, cos, sin)
    torch.musa.current_stream().wait_stream(stream)
    positions = adapter.positions_cache.get(2, 257, x.device)
    assert adapter.positions_cache.get(2, 257, x.device) is positions
    query.copy_(x)
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        captured = adapter(query, cos, sin)
    other_x, other_cos, other_sin = _inputs(4, 33)
    adapter(other_x, other_cos, other_sin)
    query.copy_(x)
    graph.replay()
    torch.musa.synchronize()
    _assert_bounded_rounding(x, cos, sin, captured)
