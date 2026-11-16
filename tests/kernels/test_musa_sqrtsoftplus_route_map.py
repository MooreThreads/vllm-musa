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


@pytest.mark.parametrize("tokens", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("renormalize,routed_scale", [(False, 1.0), (True, 1.5)])
def test_plain_topk_matches_route_map_with_changed_graph_inputs(
    tokens, dtype, padding, renormalize, routed_scale
):
    from vllm_musa.jit_kernel.csrc.sqrtsoftplus_topk import sqrtsoftplus_topk

    torch.manual_seed(17085)
    logits = torch.randn(tokens, 384, dtype=dtype, device="musa")
    bias = torch.linspace(-0.3, 0.3, 384, device="musa")
    hidden = torch.randn(tokens, 5120, dtype=torch.bfloat16, device="musa")
    mask = torch.zeros(tokens, dtype=torch.bool, device="musa")
    if padding:
        mask[0] = True

    def invoke():
        plain = sqrtsoftplus_topk(
            logits,
            bias,
            6,
            renormalize,
            routed_scale,
            is_padding=mask,
            with_route_map=False,
            x=hidden,
        )
        mapped = sqrtsoftplus_topk(
            logits,
            bias,
            6,
            renormalize,
            routed_scale,
            is_padding=mask,
            with_route_map=True,
            x=hidden,
        )
        return plain, mapped

    for _ in range(3):
        invoke()
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        plain, mapped = invoke()
    assert plain[2] is None and plain[3] is None
    assert mapped[2] is not None and mapped[3] is not None
    for _ in range(3):
        logits.copy_(torch.randn_like(logits))
        hidden.copy_(torch.randn_like(hidden))
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(plain[0], mapped[0], rtol=0, atol=0)
        torch.testing.assert_close(plain[1], mapped[1], rtol=0, atol=0)
        if padding:
            assert (plain[1][0] == -1).all()
            assert (plain[0][0] == 0).all()
