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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize(
    "case", ["random", "ties", "negative_bias", "sentinel_bias", "extreme"]
)
def test_single_row_topk_matches_two_row_fallback(dtype, renormalize, case):
    from vllm_musa.jit_kernel.csrc.sqrtsoftplus_topk import sqrtsoftplus_topk

    torch.manual_seed(17185)
    logits = torch.randn(1, 384, dtype=dtype, device="musa")
    bias = torch.randn(384, device="musa") * 0.3
    if case == "ties":
        logits.zero_()
        bias.zero_()
    elif case == "negative_bias":
        bias.fill_(-20000)
    elif case == "sentinel_bias":
        bias.fill_(-3.2e38)
    elif case == "extreme":
        logits.copy_(torch.linspace(-120, 120, 384, device="musa").to(dtype))
    padding = torch.zeros(1, dtype=torch.bool, device="musa")
    duplicated = torch.empty(2, 384, dtype=dtype, device="musa")
    duplicated_padding = torch.zeros(2, dtype=torch.bool, device="musa")
    graph = torch.musa.MUSAGraph()
    for _ in range(3):
        sqrtsoftplus_topk(logits, bias, 6, renormalize, 1.5, is_padding=padding)
    torch.musa.synchronize()
    with torch.musa.graph(graph):
        actual = sqrtsoftplus_topk(
            logits, bias, 6, renormalize, 1.5, is_padding=padding
        )
    for masked in (False, True, False):
        padding.fill_(masked)
        duplicated.copy_(logits.expand(2, -1))
        duplicated_padding.fill_(masked)
        graph.replay()
        expected = sqrtsoftplus_topk(
            duplicated, bias, 6, renormalize, 1.5, is_padding=duplicated_padding
        )
        torch.musa.synchronize()
        torch.testing.assert_close(actual[0], expected[0][:1], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1][:1], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("fold_shared", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_single_row_image_bias_matches_two_row_fallback(
    dtype, input_dtype, fold_shared, renormalize
):
    from vllm_musa.jit_kernel.csrc.sqrtsoftplus_topk import sqrtsoftplus_topk

    torch.manual_seed(17285)
    logits = torch.randn(1, 384, dtype=dtype, device="musa")
    bias = torch.randn(384, device="musa") * 0.3
    image_bias = torch.randn(384, device="musa") * 0.7
    input_ids = torch.full((1,), 17, dtype=input_dtype, device="musa")
    padding = torch.zeros(1, dtype=torch.bool, device="musa")
    options = dict(
        shared_id=384 if fold_shared else -1,
        shared_weight=0.75,
        bias_vl=image_bias,
        image_sentinel_lo=1000,
    )
    graph = torch.musa.MUSAGraph()
    for _ in range(3):
        sqrtsoftplus_topk(
            logits,
            bias,
            6,
            renormalize,
            1.5,
            is_padding=padding,
            input_ids=input_ids,
            **options,
        )
    torch.musa.synchronize()
    with torch.musa.graph(graph):
        actual = sqrtsoftplus_topk(
            logits,
            bias,
            6,
            renormalize,
            1.5,
            is_padding=padding,
            input_ids=input_ids,
            **options,
        )
    for masked, token in (
        (False, 17),
        (False, 1001),
        (False, 1007),
        (True, 1001),
        (False, 17),
    ):
        padding.fill_(masked)
        input_ids.fill_(token)
        graph.replay()
        expected = sqrtsoftplus_topk(
            logits.expand(2, -1).contiguous(),
            bias,
            6,
            renormalize,
            1.5,
            is_padding=padding.expand(2).contiguous(),
            input_ids=input_ids.expand(2).contiguous(),
            **options,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(actual[0], expected[0][:1], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1][:1], rtol=0, atol=0)
