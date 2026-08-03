from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_musa")


requires_mp31 = pytest.mark.skipif(
    not hasattr(torch, "musa")
    or not torch.musa.is_available()
    or tuple(torch.musa.get_device_capability()) != (3, 1),
    reason="requires an mp31 MUSA device",
)


@requires_mp31
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5])
def test_requested_16x8_gemv_matches_reference(rows: int) -> None:
    from vllm_musa import _custom_ops as musa_ops

    vllm_musa = pytest.importorskip("vllm_musa")
    vllm_musa.register_custom_ops()
    torch.manual_seed(60083 + rows)

    hidden_size = 128
    output_size = 32
    a = torch.randn((rows, hidden_size), device="musa", dtype=torch.bfloat16)
    b = torch.randn(
        (1, output_size, hidden_size), device="musa", dtype=torch.bfloat16
    )
    c = torch.empty((rows, output_size), device="musa", dtype=torch.bfloat16)
    topk_weights = torch.ones((rows, 1), device="musa", dtype=torch.float32)
    topk_ids = torch.zeros((rows, 1), device="musa", dtype=torch.int32)

    musa_ops.musa_fused_gemv_moe(
        a,
        b,
        c,
        None,
        None,
        topk_weights,
        topk_ids,
        False,
        1,
        False,
        False,
        block_n=16,
        block_k=8,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a.float(), b[0].float().transpose(0, 1)).to(
        torch.bfloat16
    )
    torch.testing.assert_close(c.float(), expected.float(), rtol=2e-2, atol=2e-2)


@requires_mp31
def test_requested_gemv_block_is_graph_static() -> None:
    from vllm_musa import _custom_ops as musa_ops

    vllm_musa = pytest.importorskip("vllm_musa")
    vllm_musa.register_custom_ops()

    rows, hidden_size, output_size = 1, 128, 32
    a = torch.empty((rows, hidden_size), device="musa", dtype=torch.bfloat16)
    b = torch.randn(
        (1, output_size, hidden_size), device="musa", dtype=torch.bfloat16
    )
    c = torch.empty((rows, output_size), device="musa", dtype=torch.bfloat16)
    topk_weights = torch.ones((rows, 1), device="musa", dtype=torch.float32)
    topk_ids = torch.zeros((rows, 1), device="musa", dtype=torch.int32)

    warmup = torch.musa.Stream()
    warmup.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(warmup):
        for seed in (3, 5, 7):
            torch.manual_seed(seed)
            a.copy_(torch.randn_like(a))
            musa_ops.musa_fused_gemv_moe(
                a,
                b,
                c,
                None,
                None,
                topk_weights,
                topk_ids,
                False,
                1,
                False,
                False,
                block_n=16,
                block_k=8,
            )
    torch.musa.current_stream().wait_stream(warmup)

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        musa_ops.musa_fused_gemv_moe(
            a,
            b,
            c,
            None,
            None,
            topk_weights,
            topk_ids,
            False,
            1,
            False,
            False,
            block_n=16,
            block_k=8,
        )

    for seed in (11, 13, 17):
        torch.manual_seed(seed)
        next_a = torch.randn_like(a)
        expected = torch.matmul(next_a.float(), b[0].float().transpose(0, 1)).to(
            torch.bfloat16
        )
        a.copy_(next_a)
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(c.float(), expected.float(), rtol=2e-2, atol=2e-2)
