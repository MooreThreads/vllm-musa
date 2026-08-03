from __future__ import annotations

import pytest
import torch

requires_musa = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA device",
)


def _reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_fp32 = x.float() + residual.float()
    residual_out = residual_fp32.to(x.dtype)
    normalized = residual_fp32 * torch.rsqrt(
        residual_fp32.pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * weight.float()).to(x.dtype), residual_out


@requires_musa
@pytest.mark.parametrize("rows", [1, 4, 16, 64, 4096, 4100])
def test_explicit_block256_matches_fp32_sum_reference(rows: int) -> None:
    from vllm_musa import _custom_ops as musa_ops

    torch.manual_seed(60083 + rows)
    eps = 1e-6
    x = torch.randn((rows, 4096), device="musa", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn((4096,), device="musa", dtype=torch.bfloat16)
    expected, expected_residual = _reference(x, residual, weight, eps)

    musa_ops.musa_fused_add_rms_norm(
        x,
        residual,
        weight,
        eps,
        block_x=256,
    )
    torch.musa.synchronize()

    torch.testing.assert_close(x.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        residual.float(),
        expected_residual.float(),
        rtol=0.0,
        atol=0.0,
    )


@requires_musa
def test_explicit_block256_replays_with_new_inputs() -> None:
    eps = 1e-6
    static_x = torch.empty((1, 4096), device="musa", dtype=torch.bfloat16)
    static_residual = torch.empty_like(static_x)
    weight = torch.randn((4096,), device="musa", dtype=torch.bfloat16)

    warmup_stream = torch.musa.Stream()
    warmup_stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(warmup_stream):
        for seed in (3, 5, 7):
            torch.manual_seed(seed)
            static_x.copy_(torch.randn_like(static_x))
            static_residual.copy_(torch.randn_like(static_residual))
            torch.ops._C_musa_ops.musa_fused_add_rms_norm(
                static_x,
                static_residual,
                weight,
                eps,
                256,
            )
    torch.musa.current_stream().wait_stream(warmup_stream)

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        torch.ops._C_musa_ops.musa_fused_add_rms_norm(
            static_x,
            static_residual,
            weight,
            eps,
            256,
        )

    for seed in (11, 13, 17):
        torch.manual_seed(seed)
        next_x = torch.randn_like(static_x)
        next_residual = torch.randn_like(static_residual)
        expected, expected_residual = _reference(next_x, next_residual, weight, eps)
        static_x.copy_(next_x)
        static_residual.copy_(next_residual)

        graph.replay()
        torch.musa.synchronize()

        torch.testing.assert_close(
            static_x.float(), expected.float(), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            static_residual.float(),
            expected_residual.float(),
            rtol=0.0,
            atol=0.0,
        )
