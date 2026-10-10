"""Numerical smoke for Paddle's FP32-compute, out-of-place vision RoPE."""

import math

import torchada  # noqa: F401
import torch

from vllm_musa.model_executor.layers.rotary_embedding.base import (
    MusaVisionApplyRotaryEmb,
)


def _reference_and_bound(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the established FP32 arithmetic plus output-rounding error bound."""
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos.float()[None, :, None, :], sin.float()[None, :, None, :]
    p1, p2, p3, p4 = x1 * cos, x2 * sin, x2 * cos, x1 * sin
    reference = torch.cat((p1 - p2, p3 + p4), dim=-1)
    magnitude = torch.cat((p1.abs() + p2.abs(), p3.abs() + p4.abs()), dim=-1)
    # Bound FP32 arithmetic by 8 unit roundoffs, then BF16 by half an output ULP.
    bound = 8 * (torch.finfo(torch.float32).eps / 2) * magnitude
    bound += torch.finfo(torch.float32).tiny
    if x.dtype != torch.float32:
        rounded = reference.to(x.dtype)
        up = torch.nextafter(rounded, torch.full_like(rounded, math.inf)).float()
        down = torch.nextafter(rounded, torch.full_like(rounded, -math.inf)).float()
        spacing = torch.maximum(
            (up - rounded.float()).abs(), (rounded.float() - down).abs()
        )
        bound += spacing / 2
    return reference, bound


def main() -> None:
    device = torch.device("cuda")
    shape = (2, 4888, 16, 72)
    torch.manual_seed(100)
    phase = torch.randn((4888, 36), device=device, dtype=torch.float32)
    cos, sin = phase.cos(), phase.sin()
    for dtype in (torch.float32, torch.bfloat16):
        base = torch.randn((2, 4888, 72, 16), device=device, dtype=dtype)
        x = base.permute(0, 1, 3, 2)
        before = x.clone()
        got = MusaVisionApplyRotaryEmb(enable_fp32_compute=True)(x, cos, sin)
        reference, bound = _reference_and_bound(
            x.detach().cpu(), cos.detach().cpu(), sin.detach().cpu()
        )
        error = (got.detach().cpu().float() - reference).abs()
        assert torch.isfinite(error).all()
        assert (error <= bound).all(), (
            f"{dtype}: {int((error > bound).sum())} elements exceed the "
            f"FP32 arithmetic/output-rounding bound; max error {error.max().item()}"
        )
        torch.testing.assert_close(x, before, rtol=0, atol=0)
        assert got.shape == shape
        assert got.dtype == dtype
        assert got.data_ptr() != x.data_ptr()


if __name__ == "__main__":
    main()
