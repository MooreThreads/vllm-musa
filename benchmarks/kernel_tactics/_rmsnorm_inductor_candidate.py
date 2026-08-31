"""Benchmark-local Triton RMSNorm candidate with no production registration."""

from __future__ import annotations

# Apply CUDA-to-MUSA compatibility before importing Torch symbols.
# isort: off
import torchada  # noqa: F401
import torch
from torch import Tensor
from torch.library import wrap_triton
# isort: on

from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    n_cols: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    input_row = input_ptr + row * n_cols
    output_row = output_ptr + row * n_cols

    values = tl.load(input_row + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / n_cols
    inverse_rms = tl.rsqrt(variance + epsilon)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normalized = values * inverse_rms * weight
    tl.store(output_row + offsets, normalized, mask=mask)


def rms_norm_triton(
    x: Tensor,
    weight: Tensor,
    epsilon: float,
    *,
    num_warps: int,
) -> Tensor:
    """Launch the exploratory candidate with a fresh output tensor."""
    hidden_size = x.shape[-1]
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(hidden_size)
    grid = (x.numel() // hidden_size,)
    wrap_triton(_rms_norm_kernel)[grid](
        output,
        x,
        weight,
        hidden_size,
        epsilon,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


__all__ = ["rms_norm_triton"]
