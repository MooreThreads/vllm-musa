# SPDX-License-Identifier: Apache-2.0
"""Fused Qwen GDN prefill state gather and initial-state masking."""

from __future__ import annotations

import os

# isort: off
import torchada  # noqa: F401
import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
# isort: on

logger = init_logger(__name__)

_FUSED_GDN_STATE_GATHER_ENV = "VLLM_MUSA_FUSED_GDN_STATE_GATHER"
_QWEN_GDN_STATE_SHAPE = (32, 128, 128)
_BLOCK_SIZE = 256
_ITEMS_PER_PROGRAM = 16


@triton.jit
def _gather_mask_gdn_state_kernel(
    state_ptr,
    state_indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    state_size: tl.constexpr,
    block_size: tl.constexpr,
    items_per_program: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    state_idx = tl.load(state_indices_ptr + seq_idx)
    has_initial_state = tl.load(has_initial_state_ptr + seq_idx)
    tile_base = tile_idx * block_size * items_per_program
    offsets = tl.arange(0, block_size)
    for item_idx in range(items_per_program):
        state_offset = tile_base + item_idx * block_size + offsets
        valid = state_offset < state_size
        value = tl.load(
            state_ptr + state_idx * state_size + state_offset,
            mask=valid & has_initial_state,
            other=0.0,
        )
        tl.store(output_ptr + seq_idx * state_size + state_offset, value, mask=valid)


def _env_enabled() -> bool:
    return os.getenv(_FUSED_GDN_STATE_GATHER_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def can_use_fused_gdn_state_gather_mask(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> bool:
    """Return whether the narrow Qwen3.5/3.6 GDN prefill contract is met."""
    return (
        _env_enabled()
        and state.dtype == torch.float32
        and state.is_contiguous()
        and state.ndim == 4
        and tuple(state.shape[1:]) == _QWEN_GDN_STATE_SHAPE
        and state_indices.dtype == torch.int32
        and state_indices.ndim == 1
        and state_indices.is_contiguous()
        and 0 < state_indices.numel() <= 64
        and has_initial_state.dtype == torch.bool
        and has_initial_state.shape == state_indices.shape
        and has_initial_state.is_contiguous()
    )


def fused_gdn_state_gather_mask(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Gather selected fp32 states and zero rows without initial state."""
    if not can_use_fused_gdn_state_gather_mask(state, state_indices, has_initial_state):
        raise ValueError("unsupported fused Qwen GDN state-gather contract")

    num_sequences = state_indices.numel()
    state_size = state[0].numel()
    output = torch.empty(
        (num_sequences, *state.shape[1:]), dtype=state.dtype, device=state.device
    )
    grid = (
        num_sequences,
        triton.cdiv(state_size, _BLOCK_SIZE * _ITEMS_PER_PROGRAM),
    )
    _gather_mask_gdn_state_kernel[grid](
        state,
        state_indices,
        has_initial_state,
        output,
        state_size=state_size,
        block_size=_BLOCK_SIZE,
        items_per_program=_ITEMS_PER_PROGRAM,
        num_warps=4,
        num_stages=1,
    )
    logger.info_once("Using fused MUSA Qwen GDN prefill state gather and mask.")
    return output
