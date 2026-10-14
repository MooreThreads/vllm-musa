# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MUSA project
"""Small fused metadata kernels for Mamba/GDN serving paths."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _mamba_align_block_table_kernel(
    block_table_ptr,
    seq_lens_ptr,
    output_ptr,
    block_table_stride_0,
    block_table_stride_1,
    output_stride_0,
    output_stride_1,
    mamba_block_size,
    num_columns: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_columns
    seq_len = tl.load(seq_lens_ptr + req_idx)
    block_col = tl.maximum((seq_len - 1) // mamba_block_size, 0) + offsets
    block_ids = tl.load(
        block_table_ptr
        + req_idx * block_table_stride_0
        + block_col * block_table_stride_1,
        mask=mask,
    )
    tl.store(
        output_ptr + req_idx * output_stride_0 + offsets * output_stride_1,
        block_ids,
        mask=mask,
    )


_COMPILED_MAMBA_ALIGN_KERNELS: dict[tuple[torch.device, int, int], object] = {}


def build_mamba_align_block_table(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    mamba_block_size: int,
    num_speculative_blocks: int,
) -> torch.Tensor:
    """Gather align-mode Mamba state block IDs in one device launch."""
    num_reqs = seq_lens.shape[0]
    num_columns = 1 + num_speculative_blocks
    assert block_table.device.type == "musa"
    assert seq_lens.device == block_table.device
    assert output.device == block_table.device
    assert block_table.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert output.dtype == torch.int32
    assert output.shape[0] >= num_reqs
    assert output.shape[1] >= num_columns

    if num_reqs == 0:
        return output[:0, :num_columns]

    block = triton.next_power_of_2(num_columns)
    runtime_args = (
        block_table,
        seq_lens,
        output,
        block_table.stride(0),
        block_table.stride(1),
        output.stride(0),
        output.stride(1),
        mamba_block_size,
    )
    cache_key = (block_table.device, num_columns, block)
    compiled_kernel = _COMPILED_MAMBA_ALIGN_KERNELS.get(cache_key)
    if compiled_kernel is None:
        compiled_kernel = _mamba_align_block_table_kernel.warmup(
            *runtime_args,
            num_columns=num_columns,
            BLOCK=block,
            num_warps=1,
            grid=(num_reqs,),
        )
        _COMPILED_MAMBA_ALIGN_KERNELS[cache_key] = compiled_kernel
    compiled_kernel[(num_reqs, 1, 1)](*runtime_args)
    return output[:num_reqs, :num_columns]
