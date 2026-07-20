"""Fused un-permute + weighted top-k reduce for the DeepGEMM grouped-MoE prefill path.

Gathers each source token's top-k expert-output rows from the grouped-GEMM
output (indexed through ``src2dst``), scales by the router weights, and reduces
into the final hidden state — one launch replacing an unfused unpermute+reduce.
"""

import triton
import triton.language as tl


@triton.jit
def post_reorder_triton_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = down_output_ptr.dtype.element_ty
    src_idx = tl.program_id(0).to(tl.int64)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk
    store_ptr = output_ptr + src_idx * hidden_size
    vec = tl.arange(0, BLOCK_SIZE)
    for start in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start + vec
        mask = offset < hidden_size
        acc = tl.zeros([BLOCK_SIZE], dtype=InDtype)
        for idx in range(topk):
            dst = tl.load(src2dst_ptr + idx)
            if dst >= 0:
                weight = tl.load(topk_weights_ptr + idx).to(InDtype)
                acc += (
                    tl.load(
                        down_output_ptr + dst.to(tl.int64) * hidden_size + offset,
                        mask=mask,
                    )
                    * weight
                )
        tl.store(store_ptr + offset, acc, mask=mask)
