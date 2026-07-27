"""Append a folded shared expert's routing column in one kernel.

The routed topk output is (tokens, top_k); a folded shared expert rides the grouped GEMM as one
extra always-selected slot, so the routing needs one more column carrying
`sigmoid(shared_gate(x))` and the shared expert's id. Done with tensor ops that is a
sigmoid + zeros_like + add + 2 cats per MoE layer; this writes all of it in a single pass.

The shared weight stays out of any renormalization (the routed weights are already normalized
among themselves when the caller runs this), matching the reference implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _extend_topk_with_shared_kernel(
    topk_weights_ptr,
    topk_ids_ptr,
    shared_logits_ptr,
    out_weights_ptr,
    out_ids_ptr,
    shared_expert_id,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < top_k

    w = tl.load(topk_weights_ptr + token * top_k + offs, mask=mask, other=0.0)
    i = tl.load(topk_ids_ptr + token * top_k + offs, mask=mask, other=0)

    out_w = out_weights_ptr + token * (top_k + 1)
    out_i = out_ids_ptr + token * (top_k + 1)
    tl.store(out_w + offs, w, mask=mask)
    tl.store(out_i + offs, i, mask=mask)

    logit = tl.load(shared_logits_ptr + token).to(tl.float32)
    shared_w = (1.0 / (1.0 + tl.exp(-logit))).to(
        shared_logits_ptr.dtype.element_ty
    )
    tl.store(out_w + top_k, shared_w.to(out_w.dtype.element_ty))
    tl.store(out_i + top_k, shared_expert_id)


def extend_topk_with_shared(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_logits: torch.Tensor,
    shared_expert_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, top_k = topk_weights.shape
    out_w = torch.empty(
        (tokens, top_k + 1), device=topk_weights.device, dtype=topk_weights.dtype
    )
    out_i = torch.empty(
        (tokens, top_k + 1), device=topk_ids.device, dtype=topk_ids.dtype
    )
    if tokens == 0:
        return out_w, out_i
    _extend_topk_with_shared_kernel[(tokens,)](
        topk_weights,
        topk_ids,
        shared_logits,
        out_w,
        out_i,
        int(shared_expert_id),
        top_k=top_k,
        BLOCK=triton.next_power_of_2(top_k),
    )
    return out_w, out_i
