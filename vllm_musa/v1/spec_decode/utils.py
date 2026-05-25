from vllm.triton_utils import tl, triton


@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,  # [num_reqs, num_sampled_tokens_per_req]
    discard_request_mask_ptr,  # [num_reqs]
    backup_next_token_ids_ptr,  # [num_reqs]
    next_token_ids_ptr,  # [num_reqs] (output)
    valid_sampled_tokens_count_ptr,  # [num_reqs] (output)
    vocab_size,  # tl.int32
    num_sampled_tokens_per_req,  # tl.int32 (num_spec_tokens + 1)
    num_reqs,  # tl.int32
    stride_sampled_token_ids,  # tl.int32 (stride for dim 0)
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Power-of-2 >= num_sampled_tokens_per_req
):
    """
    Fused kernel for Eagle prepare_next_token_ids_padded. This kernel computes the
    number of valid (1 + accepted) tokens for each request, and the corresponding
    "next" token id to sample from during speculative decoding. This is the
    "last accepted token" from the sampled tokens, or the backup token if no
    tokens were accepted or if the request is marked as discarded.
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Check if this request is discarded.
    is_discarded = tl.load(discard_request_mask_ptr + req_idx)

    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        # ==================== MUSA ADAPTATION ====================
        valid_count = tl.full((), 0, dtype=tl.int32)
        # ========================== END ==========================
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
    else:
        # Count the number of valid tokens among the sampled tokens.
        token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
        token_mask = token_offs < num_sampled_tokens_per_req

        row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
        token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

        # Rejected tokens are -1, valid tokens are in [0, vocab_size)
        is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask
        # ==================== MUSA ADAPTATION ====================
        valid_count = tl.sum(is_valid_mask.to(tl.int32))
        # ========================== END ==========================

        if valid_count > 0:
            # Guaranteed to be well-defined since
            # valid_count > 0 implies is_valid_mask is not empty
            last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))

            # Select the token at that index, using a sum trick since
            # we don't want to load again to access token_ids[last_valid_index].
            last_valid_token = tl.sum(
                tl.where(token_offs == last_valid_index, token_ids, 0)
            )
            tl.store(next_token_ids_ptr + req_idx, last_valid_token)
        else:
            # No valid tokens found, use backup token
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            tl.store(next_token_ids_ptr + req_idx, backup_token)

        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


import torch
import vllm.v1.spec_decode.utils

vllm.v1.spec_decode.utils.eagle_prepare_next_token_padded_kernel = (
    eagle_prepare_next_token_padded_kernel
)


# MUSA-0109 / MUSA-0090 reproduction fix (2026-05-17): the upstream
# `update_num_computed_tokens_for_batch_change` is wrapped with
# `@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)`,
# and on MUSA the simple_compile_backend is Inductor. When the MUSA-0090
# EagleFullLoopRunner captures a CUDAGraph and the next execute_model triggers
# Inductor's Triton precompile of this function, the MUSA driver returns
# "unknown error" (likely due to capture state interaction).
# See generated/musa0109/musa0090-reproduction.md for the full trace.
#
# Workaround: replace this function with an eager (no torch.compile) version.
# The function does only 5-6 small tensor ops on small tensors — eager is fine.


def _musa_update_num_computed_tokens_for_batch_change(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    """Eager (no torch.compile) version of update_num_computed_tokens_for_batch_change.

    Same semantics as upstream; just avoids Inductor precompile on MUSA which
    crashes after a MUSA-0090 EagleFullLoopRunner CUDAGraph replay.
    """
    gather_indices = prev_positions.clamp(min=0)
    valid_counts = valid_sampled_token_count[gather_indices]
    prev_computed = num_computed_tokens[gather_indices]
    prev_drafts = prev_num_draft_tokens[gather_indices]

    participating = (prev_positions >= 0) & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.int()

    n = prev_positions.shape[0]
    num_computed_tokens[:n].copy_(
        torch.where(participating, corrected, cpu_num_computed_tokens)
    )
    num_accepted_tokens.copy_(
        torch.where(participating, valid_counts, num_accepted_tokens)
    )


vllm.v1.spec_decode.utils.update_num_computed_tokens_for_batch_change = (
    _musa_update_num_computed_tokens_for_batch_change
)
