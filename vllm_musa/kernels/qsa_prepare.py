# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused QSA prepare kernel for Qwen4Exp on MUSA.

Started as a copy of ``nvidia/ops/qsa_prepare.py`` so the MUSA path can keep
the single-launch contract while diverging where MUSA's Triton needs it. The
only intentional divergence is the raw-state source select: the NVIDIA kernel
selects one pointer with ``tl.where`` and then loads it, which MUSA's Triton
does not accept reliably; this copy issues two masked loads instead (the same
choice the AMD copy makes).
"""


import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _norm_rope(
    x,
    pos_t,
    pos_h,
    pos_w,
    cos_sin_ptr,
    cos_sin_stride,
    norm_weight_ptr,
    eps,
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    """Apply Gemma RMSNorm and selected-axis NeoX RoPE to register rows."""
    TILE_T: tl.constexpr = x.shape[0]
    TILE_H: tl.constexpr = x.shape[1]
    D: tl.constexpr = x.shape[2]
    ROWS: tl.constexpr = TILE_T * TILE_H
    HALF: tl.constexpr = D // 2
    QUARTER: tl.constexpr = D // 4
    pairs = tl.arange(0, QUARTER)
    if IS_MROPE:
        # Qwen interleaves temporal, height, and width rotary pairs. Each axis
        # still indexes the same position-major cos/sin table.
        h_mask = ((pairs % 3) == 1) & (pairs <= 3 * MROPE_H)
        w_mask = ((pairs % 3) == 2) & (pairs <= 3 * MROPE_W)
        t_mask = ~(h_mask | w_mask)
        # MUSA Triton rejects indexing a Python tuple with a static_range
        # variable ("unsupported tensor index"), so unroll the three axes.
        base = cos_sin_ptr + pairs[None, :]
        cos = (
            tl.load(base + pos_t[:, None] * cos_sin_stride, mask=t_mask[None, :], other=0)
            + tl.load(base + pos_h[:, None] * cos_sin_stride, mask=h_mask[None, :], other=0)
            + tl.load(base + pos_w[:, None] * cos_sin_stride, mask=w_mask[None, :], other=0)
        )
        sin = (
            tl.load(
                base + pos_t[:, None] * cos_sin_stride + QUARTER,
                mask=t_mask[None, :],
                other=0,
            )
            + tl.load(
                base + pos_h[:, None] * cos_sin_stride + QUARTER,
                mask=h_mask[None, :],
                other=0,
            )
            + tl.load(
                base + pos_w[:, None] * cos_sin_stride + QUARTER,
                mask=w_mask[None, :],
                other=0,
            )
        )
    else:
        cos = tl.load(cos_sin_ptr + pos_t[:, None] * cos_sin_stride + pairs[None, :])
        sin = tl.load(
            cos_sin_ptr + pos_t[:, None] * cos_sin_stride + QUARTER + pairs[None, :]
        )

    cos = tl.reshape(
        tl.broadcast_to(cos[:, None, :], (TILE_T, TILE_H, QUARTER)),
        (ROWS, QUARTER),
    )
    sin = tl.reshape(
        tl.broadcast_to(sin[:, None, :], (TILE_T, TILE_H, QUARTER)),
        (ROWS, QUARTER),
    )
    x = tl.reshape(x, (ROWS, D)).to(tl.float32)
    weight = tl.load(norm_weight_ptr + tl.arange(0, D)).to(tl.float32) + 1.0
    rrms = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    y = (x * rrms[:, None] * weight[None, :]).to(cos.dtype)
    rotated, passthrough = tl.split(
        tl.permute(tl.reshape(y, (ROWS, 2, HALF)), (0, 2, 1))
    )
    r0, r1 = tl.split(tl.permute(tl.reshape(rotated, (ROWS, 2, QUARTER)), (0, 2, 1)))
    out0 = r0 * cos - r1 * sin
    out1 = r1 * cos + r0 * sin
    rotated = tl.reshape(tl.permute(tl.join(out0, out1), (0, 2, 1)), (ROWS, HALF))
    result = tl.reshape(tl.permute(tl.join(rotated, passthrough), (0, 2, 1)), (ROWS, D))
    return tl.reshape(result, (TILE_T, TILE_H, D))


@triton.jit
def _to_dst_dtype(x, dst, scale):
    """Round to BF16 like the unfused path, then scale for an FP8 destination."""
    out_ty = dst.dtype.element_ty
    x = x.to(tl.bfloat16)
    if out_ty == tl.float8e4nv:
        x = x.to(tl.float32) / scale
    return x.to(out_ty)


@triton.jit
def _store_rotated(dst, y, o1, o2, scale):
    """Store a normalized head whose first ``2 * len(o1)`` dims are rotated."""
    HALF: tl.constexpr = o1.shape[0]
    dims = tl.arange(0, y.shape[0])
    rot = tl.arange(0, HALF)
    tl.store(dst + dims, _to_dst_dtype(y, dst, scale), mask=dims >= 2 * HALF)
    tl.store(dst + rot, _to_dst_dtype(o1, dst, scale))
    tl.store(dst + HALF + rot, _to_dst_dtype(o2, dst, scale))


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_state_blocks",
        "num_compressed_blocks",
        "num_k_work",
    ]
)
def _qsa_prepare_kernel(
    q_ptr,
    q_stride_token,
    k_ptr,
    k_stride_token,
    pos_ptr,
    pos_stride_axis,
    pos_stride_token,
    cos_sin_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    eps,
    q_out_ptr,
    q_out_stride_token,
    q_out_stride_head,
    state_cache_ptr,
    state_cache_stride_block,
    state_cache_stride_token,
    state_slots_ptr,
    state_table_ptr,
    state_table_stride_req,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    k_work_metadata_ptr,
    compressed_cache_ptr,
    compressed_cache_stride_block,
    compressed_cache_stride_token,
    num_tokens,
    num_state_blocks,
    num_compressed_blocks,
    num_k_work,
    HQ: tl.constexpr,
    D: tl.constexpr,
    TILE_T_Q: tl.constexpr,
    TILE_H_Q: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    COMP_PAGE_SIZE: tl.constexpr,
    IS_2D_POSITIONS: tl.constexpr,
    IS_K_MROPE: tl.constexpr,
    CACHE_HAS_ROPE_POS: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    MAX_ROPE_POS: tl.constexpr,
    main_qkv_ptr,
    main_qkv_stride_token,
    main_q_norm_weight_ptr,
    main_k_norm_weight_ptr,
    main_eps,
    main_q_out_ptr,
    main_gate_out_ptr,
    main_cache_ptr,
    main_cache_stride_block,
    main_cache_stride_token,
    main_cache_stride_head,
    main_slots_ptr,
    main_k_scale,
    main_v_scale,
    MAIN_HQ: tl.constexpr,
    MAIN_HK: tl.constexpr,
    MAIN_D: tl.constexpr,
    MAIN_PAGE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_index_work = num_k_work + tl.cdiv(num_tokens, TILE_T_Q) * tl.cdiv(HQ, TILE_H_Q)
    if pid >= num_index_work:
        # Main attention: one program per (token, Q or KV head) after the
        # indexer work. RoPE covers the first D // 2 dims, the width of the
        # shared cos/sin table.
        main_pid = pid - num_index_work
        token = (main_pid // (MAIN_HQ + MAIN_HK)).to(tl.int64)
        head = main_pid % (MAIN_HQ + MAIN_HK)
        HALF: tl.constexpr = D // 4
        dims = tl.arange(0, MAIN_D)
        rot = tl.arange(0, HALF)
        row = main_qkv_ptr + token * main_qkv_stride_token
        is_k = head >= MAIN_HQ
        kv_head = head - MAIN_HQ
        if is_k:
            src = row + (2 * MAIN_HQ + kv_head) * MAIN_D
            weight_ptr = main_k_norm_weight_ptr
        else:
            src = row + 2 * head * MAIN_D
            weight_ptr = main_q_norm_weight_ptr
        x = tl.load(src + dims).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / MAIN_D + main_eps)
        w = tl.load(weight_ptr + dims).to(tl.float32) + 1.0
        y = x * inv_rms * w
        x1 = tl.load(src + rot).to(tl.float32)
        x2 = tl.load(src + HALF + rot).to(tl.float32)
        w1 = tl.load(weight_ptr + rot).to(tl.float32) + 1.0
        w2 = tl.load(weight_ptr + HALF + rot).to(tl.float32) + 1.0
        x1 = (x1 * inv_rms * w1).to(tl.bfloat16).to(tl.float32)
        x2 = (x2 * inv_rms * w2).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(pos_ptr + token * pos_stride_token).to(tl.int64)
        if IS_2D_POSITIONS:
            pos_h = tl.load(pos_ptr + pos_stride_axis + token * pos_stride_token)
            pos_w = tl.load(pos_ptr + 2 * pos_stride_axis + token * pos_stride_token)
            is_h = (rot % 3 == 1) & (rot < 3 * MROPE_H)
            is_w = (rot % 3 == 2) & (rot < 3 * MROPE_W)
            pos = tl.where(
                is_h, pos_h.to(tl.int64), tl.where(is_w, pos_w.to(tl.int64), pos)
            )
        cos = tl.load(cos_sin_ptr + pos * (D // 2) + rot).to(tl.float32)
        sin = tl.load(cos_sin_ptr + pos * (D // 2) + HALF + rot).to(tl.float32)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_k:
            slot = tl.load(main_slots_ptr + token).to(tl.int64)
            if slot >= 0:
                dst = (
                    main_cache_ptr
                    + (slot // MAIN_PAGE_SIZE) * main_cache_stride_block
                    + (slot % MAIN_PAGE_SIZE) * main_cache_stride_token
                    + kv_head * main_cache_stride_head
                )
                _store_rotated(dst, y, o1, o2, main_k_scale)
                v = tl.load(src + MAIN_HK * MAIN_D + dims)
                tl.store(dst + MAIN_D + dims, _to_dst_dtype(v, dst, main_v_scale))
        else:
            out = (token * MAIN_HQ + head) * MAIN_D
            _store_rotated(main_q_out_ptr + out, y, o1, o2, None)
            gate = tl.load(src + MAIN_D + dims)
            tl.store(main_gate_out_ptr + out + dims, gate)
        return
    # K work occupies the first programs; the remaining programs tile Q. This
    # keeps both paths in one launch while leaving their register shapes
    # independent.
    if pid >= num_k_work:
        q_pid = pid - num_k_work
        num_head_tiles: tl.constexpr = tl.cdiv(HQ, TILE_H_Q)
        token_tile = q_pid // num_head_tiles
        head_tile = q_pid % num_head_tiles
        tokens = (token_tile * TILE_T_Q + tl.arange(0, TILE_T_Q)).to(tl.int64)
        heads = head_tile * TILE_H_Q + tl.arange(0, TILE_H_Q)
        valid_tokens = tokens < num_tokens
        valid_heads = heads < HQ
        dims = tl.arange(0, D)
        mask = valid_tokens[:, None, None] & valid_heads[None, :, None]
        x = tl.load(
            q_ptr
            + tokens[:, None, None] * q_stride_token
            + heads[None, :, None] * D
            + dims[None, None, :],
            mask=mask,
            other=0.0,
        )
        pos_t = tl.load(pos_ptr + tokens * pos_stride_token, mask=valid_tokens, other=0)
        if IS_2D_POSITIONS:
            pos_h = tl.load(
                pos_ptr + pos_stride_axis + tokens * pos_stride_token,
                mask=valid_tokens,
                other=0,
            )
            pos_w = tl.load(
                pos_ptr + 2 * pos_stride_axis + tokens * pos_stride_token,
                mask=valid_tokens,
                other=0,
            )
        else:
            pos_h = pos_t
            pos_w = pos_t
        y = _norm_rope(
            x,
            pos_t,
            pos_h,
            pos_w,
            cos_sin_ptr,
            D // 2,
            q_norm_weight_ptr,
            eps,
            IS_2D_POSITIONS,
            MROPE_H,
            MROPE_W,
        )
        tl.store(
            q_out_ptr
            + tokens[:, None, None] * q_out_stride_token
            + heads[None, :, None] * q_out_stride_head
            + dims[None, None, :],
            y,
            mask=mask,
        )

    if pid < num_k_work:
        # One work item owns one completed compression group. Work item zero
        # additionally commits the request's current raw-K suffix below.
        work_metadata = tl.load(k_work_metadata_ptr + pid * 2 + tl.arange(0, 2))
        request, work_in_request = tl.split(work_metadata)
        if request < 0:
            return

        query_start = tl.load(query_start_loc_ptr + request)
        query_end = tl.load(query_start_loc_ptr + request + 1)
        query_len = query_end - query_start
        chunk_end = tl.load(logical_positions_ptr + query_end - 1)
        chunk_start = chunk_end - query_len + 1
        num_groups = (chunk_end + 1) // COMPRESS_RATIO - chunk_start // COMPRESS_RATIO
        dims = tl.arange(0, D)

        if work_in_request < num_groups:
            first_boundary = (
                (chunk_start + COMPRESS_RATIO) // COMPRESS_RATIO
            ) * COMPRESS_RATIO - 1
            end_position = first_boundary + work_in_request * COMPRESS_RATIO
            boundary_token = query_start + end_position - chunk_start
            valid_token = (
                (boundary_token >= query_start)
                & (boundary_token < query_end)
                & (boundary_token < num_tokens)
            )
            compressed_slot = tl.load(
                compressed_slots_ptr + boundary_token,
                mask=valid_token,
                other=-1,
            )
            valid = (
                valid_token
                & (compressed_slot >= 0)
                & (compressed_slot < num_compressed_blocks * COMP_PAGE_SIZE)
            )
            state_block = tl.load(state_table_ptr + request * state_table_stride_req)
            state_block_valid = (state_block >= 0) & (state_block < num_state_blocks)
            safe_state_block = tl.maximum(state_block, 0).to(tl.int64)
            group_offsets = tl.arange(0, COMPRESS_RATIO)
            source_positions = end_position - (COMPRESS_RATIO - 1) + group_offsets
            source_in_chunk = source_positions >= chunk_start
            source_tokens = query_start + source_positions - chunk_start
            source_tokens_valid = (
                (source_tokens >= query_start)
                & (source_tokens < query_end)
                & (source_tokens < num_tokens)
            )
            current_base = (
                k_ptr
                + tl.maximum(source_tokens, 0).to(tl.int64)[:, None] * k_stride_token
            )
            cached_base = (
                state_cache_ptr
                + safe_state_block * state_cache_stride_block
                + (source_positions % STATE_SIZE)[:, None] * state_cache_stride_token
            )
            # Only the first completed group can cross the chunk boundary.
            # Issue one masked load per source instead of selecting a pointer
            # with tl.where, which MUSA Triton rejects.
            current = tl.load(
                current_base + dims[None, :],
                mask=valid & source_in_chunk[:, None] & source_tokens_valid[:, None],
                other=0.0,
            )
            cached = tl.load(
                cached_base + dims[None, :],
                mask=valid & (~source_in_chunk)[:, None] & state_block_valid,
                other=0.0,
            )
            source = (current + cached).to(tl.float32)
            # Match the unfused path's BF16 pooled tensor before RMSNorm.
            pooled = (
                (tl.sum(source, axis=0) / COMPRESS_RATIO).to(tl.bfloat16).to(tl.float32)
            )

            first_position = end_position - (COMPRESS_RATIO - 1)
            if CACHE_HAS_ROPE_POS:
                # RoPE uses the first token in the pooled group. Its exact MRoPE
                # coordinates may live in this chunk or the raw-state ring.
                first_in_chunk = first_position >= chunk_start
                first_token = query_start + first_position - chunk_start
                first_token_valid = (
                    (first_token >= query_start)
                    & (first_token < query_end)
                    & (first_token < num_tokens)
                )
                safe_first_token = tl.maximum(first_token, 0)
                load_current_position = first_in_chunk & first_token_valid
                first_pos_t = tl.load(
                    pos_ptr + safe_first_token * pos_stride_token,
                    mask=load_current_position,
                    other=0,
                )
                if IS_2D_POSITIONS:
                    first_pos_h = tl.load(
                        pos_ptr + pos_stride_axis + safe_first_token * pos_stride_token,
                        mask=load_current_position,
                        other=0,
                    )
                    first_pos_w = tl.load(
                        pos_ptr
                        + 2 * pos_stride_axis
                        + safe_first_token * pos_stride_token,
                        mask=load_current_position,
                        other=0,
                    )
                else:
                    first_pos_h = first_pos_t
                    first_pos_w = first_pos_t
                tail = (
                    state_cache_ptr
                    + safe_state_block * state_cache_stride_block
                    + (first_position % STATE_SIZE) * state_cache_stride_token
                    + D
                ).to(tl.pointer_type(tl.int64))
                load_cached_position = ~first_in_chunk & state_block_valid
                cached_pos_t = tl.load(tail, mask=load_cached_position, other=0)
                cached_pos_h = tl.load(tail + 1, mask=load_cached_position, other=0)
                cached_pos_w = tl.load(tail + 2, mask=load_cached_position, other=0)
                pos_t = tl.where(first_in_chunk, first_pos_t.to(tl.int64), cached_pos_t)
                pos_h = tl.where(first_in_chunk, first_pos_h.to(tl.int64), cached_pos_h)
                pos_w = tl.where(first_in_chunk, first_pos_w.to(tl.int64), cached_pos_w)
                # Recycled side-cache pages can carry stale MRoPE tails. Keep
                # those values from becoming unchecked rotary-cache addresses.
                pos_t = tl.where(
                    (pos_t >= 0) & (pos_t < MAX_ROPE_POS), pos_t, first_position
                )
                pos_h = tl.where(
                    (pos_h >= 0) & (pos_h < MAX_ROPE_POS), pos_h, first_position
                )
                pos_w = tl.where(
                    (pos_w >= 0) & (pos_w < MAX_ROPE_POS), pos_w, first_position
                )
            else:
                pos_t = first_position
                pos_h = first_position
                pos_w = first_position
            y = _norm_rope(
                tl.reshape(pooled, (1, 1, D)),
                pos_t + tl.arange(0, 1),
                pos_h + tl.arange(0, 1),
                pos_w + tl.arange(0, 1),
                cos_sin_ptr,
                D // 2,
                k_norm_weight_ptr,
                eps,
                IS_K_MROPE,
                MROPE_H,
                MROPE_W,
            )
            compressed_block = (compressed_slot // COMP_PAGE_SIZE).to(tl.int64)
            compressed_row = compressed_slot % COMP_PAGE_SIZE
            tl.store(
                compressed_cache_ptr
                + compressed_block * compressed_cache_stride_block
                + compressed_row * compressed_cache_stride_token
                + dims,
                tl.reshape(y, (D,)),
                mask=valid,
            )

        if work_in_request == 0:
            # This CTA may have just read historical rows from the circular buffer.
            # Keep every lane past those loads before overwriting the same ring.
            tl.debug_barrier()
            # One CTA per request commits only the suffix retained by the ring.
            num_state_rows = tl.minimum(query_len, STATE_SIZE)
            for state_offset in tl.range(0, num_state_rows):
                token = (query_end - num_state_rows + state_offset).to(tl.int64)
                valid_token = (
                    (token >= query_start) & (token < query_end) & (token < num_tokens)
                )
                slot = tl.load(state_slots_ptr + token, mask=valid_token, other=-1)
                valid_slot = (
                    valid_token & (slot >= 0) & (slot < num_state_blocks * STATE_SIZE)
                )
                safe_slot = tl.maximum(slot, 0)
                state_row = (
                    state_cache_ptr
                    + (safe_slot // STATE_SIZE).to(tl.int64) * state_cache_stride_block
                    + (safe_slot % STATE_SIZE) * state_cache_stride_token
                )
                k = tl.load(
                    k_ptr + tl.maximum(token, 0) * k_stride_token + dims,
                    mask=valid_slot,
                    other=0.0,
                )
                tl.store(state_row + dims, k, mask=valid_slot)
                if CACHE_HAS_ROPE_POS:
                    pos_t = tl.load(
                        pos_ptr + tl.maximum(token, 0) * pos_stride_token,
                        mask=valid_slot,
                        other=0,
                    )
                    if IS_2D_POSITIONS:
                        pos_h = tl.load(
                            pos_ptr
                            + pos_stride_axis
                            + tl.maximum(token, 0) * pos_stride_token,
                            mask=valid_slot,
                            other=0,
                        )
                        pos_w = tl.load(
                            pos_ptr
                            + 2 * pos_stride_axis
                            + tl.maximum(token, 0) * pos_stride_token,
                            mask=valid_slot,
                            other=0,
                        )
                    else:
                        pos_h = pos_t
                        pos_w = pos_t
                    tail = (state_row + D).to(tl.pointer_type(tl.int64))
                    tl.store(tail, pos_t.to(tl.int64), mask=valid_slot)
                    tl.store(tail + 1, pos_h.to(tl.int64), mask=valid_slot)
                    tl.store(tail + 2, pos_w.to(tl.int64), mask=valid_slot)


def qsa_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
    state_cache: torch.Tensor,
    state_slots: torch.Tensor,
    state_block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_slots: torch.Tensor,
    k_work_metadata: torch.Tensor,
    *,
    compress_ratio: int,
    mrope_section: tuple[int, int, int] | None,
    rope_pos_offset: int | None,
    main_qkv: torch.Tensor,
    main_q_norm_weight: torch.Tensor,
    main_k_norm_weight: torch.Tensor,
    main_eps: float,
    main_kv_cache: torch.Tensor,
    main_slot_mapping: torch.Tensor,
    main_k_scale: float,
    main_v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Q, compress K, then update the circular raw state.

    Also prepares the main attention (QK-norm/RoPE, gate copy, K/V cache write)
    and returns its Q and gate.
    """
    num_tokens = q.shape[0]
    main_head_dim = main_kv_cache.shape[-1] // 2
    num_main_kv_heads = main_kv_cache.shape[2]
    num_main_q_heads = main_qkv.shape[1] // (2 * main_head_dim) - num_main_kv_heads
    main_q_out = main_qkv.new_empty(num_tokens, num_main_q_heads, main_head_dim)
    main_gate_out = torch.empty_like(main_q_out)
    if num_tokens == 0:
        return main_q_out, main_gate_out
    num_q_heads, head_dim = q_out.shape[1:]
    assert cos_sin_cache.shape[-1] * 2 == head_dim
    assert q.shape == (num_tokens, num_q_heads * head_dim)
    assert k.shape == (num_tokens, head_dim)
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert q_out.stride(-1) == 1
    assert cos_sin_cache.is_contiguous()
    assert state_cache.stride(-1) == 1
    assert compressed_cache.stride(-1) == 1
    assert k_work_metadata.ndim == 2 and k_work_metadata.shape[1] == 2
    is_2d_positions = positions.ndim == 2
    is_k_mrope = bool(mrope_section)
    cache_has_rope_pos = rope_pos_offset is not None
    assert rope_pos_offset is None or rope_pos_offset == head_dim
    if is_2d_positions:
        assert positions.shape == (3, num_tokens)
        assert is_k_mrope
        pos_stride_axis, pos_stride_token = positions.stride()
    else:
        assert positions.shape == (num_tokens,)
        pos_stride_axis, pos_stride_token = 0, positions.stride(0)
    section = mrope_section if mrope_section is not None else (0, 0, 0)
    assert len(section) == 3
    qkv_width = 2 * (num_main_q_heads + num_main_kv_heads) * main_head_dim
    assert main_qkv.shape == (num_tokens, qkv_width), (
        f"main_qkv={tuple(main_qkv.shape)} width={qkv_width} tokens={num_tokens} "
        f"hd={main_head_dim} q={num_main_q_heads} kv={num_main_kv_heads} "
        f"kv_cache={tuple(main_kv_cache.shape)}"
    )
    assert main_qkv.stride(-1) == 1
    assert main_slot_mapping.shape == (num_tokens,)

    if num_tokens <= 4096:
        TILE_T_Q, TILE_H_Q = 2, 2
    else:
        TILE_T_Q, TILE_H_Q = 2, 4
    num_k_work = k_work_metadata.shape[0]
    num_q_work = triton.cdiv(num_tokens, TILE_T_Q) * triton.cdiv(num_q_heads, TILE_H_Q)
    num_main_work = num_tokens * (num_main_q_heads + num_main_kv_heads)
    _qsa_prepare_kernel[(num_k_work + num_q_work + num_main_work,)](
        q,
        q.stride(0),
        k,
        k.stride(0),
        positions,
        pos_stride_axis,
        pos_stride_token,
        cos_sin_cache,
        q_norm_weight,
        k_norm_weight,
        eps,
        q_out,
        q_out.stride(0),
        q_out.stride(1),
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        state_slots,
        state_block_table,
        state_block_table.stride(0),
        query_start_loc,
        logical_positions,
        compressed_slots,
        k_work_metadata,
        compressed_cache,
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        num_tokens,
        state_cache.shape[0],
        compressed_cache.shape[0],
        num_k_work,
        HQ=num_q_heads,
        D=head_dim,
        TILE_T_Q=TILE_T_Q,
        TILE_H_Q=TILE_H_Q,
        COMPRESS_RATIO=compress_ratio,
        STATE_SIZE=state_cache.shape[1],
        COMP_PAGE_SIZE=compressed_cache.shape[1],
        IS_2D_POSITIONS=is_2d_positions,
        IS_K_MROPE=is_k_mrope,
        CACHE_HAS_ROPE_POS=cache_has_rope_pos,
        MROPE_H=section[1],
        MROPE_W=section[2],
        MAX_ROPE_POS=cos_sin_cache.shape[0],
        main_qkv_ptr=main_qkv,
        main_qkv_stride_token=main_qkv.stride(0),
        main_q_norm_weight_ptr=main_q_norm_weight,
        main_k_norm_weight_ptr=main_k_norm_weight,
        main_eps=main_eps,
        main_q_out_ptr=main_q_out,
        main_gate_out_ptr=main_gate_out,
        main_cache_ptr=main_kv_cache,
        main_cache_stride_block=main_kv_cache.stride(0),
        main_cache_stride_token=main_kv_cache.stride(1),
        main_cache_stride_head=main_kv_cache.stride(2),
        main_slots_ptr=main_slot_mapping,
        main_k_scale=main_k_scale,
        main_v_scale=main_v_scale,
        MAIN_HQ=num_main_q_heads,
        MAIN_HK=num_main_kv_heads,
        MAIN_D=main_head_dim,
        MAIN_PAGE_SIZE=main_kv_cache.shape[1],
        num_warps=1,
    )
    return main_q_out, main_gate_out


__all__ = ["qsa_prepare"]


def warmup_qsa_prepare(
    *,
    indexer,
    owner,
    num_k_work: int,
    num_tokens: int,
    is_2d_positions: bool,
) -> None:
    """Compile one fused-prepare specialization before graph capture.

    The QSA indexer runs inside ``eager_break_during_capture`` segments, so a
    lazy Triton JIT there would run Python work on the capturing stream. All
    slot mappings are ``-1`` so the dummy launch writes no cache rows.
    """
    import torch

    device = indexer.q_layernorm.weight.device
    dtype = indexer.rotary_emb.cos_sin_cache.dtype
    index_heads = int(indexer.index_n_heads)
    index_head_dim = int(indexer.index_head_dim)
    compress_ratio = int(indexer.compress_ratio)

    q = torch.zeros(num_tokens, index_heads * index_head_dim, device=device, dtype=dtype)
    k = torch.zeros(num_tokens, index_head_dim, device=device, dtype=dtype)
    if is_2d_positions:
        positions = torch.zeros(3, num_tokens, device=device, dtype=torch.int64)
    else:
        positions = torch.zeros(num_tokens, device=device, dtype=torch.int64)
    q_out = torch.empty(num_tokens, index_heads, index_head_dim, device=device, dtype=dtype)
    state_cache = indexer.raw_key_cache.key_cache
    state_slots = torch.full((num_tokens,), -1, device=device, dtype=torch.int32)
    state_block_table = torch.zeros(1, 1, device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, num_tokens], device=device, dtype=torch.int32)
    logical_positions = torch.zeros(num_tokens, device=device, dtype=torch.int64)
    compressed_cache = indexer.compressed_key_cache.kv_cache
    compressed_slots = torch.full((num_tokens,), -1, device=device, dtype=torch.int32)
    k_work_metadata = torch.full((num_k_work, 2), -1, device=device, dtype=torch.int32)

    main_kv_cache = owner.kv_cache.transpose(1, 2)
    if owner.kv_cache_dtype in ("fp8", "fp8_e4m3"):
        main_kv_cache = main_kv_cache.view(torch.float8_e4m3fn)
    # Mirror the kernel's own derivation from the live cache so this warmup
    # compiles the specialization the real forward pass will launch. The main
    # attention head_dim (256) differs from the indexer head_dim (128), and the
    # number of per-rank KV heads depends on TP, so neither can be assumed.
    main_head_dim = main_kv_cache.shape[-1] // 2
    main_kv_heads = main_kv_cache.shape[-2]
    main_q_heads = owner.num_heads
    assert main_head_dim == owner.head_dim, (main_head_dim, owner.head_dim)
    qkv_width = 2 * (main_q_heads + main_kv_heads) * main_head_dim
    main_qkv = torch.zeros(num_tokens, qkv_width, device=device, dtype=dtype)
    main_slots = torch.full((num_tokens,), -1, device=device, dtype=torch.int32)

    qsa_prepare(
        q,
        k,
        positions,
        indexer.rotary_emb.cos_sin_cache,
        indexer.q_layernorm.weight,
        indexer.k_layernorm.weight,
        indexer.q_layernorm.variance_epsilon,
        q_out,
        state_cache,
        state_slots,
        state_block_table,
        query_start_loc,
        logical_positions,
        compressed_cache,
        compressed_slots,
        k_work_metadata,
        compress_ratio=compress_ratio,
        mrope_section=getattr(indexer.rotary_emb, "mrope_section", None),
        rope_pos_offset=(
            indexer.raw_key_cache.rope_position_offset
            if indexer.raw_key_cache.rope_position_cache is not None
            else None
        ),
        main_qkv=main_qkv,
        main_q_norm_weight=owner.q_norm.weight,
        main_k_norm_weight=owner.k_norm.weight,
        main_eps=owner.q_norm.variance_epsilon,
        main_kv_cache=main_kv_cache,
        main_slot_mapping=main_slots,
        main_k_scale=owner._k_scale_float,
        main_v_scale=owner._v_scale_float,
    )


__all__ = ["qsa_prepare", "warmup_qsa_prepare"]
