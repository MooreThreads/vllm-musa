from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton

from vllm_musa import _custom_ops as musa_ops


@triton.jit
def _max_with_nan(left, right):
    return tl.maximum(left, right, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _candidate_block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    counts,
    stride_row,
    stride_start,
    stride_end,
    stride_score,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.minimum(tl.load(ends + row // ROW_REPEAT * stride_end), WIDTH)
    count = tl.cdiv(tl.maximum(end - start, 0), BLOCK_SIZE)
    if tl.program_id(1) == 0:
        tl.store(counts + row, count)
    first = tl.program_id(1) * TILE
    if first >= count:
        return
    blocks = first + tl.arange(0, TILE)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    values = tl.load(
        logits + row * stride_row + cols,
        (blocks[:, None] < count)
        & (offsets[None, :] < BLOCK_SIZE)
        & (cols >= 0)
        & (cols < end),
        other=-float("inf"),
    )
    reduced = tl.reduce(values, 1, _max_with_nan)
    reduced = tl.where(blocks == count - 1, float("inf"), reduced)
    tl.store(scores + row * stride_score + blocks, reduced, blocks < count)


def _topk(
    logits: torch.Tensor, ends: torch.Tensor, output: torch.Tensor, topk: int
) -> None:
    musa_ops.deepseek_v4_sparse_indexer_topk_decode(logits, ends, output, topk)


@triton.jit
def _store_candidate_blocks_kernel(
    scores,
    columns,
    counts,
    output,
    stride_score,
    stride_column,
    stride_output_row,
    stride_output_col,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
):
    """Preserve native short-row order and sort truncated selections."""
    row = tl.program_id(0)
    slots = tl.arange(0, TILE)
    column = tl.load(columns + row * stride_column + slots, slots < TOPK, other=-1)
    valid = (slots < TOPK) & (column >= 0) & (column < WIDTH)
    value = tl.load(scores + row * stride_score + column, valid, other=-float("inf"))
    selected = tl.where(valid & (value != -float("inf")), column, 2147483647).to(
        tl.int32
    )
    if tl.load(counts + row) > TOPK:
        selected = tl.sort(selected)
    tl.store(
        output + row * stride_output_row + slots * stride_output_col,
        tl.where(selected != 2147483647, selected, -1),
        slots < TOPK,
    )


def select_candidate_blocks(
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    block_size: int,
    output: torch.Tensor,
    row_repeat: int = 1,
) -> None:
    """Publish visible blocks in token order, keeping the newest block pinned."""
    rows, width = logits.shape
    num_blocks = triton.cdiv(width, block_size)
    scores = torch.empty((rows, num_blocks), device=logits.device, dtype=torch.float32)
    counts = torch.empty((rows,), device=logits.device, dtype=torch.int32)
    _candidate_block_scores_kernel[(rows, triton.cdiv(num_blocks, 32))](
        logits,
        row_starts,
        row_ends,
        scores,
        counts,
        logits.stride(0),
        row_starts.stride(0) if row_starts is not None else 0,
        row_ends.stride(0),
        scores.stride(0),
        width,
        block_size,
        row_starts is not None,
        row_repeat,
        32,
    )
    columns = torch.empty(output.shape, device=output.device, dtype=output.dtype)
    _topk(scores, counts, columns, output.shape[1])
    _store_candidate_blocks_kernel[(rows,)](
        scores,
        columns,
        counts,
        output,
        scores.stride(0),
        columns.stride(0),
        *output.stride(),
        num_blocks,
        output.shape[1],
        triton.next_power_of_2(output.shape[1]),
    )


@triton.jit
def _compact_candidate_blocks_kernel(
    candidates,
    lengths,
    output,
    ends,
    stride_candidate_row,
    stride_candidate_col,
    stride_length,
    NUM_CANDIDATES: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    SORT_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, SORT_SIZE)
    blocks = tl.load(
        candidates + row * stride_candidate_row + slots * stride_candidate_col,
        slots < NUM_CANDIDATES,
        other=-1,
    )
    length = tl.load(lengths + row * stride_length)
    valid = (blocks >= 0) & (blocks < tl.cdiv(length, CANDIDATE_BLOCK))
    ordered = tl.sort(tl.where(valid, blocks, 2147483647).to(tl.int32))
    tl.store(
        output + row * NUM_CANDIDATES + slots,
        ordered,
        slots < NUM_CANDIDATES,
    )
    tl.debug_barrier()
    previous = tl.load(
        output + row * NUM_CANDIDATES + slots - 1,
        (slots > 0) & (slots < NUM_CANDIDATES),
        other=2147483647,
    )
    unique = ((slots == 0) | (ordered != previous)) & (ordered != 2147483647)
    compact = tl.cumsum(unique.to(tl.int32), 0) - 1
    tl.store(ends + row, tl.sum(unique.to(tl.int32), 0) * CANDIDATE_BLOCK)
    tl.debug_barrier()
    tl.store(output + row * NUM_CANDIDATES + slots, -1, slots < NUM_CANDIDATES)
    tl.debug_barrier()
    tl.store(output + row * NUM_CANDIDATES + compact, ordered, unique)


@triton.jit
def _paged_candidate_mqa_kernel(
    query,
    cache,
    weights,
    block_table,
    requests,
    lengths,
    candidates,
    ends,
    logits,
    stride_q_row,
    stride_q_head,
    stride_cache,
    stride_weight,
    stride_weight_head,
    stride_table,
    stride_length,
    stride_end,
    request_row_offset,
    REQUEST_REPEAT: tl.constexpr,
    HAS_REQUESTS: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    WIDTH: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    SPARSE: tl.constexpr,
    FUSE_BOUNDS: tl.constexpr,
    TILE: tl.constexpr,
    IDENTITY_WHEN_FULL: tl.constexpr = False,
):
    row = tl.program_id(0)
    length = tl.load(lengths + row * stride_length)
    if SPARSE and FUSE_BOUNDS:
        visible_width = WIDTH
        if IDENTITY_WHEN_FULL:
            visible_width = tl.minimum(length, WIDTH)
            if tl.program_id(1) == 0:
                tl.store(ends + row * stride_end, visible_width)
        elif tl.program_id(1) == 0:
            slots = tl.arange(0, triton.next_power_of_2(NUM_CANDIDATES))
            blocks = tl.load(
                candidates + row * NUM_CANDIDATES + slots,
                slots < NUM_CANDIDATES,
                other=-1,
            )
            valid = (blocks >= 0) & (blocks < tl.cdiv(length, CANDIDATE_BLOCK))
            tl.store(
                ends + row * stride_end,
                tl.max(tl.where(valid, slots + 1, 0), 0) * CANDIDATE_BLOCK,
            )
    else:
        visible_width = tl.minimum(tl.load(ends + row * stride_end), WIDTH)
    if tl.program_id(1) * TILE >= visible_width:
        return
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    if SPARSE:
        block = tl.load(
            candidates + row * NUM_CANDIDATES + cols // CANDIDATE_BLOCK,
            cols < WIDTH,
            other=-1,
        )
        token = tl.where(
            block >= 0, block * CANDIDATE_BLOCK + cols % CANDIDATE_BLOCK, -1
        )
    else:
        token = cols
    valid = (cols < WIDTH) & (token >= 0) & (token < length)
    valid = valid & (token // CACHE_BLOCK < TABLE_WIDTH)
    result = tl.full((TILE,), -float("inf"), tl.float32)
    if tl.sum(valid.to(tl.int32), 0) > 0:
        if HAS_REQUESTS:
            request = tl.load(requests + row)
        else:
            request = (row + request_row_offset) // REQUEST_REPEAT
        page = tl.load(
            block_table + request * stride_table + token // CACHE_BLOCK,
            valid,
            other=-1,
        )
        valid = valid & (page >= 0) & (page < NUM_PAGES)
        base = cache + tl.where(valid, page, 0).to(tl.int64) * stride_cache
        offset = token % CACHE_BLOCK
        dimensions = tl.arange(0, 128)
        heads = tl.arange(0, 32)
        keys = tl.load(
            base[:, None] + offset[:, None] * 128 + dimensions[None, :],
            valid[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        scales = tl.load(
            (base + CACHE_BLOCK * 128 + offset * 4).to(tl.pointer_type(tl.float32)),
            valid,
            other=0.0,
        )
        queries = tl.load(
            query
            + row * stride_q_row
            + dimensions[:, None]
            + heads[None, :] * stride_q_head,
        )
        score = tl.dot(keys, queries)
        weight = tl.load(weights + row * stride_weight + heads * stride_weight_head)
        result = tl.sum(tl.maximum(score, 0.0) * weight[None, :], 1) * scales
        result = tl.where(valid, result, -float("inf"))
    tl.store(logits + row * WIDTH + cols, result, cols < WIDTH)


@triton.jit
def _remap_candidate_topk_kernel(
    candidates,
    logits,
    columns,
    output,
    stride_column,
    stride_output,
    WIDTH: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
    SPARSE: tl.constexpr = True,
):
    row = tl.program_id(0)
    slots = tl.arange(0, TILE)
    column = tl.load(columns + row * stride_column + slots, slots < TOPK, other=-1)
    valid = (slots < TOPK) & (column >= 0) & (column < WIDTH)
    if SPARSE:
        block = tl.load(
            candidates + row * NUM_CANDIDATES + column // CANDIDATE_BLOCK,
            valid,
            other=-1,
        )
        token = tl.where(
            block >= 0, block * CANDIDATE_BLOCK + column % CANDIDATE_BLOCK, -1
        )
    else:
        token = column
    score = tl.load(logits + row * WIDTH + column, valid, other=-float("inf"))
    valid = valid & (token >= 0) & (score != -float("inf"))
    count = tl.sum(valid.to(tl.int32), 0)
    if count == TOPK:
        tl.store(output + row * stride_output + slots, token, slots < TOPK)
    else:
        compact = tl.cumsum(valid.to(tl.int32), 0) - 1
        tl.store(output + row * stride_output + compact, token, valid)
        tl.store(
            output + row * stride_output + slots,
            -1,
            (slots < TOPK) & (slots >= count),
        )


def _sparse_width(num_candidates: int, block_size: int, dense_width: int) -> int:
    width = num_candidates * block_size
    return width if width < dense_width else 0


def _score_and_select_rows(
    query: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    requests: torch.Tensor | None,
    lengths: torch.Tensor,
    candidates: torch.Tensor,
    candidate_block_size: int,
    output: torch.Tensor,
    dense_width: int,
    candidate_write: bool,
    candidates_sorted_unique: bool = False,
    request_repeat: int = 1,
    request_row_offset: int = 0,
    candidates_identity_when_full: bool = False,
) -> None:
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
    )

    rows = query.shape[0]
    if rows == 0:
        return
    width = _sparse_width(candidates.shape[1], candidate_block_size, dense_width)
    sparse = (
        not candidate_write
        and width > 0
        and os.getenv("VLLM_MUSA_CANDIDATE_SPARSE_MQA", "1") != "0"
    )
    width = width if sparse else dense_width
    if width == 0:
        output.fill_(-1)
        if candidate_write:
            candidates.fill_(-1)
        return
    blocks = (
        torch.empty((rows, candidates.shape[1]), device=query.device, dtype=torch.int32)
        if sparse and not candidates_sorted_unique
        else None
    )
    ends = (
        torch.empty((rows,), device=query.device, dtype=torch.int32)
        if sparse
        else lengths
    )
    if sparse:
        if candidates_sorted_unique:
            blocks = candidates.contiguous()
        else:
            _compact_candidate_blocks_kernel[(rows,)](
                candidates,
                lengths,
                blocks,
                ends,
                *candidates.stride(),
                lengths.stride(0),
                candidates.shape[1],
                candidate_block_size,
                triton.next_power_of_2(candidates.shape[1]),
            )
    logits = torch.empty((rows, width), device=query.device, dtype=torch.float32)
    _paged_candidate_mqa_kernel[(rows, triton.cdiv(width, 64))](
        query,
        cache,
        weights,
        block_table,
        requests,
        lengths,
        blocks,
        ends,
        logits,
        query.stride(0),
        query.stride(1),
        cache.stride(0),
        *weights.stride(),
        block_table.stride(0),
        lengths.stride(0),
        ends.stride(0),
        request_row_offset,
        request_repeat,
        requests is not None,
        cache.shape[0],
        block_table.shape[1],
        width,
        cache.shape[1],
        candidate_block_size,
        candidates.shape[1],
        sparse,
        candidates_sorted_unique,
        64,
        candidates_identity_when_full,
    )
    if candidate_write:
        select_candidate_blocks(logits, None, lengths, candidate_block_size, candidates)
    elif not sparse:
        apply_candidate_mask(logits, None, lengths, candidates, candidate_block_size)
    if candidate_write:
        _topk(logits, lengths, output, output.shape[1])
        return
    columns = torch.empty_like(output)
    _topk(logits, ends, columns, output.shape[1])
    _remap_candidate_topk_kernel[(rows,)](
        blocks,
        logits,
        columns,
        output,
        columns.stride(0),
        output.stride(0),
        width,
        candidates.shape[1],
        candidate_block_size,
        output.shape[1],
        triton.next_power_of_2(output.shape[1]),
        sparse,
    )


def _score_and_select(
    query: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    requests: torch.Tensor | None,
    lengths: torch.Tensor,
    candidates: torch.Tensor,
    candidate_block_size: int,
    output: torch.Tensor,
    dense_width: int,
    candidate_write: bool,
    candidates_sorted_unique: bool = False,
    request_repeat: int = 1,
    candidates_identity_when_full: bool = False,
) -> None:
    sparse_width = _sparse_width(candidates.shape[1], candidate_block_size, dense_width)
    width = (
        sparse_width
        if not candidate_write
        and sparse_width > 0
        and os.getenv("VLLM_MUSA_CANDIDATE_SPARSE_MQA", "1") != "0"
        else dense_width
    )
    tile_rows = max(1, (256 << 20) // max(width * 4, 1))
    for start in range(0, query.shape[0], tile_rows):
        stop = min(start + tile_rows, query.shape[0])
        _score_and_select_rows(
            query[start:stop],
            cache,
            weights[start:stop],
            block_table,
            requests[start:stop] if requests is not None else None,
            lengths[start:stop],
            candidates[start:stop],
            candidate_block_size,
            output[start:stop],
            dense_width,
            candidate_write,
            candidates_sorted_unique,
            request_repeat,
            start,
            candidates_identity_when_full,
        )


def try_candidate_indexer(
    query: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    metadata,
    candidates: torch.Tensor,
    candidate_block_size: int,
    output: torch.Tensor,
    max_model_len: int,
    use_fp4_cache: bool,
    candidate_write: bool,
) -> bool:
    """Run the FP8 two-level indexer without gathering the full paged K cache."""
    if (
        use_fp4_cache
        or query.ndim != 3
        or query.dtype != torch.float8_e4m3fn
        or query.shape[1:] != (32, 128)
        or query.stride(-1) != 1
        or cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[2] != 132
        or cache.stride(1) != 132
        or cache.stride(2) != 1
        or cache.stride(0) < cache.shape[1] * 132
        or weights.dtype != torch.float32
        or weights.ndim != 2
        or weights.shape[0] < query.shape[0]
        or weights.shape[1] != query.shape[1]
        or output.ndim != 2
        or output.dtype not in (torch.int32, torch.int64)
        or output.shape[0] < query.shape[0]
        or not 0 < output.shape[1] <= 512
        or output.stride(1) != 1
        or candidates.ndim != 2
        or candidates.dtype not in (torch.int32, torch.int64)
        or candidates.shape[0] < query.shape[0]
        or candidates.shape[1] == 0
        or candidates.shape[1] > 2048
        or candidate_block_size not in (8, 16)
        or max_model_len <= 0
        or (metadata.decode is not None and metadata.decode.requires_padding)
    ):
        return False
    if metadata.num_prefills > 0 and metadata.prefill is not None:
        for chunk in metadata.prefill.chunks:
            start, stop = chunk.token_start, chunk.token_end
            cumulative = getattr(chunk, "local_cu_seq_lens", None)
            if cumulative is None:
                cumulative = chunk.cu_seq_lens
            requests = torch.searchsorted(
                cumulative[1:], chunk.cu_seqlen_ks.contiguous(), right=True
            ).to(torch.int32)
            lengths = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).contiguous()
            _score_and_select(
                query[start:stop],
                cache,
                weights[start:stop],
                chunk.block_table,
                requests,
                lengths,
                candidates[start:stop],
                candidate_block_size,
                output[start:stop],
                min(
                    int(chunk.total_seq_lens),
                    chunk.block_table.shape[1] * cache.shape[1],
                ),
                candidate_write,
                candidates_sorted_unique=True,
            )
    if metadata.num_decodes > 0 and metadata.decode is not None:
        rows = metadata.num_decode_tokens
        decode = metadata.decode
        repeat = decode.seq_lens.shape[-1] if decode.seq_lens.ndim == 2 else 1
        _score_and_select(
            query[:rows],
            cache,
            weights[:rows],
            decode.block_table,
            None,
            decode.seq_lens.reshape(-1)[:rows],
            candidates[:rows],
            candidate_block_size,
            output[:rows],
            min(max_model_len, decode.block_table.shape[1] * cache.shape[1]),
            candidate_write,
            candidates_sorted_unique=True,
            request_repeat=repeat,
            candidates_identity_when_full=True,
        )
    return True
