# SPDX-License-Identifier: Apache-2.0
"""Small-token FP8 MoE on FP8 dots, reading each routed expert's weights once.

Consecutive decode or verify tokens share many routed experts. Each program owns
one distinct expert of the batch, gathers the routed rows sent to it (at most 16
rows of the MMA tile) and multiplies them by that expert's weight tile, one
32-wide scale group per FP8 dot. The gate/up projection quantizes the hidden
states per 32-wide group in-kernel and splits K; a second kernel sums the K
splits, rounds gate and up to BF16 and quantizes their clamped SwiGLU for the
down projection. The down projection writes one partial per routed row and a
last kernel sums each token's routed rows and rounds once.
"""

from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton

_FP8_MAX = 448.0
_QUANT_EPS = 1e-10
_MAX_ROUTED = 64
# Below four tokens the routed-row GEMVs win unless the router supplies the route map
# and the quantized activation.
MIN_TOKENS = 4
_ENABLED = os.environ.get("VLLM_MUSA_FP8_MOE_MMA", "1") != "0"

# (SPLIT, BLOCK_N, num_warps) for the gate/up projection and (BLOCK_N, num_warps)
# for the down projection, measured on MTT S5000 for the DeepSeek-V4.1 TP8 experts.
_W1_CONFIG = (8, 64, 4)
_W2_CONFIG = (64, 4)


@triton.jit
def _moe_route_map_kernel(
    ids_ptr,
    map_ptr,
    R,
    E,
    MAX_R: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Distinct routed experts in first-use order.

    ``map[u] = [expert, rows, row_0, ...]`` with the routed rows of expert
    ``u`` packed from ``row_0``; ``map[MAX_R] = [count]``.
    """
    offs_r = tl.arange(0, MAX_R)
    ids = tl.load(ids_ptr + offs_r, mask=offs_r < R, other=-1)
    valid = (offs_r < R) & (ids >= 0) & (ids < E)
    eq = (ids[:, None] == ids[None, :]) & valid[:, None] & valid[None, :]
    earlier = offs_r[None, :] < offs_r[:, None]
    first = valid & (tl.sum((eq & earlier).to(tl.int32), axis=1) == 0)
    uniq = tl.cumsum(first.to(tl.int32), axis=0) - 1
    # rank of each row among the rows of its expert
    rank = tl.sum((eq & earlier).to(tl.int32), axis=1)
    # the distinct-expert index of each row = index of the first row with its expert
    owner = tl.sum(
        tl.where(
            eq & (earlier | (offs_r[None, :] == offs_r[:, None])) & first[None, :],
            uniq[None, :],
            0,
        ),
        axis=1,
    )
    width = 2 + BLOCK_M
    tl.store(map_ptr + owner * width + 2 + rank, offs_r, mask=valid & (rank < BLOCK_M))
    tl.store(map_ptr + uniq * width, ids, mask=first)
    count = tl.sum(eq.to(tl.int32), axis=1)
    tl.store(map_ptr + uniq * width + 1, count, mask=first)
    tl.store(map_ptr + MAX_R * width, tl.sum(first.to(tl.int32), axis=0))


@triton.jit
def _expert_rows(
    map_ptr, u, TOPK: tl.constexpr, MAX_R: tl.constexpr, BLOCK_M: tl.constexpr
):
    """(expert, distinct experts, routed rows, tokens, row valid) of distinct
    expert ``u``."""
    width = 2 + BLOCK_M
    n_uniq = tl.load(map_ptr + MAX_R * width)
    base = map_ptr + tl.minimum(u, MAX_R - 1) * width
    expert = tl.load(base)
    count = tl.load(base + 1)
    offs_i = tl.arange(0, BLOCK_M)
    row_valid = (offs_i < count) & (u < n_uniq)
    row = tl.load(base + 2 + offs_i, mask=row_valid, other=0)
    return expert, n_uniq, row, row // TOPK, row_valid


@triton.jit
def _moe_w1_mma_kernel(
    x_ptr,
    w_ptr,
    ws_ptr,
    map_ptr,
    part_ptr,
    R,
    E,
    N,
    stride_xm,
    stride_we,
    stride_wn,
    stride_wse,
    stride_wsn,
    TOPK: tl.constexpr,
    MAX_R: tl.constexpr,
    K_PER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_EPS: tl.constexpr,
):
    u = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    expert, n_uniq, row, tok, row_valid = _expert_rows(map_ptr, u, TOPK, MAX_R, 16)
    if u < n_uniq:
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, 32)
        e64 = expert.to(tl.int64)
        x_rows = x_ptr + tok[:, None].to(tl.int64) * stride_xm + offs_k[None, :]
        w_cols = (
            w_ptr
            + e64 * stride_we
            + offs_n[None, :].to(tl.int64) * stride_wn
            + offs_k[:, None]
        )
        ws_cols = ws_ptr + e64 * stride_wse + (offs_n // 32).to(tl.int64) * stride_wsn
        acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        k_start = pid_k * K_PER
        for kk in range(k_start, k_start + K_PER, 32):
            x = tl.load(x_rows + kk, mask=row_valid[:, None], other=0.0).to(tl.float32)
            a_s = tl.maximum(tl.max(tl.abs(x), axis=1), QUANT_EPS) / FP8_MAX
            a = tl.clamp(x / a_s[:, None], -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            b = tl.load(w_cols + kk)
            b_s = tl.load(ws_cols + kk // 32)
            acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        parts = part_ptr + (pid_k * R + row[:, None]).to(tl.int64) * N + offs_n[None, :]
        tl.store(parts, acc, mask=row_valid[:, None])


@triton.jit
def _moe_w1_mma_fp8_kernel(
    xq_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    map_ptr,
    part_ptr,
    R,
    E,
    N,
    stride_xm,
    stride_xsm,
    stride_we,
    stride_wn,
    stride_wse,
    stride_wsn,
    TOPK: tl.constexpr,
    MAX_R: tl.constexpr,
    K_PER: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``_moe_w1_mma_kernel`` on an activation already quantized per 32-wide group."""
    u = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    expert, n_uniq, row, tok, row_valid = _expert_rows(map_ptr, u, TOPK, MAX_R, 16)
    if u < n_uniq:
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, 32)
        e64 = expert.to(tl.int64)
        x_rows = xq_ptr + tok[:, None].to(tl.int64) * stride_xm + offs_k[None, :]
        xs_rows = xs_ptr + tok.to(tl.int64) * stride_xsm
        w_cols = (
            w_ptr
            + e64 * stride_we
            + offs_n[None, :].to(tl.int64) * stride_wn
            + offs_k[:, None]
        )
        ws_cols = ws_ptr + e64 * stride_wse + (offs_n // 32).to(tl.int64) * stride_wsn
        acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        k_start = pid_k * K_PER
        for kk in range(k_start, k_start + K_PER, 32):
            a = tl.load(x_rows + kk, mask=row_valid[:, None], other=0.0)
            a_s = tl.load(xs_rows + kk // 32, mask=row_valid, other=0.0)
            b = tl.load(w_cols + kk)
            b_s = tl.load(ws_cols + kk // 32)
            acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        parts = part_ptr + (pid_k * R + row[:, None]).to(tl.int64) * N + offs_n[None, :]
        tl.store(parts, acc, mask=row_valid[:, None])


@triton.jit
def _moe_w1_swiglu_quant_kernel(
    part_ptr,
    aq_ptr,
    as_ptr,
    R,
    N,
    INTER: tl.constexpr,
    SPLIT: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_EPS: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    offs = c * 32 + tl.arange(0, 32)
    gate = tl.zeros((32,), tl.float32)
    up = tl.zeros((32,), tl.float32)
    for s in tl.static_range(SPLIT):
        base = part_ptr + (s * R + r).to(tl.int64) * N
        gate += tl.load(base + offs)
        up += tl.load(base + INTER + offs)
    # [gate | up] are BF16 outputs; the SwiGLU rounds like silu_and_mul_with_clamp.
    gate = tl.minimum(gate.to(tl.bfloat16).to(tl.float32), SWIGLU_LIMIT)
    up = up.to(tl.bfloat16).to(tl.float32)
    up = tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    v = (silu * up).to(tl.bfloat16).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(v), axis=0), QUANT_EPS) / FP8_MAX
    q = tl.clamp(v / s, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    tl.store(aq_ptr + r.to(tl.int64) * INTER + offs, q)
    tl.store(as_ptr + r * (INTER // 32) + c, s)


@triton.jit
def _moe_w2_mma_kernel(
    aq_ptr,
    as_ptr,
    w_ptr,
    ws_ptr,
    map_ptr,
    rw_ptr,
    part_ptr,
    R,
    E,
    N,
    stride_we,
    stride_wn,
    stride_wse,
    stride_wsn,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    u = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert, n_uniq, row, tok, row_valid = _expert_rows(map_ptr, u, TOPK, MAX_R, 16)
    if u < n_uniq:
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, 32)
        e64 = expert.to(tl.int64)
        a_rows = aq_ptr + row[:, None].to(tl.int64) * K + offs_k[None, :]
        w_cols = (
            w_ptr
            + e64 * stride_we
            + offs_n[None, :].to(tl.int64) * stride_wn
            + offs_k[:, None]
        )
        ws_cols = ws_ptr + e64 * stride_wse + (offs_n // 32).to(tl.int64) * stride_wsn
        acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        # A runtime loop: the unrolled one is slower on MUSA.
        for g in range(0, K // 32):
            a = tl.load(a_rows + g * 32, mask=row_valid[:, None], other=0.0)
            a_s = tl.load(as_ptr + row * (K // 32) + g, mask=row_valid, other=0.0)
            b = tl.load(w_cols + g * 32)
            b_s = tl.load(ws_cols + g)
            acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        rw = tl.load(rw_ptr + row, mask=row_valid, other=0.0)
        parts = part_ptr + row[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(parts, acc * rw[:, None], mask=row_valid[:, None])


@triton.jit
def _moe_w2_slot_sum_kernel(
    part_ptr,
    ids_ptr,
    out_ptr,
    N,
    E,
    stride_om,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(TOPK):
        r = t * TOPK + s
        e = tl.load(ids_ptr + r)
        ok = (e >= 0) & (e < E)
        acc += tl.load(part_ptr + r.to(tl.int64) * N + offs, mask=mask & ok, other=0.0)
    tl.store(
        out_ptr + t * stride_om + offs, acc.to(out_ptr.dtype.element_ty), mask=mask
    )


def fp8_moe_mma_supported(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    """Shapes this path handles: 32x32 blocks, at most 64 routed rows, K and N tiled."""
    if not _ENABLED or topk_ids.dim() != 2:
        return False
    tokens, top_k = topk_ids.shape
    e, n1, h = w1.shape
    inter = n1 // 2
    split, block_n1, _ = _W1_CONFIG
    block_n2, _ = _W2_CONFIG
    return (
        0 < tokens * top_k <= _MAX_ROUTED
        and tokens <= 16
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.stride(-1) == 1
        and w2.shape == (e, h, inter)
        and tuple(w1_scale.shape) == (e, n1 // 32, h // 32)
        and tuple(w2_scale.shape) == (e, h // 32, inter // 32)
        and w1_scale.dtype == torch.float32
        and w2_scale.dtype == torch.float32
        and w1.is_contiguous()
        and w2.is_contiguous()
        and w1_scale.is_contiguous()
        and w2_scale.is_contiguous()
        and inter % 32 == 0
        and n1 % block_n1 == 0
        and h % (split * 32) == 0
        and h % block_n2 == 0
    )


def fp8_moe_mma(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """FP8 MoE with the clamped SwiGLU for at most 64 routed rows."""
    tokens, top_k = topk_ids.shape
    e, n1, h = w1.shape
    inter = n1 // 2
    rows = tokens * top_k
    device = hidden_states.device
    ids = topk_ids.reshape(-1).to(torch.int32).contiguous()
    weights = topk_weights.reshape(-1).to(torch.float32).contiguous()
    split, block_n1, warps1 = _W1_CONFIG
    block_n2, warps2 = _W2_CONFIG

    # The MUSA sqrt-softplus router writes this map along with the ids.
    route_map = getattr(topk_ids, "musa_route_map", None)
    if route_map is None:
        route_map = torch.empty(
            ((_MAX_ROUTED + 1) * (2 + 16),), dtype=torch.int32, device=device
        )
        _moe_route_map_kernel[(1,)](
            ids, route_map, rows, e, MAX_R=_MAX_ROUTED, BLOCK_M=16, num_warps=4
        )
    part1 = torch.empty((split, rows, n1), dtype=torch.float32, device=device)
    # The router may have quantized this activation already.
    route_x = getattr(topk_ids, "musa_route_x", None)
    if route_x is not None and route_x[0] is hidden_states:
        xq, xs = route_x[1], route_x[2]
        _moe_w1_mma_fp8_kernel[(rows, n1 // block_n1, split)](
            xq,
            xs,
            w1,
            w1_scale,
            route_map,
            part1,
            rows,
            e,
            n1,
            xq.stride(0),
            xs.stride(0),
            w1.stride(0),
            w1.stride(1),
            w1_scale.stride(0),
            w1_scale.stride(1),
            TOPK=top_k,
            MAX_R=_MAX_ROUTED,
            K_PER=h // split,
            BLOCK_N=block_n1,
            num_warps=warps1,
            num_stages=1,
        )
    else:
        _moe_w1_mma_kernel[(rows, n1 // block_n1, split)](
            hidden_states,
            w1,
            w1_scale,
            route_map,
            part1,
            rows,
            e,
            n1,
            hidden_states.stride(0),
            w1.stride(0),
            w1.stride(1),
            w1_scale.stride(0),
            w1_scale.stride(1),
            TOPK=top_k,
            MAX_R=_MAX_ROUTED,
            K_PER=h // split,
            BLOCK_N=block_n1,
            FP8_MAX=_FP8_MAX,
            QUANT_EPS=_QUANT_EPS,
            num_warps=warps1,
            num_stages=1,
        )
    aq = torch.empty((rows, inter), dtype=torch.float8_e4m3fn, device=device)
    a_scale = torch.empty((rows, inter // 32), dtype=torch.float32, device=device)
    _moe_w1_swiglu_quant_kernel[(rows, inter // 32)](
        part1,
        aq,
        a_scale,
        rows,
        n1,
        INTER=inter,
        SPLIT=split,
        SWIGLU_LIMIT=float(swiglu_limit),
        FP8_MAX=_FP8_MAX,
        QUANT_EPS=_QUANT_EPS,
        num_warps=1,
    )
    part2 = torch.empty((rows, h), dtype=torch.float32, device=device)
    _moe_w2_mma_kernel[(rows, h // block_n2)](
        aq,
        a_scale,
        w2,
        w2_scale,
        route_map,
        weights,
        part2,
        rows,
        e,
        h,
        w2.stride(0),
        w2.stride(1),
        w2_scale.stride(0),
        w2_scale.stride(1),
        K=inter,
        TOPK=top_k,
        MAX_R=_MAX_ROUTED,
        BLOCK_N=block_n2,
        num_warps=warps2,
    )
    out = torch.empty_like(hidden_states)
    _moe_w2_slot_sum_kernel[(tokens, triton.cdiv(h, 1024))](
        part2,
        ids,
        out,
        h,
        e,
        out.stride(0),
        TOPK=top_k,
        BLOCK=1024,
        num_warps=4,
    )
    return out
