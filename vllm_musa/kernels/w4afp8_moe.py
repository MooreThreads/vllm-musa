from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _initialize_ragged_indices(
    indices_ptr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(indices_ptr + rows, tl.where(rows % 128 == 0, 0, -1), rows < ROWS)


@triton.jit
def _assign_expert_rows(
    topk_ids_ptr,
    expert_map_ptr,
    counts_ptr,
    local_ids_ptr,
    slots_ptr,
    ROUTES: tl.constexpr,
    TOTAL_EXPERTS: tl.constexpr,
    LOCAL_EXPERTS: tl.constexpr,
    HAS_MAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    routes = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    global_ids = tl.load(topk_ids_ptr + routes, routes < ROUTES, other=-1).to(tl.int32)
    valid = (routes < ROUTES) & (global_ids >= 0) & (global_ids < TOTAL_EXPERTS)
    if HAS_MAP:
        local_ids = tl.load(expert_map_ptr + global_ids, valid, other=-1)
    else:
        local_ids = global_ids
    valid = valid & (local_ids >= 0) & (local_ids < LOCAL_EXPERTS)
    slots = tl.atomic_add(counts_ptr + local_ids, 1, mask=valid, sem="relaxed")
    tl.store(local_ids_ptr + routes, tl.where(valid, local_ids, -1), routes < ROUTES)
    tl.store(slots_ptr + routes, tl.where(valid, slots, -1), routes < ROUTES)


@triton.jit
def _scatter_quantized_rows(
    input_ptr,
    local_ids_ptr,
    slots_ptr,
    offsets_ptr,
    quantized_ptr,
    scales_ptr,
    indices_ptr,
    TOPK: tl.constexpr,
    HIDDEN: tl.constexpr,
    MASKED: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    route = tl.program_id(0).to(tl.int64)
    expert = tl.load(local_ids_ptr + route)
    slot = tl.load(slots_ptr + route)
    if expert >= 0:
        if MASKED:
            destination = expert * CAPACITY + slot
        else:
            destination = tl.load(offsets_ptr + expert) + slot
        destination = destination.to(tl.int64)
        columns = tl.arange(0, BLOCK)
        values = tl.load(
            input_ptr + (route // TOPK) * HIDDEN + columns, columns < HIDDEN, other=0
        ).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(values), 0), 1.0e-4) / 448.0
        quantized = tl.maximum(tl.minimum(values / scale, 448.0), -448.0)
        tl.store(
            quantized_ptr + destination * HIDDEN + columns,
            quantized.to(quantized_ptr.dtype.element_ty),
            columns < HIDDEN,
        )
        tl.store(scales_ptr + destination, scale)
        if not MASKED:
            tl.store(indices_ptr + destination, expert)


@triton.jit
def _swiglu_quantized_rows(
    gateup_ptr,
    quantized_ptr,
    scales_ptr,
    counts_ptr,
    INTERMEDIATE: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    active = True
    if CAPACITY > 0:
        active = row % CAPACITY < tl.load(counts_ptr + row // CAPACITY)
    if active:
        columns = tl.arange(0, BLOCK)
        gate = tl.load(
            gateup_ptr + row * 2 * INTERMEDIATE + columns,
            columns < INTERMEDIATE,
            other=0,
        ).to(tl.float32)
        up = tl.load(
            gateup_ptr + row * 2 * INTERMEDIATE + INTERMEDIATE + columns,
            columns < INTERMEDIATE,
            other=0,
        ).to(tl.float32)
        if LIMIT is not None:
            gate = tl.minimum(gate, LIMIT)
            up = tl.maximum(tl.minimum(up, LIMIT), -LIMIT)
        values = (
            (gate / (1.0 + tl.exp(-gate)) * up)
            .to(gateup_ptr.dtype.element_ty)
            .to(tl.float32)
        )
        values = tl.where(columns < INTERMEDIATE, values, 0.0)
        scale = tl.maximum(tl.max(tl.abs(values), 0), 1.0e-4) / 448.0
        quantized = tl.maximum(tl.minimum(values / scale, 448.0), -448.0)
        tl.store(
            quantized_ptr + row * INTERMEDIATE + columns,
            quantized.to(quantized_ptr.dtype.element_ty),
            columns < INTERMEDIATE,
        )
        tl.store(scales_ptr + row, scale)


@triton.jit
def _combine_expert_rows(
    expert_output_ptr,
    local_ids_ptr,
    slots_ptr,
    offsets_ptr,
    weights_ptr,
    output_ptr,
    TOPK: tl.constexpr,
    HIDDEN: tl.constexpr,
    MASKED: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    accumulator = tl.full((BLOCK,), 0, tl.float32)
    for selection in range(TOPK):
        route = token * TOPK + selection
        expert = tl.load(local_ids_ptr + route)
        slot = tl.load(slots_ptr + route)
        if expert >= 0:
            if MASKED:
                source = expert * CAPACITY + slot
            else:
                source = tl.load(offsets_ptr + expert) + slot
            source = source.to(tl.int64)
            values = tl.load(
                expert_output_ptr + source * HIDDEN + columns, columns < HIDDEN, other=0
            ).to(tl.float32)
            weight = tl.load(weights_ptr + route).to(tl.float32)
            accumulator += values * weight
    tl.store(output_ptr + token * HIDDEN + columns, accumulator, columns < HIDDEN)


def w4afp8_moe(
    hidden_states: torch.Tensor,
    gateup_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gateup_scale: torch.Tensor,
    down_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
    *,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Execute routed W4AFP8 experts without expanding packed INT4 weights."""
    if hidden_states.ndim != 2 or gateup_weight.ndim != 3:
        raise ValueError("W4AFP8 expects hidden states [T,H] and weights [E,N,K/2]")
    tokens, hidden = hidden_states.shape
    experts, doubled_intermediate, packed_hidden = gateup_weight.shape
    intermediate = doubled_intermediate // 2
    if (
        topk_ids.ndim != 2
        or topk_ids.shape[0] != tokens
        or topk_weights.shape != topk_ids.shape
        or topk_ids.shape[1] == 0
    ):
        raise ValueError(
            "W4AFP8 routing IDs and weights must have matching [T,topk] shapes"
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("W4AFP8 routing IDs must be INT32 or INT64")
    if expert_map is not None and (
        expert_map.ndim != 1 or expert_map.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("W4AFP8 expert map must be a one-dimensional integer tensor")
    topk = topk_ids.shape[1]
    if (
        experts == 0
        or hidden == 0
        or intermediate == 0
        or doubled_intermediate % 2
        or hidden != packed_hidden * 2
        or hidden % 128
        or intermediate % 128
    ):
        raise ValueError("W4AFP8 requires packed INT4 K/2 and group128 dimensions")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("MUSA W4AFP8 currently requires BF16 hidden states")
    if gateup_weight.dtype != torch.int8 or down_weight.dtype != torch.int8:
        raise ValueError("W4AFP8 weights must be signed INT4 packed into INT8")
    if down_weight.shape != (experts, hidden, intermediate // 2):
        raise ValueError("W4AFP8 Down weight shape does not match GateUp")
    for scale, expected in (
        (gateup_scale, (experts, doubled_intermediate, hidden // 128)),
        (down_scale, (experts, hidden, intermediate // 128)),
    ):
        if scale.shape != expected or scale.dtype != torch.bfloat16:
            raise ValueError("W4AFP8 weight scales must be BF16 [E,N,K/128]")
    if tokens == 0:
        return torch.empty_like(hidden_states)
    import mate.gemm

    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    if expert_map is not None:
        expert_map = expert_map.contiguous()
    device = hidden_states.device
    routes = tokens * topk
    masked = routes <= 128
    small_masked = masked and routes <= 32
    capacity = triton.next_power_of_2(routes) if small_masked else 128
    active_expert_upper = min(experts, routes)
    rows = (
        experts * capacity
        if masked
        else triton.cdiv(routes + active_expert_upper * 127, 128) * 128
    )
    counts = torch.zeros(experts, dtype=torch.int32, device=device)
    local_ids = torch.empty(routes, dtype=torch.int32, device=device)
    slots = torch.empty_like(local_ids)
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
    indices = torch.empty(rows, dtype=torch.int32, device=device)
    total_experts = expert_map.numel() if expert_map is not None else experts
    _assign_expert_rows[(triton.cdiv(routes, 256),)](
        topk_ids,
        expert_map if expert_map is not None else counts,
        counts,
        local_ids,
        slots,
        routes,
        total_experts,
        experts,
        expert_map is not None,
        256,
    )
    if not masked:
        _initialize_ragged_indices[(triton.cdiv(rows, 256),)](indices, rows, 256)
        torch.cumsum(
            ((counts + 127) // 128) * 128, dim=0, dtype=torch.int32, out=offsets[1:]
        )
    quantized = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=device)
    if not masked:
        quantized.view(torch.uint8).zero_()
    scale_allocator = torch.empty if masked else torch.ones
    activation_scales = scale_allocator(rows, dtype=torch.float32, device=device)
    _scatter_quantized_rows[(routes,)](
        hidden_states.contiguous(),
        local_ids,
        slots,
        offsets,
        quantized,
        activation_scales,
        indices,
        topk,
        hidden,
        masked,
        capacity,
        triton.next_power_of_2(hidden),
        num_warps=8,
    )

    def gemm(
        activation: torch.Tensor,
        scales: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if masked:
            mate.gemm.masked_moe_gemm_mixed_dtype(
                (
                    activation.view(experts, capacity, -1),
                    scales.view(experts, capacity, 1),
                ),
                (weight, weight_scales),
                counts,
                output.view(experts, capacity, -1),
                expect_tokens=routes,
                enable_overlap=False,
                signal=None,
                mixed_dtype=mate.gemm.GemmMixedDType.S4FP8,
                backend="auto",
                a_quant_recipe=(1, -1),
                b_quant_recipe=(1, 128),
            )
        else:
            mate.gemm.ragged_moe_gemm_mixed_dtype(
                (activation, scales.view(-1, 1)),
                (weight, weight_scales),
                indices,
                output,
                alignment_m=128,
                mixed_dtype=mate.gemm.GemmMixedDType.S4FP8,
                backend="mubin",
                a_quant_recipe=(1, -1),
                b_quant_recipe=(1, 128),
            )

    gateup_allocator = torch.empty if masked else torch.zeros
    gateup = gateup_allocator(
        (rows, doubled_intermediate), dtype=torch.bfloat16, device=device
    )
    gemm(quantized, activation_scales, gateup_weight, gateup_scale, gateup)
    intermediate_quantized = torch.empty(
        (rows, intermediate), dtype=torch.float8_e4m3fn, device=device
    )
    intermediate_scales = torch.empty(rows, dtype=torch.float32, device=device)
    _swiglu_quantized_rows[(rows,)](
        gateup,
        intermediate_quantized,
        intermediate_scales,
        counts,
        intermediate,
        swiglu_limit,
        triton.next_power_of_2(intermediate),
        capacity if masked else 0,
        num_warps=8,
    )
    expert_output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
    gemm(
        intermediate_quantized,
        intermediate_scales,
        down_weight,
        down_scale,
        expert_output,
    )
    output = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    _combine_expert_rows[(tokens, triton.cdiv(hidden, 256))](
        expert_output,
        local_ids,
        slots,
        offsets,
        topk_weights,
        output,
        topk,
        hidden,
        masked,
        capacity,
        256,
    )
    return output
