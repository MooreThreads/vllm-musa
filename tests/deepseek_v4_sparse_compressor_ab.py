# SPDX-License-Identifier: Apache-2.0
"""MUSA correctness and eager event-time A/B for the 512-d sparse compressor."""

from __future__ import annotations

import argparse
import json

import torch
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_sparse_attn,
)

from vllm_musa import _custom_ops as musa_ops
from vllm_musa.kernels.deepseek_v4_sparse_compressor import _MAX_DECODE_ROWS

HEAD_DIM = 512
ROPE_DIM = 64
STATE_BLOCK_SIZE = 4
STATE_WIDTH = 1024
COMPRESS_RATIO = 4
TOKEN_STRIDE = 576
SCALE_DIM = 8
KV_TOKEN_BYTES = TOKEN_STRIDE + SCALE_DIM


def make_inputs(
    capture_rows: int,
    active_rows: int,
    kv_block_size: int,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    if not 0 < active_rows <= capture_rows <= _MAX_DECODE_ROWS:
        raise ValueError(
            f"expected 0 < active_rows <= capture_rows <= {_MAX_DECODE_ROWS}"
        )
    torch.manual_seed(20260916)
    device = torch.device("musa")
    max_position = 4 * (active_rows + 16) - 1
    logical_state_blocks = (max_position + 1) // STATE_BLOCK_SIZE
    num_state_blocks = capture_rows * logical_state_blocks

    state_cache = torch.randn(
        (num_state_blocks, STATE_BLOCK_SIZE, 2 * STATE_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    block_table = torch.arange(
        num_state_blocks, dtype=torch.int32, device=device
    ).reshape(capture_rows, logical_state_blocks)
    token_to_req = torch.arange(capture_rows, dtype=torch.int32, device=device)
    positions = torch.full((capture_rows,), 3, dtype=torch.int64, device=device)
    positions[:active_rows] = (
        torch.arange(active_rows, dtype=torch.int64, device=device) * 4 + 31
    )

    state_slots = torch.full((capture_rows,), -1, dtype=torch.int64, device=device)
    active_positions = positions[:active_rows]
    active_pages = block_table[
        torch.arange(active_rows, device=device),
        active_positions // STATE_BLOCK_SIZE,
    ].to(torch.int64)
    state_slots[:active_rows] = (
        active_pages * STATE_BLOCK_SIZE + active_positions % STATE_BLOCK_SIZE
    )

    rms_weight = (
        torch.randn((HEAD_DIM,), dtype=torch.float32, device=device) * 0.1 + 1.0
    ).to(weight_dtype)
    cos_sin = torch.randn(
        (max_position + 1, ROPE_DIM), dtype=torch.float32, device=device
    )
    num_kv_blocks = (active_rows + kv_block_size - 1) // kv_block_size + 1
    kv_cache = torch.full(
        (num_kv_blocks, kv_block_size, KV_TOKEN_BYTES),
        0xCD,
        dtype=torch.uint8,
        device=device,
    )
    kv_slots = torch.full((capture_rows,), -1, dtype=torch.int64, device=device)
    kv_slots[:active_rows] = torch.arange(active_rows, dtype=torch.int64, device=device)
    return (
        state_cache,
        token_to_req,
        positions,
        state_slots,
        block_table,
        rms_weight,
        cos_sin,
        kv_cache,
        kv_slots,
    )


def run_triton(inputs: tuple[torch.Tensor, ...]) -> None:
    (
        state_cache,
        token_to_req,
        positions,
        state_slots,
        block_table,
        rms_weight,
        cos_sin,
        kv_cache,
        kv_slots,
    ) = inputs
    _fused_kv_compress_norm_rope_insert_sparse_attn[(state_slots.numel(),)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req,
        positions,
        state_slots,
        block_table,
        block_table.stride(0),
        STATE_BLOCK_SIZE,
        rms_weight,
        1.0e-6,
        cos_sin,
        cos_sin.stride(0),
        kv_cache,
        kv_slots,
        kv_cache.shape[1],
        HEAD_SIZE=HEAD_DIM,
        TRITON_BLOCK_SIZE=HEAD_DIM,
        STATE_WIDTH=STATE_WIDTH,
        COMPRESS_RATIO=COMPRESS_RATIO,
        OVERLAP=True,
        ROPE_HEAD_DIM=ROPE_DIM,
        FP8_MAX=448.0,
        QUANT_BLOCK=64,
        TOKEN_STRIDE=TOKEN_STRIDE,
        SCALE_DIM=SCALE_DIM,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        SANITIZE_CACHE_NANS=False,
        num_warps=4,
    )


def run_native(inputs: tuple[torch.Tensor, ...]) -> None:
    (
        state_cache,
        token_to_req,
        positions,
        state_slots,
        block_table,
        rms_weight,
        cos_sin,
        kv_cache,
        kv_slots,
    ) = inputs
    musa_ops.deepseek_v4_sparse_compress_cache(
        state_cache,
        token_to_req,
        positions,
        state_slots,
        block_table,
        rms_weight,
        cos_sin,
        kv_cache,
        kv_slots,
        1.0e-6,
        STATE_BLOCK_SIZE,
        STATE_WIDTH,
        kv_cache.shape[1],
        COMPRESS_RATIO,
        TOKEN_STRIDE,
        SCALE_DIM,
        64,
    )


def clone_with_cache(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    cloned = list(inputs)
    cloned[7] = inputs[7].clone()
    return tuple(cloned)


def event_time_us(fn, inputs, warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        fn(inputs)
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn(inputs)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-rows", type=int, default=5)
    parser.add_argument("--active-rows", type=int, default=1)
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument(
        "--weight-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    weight_dtype = getattr(torch, args.weight_dtype)
    base_inputs = make_inputs(
        args.capture_rows, args.active_rows, args.kv_block_size, weight_dtype
    )
    triton_inputs = clone_with_cache(base_inputs)
    native_inputs = clone_with_cache(base_inputs)
    run_triton(triton_inputs)
    run_native(native_inputs)
    torch.musa.synchronize()

    triton_cache = triton_inputs[7]
    native_cache = native_inputs[7]
    cache_equal = bool(torch.equal(triton_cache, native_cache))
    result = {
        "capture_rows": args.capture_rows,
        "active_rows": args.active_rows,
        "kv_block_size": args.kv_block_size,
        "weight_dtype": args.weight_dtype,
        "cache_equal": cache_equal,
        "cache_mismatch_count": int(
            torch.count_nonzero(triton_cache != native_cache).item()
        ),
        "native_finite": bool(torch.isfinite(native_cache.to(torch.float32)).all()),
        "native_not_poison": bool((native_cache != 0).any()),
    }
    if not cache_equal:
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(1)

    triton_us = event_time_us(
        run_triton, clone_with_cache(base_inputs), args.warmups, args.iterations
    )
    native_us = event_time_us(
        run_native, clone_with_cache(base_inputs), args.warmups, args.iterations
    )
    result.update(
        {
            "triton_event_us": triton_us,
            "native_event_us": native_us,
            "speedup": triton_us / native_us if native_us else None,
        }
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
