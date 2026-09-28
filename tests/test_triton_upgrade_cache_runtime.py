# SPDX-License-Identifier: Apache-2.0
"""Packed-cache and DeepSeek compressor checks for the Triton upgrade."""

import math
import struct

import pytest

pytest.importorskip("torchada")
import torch


@pytest.fixture(scope="module", autouse=True)
def musa_device():
    if not getattr(torch.version, "musa", None):
        pytest.skip("requires MUSA PyTorch")
    assert torch.musa.is_available()
    torch.musa.set_device(0)


def _pack_bits(values: list[int], bits: int) -> bytes:
    packed = sum(value << (bits * i) for i, value in enumerate(values))
    return packed.to_bytes(math.ceil(len(values) * bits / 8), "little")


@pytest.mark.parametrize("key_fp8", [False, True])
@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("splits", [1, 4])
def test_turboquant_packed_cache_against_cpu(key_fp8, bits, splits):
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _tq_full_dequant_kv,
        triton_turboquant_decode_attention,
    )

    torch.manual_seed(18025)
    dim, length, page = 64, 17, 16
    centroids = torch.linspace(-1, 1, 2**bits)
    indices = torch.randint(2**bits, (length, dim))
    value_indices = torch.randint(2**bits, (length, dim))
    norms = torch.linspace(0.5, 1.5, length).half()
    fp8_keys = (torch.randn(length, dim) / 3).to(torch.float8_e4m3fn)
    kps = dim if key_fp8 else math.ceil(dim * bits / 8) + 2
    vbytes = math.ceil(dim * bits / 8)
    cache = torch.zeros(2, page, 1, kps + vbytes + 4, dtype=torch.uint8)
    for token in range(length):
        if key_fp8:
            key_bytes = bytes(fp8_keys[token].view(torch.uint8).tolist())
        else:
            key_bytes = _pack_bits(indices[token].tolist(), bits)
            key_bytes += struct.pack("<e", norms[token].item())
        value_bytes = _pack_bits(value_indices[token].tolist(), bits)
        value_bytes += struct.pack("<ee", 0.125, -0.5)
        cache[token // page, token % page, 0] = torch.tensor(
            list(key_bytes + value_bytes), dtype=torch.uint8
        )
    if key_fp8:
        keys = fp8_keys.float()
    else:
        keys = centroids[indices]
        keys = keys / keys.norm(dim=-1, keepdim=True) * norms.float()[:, None]
    values = value_indices.float() * 0.125 - 0.5
    q = torch.randn(1, 2, dim).half()
    gpu_cache = cache.to("musa")
    table = torch.tensor([[0, 1]], device="musa", dtype=torch.int32)
    gpu_centroids = centroids.to("musa")
    out = triton_turboquant_decode_attention(
        q.to("musa"),
        gpu_cache,
        table,
        torch.tensor([length], device="musa", dtype=torch.int32),
        torch.eye(dim, device="musa"),
        gpu_centroids,
        dim**-0.5,
        bits,
        kps,
        bits,
        key_fp8=key_fp8,
        norm_correction=True,
        max_num_kv_splits=splits,
    )
    scores = torch.einsum("bhd,sd->bhs", q.float(), keys) * (dim**-0.5)
    expected = torch.einsum("bhs,sd->bhd", scores.softmax(-1), values)
    torch.testing.assert_close(out.cpu().float(), expected, atol=0.002, rtol=0.005)

    # Also execute the independent bulk-dequant path, not just decode stage 1.
    k_out = torch.empty(1, 1, length, dim, device="musa", dtype=torch.float16)
    v_out = torch.empty_like(k_out)
    _tq_full_dequant_kv[(length, 1)](
        gpu_cache,
        table,
        gpu_centroids,
        k_out,
        v_out,
        *k_out.stride()[:3],
        *v_out.stride()[:3],
        *gpu_cache.stride()[:3],
        table.stride(0),
        HEAD_DIM=dim,
        BLOCK_SIZE=page,
        NUM_KV_HEADS=1,
        MSE_BYTES=math.ceil(dim * bits / 8),
        KPS=kps,
        VQB=bits,
        VAL_DATA_BYTES=vbytes,
        MSE_BITS=bits,
        KEY_FP8=int(key_fp8),
        BLOCK_D=dim,
        NORM_CORRECTION=1,
        FP8_E4B15=0,
    )
    torch.testing.assert_close(k_out.cpu().float()[0, 0], keys, atol=0.001, rtol=0.002)
    torch.testing.assert_close(v_out.cpu().float()[0, 0], values, atol=0, rtol=0)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("position", [3, 7])
def test_deepseek_compress_quant_cache_against_cpu(overlap, position):
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        _fused_kv_compress_norm_rope_insert_indexer_attn,
    )

    torch.manual_seed(18054)
    dim, ratio, page, state_width = 128, 4, 4, 256
    state = torch.randn(2, page, 2 * state_width).float()
    weight = torch.linspace(0.8, 1.2, dim)
    angles = torch.arange(8).float()[:, None] * torch.linspace(0.01, 0.03, 32)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1)
    # Three requests: one real boundary, one negative slot, one non-boundary.
    positions = torch.tensor([position, position, position - 1], dtype=torch.int64)
    state_slots = torch.tensor([0, -1, 0], dtype=torch.int64)
    output_slots = torch.tensor([0, 1, 2], dtype=torch.int64)
    kv = torch.full((1, page * (128 + 4)), 0x5A, device="musa", dtype=torch.uint8)
    gpu_state = state.to("musa")
    _fused_kv_compress_norm_rope_insert_indexer_attn[(3,)](
        gpu_state,
        gpu_state.stride(0),
        gpu_state.stride(1),
        torch.zeros(3, device="musa", dtype=torch.int32),
        positions.to("musa"),
        state_slots.to("musa"),
        torch.tensor([[0, 1]], device="musa", dtype=torch.int32),
        2,
        page,
        weight.to("musa"),
        1e-6,
        cos_sin.to("musa"),
        cos_sin.stride(0),
        kv,
        output_slots.to("musa"),
        page,
        HEAD_SIZE=dim,
        TRITON_BLOCK_SIZE=dim,
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=64,
        FP8_MAX=448.0,
        QUANT_BLOCK=128,
        TOKEN_STRIDE=128,
        SCALE_DIM=4,
        KV_BLOCK_STRIDE=kv.stride(0),
        num_warps=1,
    )
    gathered, scores = [], []
    for row, pos in enumerate(
        range(position - (1 + overlap) * ratio + 1, position + 1)
    ):
        offset = dim if row >= ratio else 0
        if pos < 0:
            gathered.append(torch.zeros(dim))
            scores.append(torch.full((dim,), -float("inf")))
        else:
            gathered.append(state[pos // page, pos % page, offset : offset + dim])
            scores.append(
                state[
                    pos // page,
                    pos % page,
                    state_width + offset : state_width + offset + dim,
                ]
            )
    compressed = (torch.stack(gathered) * torch.stack(scores).softmax(0)).sum(0)
    normed = compressed * torch.rsqrt(compressed.square().mean() + 1e-6) * weight
    rotated = normed.clone()
    even, odd = normed[64::2], normed[65::2]
    cos, sin = cos_sin[position // ratio * ratio].chunk(2)
    rotated[64::2] = even * cos - odd * sin
    rotated[65::2] = odd * cos + even * sin
    rounded = rotated.bfloat16().float()
    scale = torch.exp2(
        torch.ceil(torch.log2(rounded.abs().max().clamp_min(1e-4) / 448))
    )
    expected = (rounded / scale).to(torch.float8_e4m3fn).float()
    raw = kv.cpu()[0]
    actual = raw[:128].view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        raw[page * 128 : page * 128 + 4].view(torch.float32)[0], scale, atol=0, rtol=0
    )
    assert torch.all(raw[128 : page * 128] == 0x5A)
    assert torch.all(raw[page * 128 + 4 :] == 0x5A)
