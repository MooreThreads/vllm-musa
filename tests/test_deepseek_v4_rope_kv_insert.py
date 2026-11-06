from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None

requires_mp31 = pytest.mark.skipif(
    torch is None
    or not hasattr(torch, "musa")
    or not torch.musa.is_available()
    or tuple(torch.musa.get_device_capability()) != (3, 1),
    reason="requires an mp31 MUSA device",
)


def _rope_cache(max_pos: int) -> torch.Tensor:
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = torch.arange(max_pos, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=1).to("musa")


@requires_mp31
@pytest.mark.parametrize("tokens", [1, 5, 40, 161, 1030, 4096])
@pytest.mark.parametrize("heads", [7, 8])
@pytest.mark.parametrize("padded", [64, 0])
@pytest.mark.parametrize("q_norm", [False, True])
def test_triton_kernel_matches_the_native_op(tokens, heads, padded, q_norm) -> None:
    import vllm_musa._custom_ops  # noqa: F401
    from vllm_musa.kernels import deepseek_v4_rope_kv_insert as rk

    torch.manual_seed(tokens + padded)
    block_size, num_blocks = 64, 80
    q = (torch.randn(tokens, heads, 512) * 4).to(torch.bfloat16)
    q[1::5] = 0  # a zero query, as the DSpark context insert passes
    q = q.to("musa")
    kv = (torch.randn(tokens, 512) * 4).to(torch.bfloat16)
    kv[::3, :64] *= 1e-6  # groups under the 1e-4 amax floor
    kv = kv.to("musa")
    pos_dtype = torch.int32 if tokens % 2 == 0 else torch.int64
    positions = torch.randint(0, 8192, (tokens,), dtype=pos_dtype).to("musa")
    slots = torch.randperm(num_blocks * block_size)[:tokens]
    slots[::7] = -1  # padding slots skip the insert
    num_slots = tokens - tokens // 4  # fewer slots than rows
    slot_mapping = slots[:num_slots].to(torch.int32 if tokens % 2 else torch.int64)
    slot_mapping = slot_mapping.to("musa")
    cache = _rope_cache(8192)
    kv_cache = torch.randint(0, 256, (num_blocks, block_size, 584), dtype=torch.uint8)
    kv_cache = kv_cache.to("musa")
    assert rk.rope_kv_insert_supported(
        q, kv, kv_cache, slot_mapping, positions, cache, block_size, padded
    )

    native_q, native_cache = q.clone(), kv_cache.clone()
    out = rk.rope_kv_insert(
        q,
        kv,
        kv_cache,
        slot_mapping,
        positions,
        cache,
        1e-6,
        block_size,
        padded,
        q_norm,
    )
    ref = torch.ops._C_musa_ops.deepseek_v4_qnorm_rope_kv_insert(
        native_q,
        kv,
        native_cache,
        slot_mapping,
        positions,
        cache,
        1e-6,
        block_size,
        padded,
        q_norm,
    )
    assert out.shape == ref.shape
    assert torch.equal(out, ref)
    assert torch.equal(kv_cache, native_cache)


@requires_mp31
def test_a_strided_rope_cache_keeps_the_native_op() -> None:
    from vllm_musa.kernels import deepseek_v4_rope_kv_insert as rk

    q = torch.zeros(4, 8, 512, dtype=torch.bfloat16, device="musa")
    kv = torch.zeros(4, 512, dtype=torch.bfloat16, device="musa")
    kv_cache = torch.zeros(2, 64, 584, dtype=torch.uint8, device="musa")
    slots = torch.arange(4, device="musa")
    positions = torch.arange(4, device="musa")
    cache = _rope_cache(16)
    assert rk.rope_kv_insert_supported(q, kv, kv_cache, slots, positions, cache, 64, 64)
    # The native op reads the cache rows at a stride of 64.
    wide = torch.zeros(16, 128, device="musa")
    wide[:, :64] = cache
    assert not rk.rope_kv_insert_supported(
        q, kv, kv_cache, slots, positions, wide[:, :64], 64, 64
    )


@requires_mp31
@pytest.mark.parametrize("tokens", [0, 1, 5, 161, 4096])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("empty_slots", [False, True])
def test_kv_only_writer_matches_fused_cache_bytes(tokens, native, empty_slots):
    from vllm_musa import _custom_ops as ops
    from vllm_musa.kernels import deepseek_v4_rope_kv_insert as rk

    torch.manual_seed(tokens)
    block_size, num_blocks = 64, 80
    kv = (torch.randn(tokens, 512) * 4).to(torch.bfloat16).to("musa")
    original_kv = kv.clone()
    positions = torch.randint(0, 8192, (tokens,), dtype=torch.int64).to("musa")
    slot_count = 0 if empty_slots else tokens - tokens // 4
    slots = torch.randperm(num_blocks * block_size)[:slot_count].to("musa")
    slots[::7] = -1
    cache = torch.randint(
        0, 256, (num_blocks, block_size, 584), dtype=torch.uint8
    ).to("musa")
    expected = cache.clone()
    rope = _rope_cache(8192)
    query = torch.zeros(tokens, 8, 512, dtype=torch.bfloat16, device="musa")
    ops.deepseek_v4_qnorm_rope_kv_insert(
        query, kv, expected, slots, positions, rope, 1e-6, block_size
    )
    if native:
        torch.ops._C_musa_ops.deepseek_v4_kv_rope_insert(
            kv, cache, slots, positions, rope, block_size
        )
    else:
        rk.kv_rope_insert(kv, cache, slots, positions, rope, block_size)
    assert torch.equal(cache, expected)
    assert torch.equal(kv, original_kv)


@requires_mp31
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "num_blocks,storage_offset,shortfall,cache_width",
    [(1, 0, 37312, 1), (2, 0, 1, 1), (2, 32, 1, 1), (1, 32, 0, 0)],
)
def test_kv_only_writer_rejects_truncated_cache_storage(
    native, num_blocks, storage_offset, shortfall, cache_width
) -> None:
    from vllm_musa import _custom_ops as ops
    from vllm_musa.kernels import deepseek_v4_rope_kv_insert as rk

    block_size, page_bytes = 64, 64 * 584
    block_stride = page_bytes + 512
    storage_bytes = (
        storage_offset + (num_blocks - 1) * block_stride + page_bytes - shortfall
    )
    storage = torch.zeros(storage_bytes, dtype=torch.uint8, device="musa")
    cache = storage.as_strided(
        (num_blocks, block_size, cache_width), (block_stride, 1, 1), storage_offset
    )
    kv = torch.zeros(1, 512, dtype=torch.bfloat16, device="musa")
    slots = torch.tensor([num_blocks * block_size - 1], device="musa")
    positions = torch.zeros(1, dtype=torch.int64, device="musa")
    rope = _rope_cache(1)
    assert not rk.rope_kv_insert_supported(
        kv.unsqueeze(1), kv, cache, slots, positions, rope, block_size, 0
    )
    writer = (
        torch.ops._C_musa_ops.deepseek_v4_kv_rope_insert
        if native
        else ops.deepseek_v4_kv_rope_insert
    )
    with pytest.raises(RuntimeError, match="kv_cache storage is too small"):
        writer(kv, cache, slots, positions, rope, block_size)
    assert torch.count_nonzero(storage) == 0


@requires_mp31
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("num_blocks", [0, 2])
@pytest.mark.parametrize("storage_offset", [0, 32])
def test_kv_only_writer_preserves_page_padding_and_storage_offset(
    native, num_blocks, storage_offset
) -> None:
    from vllm_musa import _custom_ops as ops
    from vllm_musa.kernels import deepseek_v4_rope_kv_insert as rk

    torch.manual_seed(4)
    block_size, page_bytes = 64, 64 * 584
    block_stride = page_bytes + 512
    storage_bytes = storage_offset
    if num_blocks:
        storage_bytes += (num_blocks - 1) * block_stride + page_bytes
    storage = torch.full((storage_bytes,), 165, dtype=torch.uint8, device="musa")
    expected_storage = storage.clone()
    cache = storage.as_strided(
        (num_blocks, block_size, 584), (block_stride, 584, 1), storage_offset
    )
    expected_cache = expected_storage.as_strided(
        cache.shape, cache.stride(), storage_offset
    )
    kv = torch.randn(4, 512, device="musa").to(torch.bfloat16)
    slots = torch.tensor([0, 63, 64, 127], device="musa")
    positions = torch.arange(4, device="musa")
    rope = _rope_cache(4)
    assert rk.rope_kv_insert_supported(
        kv.unsqueeze(1), kv, cache, slots, positions, rope, block_size, 0
    )
    query = torch.zeros(4, 8, 512, dtype=torch.bfloat16, device="musa")
    ops.deepseek_v4_qnorm_rope_kv_insert(
        query, kv, expected_cache, slots, positions, rope, 1e-6, block_size
    )
    writer = (
        torch.ops._C_musa_ops.deepseek_v4_kv_rope_insert
        if native
        else ops.deepseek_v4_kv_rope_insert
    )
    writer(kv, cache, slots, positions, rope, block_size)
    assert torch.equal(storage, expected_storage)


@requires_mp31
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("block_size", [-1, 0])
def test_kv_only_writer_rejects_nonpositive_block_size(native, block_size) -> None:
    from vllm_musa import _custom_ops as ops
    from vllm_musa.kernels import deepseek_v4_rope_kv_insert as rk

    kv = torch.zeros(1, 512, dtype=torch.bfloat16, device="musa")
    cache = torch.empty(0, 0, 584, dtype=torch.uint8, device="musa")
    slots = torch.empty(0, dtype=torch.int64, device="musa")
    positions = torch.zeros(1, dtype=torch.int64, device="musa")
    rope = _rope_cache(1)
    assert not rk.rope_kv_insert_supported(
        kv.unsqueeze(1), kv, cache, slots, positions, rope, block_size, 0
    )
    writer = (
        torch.ops._C_musa_ops.deepseek_v4_kv_rope_insert
        if native
        else ops.deepseek_v4_kv_rope_insert
    )
    with pytest.raises(RuntimeError, match="cache_block_size must be positive"):
        writer(kv, cache, slots, positions, rope, block_size)


def _dispatch_inputs() -> tuple[torch.Tensor, ...]:
    """Five rows of a supported call, with three padded slots past them."""
    torch.manual_seed(5)
    q = (torch.randn(5, 8, 512) * 4).to(torch.bfloat16).to("musa")
    kv = (torch.randn(5, 512) * 4).to(torch.bfloat16).to("musa")
    kv_cache = torch.randint(0, 256, (4, 64, 584), dtype=torch.uint8).to("musa")
    slots = torch.randperm(4 * 64)[:8].to("musa")
    positions = torch.randint(0, 256, (5,)).to("musa")
    return q, kv, kv_cache, slots, positions, _rope_cache(256)


@requires_mp31
def test_supported_calls_take_the_triton_kernel(monkeypatch) -> None:
    from vllm_musa import _custom_ops as ops
    from vllm_musa.kernels import deepseek_v4_rope_kv_insert as rk

    seen, result = [], object()

    def spy(q, kv, kv_cache, slot_mapping, *args, **kwargs):
        seen.append(slot_mapping)
        return result

    monkeypatch.setattr(rk, "rope_kv_insert", spy)
    q, kv, kv_cache, slots, positions, cache = _dispatch_inputs()
    out = ops.deepseek_v4_qnorm_rope_kv_insert(
        q, kv, kv_cache, slots, positions, cache, 1e-6, 64, 64, apply_q_norm=False
    )
    assert out is result
    # The padded slots are dropped before the kernel.
    assert len(seen) == 1 and torch.equal(seen[0], slots[:5])


@requires_mp31
def test_unsupported_calls_take_the_native_op(monkeypatch) -> None:
    from vllm_musa import _custom_ops as ops
    from vllm_musa.kernels import deepseek_v4_rope_kv_insert as rk

    def no_kernel(*args, **kwargs):
        raise AssertionError("an unsupported call reached the Triton kernel")

    monkeypatch.setattr(rk, "rope_kv_insert", no_kernel)
    monkeypatch.setattr(rk, "_ENABLED", False)
    q, kv, kv_cache, slots, positions, cache = _dispatch_inputs()
    native_q, native_cache = q.clone(), kv_cache.clone()
    out = ops.deepseek_v4_qnorm_rope_kv_insert(
        q, kv, kv_cache, slots, positions, cache, 1e-6, 64, 64, apply_q_norm=False
    )
    ref = torch.ops._C_musa_ops.deepseek_v4_qnorm_rope_kv_insert(
        native_q, kv, native_cache, slots[:5], positions, cache, 1e-6, 64, 64, False
    )
    assert torch.equal(out, ref)
    assert torch.equal(kv_cache, native_cache)
