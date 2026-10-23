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
@pytest.mark.parametrize("padded", [64, 0])
@pytest.mark.parametrize("q_norm", [False, True])
def test_triton_kernel_matches_the_native_op(tokens, padded, q_norm) -> None:
    import vllm_musa._custom_ops  # noqa: F401
    from vllm_musa.kernels import deepseek_v4_rope_kv_insert as rk

    torch.manual_seed(tokens + padded)
    heads, block_size, num_blocks = 8, 64, 80
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
