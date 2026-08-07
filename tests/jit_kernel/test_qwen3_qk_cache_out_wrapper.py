# SPDX-License-Identifier: Apache-2.0
"""MUSA wrapper-level checks for the dense Qwen3 cache-out provider."""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("VLLM_MUSA_JIT_CACHE_DIR", "/tmp/vllm_musa_pytest_jit_cache")
os.environ.setdefault("VLLM_MUSA_ARCH_LIST", "31")

pytest.importorskip("torchada")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module", autouse=True)
def _musa_device() -> None:
    pytest.importorskip("torch_musa")
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.musa.set_device(0)


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    normed = (x.float() * torch.rsqrt(variance + 1e-6) * weight.float()).to(x.dtype)
    cos_sin = cos_sin_cache.index_select(0, positions[0])
    half = cos_sin.shape[-1] // 2
    cos = cos_sin[:, None, :half].float()
    sin = cos_sin[:, None, half:].float()
    x1, x2 = normed[..., :half].float(), normed[..., half:].float()
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)


@pytest.mark.parametrize("q_heads", [16, 32])
def test_qwen3_cache_out_wrapper_registers_and_preserves_padding(q_heads: int) -> None:
    from vllm_musa.jit_kernel.csrc.norm import fused_qk_rmsnorm_mrope_cache_out

    device = torch.device("musa")
    tokens, kv_heads, head_dim = 2, 8, 128
    qkv = torch.randn(
        (tokens, (q_heads + 2 * kv_heads) * head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    q_flat, k_flat, v_flat = qkv.split(
        [q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim], dim=-1
    )
    q = q_flat.view(tokens, q_heads, head_dim)
    k = k_flat.view(tokens, kv_heads, head_dim)
    v = v_flat.view(tokens, kv_heads, head_dim)
    q_weight = torch.randn((head_dim,), device=device, dtype=torch.bfloat16)
    k_weight = torch.randn((head_dim,), device=device, dtype=torch.bfloat16)
    positions = torch.arange(tokens, device=device, dtype=torch.int64).repeat(3, 1)
    angles = torch.randn((8, head_dim // 2), device=device, dtype=torch.float32)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    slot_mapping = torch.tensor([0, -1], device=device, dtype=torch.int32)
    sentinel = -9.0
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    key_cache = torch.full(
        (4, kv_heads * head_dim), sentinel, device=device, dtype=torch.bfloat16
    )
    value_cache = torch.full_like(key_cache, sentinel)

    fused_qk_rmsnorm_mrope_cache_out(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        key_cache,
        value_cache,
        slot_mapping,
        True,
        64,
        0,
        0,
        False,
    )
    torch.musa.synchronize()

    torch.testing.assert_close(
        q_out.float(),
        _reference(q, q_weight, positions, cos_sin_cache).float(),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        k_out.float(),
        _reference(k, k_weight, positions, cos_sin_cache).float(),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(key_cache[0], k_out[0].reshape(-1))
    torch.testing.assert_close(value_cache[0], v[0].reshape(-1))
    assert torch.equal(key_cache[1:], torch.full_like(key_cache[1:], sentinel))
    assert torch.equal(value_cache[1:], torch.full_like(value_cache[1:], sentinel))


@pytest.mark.parametrize("tokens", [1, 2, 8, 16])
def test_qwen35_cache_out_matches_no_cache_across_page_boundaries(
    tokens: int,
) -> None:
    from vllm_musa.jit_kernel.csrc.norm import (
        fused_qk_rmsnorm_mrope,
        fused_qk_rmsnorm_mrope_cache_out,
    )

    device = torch.device("musa")
    q_heads, kv_heads, head_dim, rot_dim = 32, 2, 256, 64
    q_weight = torch.randn((head_dim,), device=device, dtype=torch.bfloat16)
    k_weight = torch.randn((head_dim,), device=device, dtype=torch.bfloat16)
    angles = torch.randn((128, rot_dim // 2), device=device, dtype=torch.float32)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(
        torch.bfloat16
    )
    sentinel = -9.0
    block_size = 64
    interleaved_storage = torch.full(
        (3, 3, 2, block_size, kv_heads, head_dim),
        sentinel,
        device=device,
        dtype=torch.bfloat16,
    )
    key_cache = interleaved_storage[:, 1, 0]
    value_cache = interleaved_storage[:, 1, 1]
    expected_key = key_cache.clone()
    expected_value = value_cache.clone()

    first_slots = [63 + index for index in range(tokens)]
    if tokens > 1:
        first_slots[-1] = -1
    second_slots = [127 - index for index in range(tokens)]
    second_slots[0] = first_slots[0]
    for step, slot_values in enumerate((first_slots, second_slots)):
        q = torch.randn(
            (tokens, q_heads, head_dim), device=device, dtype=torch.bfloat16
        )
        k = torch.randn(
            (tokens, kv_heads, head_dim), device=device, dtype=torch.bfloat16
        )
        v = torch.randn_like(k)
        positions = (
            torch.arange(step * tokens, (step + 1) * tokens, device=device)
            .to(torch.int64)
            .repeat(3, 1)
        )
        slots = torch.tensor(slot_values, device=device, dtype=torch.int32)
        q_ref, k_ref = fused_qk_rmsnorm_mrope(
            q,
            k,
            q_weight,
            k_weight,
            positions,
            cos_sin_cache,
            True,
            11,
            11,
            10,
            True,
            1e-6,
            True,
        )
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        fused_qk_rmsnorm_mrope_cache_out(
            q,
            k,
            v,
            q_weight,
            k_weight,
            positions,
            cos_sin_cache,
            q_out,
            k_out,
            key_cache,
            value_cache,
            slots,
            True,
            11,
            11,
            10,
            True,
            1e-6,
            True,
        )
        torch.musa.synchronize()

        torch.testing.assert_close(q_out, q_ref)
        torch.testing.assert_close(k_out, k_ref)
        for token, slot in enumerate(slot_values):
            if slot >= 0:
                block, offset = divmod(slot, block_size)
                expected_key[block, offset].copy_(k_out[token])
                expected_value[block, offset].copy_(v[token])

    torch.testing.assert_close(key_cache, expected_key)
    torch.testing.assert_close(value_cache, expected_value)


def test_qwen3_hnd_runtime_falls_back_before_cache_view(monkeypatch) -> None:
    from vllm_musa.jit_kernel.csrc import norm
    from vllm_musa.v1.attention.backends import flash_attn

    def fail_if_cache_out(*args, **kwargs):
        raise AssertionError("HND must not enter the flat-NHD cache-out provider")

    monkeypatch.setattr(norm, "fused_qk_rmsnorm_mrope_cache_out", fail_if_cache_out)
    monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: "HND")

    updates: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record_update(layer, key, value, kv_cache, slot_mapping) -> None:
        del layer, kv_cache, slot_mapping
        updates.append((key, value))

    impl = SimpleNamespace(
        num_kv_heads=8,
        head_size=128,
        do_kv_cache_update=record_update,
    )
    device = torch.device("musa")
    q = torch.randn((2, 16, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((2, 8, 128), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q_weight = torch.randn((128,), device=device, dtype=torch.bfloat16)
    k_weight = torch.randn((128,), device=device, dtype=torch.bfloat16)
    positions = torch.arange(2, device=device, dtype=torch.int64)
    angles = torch.randn((8, 64), device=device, dtype=torch.float32)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    kv_cache = torch.empty((2, 4, 8, 64, 128), device=device, dtype=torch.bfloat16)
    slot_mapping = torch.tensor([0, -1], device=device, dtype=torch.int32)

    flash_attn.FlashAttentionImpl.do_qwen3_qk_rope_and_kv_cache_update(
        impl,
        object(),
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        kv_cache,
        slot_mapping,
    )
    torch.musa.synchronize()

    assert len(updates) == 1
    assert updates[0][0] is k_out
    assert updates[0][1] is v
    torch.testing.assert_close(
        q_out.float(),
        _reference(q, q_weight, positions.repeat(3, 1), cos_sin_cache).float(),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        k_out.float(),
        _reference(k, k_weight, positions.repeat(3, 1), cos_sin_cache).float(),
        atol=2e-2,
        rtol=2e-2,
    )
