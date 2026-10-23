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


def _reference(o, positions, cache):
    """FP32 inverse RoPE of each head and its power-of-two per-128 scales."""
    x = o.float().clone()
    cos, sin = cache[positions, :32][:, None, :], cache[positions, 32:][:, None, :]
    even, odd = x[..., 448::2].clone(), x[..., 449::2].clone()
    x[..., 448::2] = even * cos + odd * sin
    x[..., 449::2] = odd * cos - even * sin
    x = x.view(*x.shape[:-1], 4, 128)
    scale = (x.abs().amax(-1) / 448.0).clamp_min(1e-10)
    return x, torch.exp2(torch.ceil(torch.log2(scale)))


@requires_mp31
@pytest.mark.parametrize("tokens", [1, 5, 40, 161, 1030, 4096])
@pytest.mark.parametrize(
    "heads,groups,strided", [(8, 1, False), (8, 1, True), (16, 2, True)]
)
def test_triton_kernel_matches_the_native_op(tokens, heads, groups, strided) -> None:
    import vllm_musa._custom_ops  # noqa: F401
    from vllm_musa.kernels import deepseek_v4_inv_rope_fp8_quant as ir

    torch.manual_seed(tokens)
    width = heads * 8 if strided else heads
    o = (torch.randn(tokens, width, 512) * 4).to(torch.bfloat16).to("musa")
    o = o[:, :heads]  # the real heads of the padded attention output
    positions = torch.randint(0, 8192, (tokens,), dtype=torch.int64).to("musa")
    cache = _rope_cache(8192)
    assert ir.inv_rope_fp8_quant_supported(o, positions, cache, 448, 64, 128, False)
    assert not ir.inv_rope_fp8_quant_supported(o, positions, cache, 448, 64, 128, True)
    q, scales = ir.inv_rope_fp8_quant(o, positions, cache, groups, heads // groups)
    native = torch.ops._C_musa_ops.deepseek_v4_fused_inv_rope_fp8_quant
    native_q, native_scales = native(
        o, positions, cache, groups, heads // groups, 448, 64, 128, False
    )
    assert q.shape == native_q.shape and q.stride() == native_q.stride()
    assert scales.shape == native_scales.shape
    assert scales.stride() == native_scales.stride()
    assert torch.equal(scales, native_scales)
    assert torch.equal(q.view(torch.uint8), native_q.view(torch.uint8))

    x, ref_scales = _reference(o, positions, cache)
    # The NoPE chunks hold BF16 values, whose scales the reference reproduces.
    nope = scales.reshape(tokens, heads, 4)[..., :3]
    assert torch.equal(nope, ref_scales[..., :3])
    # Within half an FP8 step of the rotated value: 2^-4 relative, 2^-10 scales below.
    s = scales.reshape(tokens, heads, 4)[..., None]
    deq = q.float().reshape(tokens, heads, 4, 128) * s
    assert ((deq - x).abs() <= x.abs() * 2**-4 + s * 2**-10).all()

    aligned = (tokens + 3) // 4 * 4
    blocks = heads // groups * 4
    padded = scales.transpose(0, 1).as_strided(
        (groups, aligned, blocks), (blocks * aligned, blocks, 1)
    )
    assert torch.count_nonzero(padded[:, tokens:]) == 0


@requires_mp31
def test_scales_just_above_a_power_of_two_match_the_native_op() -> None:
    import vllm_musa._custom_ops  # noqa: F401
    from vllm_musa.kernels import deepseek_v4_inv_rope_fp8_quant as ir

    # A rotated value k ulps above 7 * 2^j puts amax / 448 just above 2^(j - 6).
    combos = [(j, k) for j in range(-16, 13) for k in range(9)]
    o = torch.zeros(len(combos), 8, 512, dtype=torch.bfloat16)
    cache = torch.zeros(len(combos), 64, dtype=torch.float32)
    cache[:, :32] = 1.0
    for t, (j, k) in enumerate(combos):
        partner = 2.0 ** (j - 10)
        o[t, 0, 448], o[t, 0, 449] = 7.0 * 2.0**j, partner
        cache[t, 32] = k * 2.0 ** (j - 21) / partner
    o, cache = o.to("musa"), cache.to("musa")
    positions = torch.arange(len(combos), device="musa")
    q, scales = ir.inv_rope_fp8_quant(o, positions, cache, 1, 8)
    native = torch.ops._C_musa_ops.deepseek_v4_fused_inv_rope_fp8_quant
    native_q, native_scales = native(o, positions, cache, 1, 8, 448, 64, 128, False)
    assert torch.equal(scales, native_scales)
    assert torch.equal(q.view(torch.uint8), native_q.view(torch.uint8))


def _dispatch_inputs() -> tuple[torch.Tensor, ...]:
    """Five tokens of 16 heads, two groups of eight."""
    torch.manual_seed(5)
    o = (torch.randn(5, 16, 512) * 4).to(torch.bfloat16).to("musa")
    positions = torch.randint(0, 256, (5,)).to("musa")
    return o, positions, _rope_cache(256)


@requires_mp31
def test_fp32_scale_calls_take_the_triton_kernel(monkeypatch) -> None:
    from vllm_musa import _custom_ops as ops
    from vllm_musa.kernels import deepseek_v4_inv_rope_fp8_quant as ir

    seen, result = [], object()

    def spy(o, positions, cos_sin_cache, n_groups, heads_per_group):
        seen.append((n_groups, heads_per_group))
        return result

    monkeypatch.setattr(ir, "inv_rope_fp8_quant", spy)
    o, positions, cache = _dispatch_inputs()
    out = ops.deepseek_v4_fused_inv_rope_fp8_quant(
        o, positions, cache, 2, 8, 448, 64, 128, False
    )
    assert out is result and seen == [(2, 8)]


@requires_mp31
def test_packed_scale_calls_take_the_native_op(monkeypatch) -> None:
    from vllm_musa import _custom_ops as ops
    from vllm_musa.kernels import deepseek_v4_inv_rope_fp8_quant as ir

    def no_kernel(*args, **kwargs):
        raise AssertionError("a packed-scale call reached the Triton kernel")

    monkeypatch.setattr(ir, "inv_rope_fp8_quant", no_kernel)
    o, positions, cache = _dispatch_inputs()
    q, scales = ops.deepseek_v4_fused_inv_rope_fp8_quant(
        o, positions, cache, 2, 8, 448, 64, 128, True
    )
    native = torch.ops._C_musa_ops.deepseek_v4_fused_inv_rope_fp8_quant
    native_q, native_scales = native(o, positions, cache, 2, 8, 448, 64, 128, True)
    assert scales.dtype == torch.int32 and torch.equal(scales, native_scales)
    assert torch.equal(q.view(torch.uint8), native_q.view(torch.uint8))
