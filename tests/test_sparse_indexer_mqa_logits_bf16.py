from __future__ import annotations

import pytest


@pytest.mark.parametrize("rows", [1, 32])
@pytest.mark.parametrize("keys", [1024, 4130, 5120])
@pytest.mark.parametrize("paged", [False, True])
def test_mate_bf16_logits_match_rounded_fp32(rows, keys, paged):
    pytest.importorskip("torchada")
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        pytest.skip("MUSA-only test")
    deep_gemm = pytest.importorskip("mate.deep_gemm")
    torch.manual_seed(3713)
    head_dim, heads = 128, 64
    query = torch.randn(rows, heads, head_dim, device="musa").to(torch.float8_e4m3fn)
    key = torch.randn(keys, head_dim, device="musa").to(torch.float8_e4m3fn)
    scales = torch.ones(keys, dtype=torch.float32, device="musa")
    weights = torch.rand(rows, heads, dtype=torch.float32, device="musa")
    ends = torch.full((rows,), keys, dtype=torch.int32, device="musa")
    if rows > 1:
        ends[:4] = torch.tensor(
            [511, 512, 513, keys - 1], dtype=torch.int32, device="musa"
        )

    if paged:
        block_size = 64
        blocks = (keys + block_size - 1) // block_size
        storage = torch.empty(
            blocks, block_size * (head_dim + 4), dtype=torch.uint8, device="musa"
        )
        packed_keys = torch.zeros(
            blocks * block_size, head_dim, dtype=torch.float8_e4m3fn, device="musa"
        )
        packed_keys[:keys].copy_(key)
        storage[:, : block_size * head_dim].copy_(
            packed_keys.view(torch.uint8).view(blocks, -1)
        )
        packed_scales = torch.ones(
            blocks, block_size, dtype=torch.float32, device="musa"
        )
        storage[:, block_size * head_dim :].copy_(packed_scales.view(torch.uint8))
        cache = storage.view(blocks, block_size, 1, head_dim + 4)
        table = torch.arange(blocks, dtype=torch.int32, device="musa").repeat(rows, 1)
        metadata = deep_gemm.get_paged_mqa_logits_metadata(
            ends, block_size, torch.musa.get_device_properties(0).multi_processor_count
        )
        arguments = (query[:, None], cache, weights, ends, table, metadata, keys, False)
        operation = deep_gemm.fp8_paged_mqa_logits
    else:
        starts = torch.zeros(rows, dtype=torch.int32, device="musa")
        arguments = (query, (key, scales), weights, starts, ends)
        operation = deep_gemm.fp8_mqa_logits

    fp32 = operation(*arguments)
    bf16 = operation(*arguments, logits_dtype=torch.bfloat16)
    assert fp32.dtype == torch.float32
    assert bf16.dtype == torch.bfloat16
    assert fp32.shape == bf16.shape
    valid = torch.arange(fp32.shape[-1], device="musa")[None, :] < ends[:, None]
    assert torch.isfinite(bf16[valid]).all()
    assert torch.equal(bf16[valid], fp32.to(torch.bfloat16)[valid])
