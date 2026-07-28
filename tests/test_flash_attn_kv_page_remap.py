# SPDX-License-Identifier: Apache-2.0
"""No-copy gates for the MUSA page-64 KV-cache remap."""

import pytest
import torch

from vllm_musa.v1.attention.backends import fa_utils


@pytest.fixture(autouse=True)
def _clear_page_arange_cache():
    fa_utils._MUSA_B64_ARANGE.clear()
    yield
    fa_utils._MUSA_B64_ARANGE.clear()


def _make_cache(*, padded: bool, block_size: int = 128) -> torch.Tensor:
    num_blocks, num_heads, head_size = 2, 2, 4
    storage_block_size = block_size + 8 if padded else block_size
    storage = torch.arange(
        num_blocks * storage_block_size * num_heads * head_size,
        dtype=torch.float32,
    ).reshape(num_blocks, storage_block_size, num_heads, head_size)
    return storage[:, :block_size]


def _call_remap(
    monkeypatch,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    layout: str = "NHD",
):
    forwarded = []
    sentinel = object()

    def fake_mate(*args, **kwargs):
        forwarded.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(
        fa_utils, "_mate_flash_attn_with_kvcache", fake_mate, raising=False
    )
    page_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    result = fa_utils.flash_attn_with_kvcache(
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=page_table,
        _musa_kv_cache_layout=layout,
    )
    assert result is sentinel
    assert len(forwarded) == 1
    return forwarded[0][1], page_table


def test_contiguous_cache_remap_is_an_alias(monkeypatch) -> None:
    key_cache = _make_cache(padded=False)
    value_cache = _make_cache(padded=False)

    forwarded, page_table = _call_remap(monkeypatch, key_cache, value_cache)

    key_pages = forwarded["k_cache"]
    value_pages = forwarded["v_cache"]
    assert key_pages.shape == (4, 64, 2, 4)
    assert value_pages.shape == (4, 64, 2, 4)
    assert key_pages.data_ptr() == key_cache.data_ptr()
    assert value_pages.data_ptr() == value_cache.data_ptr()
    assert torch.equal(
        forwarded["page_table"],
        torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=torch.int32),
    )
    assert forwarded["page_table"] is not page_table


def test_actual_hybrid_block_ratio_is_an_alias(monkeypatch) -> None:
    key_cache = _make_cache(padded=False, block_size=1088)
    value_cache = _make_cache(padded=False, block_size=1088)

    forwarded, _ = _call_remap(monkeypatch, key_cache, value_cache)

    assert forwarded["k_cache"].shape == (34, 64, 2, 4)
    assert forwarded["v_cache"].shape == (34, 64, 2, 4)
    assert forwarded["k_cache"].data_ptr() == key_cache.data_ptr()
    assert forwarded["v_cache"].data_ptr() == value_cache.data_ptr()


def test_hnd_cache_falls_back_without_rewriting_page_table(monkeypatch) -> None:
    key_cache = torch.empty((2, 128, 64, 4))
    value_cache = torch.empty((2, 128, 64, 4))

    forwarded, page_table = _call_remap(
        monkeypatch, key_cache, value_cache, layout="HND"
    )

    assert forwarded["k_cache"] is key_cache
    assert forwarded["v_cache"] is value_cache
    assert forwarded["page_table"] is page_table


@pytest.mark.parametrize("block_size", [64, 96])
def test_unsupported_block_size_falls_back(monkeypatch, block_size: int) -> None:
    key_cache = _make_cache(padded=False, block_size=block_size)
    value_cache = _make_cache(padded=False, block_size=block_size)

    forwarded, page_table = _call_remap(monkeypatch, key_cache, value_cache)

    assert forwarded["k_cache"] is key_cache
    assert forwarded["v_cache"] is value_cache
    assert forwarded["page_table"] is page_table


@pytest.mark.parametrize(
    ("padded_key", "padded_value"),
    [(True, False), (False, True), (True, True)],
)
def test_padded_cache_falls_back_without_copy(
    monkeypatch, padded_key: bool, padded_value: bool
) -> None:
    key_cache = _make_cache(padded=padded_key)
    value_cache = _make_cache(padded=padded_value)

    forwarded, page_table = _call_remap(monkeypatch, key_cache, value_cache)

    assert forwarded["k_cache"] is key_cache
    assert forwarded["v_cache"] is value_cache
    assert forwarded["page_table"] is page_table


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA device",
)
def test_musa_padded_hybrid_cache_does_not_allocate(monkeypatch) -> None:
    num_blocks, block_size, padding, num_heads, head_size = 2, 1088, 16, 2, 4
    page_stride = (block_size + padding) * num_heads * head_size
    inner_strides = (num_heads * head_size, head_size, 1)

    def make_cache() -> torch.Tensor:
        storage = torch.empty(
            num_blocks * page_stride,
            dtype=torch.float32,
            device="musa",
        )
        return torch.as_strided(
            storage,
            size=(num_blocks, block_size, num_heads, head_size),
            stride=(page_stride, *inner_strides),
        )

    key_cache = make_cache()
    value_cache = make_cache()
    torch.musa.synchronize()
    allocated_before = torch.musa.memory_allocated()

    forwarded, page_table = _call_remap(monkeypatch, key_cache, value_cache)

    torch.musa.synchronize()
    assert torch.musa.memory_allocated() == allocated_before
    assert forwarded["k_cache"] is key_cache
    assert forwarded["v_cache"] is value_cache
    assert forwarded["page_table"] is page_table
