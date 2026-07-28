# SPDX-License-Identifier: Apache-2.0
"""No-copy gates for the MUSA page-64 KV-cache remap."""

import pytest
import torch

from vllm_musa.v1.attention.backends import fa_utils

pytestmark = pytest.mark.skipif(
    not hasattr(fa_utils, "_mate_flash_attn_with_kvcache"),
    reason="MATE FlashAttention is only available on MUSA",
)


@pytest.fixture(autouse=True)
def _clear_page_arange_cache():
    fa_utils._MUSA_B64_ARANGE.clear()
    yield
    fa_utils._MUSA_B64_ARANGE.clear()


def _make_cache(*, padded: bool) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_size = 2, 128, 2, 4
    storage_block_size = block_size + 8 if padded else block_size
    storage = torch.arange(
        num_blocks * storage_block_size * num_heads * head_size,
        dtype=torch.float32,
    ).reshape(num_blocks, storage_block_size, num_heads, head_size)
    return storage[:, :block_size]


def _call_remap(monkeypatch, key_cache: torch.Tensor, value_cache: torch.Tensor):
    forwarded = []
    sentinel = object()

    def fake_mate(*args, **kwargs):
        forwarded.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(fa_utils, "_mate_flash_attn_with_kvcache", fake_mate)
    page_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    result = fa_utils.flash_attn_with_kvcache(
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=page_table,
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
