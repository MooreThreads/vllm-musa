# SPDX-License-Identifier: Apache-2.0
"""Source contracts for independent Mamba and attention page sizes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "vllm_musa" / "patches" / "series"


def _read(name: str) -> str:
    return (SERIES / name).read_text()


def test_separate_pool_keeps_attention_and_mamba_page_sizes_independent():
    source = _read("0083-MUSA-vllm.v1.core.kv_cache_utils.patch")
    platform = (ROOT / "vllm_musa" / "platform.py").read_text()

    assert "musa_mamba_separate_pool_enabled()" in source
    assert "groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)" in source
    assert "filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)" in source
    assert "separate_mamba_pages = (" in platform
    assert 'cache_config.mamba_cache_mode == "none"' in platform
    assert 'VLLM_MUSA_MAMBA_PAGE64' not in platform
    assert 'target_block_size = (' in platform
    assert 'else 64' in platform
    assert "cache_config.mamba_page_size_padded = None" in platform


def test_mixed_page_memory_check_accounts_for_each_cache_family():
    source = _read(
        "0088-MUSA-vllm.v1.core.kv_cache_utils-mixed-page-memory.patch"
    )

    assert "mamba_required_bytes = mamba_page * mamba_num_blocks" in source
    assert "attn_required_bytes = attn_group_size * attn_page" in source
    assert "return mamba_required_bytes + attn_required_bytes" in source
    assert "get_uniform_page_size() across both cache families" in source
