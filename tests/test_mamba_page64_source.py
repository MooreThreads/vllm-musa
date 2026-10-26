# SPDX-License-Identifier: Apache-2.0
"""Source contracts for independent Mamba and attention page sizes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "vllm_musa" / "patches" / "series"


def _read(name: str) -> str:
    return (SERIES / name).read_text()


def _read_entry(prefix: str) -> str:
    matches = sorted(SERIES.glob(f"{prefix}*.patch"))
    assert len(matches) == 1, matches
    return matches[0].read_text()


def test_separate_pool_keeps_attention_and_mamba_page_sizes_independent():
    source = _read("0001-MUSA-base-platform-port.patch")
    fastpath = _read_entry("0057")
    platform = (ROOT / "vllm_musa" / "platform.py").read_text()
    kernel = (ROOT / "vllm_musa" / "jit_kernel" / "mamba_metadata.py").read_text()

    assert "musa_mamba_separate_pool_enabled()" in source
    assert "_musa_separated_mamba_attn_tensors" in source
    assert "separate_mamba_pages = (" in platform
    assert 'cache_config.mamba_cache_mode == "none"' not in platform
    assert 'cache_config.mamba_cache_mode == "align"' in platform
    assert "Mamba scheduler blocks at 64 tokens" in platform
    assert 'VLLM_MUSA_MAMBA_PAGE64' not in platform
    assert 'target_block_size = (' in platform
    assert 'else 64' in platform
    assert "cache_config.mamba_block_size = 64" in platform
    assert "cache_config.mamba_page_size_padded = None" in platform
    assert "vllm/v1/engine/core.py" in fastpath
    assert 'mamba_cache_mode == "align"' in fastpath
    assert "if not isinstance(g.kv_cache_spec, MambaSpec)" in fastpath
    assert "vllm_musa.jit_kernel.mamba_metadata" in fastpath
    assert "def build_mamba_align_block_table(" in kernel


def test_mixed_page_memory_check_accounts_for_each_cache_family():
    plan = _read("0001-MUSA-base-platform-port.patch")
    accounting = _read_entry("0056")

    assert "_musa_mamba_pool_blocks_per_request" in accounting
    assert "max_num_blocks_per_req" in accounting
    assert "prefix_cache_retention_interval" in accounting
    assert "max_concurrent_batches" in accounting
    assert "mamba_bytes = mamba_page * mamba_num_blocks * len(mamba_layers)" in plan
    assert "remaining = available_memory - mamba_bytes" in plan
    # The capacity estimate charges the whole fixed Mamba pool plus one
    # full-length attention request instead of one resident footprint.
    assert "_max_memory_usage_bytes_from_groups" in accounting
    assert (
        "if _musa_separate_pool_eligible(vllm_config, kv_cache_groups):"
        in accounting
    )
    assert (
        "mamba_required_bytes = mamba_page * mamba_num_blocks * len(mamba_layers)"
        in accounting
    )
    assert (
        "attn_required_bytes = attn_group_size * attn_page * attn_blocks_needed"
        in accounting
    )
    assert "return mamba_required_bytes + attn_required_bytes" in accounting


def test_group_builder_keeps_attention_page_independent():
    bypass = _read_entry("0056")
    assert "musa_mamba_separate_pool_enabled()" in bypass
    assert "and vllm_config.kv_transfer_config is None" in bypass
    assert "isinstance(spec, MambaSpec)" in bypass
    assert "not isinstance(spec, MambaSpec) for spec in filtered_spec.values()" in bypass
