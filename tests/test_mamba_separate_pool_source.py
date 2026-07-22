# SPDX-License-Identifier: Apache-2.0
"""Source contracts for MUSA's per-group Mamba state pools."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "vllm_musa" / "patches" / "series"


def _read(name: str) -> str:
    return (SERIES / name).read_text()


def test_mamba_state_capacity_is_per_group_not_global_block_ids():
    source = _read("0083-MUSA-vllm.v1.core.kv_cache_utils.patch")

    assert "mamba_blocks_per_request = max(" in source
    assert "mamba_num_blocks = max_num_seqs * mamba_blocks_per_request + 1" in source
    assert "max_num_seqs + 8" not in source
    assert "len(mamba_groups) * (max_num_seqs + 8)" not in source
    assert "mamba_page * mamba_num_blocks * len(mamba_layers)" in source


def test_each_mamba_group_gets_an_isolated_local_block_pool():
    source = _read("0084-MUSA-vllm.v1.core.kv_cache_coordinator.patch")

    assert "+        self.musa_mamba_block_pools = {}" in source
    assert "self.musa_mamba_block_pools[_musa_group_id] = BlockPool(" in source
    assert "+                    self.musa_mamba_block_pools.get(i)" in source
    assert "+                        i in self.musa_mamba_block_pools" in source
    assert "+        self.musa_mamba_block_pool = None" not in source


def test_mamba_pool_count_is_documented_as_per_group_address_space():
    source = _read("0082-MUSA-vllm.v1.kv_cache_interface.patch")

    assert "The count is per Mamba group" in source
    assert "each group owns a separate BlockPool/address space" in source
