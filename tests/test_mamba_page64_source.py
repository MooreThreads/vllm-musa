# SPDX-License-Identifier: Apache-2.0
"""Source contracts for independent Mamba and attention page sizes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "vllm_musa" / "patches" / "series"


def _read_entry(prefix: str) -> str:
    matches = sorted(SERIES.glob(f"{prefix}*.patch"))
    assert len(matches) == 1, matches
    return matches[0].read_text()


def test_attention_and_mamba_scheduler_blocks_stay_independent():
    """Align-mode hybrid caching keeps both scheduler blocks on 64-token units."""
    fastpath = _read_entry("0200")
    platform = (ROOT / "vllm_musa" / "platform.py").read_text()
    kernel = (ROOT / "vllm_musa" / "jit_kernel" / "mamba_metadata.py").read_text()

    assert "separate_mamba_pages = (" in platform
    assert 'cache_config.mamba_cache_mode == "none"' not in platform
    assert 'cache_config.mamba_cache_mode == "align"' in platform
    assert "Mamba scheduler blocks at 64 tokens" in platform
    assert 'VLLM_MUSA_MAMBA_PAGE64' not in platform
    assert "cache_config.mamba_block_size = 64" in platform
    assert "cache_config.mamba_page_size_padded = None" in platform
    assert "vllm/v1/engine/core.py" in fastpath
    assert 'mamba_cache_mode == "align"' in fastpath
    assert "if not isinstance(g.kv_cache_spec, MambaSpec)" in fastpath
    assert "vllm_musa.jit_kernel.mamba_metadata" in fastpath
    assert "def build_mamba_align_block_table(" in kernel
