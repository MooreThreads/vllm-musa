# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the packed Mamba pool and in-place GDN state updates."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "vllm_musa" / "patches" / "series"
UPSTREAM = ROOT / "third_party" / "vllm" / "vllm"
GDN = (
    ROOT
    / "vllm_musa"
    / "model_executor"
    / "layers"
    / "mamba"
    / "gdn"
    / "qwen_gdn_linear_attn.py"
)


def test_series_never_introduces_a_dedicated_mamba_pool():
    """The shared pool must not reintroduce a dedicated Mamba allocator."""
    forbidden = (
        "separate_musa_pools",
        "separate_pool_eligible",
        "DEDICATED_MAMBA_POOL",
    )
    for entry in sorted(SERIES.glob("*.patch")):
        text = entry.read_text()
        for token in forbidden:
            assert token not in text, f"{entry.name} reintroduces {token}"

    kv_cache_utils = (UPSTREAM / "v1" / "core" / "kv_cache_utils.py").read_text()
    worker_utils = (UPSTREAM / "v1" / "worker" / "utils.py").read_text()
    assert "musa_mamba" not in kv_cache_utils
    assert "separate_pool" not in kv_cache_utils
    assert "separate_musa_pools" not in worker_utils
    # The allocator keeps upstream's single shared backing allocation.
    assert "assert len(sizes) == 1" in worker_utils
    assert "raw_size = sizes.pop()" in worker_utils


def test_gdn_prefill_hands_mate_the_state_pool():
    """Prefill updates the paged pool in place instead of gathering per request."""
    source = GDN.read_text()

    assert "def _mate_state_pool_zero_copy(" in source
    assert 'mate_kwargs["initial_state"] = ssm_state' in source
    assert 'mate_kwargs["output_state"] = ssm_state' in source
    assert 'mate_kwargs["state_indices"] = state_indices' in source
    assert "if not pool_direct:" in source
    # The gather/scatter hop survives only on the copied fallback path.
    assert "initial_state = ssm_state[state_indices].to(torch.float32)" in source


def test_gdn_decode_does_not_require_a_contiguous_pool():
    source = GDN.read_text()

    assert "_musa_sep and ssm_state.is_contiguous()" not in source
    assert "if _mate_state_pool_zero_copy(ssm_state):" in source


def test_mate_decode_is_the_default_not_a_switch():
    """Decode must reach mate without an environment gate."""
    source = GDN.read_text()

    assert "VLLM_MUSA_ENABLE_MATE_GDN_DECODE" not in source
    assert "_musa_enable_mate_gdn_decode" not in source
    # Prefill and decode share the same ungated dispatch.
    assert "if attn_metadata.num_prefills <= 0:" in source
    assert "if self._try_mate_decode(" in source
