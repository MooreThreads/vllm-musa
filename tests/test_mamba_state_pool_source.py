# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the MUSA per-state Mamba block pool.

The carve is a build-time series member, so this test reads the patch text the
build installs. The regression it guards is concrete: the v0.28 series carved a
contiguous pool per state, the v0.32 re-cut dropped it, and the packed page
stride that came back made the native SSU reject the state view outright.

Pinned here:

* the pool is sliced state-major, so the block stride is the state's own dense
  size and not the page size;
* the carve is gated on a Mamba spec inside the MUSA separate-pool layout and
  uses the Mamba block count, not the attention one;
* both runner paths flatten the per-layer pool list.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "vllm_musa" / "patches" / "series"

POOL = "0211-MUSA-one-contiguous-block-pool-per-Mamba-state.patch"

SLICE = "+        pools.append(flat[offset:end].view(dtype).view(num_blocks, *shape))"
GATE = "+        if isinstance(spec, MambaSpec) and separate_musa_pools:"


def _read() -> str:
    return (SERIES / POOL).read_text()


def test_pool_member_slices_each_state_out_of_the_page():
    source = _read()

    assert "+def _musa_mamba_state_pools(" in source
    assert "+    for shape, dtype in zip(spec.shapes, spec.dtypes):" in source
    assert SLICE in source
    # The block extent is the state's own byte size; a page term would put the
    # packed stride back into the view.
    assert "+        end = offset + num_blocks * prod(shape) * get_dtype_size(dtype)" in source
    assert "+            raise RuntimeError(" in source


def test_pool_member_gates_on_a_mamba_spec_in_the_separate_pool_layout():
    source = _read()

    assert GATE in source
    # The carve must use the Mamba block count and finish before the attention
    # branch picks a kernel block size.
    assert source.index("         num_blocks = kv_cache_config.num_blocks_of(tensor)") < (
        source.index(GATE)
    )
    assert source.index(GATE) < source.index("         kernel_block_size = None")
    # One pool tensor per layer: sharing a tensor would hand two layers the
    # same pools.
    assert '+            assert len(tensor.layers) == 1, "the MUSA Mamba pool' in source


def test_pool_member_flattens_the_layer_pools_in_both_runner_paths():
    source = _read()

    # bind_kv_cache() feeds runner_kv_caches; the v1 GPU runner rebuilds its
    # own list from the cached dict.
    assert source.count("+            if isinstance(cache, (list, tuple)):") == 1
    assert "+                runner_kv_caches.extend(cache)" in source
    assert "vllm/v1/worker/gpu/model_runner.py" in source
    assert "+            for state in (cache if isinstance(cache, (list, tuple)) else (cache,))" in source
