# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the MUSA per-state Mamba block pool.

The carve ships as a series member, so this test reads the patch text the build
installs rather than an imported module.

Pinned here:

* the pool is selected from page-major packed bytes with the correct page
  stride for each state;
* the carve is gated on a Mamba spec in the shared backing allocation and uses
  the Mamba block count, not the attention one;
* both runner paths flatten the per-layer pool list.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "vllm_musa" / "patches" / "series"

POOL = "0211-MUSA-one-contiguous-block-pool-per-Mamba-state.patch"

SLICE = "+        state = pages[:, offset:end].view(dtype)"
GATHER = "+    pages = buf.view(num_blocks, page_bytes)"
GATE = "+        if isinstance(spec, MambaSpec):"


def _read() -> str:
    return (SERIES / POOL).read_text()


def test_pool_member_slices_each_state_out_of_the_page():
    source = _read()

    assert "+def _musa_mamba_state_pools(" in source
    assert "+    for shape, dtype in zip(spec.shapes, spec.dtypes):" in source
    assert GATHER in source
    assert SLICE in source
    # Each state selects the corresponding byte range in every page while
    # preserving the page stride required by the fused backing allocation.
    assert (
        "+        state_bytes = prod(shape) * get_dtype_size(dtype)"
        in source
    )
    assert "+        end = offset + state_bytes" in source
    assert "+            raise RuntimeError(" in source


def test_pool_member_gates_on_a_mamba_spec_in_the_fused_pool_layout():
    source = _read()

    assert GATE in source
    # The carve must use the Mamba block count and finish before the attention
    # branch picks a kernel block size.
    assert source.index(
        "         num_blocks = kv_cache_config.num_blocks_of(tensor)"
    ) < (source.index(GATE))
    assert source.index(GATE) < source.index("         kernel_block_size = None")
    # Fused backing allocations contain multiple Mamba layers; each layer must
    # receive a distinct dense view rather than sharing the first layer's pool.
    assert "+                            tensor.offset + layer_idx * tensor.layer_stride," in source
    assert "+                for layer_idx, name in enumerate(tensor.layers)" in source


def test_pool_member_flattens_the_layer_pools_in_both_runner_paths():
    source = _read()

    # bind_kv_cache() feeds runner_kv_caches; the v1 GPU runner rebuilds its
    # own list from the cached dict.
    assert source.count("+            if isinstance(cache, (list, tuple)):") == 1
    assert "+                runner_kv_caches.extend(cache)" in source
    assert "vllm/v1/worker/gpu/model_runner.py" in source
    assert (
        "+            for state in (cache if isinstance(cache, (list, tuple)) else (cache,))"
        in source
    )


def test_page_major_sentinel_requires_per_block_state_gather():
    """A flat state-major slice must not mix adjacent packed pages."""
    import torch

    num_blocks = 3
    state_bytes = (2, 1)
    page_bytes = 4
    packed = torch.empty(num_blocks, page_bytes, dtype=torch.int8)
    for block in range(num_blocks):
        packed[block, :2] = 10 + block
        packed[block, 2:] = 20 + block

    gathered = packed[:, :2].view(num_blocks, *state_bytes)
    assert gathered[:, 0, 0].tolist() == [10, 11, 12]
    assert gathered.stride(0) == page_bytes
