# SPDX-License-Identifier: Apache-2.0
"""Check Philox lane values and 64-bit counter handling on MUSA Triton."""

import pytest

pytest.importorskip("torchada")
import torch
import triton
import triton.language as tl


@triton.jit
def _rng_lanes(OFFSETS, OUTPUT, SEED: tl.constexpr, NARROW: tl.constexpr):
    i = tl.arange(0, 8)
    offset = tl.load(OFFSETS + i)
    if NARROW:
        offset = offset.to(tl.uint32)
    a, b, c, d = tl.randint4x(SEED, offset)
    # Explicit equal-width bitcasts cover the contract used by tl_rand64.
    tl.store(OUTPUT + i * 4, a.to(tl.uint32, bitcast=True).to(tl.int64))
    tl.store(OUTPUT + i * 4 + 1, b.to(tl.uint32, bitcast=True).to(tl.int64))
    tl.store(OUTPUT + i * 4 + 2, c.to(tl.uint32, bitcast=True).to(tl.int64))
    tl.store(OUTPUT + i * 4 + 3, d.to(tl.uint32, bitcast=True).to(tl.int64))


def _philox_cpu(seed: int, counter: int) -> list[int]:
    mask = 0xFFFFFFFF
    c0, c1, c2, c3 = counter & mask, (counter >> 32) & mask, 0, 0
    k0, k1 = seed & mask, (seed >> 32) & mask
    for _ in range(10):
        product_a, product_b = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (
            ((product_b >> 32) ^ c1 ^ k0) & mask,
            product_b & mask,
            ((product_a >> 32) ^ c3 ^ k1) & mask,
            product_a & mask,
        )
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return [c0, c1, c2, c3]


@pytest.mark.parametrize("narrow", [False, True])
def test_randint4x_integer_lanes_against_cpu(narrow: bool) -> None:
    if not getattr(torch.version, "musa", None):
        pytest.skip("requires MUSA PyTorch")
    assert torch.musa.is_available()
    offsets = [-1, 0, 1, 2**31, 2**32 - 1, 2**32, 2**40 + 7, 2**63 - 1]
    seed = 0x123456789ABCDEF0
    result = torch.empty(8, 4, device="musa", dtype=torch.int64)
    _rng_lanes[(1,)](
        torch.tensor(offsets, device="musa", dtype=torch.int64), result, seed, narrow
    )
    expected = torch.tensor(
        [
            _philox_cpu(seed, value & 0xFFFFFFFF if narrow else value)
            for value in offsets
        ]
    )
    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
