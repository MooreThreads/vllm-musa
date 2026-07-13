# SPDX-License-Identifier: Apache-2.0
"""Fused z/b/a split for the strided-QKV GDN prefill path.

One TileLang kernel writes contiguous ``z`` (gate), ``b`` and ``a`` out of the
``mixed_qkvz`` / ``mixed_ba`` projections, replacing the three separate copies
the strided path would otherwise issue per GDN layer:
  - the strided-``z`` ``CopyLastContiguous`` in the output projection,
  - ``b.contiguous()`` and ``a.contiguous()``.

``mixed_qkv`` is intentionally left as a strided view of ``mixed_qkvz`` (the
strided path passes it to conv/MATE without materializing it), so unlike
SGLang's ``fused_qkvzba`` this kernel does NOT copy the large qkv block.

Adapted from SGLang's ``fused_qkvzba_split_reshape_cat_contiguous`` (row kernel)
with the qkv writes dropped for the MUSA strided path.
"""

import functools

import tilelang
import tilelang.language as T
import torch

from vllm_musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS,
    MUSA_COMPILE_FLAGS,
    tilelang_dtype,
)

_PASS_CONFIGS = dict(MUSA_COMMON_PASS_CONFIGS)
for _key, _value in (
    ("TL_DISABLE_SAFE_COPY_PREDICATION", True),
    ("TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION", True),
    ("TL_CONFIG_INDEX_BITWIDTH", 32),
):
    if hasattr(tilelang.PassConfigKey, _key):
        _PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value

__all__ = ["fused_zba"]


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _fused_zba_kernel(
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
    input_dtype: str,
    ba_dtype: str,
    block_elems: int,
):
    m = T.dynamic("m")
    total_q = num_heads_qk * head_qk
    total_v = num_heads_v * head_v
    qkv_dim = total_q * 2 + total_v
    total_qkvz = qkv_dim + total_v
    total_ba = num_heads_v * 2
    v_blocks = T.ceildiv(total_v, block_elems)
    ba_blocks = T.ceildiv(total_ba, block_elems)

    @T.prim_func
    def zba_contiguous(
        z: T.Tensor((m, num_heads_v, head_v), input_dtype),
        b: T.Tensor((m, num_heads_v), ba_dtype),
        a: T.Tensor((m, num_heads_v), ba_dtype),
        mixed_qkvz: T.Tensor((m, total_qkvz), input_dtype),
        mixed_ba: T.Tensor((m, total_ba), ba_dtype),
    ):
        with T.Kernel(m, threads=block_elems) as row:
            for block in T.serial(v_blocks):
                for i in T.Parallel(block_elems):
                    offset = block * block_elems + i
                    if offset < total_v:
                        z[row, offset // head_v, offset % head_v] = mixed_qkvz[
                            row, qkv_dim + offset
                        ]

            for block in T.serial(ba_blocks):
                for i in T.Parallel(block_elems):
                    offset = block * block_elems + i
                    if offset < num_heads_v:
                        b[row, offset] = mixed_ba[row, offset]
                    ba_offset = offset + num_heads_v
                    if ba_offset < total_ba:
                        a[row, offset] = mixed_ba[row, ba_offset]

    return zba_contiguous


_fused_zba_kernel.mode = "lazy"


def _block_elems(total_v: int) -> int:
    be = 1
    while be < total_v and be < 1024:
        be <<= 1
    return max(be, 32)


def fused_zba(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Split contiguous z/b/a out of mixed_qkvz / mixed_ba in one kernel.

    Layout (Qwen3.5): mixed_qkvz = [q, k, v, z]; mixed_ba = [b, a] (chunk(2)).
    Returns contiguous z [m, num_heads_v, head_v], b/a [m, num_heads_v].
    """
    m = mixed_qkvz.shape[0]
    z = torch.empty(
        (m, num_heads_v, head_v), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
    )
    b = torch.empty((m, num_heads_v), dtype=mixed_ba.dtype, device=mixed_ba.device)
    a = torch.empty_like(b)
    if m == 0:
        return z, b, a
    kernel = _fused_zba_kernel(
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        tilelang_dtype(mixed_qkvz.dtype),
        tilelang_dtype(mixed_ba.dtype),
        _block_elems(num_heads_v * head_v),
    )
    kernel(z, b, a, mixed_qkvz.contiguous(), mixed_ba.contiguous())
    return z, b, a
