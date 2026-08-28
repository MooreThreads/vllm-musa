# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA FlashMLA sparse backend shim.

Registers the sparse-MLA backend so the MLA selector can pick it for DSA
(index_topk) models such as GLM-5.2, and routes the sparse FlashMLA ops
(metadata + kernels) from the CUDA `vllm._flashmla_C` path to the MUSA
`flash_mla` library.
"""

from os import getenv

import torch
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_musa.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd as _musa_flash_mla_sparse_fwd,
    is_flashmla_sparse_supported,
)

# MUSA: the upstream sparse backend imports FlashMLA ops from the core module,
# which is backed by the CUDA `vllm._flashmla_C` (not built on MUSA). Rebind
# those names in the core sparse module to the MUSA `flash_mla` equivalents so
# the sparse metadata builder and impl execute on MUSA.
import vllm.v1.attention.backends.mla.flashmla_sparse as _core_sparse
from vllm_musa.v1.attention.ops.flashmla import (
    FlashMLASchedMeta as _musa_sched_meta,
    flash_mla_with_kvcache as _musa_mla_kvcache,
    get_mla_metadata as _musa_get_mla_metadata,
)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _can_use_tilelang_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    d_v: int,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor | None,
) -> bool:
    if getenv("VLLM_MUSA_SPARSE_MLA_TILELANG", "1") != "1":
        return False
    if attn_sink is not None:
        return False
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        return False
    if indices.dtype != torch.int32:
        return False
    if q.dim() != 3 or kv.dim() != 3 or indices.dim() != 3:
        return False
    if q.device != kv.device or q.device != indices.device:
        return False
    if not q.is_contiguous() or not kv.is_contiguous() or not indices.is_contiguous():
        return False
    if indices.shape[1] != 1 or kv.shape[1] != 1:
        return False
    if q.shape[0] != indices.shape[0] or q.shape[2] != kv.shape[2]:
        return False
    if d_v != 512 or indices.shape[-1] != 2048:
        return False
    tail_dim = q.shape[2] - d_v
    if not _is_power_of_two(tail_dim):
        return False
    if topk_length is not None:
        if topk_length.dtype != torch.int32:
            return False
        if topk_length.device != indices.device:
            return False
        if topk_length.numel() != indices.shape[0] * indices.shape[1]:
            return False
    return True


def _musa_backend_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _can_use_tilelang_sparse_prefill(q, kv, indices, d_v, attn_sink, topk_length):
        from vllm_musa.v1.attention.ops.sparse_mla_tilelang import sparse_mla_fwd_bf16

        # This monkeypatch is private to FlashMLASparseImpl._bf16_flash_mla_kernel,
        # which consumes only the first return value. Keep the public op on the
        # native flash_mla path so callers that need aux tensors keep that contract.
        result = sparse_mla_fwd_bf16(
            q, kv, indices, sm_scale, d_v=d_v, topk_length=topk_length
        )
        if out is not None:
            out.copy_(result)
            result = out
        aux = q.new_empty(0)
        return result, aux, aux

    return _musa_flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
    )


_core_sparse.get_mla_metadata = _musa_get_mla_metadata
_core_sparse.flash_mla_sparse_fwd = _musa_backend_sparse_fwd
_core_sparse.flash_mla_with_kvcache = _musa_mla_kvcache
_core_sparse.FlashMLASchedMeta = _musa_sched_meta


@register_backend(AttentionBackendEnum.FLASHMLA_SPARSE)
class MUSAFlashMLASparseBackend(FlashMLASparseBackend):
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # MATE 0.2.6 supports both the DeepSeek-V3.2 layout (512 NoPE +
        # 64 RoPE) and GLM-5.3-Flash's MLA-nope layout (512 NoPE + 0 RoPE).
        return [512, 576]

    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        # get_kv_cache_shape is block-first for both the ordinary and packed
        # layouts. Do not use the generic sentinel-shape probe: the 512-wide
        # layout folds two 64-token kernel pages into one allocator page.
        return 0

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if head_size == 512 and block_size == 64:
            if num_blocks % 2 != 0:
                raise ValueError(
                    "GLM-5.3 packed sparse MLA requires an even number of "
                    f"64-token kernel blocks, got {num_blocks}."
                )
            # Preserve the exact element/byte count:
            #   N * 64 * 512 == (N / 2) * 128 * 512.
            # The block table still addresses 64-token kernel pages; the
            # sparse-index remap resolves IDs 2b/2b+1 into the two subpages.
            return (num_blocks // 2, block_size * 2, head_size)
        return (num_blocks, block_size, head_size)

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 3 and is_flashmla_sparse_supported()[0]

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        return is_flashmla_sparse_supported()[1]
