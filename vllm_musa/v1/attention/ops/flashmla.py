# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_FLASHMLA_IMPORT_ERROR: Exception | None = None
try:
    import flash_mla as _flash_mla
except Exception as exc:
    _FLASHMLA_IMPORT_ERROR = exc
    _flash_mla = None
else:
    _REQUIRED_FLASHMLA_ATTRS = (
        "FlashMLASchedMeta",
        "flash_mla_sparse_fwd",
        "flash_mla_with_kvcache",
        "get_mla_metadata",
    )
    for _attr in _REQUIRED_FLASHMLA_ATTRS:
        if not hasattr(_flash_mla, _attr):
            _FLASHMLA_IMPORT_ERROR = AttributeError(
                f"flash_mla is missing required attribute {_attr!r}"
            )
            _flash_mla = None
            break


class _UnavailableFlashMLASchedMeta:
    def __init__(
        self,
        tile_scheduler_metadata: torch.Tensor | None = None,
        num_splits: torch.Tensor | None = None,
    ) -> None:
        self.tile_scheduler_metadata = tile_scheduler_metadata
        self.num_splits = num_splits


FlashMLASchedMeta = (
    _flash_mla.FlashMLASchedMeta
    if _flash_mla is not None
    else _UnavailableFlashMLASchedMeta
)


def _flashmla_unavailable_reason() -> str:
    if _FLASHMLA_IMPORT_ERROR is None:
        return "flash_mla is not available."
    return (
        "flash_mla is not available: "
        f"{type(_FLASHMLA_IMPORT_ERROR).__name__}: {_FLASHMLA_IMPORT_ERROR}"
    )


def _raise_flashmla_unavailable(*_args, **_kwargs):
    raise RuntimeError(_flashmla_unavailable_reason())


def _require_flashmla():
    if _flash_mla is None:
        _raise_flashmla_unavailable()
    return _flash_mla


def _is_flashmla_available() -> tuple[bool, str | None]:
    if _flash_mla is None:
        return False, _flashmla_unavailable_reason()
    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    is_available, reason = _is_flashmla_available()
    if not is_available:
        return False, reason
    if not current_platform.is_musa():
        return False, "FlashMLA Dense is only supported on MUSA devices."
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    is_available, reason = _is_flashmla_available()
    if not is_available:
        return False, reason
    device_capability = current_platform.get_device_capability()
    if device_capability is None or device_capability[0] != 3:
        return False, "FlashMLA Sparse is only supported on MUSA devices."
    return True, None


def _coerce_flashmla_sched_meta(
    tile_scheduler_metadata: FlashMLASchedMeta | torch.Tensor,
    num_splits: torch.Tensor | None,
) -> FlashMLASchedMeta:
    if isinstance(tile_scheduler_metadata, FlashMLASchedMeta):
        if num_splits is not None:
            tile_scheduler_metadata.num_splits = num_splits
        return tile_scheduler_metadata

    return FlashMLASchedMeta(
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
    )


def get_mla_metadata(*args, **kwargs):
    flash_mla = _require_flashmla()
    return flash_mla.get_mla_metadata(*args, **kwargs)


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flash_mla = _require_flashmla()

    result, max_logits, lse = flash_mla.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    if out is not None:
        out.copy_(result)
        result = out
    return result, max_logits, lse


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta | torch.Tensor,
    num_splits: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flash_mla = _require_flashmla()
    scheduler_metadata = _coerce_flashmla_sched_meta(
        tile_scheduler_metadata, num_splits
    )
    result, softmax_lse = flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=scheduler_metadata,
        num_splits=None,
        softmax_scale=softmax_scale,
        causal=causal,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )
    if out is not None:
        out.copy_(result)
        result = out
    return result, softmax_lse


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _flash_mla is None:
        _raise_flashmla_unavailable()
    scheduler_metadata, _ = get_mla_metadata(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        is_fp8_kvcache=True,
    )
    if (
        scheduler_metadata.tile_scheduler_metadata is None
        or scheduler_metadata.num_splits is None
    ):
        raise RuntimeError("FlashMLA FP8 decode metadata was not initialized.")
    return scheduler_metadata.tile_scheduler_metadata, scheduler_metadata.num_splits


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if descale_q is not None or descale_k is not None:
        raise NotImplementedError(
            "MUSA FlashMLA FP8 dense path does not support explicit descale tensors."
        )
    return flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=FlashMLASchedMeta(
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
        ),
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        causal=causal,
        is_fp8_kvcache=True,
    )
