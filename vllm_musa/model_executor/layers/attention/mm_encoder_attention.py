# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _is_current_stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except (AttributeError, RuntimeError):
        return False


def _alloc_attn_out(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # Zero-initialize during CUDA graph capture so unvisited padding tokens
    # in budget-captured encoder graphs (cudagraph_mm_encoder) cannot emit NaNs.
    if _is_current_stream_capturing():
        return torch.zeros(shape, dtype=dtype, device=device)
    return torch.empty(shape, dtype=dtype, device=device)


def _musa_flash_attn_maxseqlen_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm_musa.v1.attention.backends.fa_utils import flash_attn_varlen_func

    q_len = q.size(1)
    if cu_seqlens is None:
        out = _alloc_attn_out(
            (batch_size, q_len, q.size(2), v.size(3)),
            dtype=q.dtype,
            device=q.device,
        )
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
            fa_version=fa_version,
            out=out,
        )

    if max_seqlen is None:
        max_seqlen_value = q_len
    else:
        with gpu_sync_allowed():
            max_seqlen_value = int(max_seqlen.item())

    q_flat = q.reshape(-1, *q.shape[2:])
    k_flat = k.reshape(-1, *k.shape[2:])
    v_flat = v.reshape(-1, *v.shape[2:])
    out = _alloc_attn_out(
        (q_flat.size(0), q_flat.size(1), v_flat.size(2)),
        dtype=q.dtype,
        device=q.device,
    )
    output = flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen_value,
        max_seqlen_k=max_seqlen_value,
        dropout_p=0.0,
        causal=False,
        softmax_scale=scale,
        fa_version=fa_version,
        out=out,
    )
    return output.reshape(batch_size, q_len, *output.shape[1:])


def _musa_flash_attn_maxseqlen_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    del k, batch_size, fa_version, scale, cu_seqlens, max_seqlen
    return torch.empty(
        (*q.shape[:3], v.size(-1)),
        dtype=q.dtype,
        device=q.device,
    )


direct_register_custom_op(
    op_name="musa_flash_attn_maxseqlen_wrapper",
    op_func=_musa_flash_attn_maxseqlen_wrapper,
    fake_impl=_musa_flash_attn_maxseqlen_wrapper_fake,
)


def _apply_sdpa_4d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    return out.transpose(1, 2).contiguous()


def _musa_torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    if cu_seqlens is None:
        return _apply_sdpa_4d(q, k, v, scale=scale, enable_gqa=enable_gqa)

    with gpu_sync_allowed():
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    q_chunks = torch.split(q, lens, dim=1)
    k_chunks = torch.split(k, lens, dim=1)
    v_chunks = torch.split(v, lens, dim=1)
    outputs = [
        _apply_sdpa_4d(q_i, k_i, v_i, scale=scale, enable_gqa=enable_gqa)
        if q_i.size(1) > 0
        else torch.zeros(
            (q_i.size(0), 0, q_i.size(2), v_i.size(3)),
            dtype=q_i.dtype,
            device=q_i.device,
        )
        for q_i, k_i, v_i in zip(q_chunks, k_chunks, v_chunks)
    ]
    return torch.cat(outputs, dim=1)


def _musa_torch_sdpa_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    del k, scale, cu_seqlens, enable_gqa
    return torch.empty(
        (*q.shape[:3], v.size(-1)),
        dtype=q.dtype,
        device=q.device,
    )


direct_register_custom_op(
    op_name="musa_torch_sdpa_wrapper",
    op_func=_musa_torch_sdpa_wrapper,
    fake_impl=_musa_torch_sdpa_wrapper_fake,
)


@MMEncoderAttention.register_oot
class MusaMMEncoderAttention(MMEncoderAttention):
    @classmethod
    def maybe_recompute_cu_seqlens(
        cls,
        attn_backend: AttentionBackendEnum,
        cu_seqlens: np.ndarray,
        hidden_size: int,
        tp_size: int,
        device: torch.device,
        fp8_padded_hidden_size: int | None = None,
    ) -> torch.Tensor:
        if attn_backend == AttentionBackendEnum.TORCH_SDPA:
            return torch.from_numpy(cu_seqlens)

        return super().maybe_recompute_cu_seqlens(
            attn_backend,
            cu_seqlens,
            hidden_size,
            tp_size,
            device,
            fp8_padded_hidden_size=fp8_padded_hidden_size,
        )

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4
        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)
        output = torch.ops.vllm.musa_torch_sdpa_wrapper(
            query,
            key,
            value,
            self.scale,
            cu_seqlens,
            self.num_heads > self.num_kv_heads,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _forward_fa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert (cu_seqlens is not None and max_seqlen is not None) or (
            cu_seqlens is None and max_seqlen is None
        ), "cu_seqlens and max_seqlen should be both set or both None."

        batch_size, query_len = query.size()[:2]
        key_value_len = key.size(1)
        is_reshaped = query.dim() != 4
        query, key, value = self.view_qkv_to_4d(
            query,
            key,
            value,
            batch_size,
            query_len,
            key_value_len,
        )

        output = torch.ops.vllm.musa_flash_attn_maxseqlen_wrapper(
            query,
            key,
            value,
            batch_size,
            self._fa_version,
            self.scale,
            cu_seqlens,
            max_seqlen,
        )
        if is_reshaped:
            output = output.reshape(batch_size, query_len, -1)
        return output

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_flash_attn_backend:
            return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
        if self.attn_backend == AttentionBackendEnum.TRITON_ATTN:
            return self._forward_triton(
                query,
                key,
                value,
                cu_seqlens,
                max_seqlen,
            )
        if self.attn_backend == AttentionBackendEnum.FLASHINFER:
            return self._forward_flashinfer(
                query,
                key,
                value,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
            )
        if self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            return self._forward_sdpa(query, key, value, cu_seqlens)
        raise ValueError(
            "Unsupported multi-modal encoder attention backend for MUSA: "
            f"{self.attn_backend}."
        )
