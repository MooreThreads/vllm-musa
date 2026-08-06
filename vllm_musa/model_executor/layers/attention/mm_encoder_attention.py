# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import einops
import numpy as np
import torch
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import AttentionBackendEnum


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
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * q_len,
            step=q_len,
            dtype=torch.int32,
            device=q.device,
        )
    max_seqlen_value = q_len if max_seqlen is None else max_seqlen.item()

    q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in (q, k, v))
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen_value,
        max_seqlen_k=max_seqlen_value,
        dropout_p=0.0,
        causal=False,
        softmax_scale=scale,
        fa_version=fa_version,
    )
    return einops.rearrange(output, "(b s) h d -> b s h d", b=batch_size)


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
    del k, v, batch_size, fa_version, scale, cu_seqlens, max_seqlen
    return torch.empty_like(q)


direct_register_custom_op(
    op_name="musa_flash_attn_maxseqlen_wrapper",
    op_func=_musa_flash_attn_maxseqlen_wrapper,
    fake_impl=_musa_flash_attn_maxseqlen_wrapper_fake,
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

    def forward_native(
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
