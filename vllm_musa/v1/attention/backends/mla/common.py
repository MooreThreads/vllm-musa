# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from abc import abstractmethod
from typing import Generic, TypeVar

import torch
import vllm.envs as envs
from tqdm import tqdm
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import is_global_first_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import mla_attention as _mla_attention
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttentionImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    MLACommonPrefillMetadata,
    accumulate_mla_context_chunk,
    dynamic_per_batched_tensor_quant,
    has_flashinfer,
    init_mla_context_partial,
    reorg_kvcache,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

from vllm_musa.v1.attention.backends.fa_utils import get_flash_attn_version

try:
    from flash_attn_interface import flash_attn_varlen_func

    is_vllm_fa = True
except ImportError as e:
    raise ImportError(
        "MUSA platform requires MATE and flash_attn_3 to be installed. Please install them first."
    ) from e


logger = init_logger(__name__)

M = TypeVar("M", bound=MLACommonMetadata)
A = TypeVar("A")

def _disabled_prefill_backend() -> bool:
    return False


use_cudnn_prefill = getattr(
    _mla_attention, "use_cudnn_prefill", _disabled_prefill_backend
)
use_flashinfer_prefill = getattr(
    _mla_attention, "use_flashinfer_prefill", _disabled_prefill_backend
)
use_trtllm_ragged_deepseek_prefill = getattr(
    _mla_attention, "use_trtllm_ragged_deepseek_prefill", _disabled_prefill_backend
)


class MUSAMLAPrefillBackend(MLAPrefillBackend):
    """Compatibility backend for vLLM v0.22 MLA prefill selection.

    MUSA keeps the prefill execution in this module's MLACommonImpl because
    mate's FlashAttention interface differs from upstream CUDA FA. v0.22 still
    requires MLAAttention to own a prefill_backend object, so provide a backend
    that participates in metadata construction while execution remains here.
    """

    supported_dtypes = [torch.float16, torch.bfloat16]
    requires_r1_mla_dimensions = False

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_compute_capability(cls, device_capability) -> bool:
        return True

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype in cls.supported_dtypes

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def validate_configuration(cls, device_capability, selector_config) -> list[str]:
        if not cls.supports_dtype(selector_config.dtype):
            return [f"dtype {selector_config.dtype} not supported"]
        return []

    def prepare_metadata(self, prefill_metadata: MLACommonPrefillMetadata) -> None:
        self._prefill_metadata = prefill_metadata

    def run_prefill_new_tokens(self, *args, **kwargs):
        raise RuntimeError("MUSA MLA prefill is executed by MLACommonImpl")

    def run_prefill_context_chunk(self, *args, **kwargs):
        raise RuntimeError("MUSA MLA prefill is executed by MLACommonImpl")


def _get_musa_mla_prefill_backend(vllm_config):
    return MUSAMLAPrefillBackend


def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor):
    # Convert from (B, N, L) to (N, B, L)
    x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)

    if self.is_aiter_triton_fp8_bmm_enabled:
        out = out.view(-1, self.num_heads, self.v_head_dim)
        # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
        x = rocm_aiter_ops.triton_fp8_bmm(
            x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True, YQ=out
        )
    else:
        # ==================== MUSA ADAPTATION ====================
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # Convert from (N, B, V) to (B, N * V)
        out_new = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        # ========================== END ==========================
        out.copy_(out_new)


class MLACommonImpl(MLAAttentionImpl[M], Generic[M]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported for MLA")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj
        self.indexer = indexer
        self.q_pad_num_heads = q_pad_num_heads

        self.supports_quant_query_input = True

        # Use flashinfer's optimized concat_mla_k kernel when available.
        # The kernel is optimized for DeepSeek V3 dimensions:
        # num_heads=128, nope_dim=128, rope_dim=64
        self._use_flashinfer_concat_mla_k = (
            has_flashinfer()
            and (self.num_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        )

        # MUSA MLA prefill runs on MATE's FlashAttention. The FlashInfer, cuDNN
        # and TRT-LLM prefill backends are unreachable here: upstream deleted
        # their prefill metadata in the v0.28 chunked-context rework, and MATE's
        # FlashAttention interface differs from upstream CUDA FA anyway. Fail
        # loudly if a future bump ever reports one of them as selectable instead
        # of silently falling back to FlashAttention.
        if (
            use_flashinfer_prefill()
            or use_trtllm_ragged_deepseek_prefill()
            or use_cudnn_prefill()
        ):
            raise RuntimeError(
                "MUSA MLA supports only the FlashAttention prefill backend; the "
                "FlashInfer/cuDNN/TRT-LLM prefill backends are unavailable since "
                "the vLLM v0.28 chunked-context rework (MUSA-100055)."
            )
        logger.debug_once("Using FlashAttention prefill for MLA")
        self._run_prefill_context_chunk = self._run_prefill_context_chunk_fa
        self._run_prefill_new_tokens = self._run_prefill_new_tokens_fa

        # Handle the differences between the flash_attn_varlen from
        # flash_attn and the one from vllm_flash_attn. The former is used on
        # RoCM and the latter has an additional parameter to control
        # FA2 vs FA3
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.vllm_flash_attn_version = get_flash_attn_version()
        if self.vllm_flash_attn_version is not None:
            # ==================== MUSA ADAPTATION ====================
            if not current_platform.is_musa():
                self.flash_attn_varlen_func = functools.partial(
                    flash_attn_varlen_func, fa_version=self.vllm_flash_attn_version
                )
            # ========================== END ==========================

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim for attention backends that do
        # not support different headdims
        # We don't need to pad V if we are on a hopper system with FA3
        self._pad_v = self.vllm_flash_attn_version is None or not (
            self.vllm_flash_attn_version == 3
            and current_platform.get_device_capability()[0] == 9
        )
        # ==================== MUSA ADAPTATION ====================
        self._pad_v &= not current_platform.is_musa()
        # ========================== END ==========================

        parallel_config = get_current_vllm_config().parallel_config
        # Avoid requiring an initialized DCP group in tests and match the
        # vLLM v0.28 MLA initialization contract.
        self.dcp_world_size: int = parallel_config.decode_context_parallel_size

        self.chunked_prefill_workspace_size = (
            MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
                get_current_vllm_config()
            )
        )
        self.cp_kv_cache_interleave_size: int = (
            parallel_config.cp_kv_cache_interleave_size
        )

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        maybe_padded_v = v
        if self._pad_v:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        if is_vllm_fa:
            kwargs["return_softmax_lse"] = return_softmax_lse
        else:
            # ROCm leverages the upstream flash_attn, which takes a parameter
            # called "return_attn_probs" instead of return_softmax_lse
            kwargs["return_attn_probs"] = return_softmax_lse
        if envs.VLLM_BATCH_INVARIANT:
            kwargs["num_splits"] = 1

        attn_out = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=maybe_padded_v,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there is multiple results
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    def _run_prefill_new_tokens_fa(
        self, prefill: MLACommonPrefillMetadata, q, k, v, return_softmax_lse
    ):
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill.query_start_loc,
            cu_seqlens_k=prefill.query_start_loc,
            max_seqlen_q=prefill.max_query_len,
            max_seqlen_k=prefill.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def _run_prefill_context_chunk_fa(self, chunk, q, k, v):
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=chunk.query_start_loc,
            cu_seqlens_k=chunk.cu_seq_lens,
            max_seqlen_q=chunk.max_query_len,
            max_seqlen_k=chunk.max_seq_len,
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(
                f"Layer '{layer}' has no recognized weight attribute: {WEIGHT_NAMES}."
            )

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(
                    layer.input_size_per_partition,
                    dtype=act_dtype,
                    device=get_layer_weight(layer).device,
                )
                dequant_weights = layer.quant_method.apply(layer, eye, bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            return layer.weight

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        if self.is_aiter_triton_fp8_bmm_enabled:
            W_K = W_UK.transpose(0, 1)  # 16 512 128
            W_V = W_UV.permute(1, 2, 0)  # 16 128 512
            self.W_K, self.W_K_scale = dynamic_per_batched_tensor_quant(
                W_K, dtype=current_platform.fp8_dtype()
            )
            self.W_V, self.W_V_scale = dynamic_per_batched_tensor_quant(
                W_V, dtype=current_platform.fp8_dtype()
            )

            # The kernel operates on non-padded inputs. Hence, pre-compiling
            # triton kernel to avoid runtime compilation for unseen batch sizes
            # Pre-compile for batch sizes 1 to 1024 to cover most use-cases.
            # On DS-R1, this step adds roughly 50s to the model loading time.
            max_batch_size = 1024  # [ToDo] Find the optimal upper limit
            pre_compilation_list = list(range(1, max_batch_size + 1))
            if is_global_first_rank():
                pre_compilation_list = tqdm(
                    pre_compilation_list,
                    desc="[Aiter Triton] Pre-compiling fp8 BMM kernel",
                    total=max_batch_size,
                )

            for m in pre_compilation_list:
                x = torch.empty(
                    (self.W_K.shape[0], m, self.W_K.shape[2]),
                    dtype=torch.bfloat16,
                    device=self.W_K.device,
                )
                rocm_aiter_ops.triton_fp8_bmm(
                    x, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
                )

                x = torch.empty(
                    (self.W_V.shape[0], m, self.W_V.shape[2]),
                    dtype=torch.bfloat16,
                    device=self.W_V.device,
                )
                rocm_aiter_ops.triton_fp8_bmm(
                    x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True
                )
        else:
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1)
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0)

    def _concat_k_nope_k_pe(
        self, k_nope: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficiently concatenate k_nope and k_pe tensors along the last dimension.

        This function avoids the performance penalty of torch.cat with expanded
        non-contiguous tensors by pre-allocating the output and using direct copies.

        Args:
            k_nope: Tensor of shape [..., nope_dim]
            k_pe: Tensor to broadcast and concatenate, typically shape [..., 1, pe_dim]
                or [..., pe_dim]

        Returns:
            Tensor of shape [..., nope_dim + pe_dim]
        """
        k = torch.empty(
            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )
        # Direct copies with efficient broadcasting
        k[..., : k_nope.shape[-1]] = k_nope
        k[..., k_nope.shape[-1] :] = k_pe
        return k

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
    ):
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        chunked_context = prefill_metadata.chunked_context
        assert chunked_context is not None

        # vLLM v0.28 schedules chunked context per request: `chunks` holds one
        # entry per (request batch, context window) pair carrying its own q/kv
        # offsets, and `empty_token_slices` marks prefills that no chunk covers.
        # Those rows must be neutralized (-inf lse) before the final merge
        # against the suffix partial, which init_mla_context_partial does.
        workspace = chunked_context.workspace

        output = None
        output_lse = None
        for chunk in chunked_context.chunks:
            toks = chunk.num_context_tokens
            ops.gather_and_maybe_dequant_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table[chunk.request_slice],
                cu_seq_lens=chunk.cu_seq_lens,
                token_to_seq=chunk.token_to_seq,
                num_tokens=toks,
                kv_cache_dtype=self.kv_cache_dtype,
                scale=k_scale,
                seq_starts=chunk.starts,
            )

            kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = self._run_prefill_context_chunk(
                chunk=chunk,
                q=q[chunk.token_slice],
                k=k,
                v=v,
            )

            if output is None:
                # A single chunk covering every prefill token is already the
                # whole context partial.
                if (
                    len(chunked_context.chunks) == 1
                    and not chunked_context.empty_token_slices
                ):
                    return attn_output, attn_softmax_lse
                output, output_lse = init_mla_context_partial(
                    chunked_context,
                    attn_output,
                    attn_softmax_lse,
                    num_tokens=q.shape[0],
                )
            accumulate_mla_context_chunk(
                chunk, attn_output, attn_softmax_lse, output, output_lse
            )

        return output, output_lse

    def _context_parallel_compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        dcp_world_size: int,
    ):
        assert k_scale is None, "DCP not support scaled kvcache now."
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        chunked_context = prefill_metadata.chunked_context
        assert chunked_context is not None
        assert chunked_context.dcp_manager is not None

        # Migrated to the v0.28 per-request chunk layout. NOTE: this DCP branch
        # is source-derived only -- no MUSA DCP MLA run validates it yet.
        output = None
        output_lse = None
        workspace = chunked_context.workspace

        for chunk in chunked_context.chunks:
            assert chunk.padded_local_seq_lens is not None
            assert chunk.local_context_lens_allranks is not None
            assert chunk.padded_local_cu_seq_lens is not None
            assert chunk.local_starts is not None

            toks = chunk.num_local_context_tokens
            ops.cp_gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace[:toks],
                block_table=prefill_metadata.block_table[chunk.request_slice],
                cu_seq_lens=chunk.padded_local_cu_seq_lens,
                batch_size=chunk.num_requests,
                seq_starts=chunk.starts,
            )
            # workspace
            # |------- N tokens --------|--------- N*dcp_size tokens ----------|
            # |<- use for loca_gather ->|<--------- use for allgather -------->|
            allgather_offset = workspace.shape[0] // (dcp_world_size + 1)
            assert allgather_offset * (dcp_world_size + 1) == workspace.shape[0]
            assert toks <= allgather_offset
            local_gathered_kvcache = workspace[:toks]
            cur_allgather_workspace = workspace[
                allgather_offset : allgather_offset * (1 + dcp_world_size)
            ]
            assert toks * dcp_world_size <= cur_allgather_workspace.shape[0]
            cur_allgather_kvcache = cur_allgather_workspace[: toks * dcp_world_size]
            chunked_context.dcp_manager.kv_gather(
                cur_allgather_kvcache, local_gathered_kvcache
            )
            assert (
                cur_allgather_kvcache.shape[-1]
                == self.kv_lora_rank + self.qk_rope_head_dim
            )
            allgatered_kv_c_normed, allgatered_k_pe = cur_allgather_kvcache.unsqueeze(
                1
            ).split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

            kv_c_normed, k_pe = reorg_kvcache(
                allgatered_kv_c_normed,
                allgatered_k_pe,
                padded_local_chunk_seq_lens_lst=chunk.padded_local_seq_lens,
                local_context_lens_allranks=chunk.local_context_lens_allranks,
                local_starts=chunk.local_starts,
                sum_seq_len=chunk.num_context_tokens,
                max_seq_len=chunk.max_seq_len,
                toks=toks,
            )

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = self._run_prefill_context_chunk(
                chunk=chunk,
                q=q[chunk.token_slice],
                k=k,
                v=v,
            )

            if output is None:
                if (
                    len(chunked_context.chunks) == 1
                    and not chunked_context.empty_token_slices
                ):
                    return attn_output, attn_softmax_lse
                output, output_lse = init_mla_context_partial(
                    chunked_context,
                    attn_output,
                    attn_softmax_lse,
                    num_tokens=q.shape[0],
                )
            accumulate_mla_context_chunk(
                chunk, attn_output, attn_softmax_lse, output, output_lse
            )

        return output, output_lse

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        assert attn_metadata.prefill is not None

        has_context = attn_metadata.prefill.chunked_context is not None
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        output_prefill = self._run_prefill_new_tokens(
            prefill=attn_metadata.prefill,
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
        )

        if has_context:
            suffix_output, suffix_lse = output_prefill
            if self.dcp_world_size > 1:
                (
                    context_output,
                    context_lse,
                ) = self._context_parallel_compute_prefill_context(
                    q,
                    kv_c_and_k_pe_cache,
                    attn_metadata,
                    k_scale=None,
                    dcp_world_size=self.dcp_world_size,
                )
            else:
                context_output, context_lse = self._compute_prefill_context(
                    q, kv_c_and_k_pe_cache, attn_metadata, k_scale
                )

            # unpad if necessary
            if self._pad_v:
                context_output = context_output[..., : v.shape[-1]]
                suffix_output = suffix_output[..., : v.shape[-1]]

            output = output.view(-1, self.num_heads, self.v_head_dim)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )
        else:
            output_prefill = output_prefill[..., : v.shape[-1]].flatten(start_dim=-2)
            output.copy_(output_prefill)

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError


import vllm.model_executor.layers.attention.mla_attention
import vllm.v1.attention.backends.mla.prefill
import vllm.v1.attention.backends.mla.prefill.selector

vllm.model_executor.layers.attention.mla_attention.MLAAttention._v_up_proj = _v_up_proj
vllm.model_executor.layers.attention.mla_attention.MLACommonImpl = MLACommonImpl
vllm.model_executor.layers.attention.mla_attention.get_mla_prefill_backend = (
    _get_musa_mla_prefill_backend
)
vllm.v1.attention.backends.mla.prefill.get_mla_prefill_backend = (
    _get_musa_mla_prefill_backend
)
vllm.v1.attention.backends.mla.prefill.selector.get_mla_prefill_backend = (
    _get_musa_mla_prefill_backend
)
