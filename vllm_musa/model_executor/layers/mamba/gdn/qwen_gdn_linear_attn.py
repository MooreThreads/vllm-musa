# SPDX-License-Identifier: Apache-2.0
"""MUSA OOT pluggable layer: use MATE GDN for Qwen3.5."""

from __future__ import annotations

import inspect

import torch
from mate.gdn_decode import gated_delta_rule_decode
from mate.gdn_prefill import chunk_gated_delta_rule
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)

logger = init_logger(__name__)

_MATE_GDN_PREFILL_HAS_OUTPUT = (
    "output" in inspect.signature(chunk_gated_delta_rule).parameters
)
_MATE_GDN_PREFILL_HAS_IS_LOG_SPACE = (
    "is_log_space" in inspect.signature(chunk_gated_delta_rule).parameters
)
_MATE_GDN_DECODE_HAS_OUTPUT = (
    "output" in inspect.signature(gated_delta_rule_decode).parameters
)


def _log_once(method_name: str, message: str, *args) -> None:
    log_method = getattr(logger, f"{method_name}_once", None)
    if log_method is None:
        log_method = getattr(logger, method_name)
    log_method(message, *args)


@QwenGatedDeltaNetAttention.register_oot
class MusaQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    """MUSA replacement for Qwen3.5 GDN attention.

    Keeps upstream construction and the qwen_gdn_attention_core call chain, but
    routes the core recurrent path through MATE kernels when available.
    """

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        attn_metadata = self._get_gdn_attention_metadata(mixed_qkv)
        if attn_metadata is None:
            return

        if attn_metadata.num_prefills <= 0:
            if self._try_mate_decode(mixed_qkv, b, a, core_attn_out, attn_metadata):
                return
            return super()._forward_core(mixed_qkv, b, a, core_attn_out)

        return self._forward_core_mate_prefill(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            attn_metadata,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # MUSA: Qwen3.5 GDN forward with a single fused z/b/a split kernel
        # (contiguous z/b/a in one launch) replacing the strided-z output-proj
        # copy + b/a contiguous copies. mixed_qkv stays a strided view (conv/MATE
        # accept it) so the large qkv block is never materialized. Qwen3-Next's
        # interleaved layout and the replicated-ba TP path keep the upstream flow.
        if self.gqa_interleaved_layout:
            return super().forward_cuda(hidden_states, output)

        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _encode_layer_name,
        )

        num_tokens = hidden_states.size(0)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        mixed_qkv = mixed_qkvz[:, :qkv_size]

        if getattr(self, "disable_tp_for_ba_proj", False) and self.tp_size > 1:
            z = mixed_qkvz[:, qkv_size:].reshape(num_tokens, -1, self.head_v_dim)
            b, a = self.split_ba(ba)
            b = b.contiguous()
            a = a.contiguous()
        else:
            from vllm_musa.jit_kernel.tilelang.gdn_fused_proj import fused_zba

            z, b, a = fused_zba(
                mixed_qkvz,
                ba,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
            )

        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            layer_name=_encode_layer_name(self.prefix),
        )

        self._output_projection(core_attn_out, z, output, num_tokens)

    def _get_gdn_attention_metadata(self, mixed_qkv: torch.Tensor):
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return None

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        return attn_metadata

    def _gdn_A_log_f32(self):
        # MUSA: cache the fp32 A_log so the captured decode does not re-cast a
        # constant parameter every layer, every step.
        t = getattr(self, "_A_log_f32_cache", None)
        if t is None:
            t = self.A_log.detach().float()
            self._A_log_f32_cache = t
        return t

    def _gdn_dt_bias_f32(self):
        # MUSA: cache the fp32 dt_bias (stored bf16) to drop the per-layer cast.
        t = getattr(self, "_dt_bias_f32_cache", None)
        if t is None:
            t = self.dt_bias.detach().float()
            self._dt_bias_f32_cache = t
        return t

    def _try_mate_decode(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata,
    ) -> bool:
        if (
            attn_metadata.spec_sequence_masks is not None
            or attn_metadata.num_decodes <= 0
        ):
            return False

        from vllm.model_executor.layers.fla.ops import (
            fused_sigmoid_gating_delta_rule_update,
        )
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        from vllm_musa.jit_kernel.tilelang.causal_conv1d import (
            musa_tilelang_causal_conv1d_update,
        )

        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        assert non_spec_state_indices_tensor is not None

        num_decode_tokens = attn_metadata.num_decode_tokens
        mixed_qkv = mixed_qkv[:num_decode_tokens]
        b = b[:num_decode_tokens]
        a = a[:num_decode_tokens]

        self_kv_cache = self.kv_cache
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        state_indices = non_spec_state_indices_tensor[:num_decode_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0),
            self.conv1d.weight.size(2),
        )
        # The upstream wrapper casts BF16 activations to the FP32 cache dtype
        # and casts the result back.  The MUSA kernel accepts those dtypes
        # directly; keep the upstream path as a structural fallback.
        mixed_qkv_tilelang = musa_tilelang_causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=state_indices,
        )
        if mixed_qkv_tilelang is None:
            from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
                causal_conv1d_update,
            )

            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices,
                validate_data=False,
            )
        else:
            mixed_qkv = mixed_qkv_tilelang

        # MUSA: mate's decode kernel reads q/k/v from strided views; split the
        # packed qkv without materializing contiguous copies.
        _qk_dim = self.key_dim // self.tp_size
        _v_dim = self.value_dim // self.tp_size
        _q, _k, _v = torch.split(mixed_qkv, [_qk_dim, _qk_dim, _v_dim], dim=-1)
        query = _q.unflatten(-1, (-1, self.head_k_dim)).unsqueeze(1)
        key = _k.unflatten(-1, (-1, self.head_k_dim)).unsqueeze(1)
        value = _v.unflatten(-1, (-1, self.head_v_dim)).unsqueeze(1)

        # MUSA: the mamba pool is page-aligned (it must not be laid out contiguously,
        # or its bytes would alias the attention KV in the shared hybrid tensor), so
        # ssm_state is not contiguous. Gather the active per-sequence states into a
        # small contiguous buffer for mate's fp32 VK decode (identity mapping), then
        # scatter the updated states back. This keeps the fast mate kernel without a
        # whole-pool contiguity copy.
        if ssm_state.dtype == torch.float32:
            try:
                import os as _os

                _musa_sep = _os.environ.get("VLLM_MUSA_MAMBA_SEPARATE_POOL", "1") == "1"
                # MUSA: write the mate decode output straight into the
                # preallocated core_attn_out buffer (bf16) to skip a per-layer copy.
                _out_view = core_attn_out[:num_decode_tokens].view(
                    num_decode_tokens,
                    1,
                    self.num_v_heads // self.tp_size,
                    self.head_v_dim,
                )
                if _musa_sep and ssm_state.is_contiguous():
                    # MUSA: separate contiguous mamba pool -> mate decodes in
                    # place (block b at b*state_numel); no gather/scatter copy.
                    mate_kwargs = {
                        "q": query,
                        "k": key,
                        "v": value,
                        "state": ssm_state,
                        "state_layout": "VK",
                        "state_indices": state_indices,
                        "scale": self.head_k_dim**-0.5,
                        "A_log": self._gdn_A_log_f32(),
                        "a": a.view(num_decode_tokens, 1, -1),
                        "dt_bias": self._gdn_dt_bias_f32(),
                        "b": b.view(num_decode_tokens, 1, -1),
                        "disable_state_update": False,
                        "use_qk_l2norm": True,
                    }
                    if _MATE_GDN_DECODE_HAS_OUTPUT:
                        mate_kwargs["output"] = _out_view
                    output, _ = gated_delta_rule_decode(**mate_kwargs)
                    _log_once(
                        "info",
                        "MUSA GDN mate in-place decode active (separate pool)",
                    )
                else:
                    active_state = ssm_state[state_indices]
                    mate_kwargs = {
                        "q": query,
                        "k": key,
                        "v": value,
                        "state": active_state,
                        "state_layout": "VK",
                        "scale": self.head_k_dim**-0.5,
                        "A_log": self._gdn_A_log_f32(),
                        "a": a.view(num_decode_tokens, 1, -1),
                        "dt_bias": self._gdn_dt_bias_f32(),
                        "b": b.view(num_decode_tokens, 1, -1),
                        "disable_state_update": False,
                        "use_qk_l2norm": True,
                    }
                    if _MATE_GDN_DECODE_HAS_OUTPUT:
                        mate_kwargs["output"] = _out_view
                    output, updated_state = gated_delta_rule_decode(**mate_kwargs)
                    ssm_state[state_indices] = updated_state
                if not _MATE_GDN_DECODE_HAS_OUTPUT:
                    core_attn_out[:num_decode_tokens] = output.view(
                        num_decode_tokens,
                        self.num_v_heads // self.tp_size,
                        self.head_v_dim,
                    )
                return True
            except Exception as e:
                _log_once(
                    "warning",
                    "MATE GDN decode failed; using recurrent fallback: %s",
                    e,
                )

        try:
            core_attn_out_non_spec, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a,
                b=b,
                dt_bias=self.dt_bias,
                q=query,
                k=key,
                v=value,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                ssm_state_indices=state_indices,
                use_qk_l2norm_in_kernel=True,
            )
        except Exception as exc:
            _log_once(
                "warning",
                "fused decode update failed (%s); using the general decode path",
                exc,
            )
            return False
        core_attn_out[:num_decode_tokens] = core_attn_out_non_spec.squeeze(0)

        return True

    def _try_mate_prefill(
        self,
        mixed_qkv_non_spec: torch.Tensor,
        a_non_spec: torch.Tensor,
        b_non_spec: torch.Tensor,
        ssm_state: torch.Tensor,
        non_spec_state_indices_tensor: torch.Tensor,
        non_spec_query_start_loc: torch.Tensor,
        has_initial_state: torch.Tensor | None,
        out: torch.Tensor | None = None,
    ):
        try:
            # MUSA: feed mate no-copy strided q/k/v views (skip the qkv
            # contiguous copy fused_post_conv_prep does) + one fused kernel for
            # the log-space gating (g, beta). mate l2-norms q/k internally.
            hk = self.num_k_heads // self.tp_size
            hv = self.num_v_heads // self.tp_size
            _kd = hk * self.head_k_dim
            _vd = hv * self.head_v_dim
            _qf, _kf, _vf = torch.split(mixed_qkv_non_spec, [_kd, _kd, _vd], dim=-1)
            q = _qf.view(_qf.shape[0], hk, self.head_k_dim)
            k = _kf.view(_kf.shape[0], hk, self.head_k_dim)
            v = _vf.view(_vf.shape[0], hv, self.head_v_dim)
            from vllm_musa.jit_kernel.fused_gdn_gating import fused_gdn_gating

            g, beta = fused_gdn_gating(self.A_log, a_non_spec, b_non_spec, self.dt_bias)
            if not _MATE_GDN_PREFILL_HAS_IS_LOG_SPACE:
                # Exp-space fallback for pre-is_log_space mate; clamp away from
                # exact zero to avoid MATE NaNs on long prefills.
                g = torch.exp(g).clamp_min(1e-30)

            fused_initial_state = None
            if has_initial_state is not None:
                from vllm_musa.jit_kernel.gdn_state_gather_mask import (
                    can_use_fused_gdn_state_gather_mask,
                    fused_gdn_state_gather_mask,
                )

                if can_use_fused_gdn_state_gather_mask(
                    ssm_state,
                    non_spec_state_indices_tensor,
                    has_initial_state,
                ):
                    fused_initial_state = fused_gdn_state_gather_mask(
                        ssm_state,
                        non_spec_state_indices_tensor,
                        has_initial_state,
                    )
                    logger.info_once(
                        "Using fused MUSA Qwen GDN prefill state gather and mask."
                    )

            state_indices = non_spec_state_indices_tensor.to(torch.int64)
            if fused_initial_state is None:
                initial_state = ssm_state[state_indices].to(torch.float32)
                if has_initial_state is not None:
                    initial_state[~has_initial_state, ...] = 0
            else:
                initial_state = fused_initial_state
            # mate compiles the GDN kernel against this dtype; int32 indexing is
            # cheaper and the offsets are bounded by the per-forward token cap.
            cu_seqlens = non_spec_query_start_loc.to(torch.int32)

            mate_kwargs = {
                "q": q,
                "k": k,
                "v": v,
                "g": g,
                "beta": beta,
                "scale": None,
                "initial_state": initial_state,
                "output_final_state": True,
                "cu_seqlens": cu_seqlens,
                "use_qk_l2norm_in_kernel": True,
            }
            if _MATE_GDN_PREFILL_HAS_IS_LOG_SPACE:
                mate_kwargs["is_log_space"] = True

            # Let mate write straight into the caller's buffer instead of
            # producing a temporary that is then copied into it.
            if out is not None and _MATE_GDN_PREFILL_HAS_OUTPUT:
                mate_kwargs["output"] = out

            output, final_state = chunk_gated_delta_rule(**mate_kwargs)
            ssm_state.index_copy_(0, state_indices, final_state.to(ssm_state.dtype))
            return output.unsqueeze(0)
        except Exception as e:
            _log_once(
                "warning",
                "MATE GDN prefill failed; using recurrent fallback: %s",
                e,
            )
            return None

    def _forward_core_mate_prefill(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata,
    ) -> None:
        from vllm.model_executor.layers.fla.ops import (
            fused_sigmoid_gating_delta_rule_update,
        )
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn,
            causal_conv1d_update,
        )

        try:
            from vllm_musa.jit_kernel.tilelang.causal_conv1d import (
                musa_tilelang_causal_conv1d_fn as causal_conv1d_fn,
            )
        except Exception:
            pass  # MUSA: fall back to Triton causal_conv1d_fn on import failure

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        self_kv_cache = self.kv_cache
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0),
            self.conv1d.weight.size(2),
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        if spec_sequence_masks is not None:
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        assert mixed_qkv_non_spec is not None
        mixed_qkv_non_spec = causal_conv1d_fn(
            mixed_qkv_non_spec.transpose(0, 1),
            conv_weights,
            self.conv1d.bias,
            activation=self.activation,
            conv_states=conv_state,
            has_initial_state=has_initial_state,
            cache_indices=non_spec_state_indices_tensor,
            query_start_loc=non_spec_query_start_loc,
            metadata=attn_metadata,
        ).transpose(0, 1)

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

        if spec_sequence_masks is not None:
            a_non_spec = a.index_select(0, non_spec_token_indx)
            b_non_spec = b.index_select(0, non_spec_token_indx)
        else:
            a_non_spec = a
            b_non_spec = b

        if spec_sequence_masks is not None:
            a_spec = a.index_select(0, spec_token_indx)
            b_spec = b.index_select(0, spec_token_indx)
            core_attn_out_spec, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a_spec,
                b=b_spec,
                dt_bias=self.dt_bias,
                q=query_spec,
                k=key_spec,
                v=value_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_spec = None

        assert non_spec_state_indices_tensor is not None
        # With no spec-decode branch the mate output IS the destination, so hand the
        # buffer down and let the kernel write it.
        mate_out = (
            core_attn_out[:num_actual_tokens]
            if (core_attn_out_spec is None and _MATE_GDN_PREFILL_HAS_OUTPUT)
            else None
        )
        core_attn_out_non_spec = self._try_mate_prefill(
            mixed_qkv_non_spec,
            a_non_spec,
            b_non_spec,
            ssm_state,
            non_spec_state_indices_tensor,
            non_spec_query_start_loc,
            has_initial_state,
            out=mate_out,
        )
        wrote_in_place = mate_out is not None and core_attn_out_non_spec is not None
        if core_attn_out_non_spec is None:
            if has_initial_state is not None:
                zero_mask = ~has_initial_state
                if bool(torch.any(zero_mask).item()):
                    ssm_state[non_spec_state_indices_tensor[zero_mask]] = 0

            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            core_attn_out_non_spec, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a_non_spec,
                b=b_non_spec,
                dt_bias=self.dt_bias,
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )

        if spec_sequence_masks is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif not wrote_in_place:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)


logger.info(
    "Registered MusaQwenGatedDeltaNetAttention as the Qwen3.5 GDN OOT "
    "pluggable layer."
)


__all__ = ["MusaQwenGatedDeltaNetAttention"]
