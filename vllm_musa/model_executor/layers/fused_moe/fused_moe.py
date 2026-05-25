# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    disable_inplace,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import dequant_mxfp6
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_Scheme
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl

from vllm_musa import _custom_ops as musa_ops

logger = init_logger(__name__)


def _musa_fp8_moe_scale_block_size(input_size: int) -> int:
    if input_size % 128 == 0:
        return 128
    if input_size % 64 == 0:
        return 64
    raise ValueError(
        "MUSA static FP8 MoE scale expansion requires the weight input "
        f"dimension to be divisible by 64 or 128, got {input_size}."
    )


def _maybe_expand_fp8_moe_per_tensor_scale(
    scale: torch.Tensor | None,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    """Expand static per-tensor FP8 MoE scales for the native MUSA GEMV op.

    The native op indexes FP8 scales as block scales with shape
    [expert, output_block, input_block]. Static FP8 checkpoints store one
    weight scale per expert, so materialize the equivalent block view here.
    """
    if scale is None or weight.dtype != torch.float8_e4m3fn or scale.dim() >= 3:
        return scale

    num_experts, output_size, input_size = weight.shape

    if scale.dim() == 0:
        per_expert_scale = scale.expand(num_experts)
    elif scale.dim() == 1:
        if scale.numel() == 1:
            per_expert_scale = scale.expand(num_experts)
        elif scale.numel() == num_experts:
            per_expert_scale = scale
        else:
            return scale
    elif scale.dim() == 2 and scale.shape == (num_experts, 1):
        per_expert_scale = scale[:, 0]
    else:
        return scale

    block_size = _musa_fp8_moe_scale_block_size(input_size)
    output_blocks = (output_size + block_size - 1) // block_size
    input_blocks = input_size // block_size
    return (
        per_expert_scale.view(num_experts, 1, 1)
        .expand(num_experts, output_blocks, input_blocks)
        .contiguous()
    )


def _supports_quant_scheme(
    weight_key,
    activation_key,
) -> bool:
    p = current_platform
    if p.is_rocm():
        from vllm.platforms.rocm import on_gfx9

        is_rocm_on_gfx9 = on_gfx9()
    else:
        is_rocm_on_gfx9 = False
    # ==================== MUSA ADAPTATION ====================
    device_supports_fp8 = is_rocm_on_gfx9 or (
        p.is_musa() and p.has_device_capability((3, 1))
    )
    # ========================== END ==========================
    if not device_supports_fp8:
        return (weight_key, activation_key) == (None, None)

    SUPPORTED_W_A = [
        (None, None),
        (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        (kFp8StaticChannelSym, kFp8DynamicTokenSym),
        (kFp8StaticTensorSym, kFp8DynamicTokenSym),
        (kFp8StaticTensorSym, kFp8StaticTensorSym),
    ]
    return (weight_key, activation_key) in SUPPORTED_W_A


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
    elif ocp_mx_scheme is not None:
        if ocp_mx_scheme in {
            "w_mxfp4_a_mxfp4",
            "w_mxfp4_a_mxfp6_e3m2",
            "w_mxfp4_a_mxfp6_e2m3",
        }:
            # 16bit activation and fp4x2 packed weight
            assert hidden_states.size(1) == w1.size(2) * 2, "hidden size mismatch"
        elif ocp_mx_scheme in {
            "w_mxfp6_e3m2_a_mxfp6_e3m2",
            "w_mxfp6_e2m3_a_mxfp6_e2m3",
        }:
            assert (
                hidden_states.size(1) == (w1.size(2) * 4) // 3
            ), "hidden size mismatch"
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")
    else:
        assert hidden_states.size(1) == w1.size(
            2
        ), f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)

    M = num_tokens

    intermediate_cache3 = torch.empty(
        (M, top_k_num, K), device=hidden_states.device, dtype=hidden_states.dtype
    )

    # The first GEMV writes activation input to cache2; the second GEMV writes
    # top-k outputs to cache3 for moe_sum.
    intermediate_cache2 = torch.empty(
        (M * top_k_num, N // 2), device=hidden_states.device, dtype=hidden_states.dtype
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    if use_fp8_w8a8:
        w1_scale = _maybe_expand_fp8_moe_per_tensor_scale(w1_scale, w1)
        w2_scale = _maybe_expand_fp8_moe_per_tensor_scale(w2_scale, w2)

    if inplace and not disable_inplace():
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    if ocp_mx_scheme is not None:
        # TODO: On platforms for which `current_platform.supports_mx()` is True
        # and for which we have a native OCP mx fused MOE kernel,
        # this dequantization step should not be done.
        if ocp_mx_scheme in {
            OCP_MX_Scheme.w_mxfp4_a_mxfp4,
            OCP_MX_Scheme.w_mxfp4_a_mxfp6_e3m2,
            OCP_MX_Scheme.w_mxfp4_a_mxfp6_e2m3,
        }:
            # Weight has to be dequantized for mxfp4 emulation.
            w1 = dequant_mxfp4(w1, w1_scale, hidden_states.dtype)
            w1_scale = None
            w2 = dequant_mxfp4(w2, w2_scale, hidden_states.dtype)
            w2_scale = None
        elif ocp_mx_scheme == OCP_MX_Scheme.w_mxfp6_e3m2_a_mxfp6_e3m2:
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        elif ocp_mx_scheme == OCP_MX_Scheme.w_mxfp6_e2m3_a_mxfp6_e2m3:
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")

    # ==================== MUSA ADAPTATION ====================
    # Due to the implementation of 0.20.0 relying on per_token_group_quant,
    # which is currently not supported by Musa, please refer to setup.py for details.
    # The version used here is 0.18.0
    logger.info_once(
        "MUSA fused MoE uses native GEMV block selection; skipping upstream "
        "Triton MoE JSON config lookup.",
        scope="global",
    )
    CHUNK_SIZE = 16384
    M = min(num_tokens, CHUNK_SIZE)
    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        curr_intermediate_cache2 = intermediate_cache2[
            : tokens_in_chunk * topk_ids.size(1)
        ]
        curr_intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
        curr_out_hidden_states = out_hidden_states[begin_chunk_idx:end_chunk_idx]

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        musa_ops.musa_fused_gemv_moe(
            curr_hidden_states,
            w1,
            curr_intermediate_cache2,
            None,
            w1_scale,
            curr_topk_weights,
            curr_topk_ids,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            use_int4_w4a16,
            use_swigelu=True,
        )
        musa_ops.musa_fused_gemv_moe(
            curr_intermediate_cache2,
            w2,
            curr_intermediate_cache3,
            None,
            w2_scale,
            curr_topk_weights,
            curr_topk_ids,
            not apply_router_weight_on_input,
            1,
            use_int4_w4a16,
            use_swigelu=False,
        )
        # ========================== END ====================
        ops.moe_sum(
            curr_intermediate_cache3.view(*curr_intermediate_cache3.size()),
            curr_out_hidden_states,
        )

    return out_hidden_states


import vllm.model_executor.layers.fused_moe.fused_moe

vllm.model_executor.layers.fused_moe.fused_moe.fused_experts_impl = fused_experts_impl
vllm.model_executor.layers.fused_moe.fused_moe.TritonExperts._supports_quant_scheme = (
    _supports_quant_scheme
)
