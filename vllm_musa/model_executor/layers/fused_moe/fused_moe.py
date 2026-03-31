# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _get_config_quant_dtype,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import (
    disable_inplace,
    moe_kernel_quantize_input,
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

    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        dtype=hidden_states.dtype,
    )

    # Note: for use_int8_w8a16 or use_int4_w4a16, the activations are
    # quantized prior to calling fused_experts.
    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        ocp_mx_scheme=ocp_mx_scheme,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
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

    qhidden_states, a1q_scale = moe_kernel_quantize_input(
        A=hidden_states,
        A_scale=a1_scale,
        quant_dtype=quant_dtype,
        per_act_token_quant=per_channel_quant,
        block_shape=block_shape,
        ocp_mx_scheme=ocp_mx_scheme,
    )
    # SPARSITY_FACTOR is a heuristic margin ensuring num_tokens * top_k
    # activates only a small fraction of total experts
    SPARSITY_FACTOR = 4
    # block quantized code path is not implemented yet.
    naive_block_assignment = (
        expert_map is None
        and num_tokens * top_k_num * SPARSITY_FACTOR <= global_num_experts
        and not (
            (use_int8_w8a16 or use_int4_w4a16)
            and block_shape is not None
            and block_shape[1] > 0
        )
    )

    if not naive_block_assignment:
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            config["BLOCK_SIZE_M"],
            global_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )
    else:
        max_num_tokens_padded = topk_ids.numel() * config["BLOCK_SIZE_M"]
        expert_ids = topk_ids.view(-1)
        num_tokens_post_padded = torch.empty(
            (1), dtype=torch.int32, device=topk_ids.device
        )
        num_tokens_post_padded.fill_(max_num_tokens_padded)
        sorted_token_ids = None

    # ==================== MUSA ADAPTATION ====================
    musa_ops.musa_fused_gemv_moe(
        qhidden_states,
        w1,
        intermediate_cache2,
        None,
        w1_scale,
        topk_weights,
        sorted_token_ids,
        apply_router_weight_on_input,
        topk_ids.shape[1],
        use_int4_w4a16,
        use_swigelu=True,
    )
    musa_ops.musa_fused_gemv_moe(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        None,
        w2_scale,
        topk_weights,
        sorted_token_ids,
        not apply_router_weight_on_input,
        1,
        use_int4_w4a16,
        use_swigelu=False,
    )
    # ========================== END ==========================
    ops.moe_sum(
        intermediate_cache3.view(*intermediate_cache3.size()),
        out_hidden_states,
    )

    return out_hidden_states


import vllm.model_executor.layers.fused_moe.fused_moe

vllm.model_executor.layers.fused_moe.fused_moe.fused_experts_impl = fused_experts_impl
vllm.model_executor.layers.fused_moe.fused_moe.TritonExperts._supports_quant_scheme = (
    _supports_quant_scheme
)
