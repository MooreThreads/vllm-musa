# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import os
import time

import torch
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import dequant_mxfp6
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_Scheme
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl

from vllm_musa import _custom_ops as musa_ops

logger = init_logger(__name__)


def disable_inplace() -> bool:
    # MUSA: in-place fused-expert output is always allowed here.
    return False


_MOE_SHAPE_INVENTORY_ENV = "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY"
_MOE_SHAPE_INVENTORY_PATH_ENV = "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_PATH"
_MOE_SHAPE_INVENTORY_MIN_TOKENS_ENV = (
    "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_MIN_TOKENS"
)
_MOE_SHAPE_INVENTORY_MAX_RECORDS_ENV = (
    "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_MAX_RECORDS"
)
_MOE_SHAPE_INVENTORY_DEFAULT_PATH = (
    "/tmp/vllm_omni_musa_outputs/deepseek_v4_moe_shape_inventory.jsonl"
)
_MOE_SHAPE_INVENTORY_RECORDS = 0
_MOE_SHAPE_INVENTORY_WARNED = False
_DEEPGEMM_PREFILL_ENV = "VLLM_MUSA_MOE_DEEPGEMM_PREFILL"
_DEEPGEMM_PREFILL_MIN_TOKENS_ENV = "VLLM_MUSA_MOE_DEEPGEMM_PREFILL_MIN_TOKENS"
_DEEPGEMM_PREFILL_WARNED = False


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _env_flag_disabled(name: str) -> bool:
    """True only when explicitly set to a falsy value; unset means enabled."""
    return os.environ.get(name, "").strip().lower() in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _tensor_meta(tensor: torch.Tensor | None) -> dict[str, object] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "is_contiguous": tensor.is_contiguous(),
    }


def _routed_token_histogram(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> dict[str, object]:
    ids_cpu = topk_ids.detach().to(device="cpu", dtype=torch.int64)
    flat_ids = ids_cpu.reshape(-1)
    valid_mask = flat_ids >= 0
    if num_experts > 0:
        valid_mask &= flat_ids < num_experts
    valid_ids = flat_ids[valid_mask]

    histogram_size = num_experts
    if histogram_size <= 0 and valid_ids.numel() > 0:
        histogram_size = int(valid_ids.max().item()) + 1

    if histogram_size > 0:
        histogram = torch.bincount(valid_ids, minlength=histogram_size).tolist()
    else:
        histogram = []

    slot_histograms = []
    for slot in range(ids_cpu.shape[1] if ids_cpu.dim() >= 2 else 0):
        slot_ids = ids_cpu[:, slot].reshape(-1)
        slot_valid = slot_ids >= 0
        if num_experts > 0:
            slot_valid &= slot_ids < num_experts
        if histogram_size > 0:
            slot_histograms.append(
                torch.bincount(slot_ids[slot_valid], minlength=histogram_size).tolist()
            )
        else:
            slot_histograms.append([])

    nonzero = [count for count in histogram if count]
    return {
        "histogram": histogram,
        "slot_histograms": slot_histograms,
        "invalid_count": int((~valid_mask).sum().item()),
        "nonzero_experts": len(nonzero),
        "max_routes_per_expert": max(nonzero) if nonzero else 0,
        "min_routes_per_nonzero_expert": min(nonzero) if nonzero else 0,
    }


def _maybe_record_deepseek_v4_moe_shape_inventory(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    block_shape: list[int] | None,
    activation: str,
    apply_router_weight_on_input: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    ocp_mx_scheme: str | None,
    per_channel_quant: bool,
    global_num_experts: int,
) -> None:
    global _MOE_SHAPE_INVENTORY_RECORDS
    global _MOE_SHAPE_INVENTORY_WARNED

    if not _env_flag_enabled(_MOE_SHAPE_INVENTORY_ENV):
        return

    num_tokens = hidden_states.size(0)
    min_tokens = _env_int(_MOE_SHAPE_INVENTORY_MIN_TOKENS_ENV, 4096)
    if num_tokens < min_tokens:
        return

    max_records = _env_int(_MOE_SHAPE_INVENTORY_MAX_RECORDS_ENV, 64)
    if max_records >= 0 and _MOE_SHAPE_INVENTORY_RECORDS >= max_records:
        return

    try:
        E, N, _ = w1.size()
        K = w2.size(1)
        num_experts = global_num_experts if global_num_experts > 0 else E
        route_stats = _routed_token_histogram(topk_ids, num_experts)
        record = {
            "event": "deepseek_v4_moe_shape_inventory",
            "time": time.time(),
            "pid": os.getpid(),
            "record_index": _MOE_SHAPE_INVENTORY_RECORDS,
            "num_tokens": num_tokens,
            "top_k": topk_ids.size(1),
            "num_local_experts": E,
            "global_num_experts": num_experts,
            "w1_intermediate_size": N,
            "w2_output_size": K,
            "hidden_states": _tensor_meta(hidden_states),
            "w1": _tensor_meta(w1),
            "w2": _tensor_meta(w2),
            "topk_weights": _tensor_meta(topk_weights),
            "topk_ids": _tensor_meta(topk_ids),
            "w1_scale": _tensor_meta(w1_scale),
            "w2_scale": _tensor_meta(w2_scale),
            "a1_scale": _tensor_meta(a1_scale),
            "a2_scale": _tensor_meta(a2_scale),
            "block_shape": block_shape,
            "activation": activation,
            "apply_router_weight_on_input": apply_router_weight_on_input,
            "use_fp8_w8a8": use_fp8_w8a8,
            "use_int8_w8a8": use_int8_w8a8,
            "use_int8_w8a16": use_int8_w8a16,
            "use_int4_w4a16": use_int4_w4a16,
            "ocp_mx_scheme": ocp_mx_scheme,
            "per_channel_quant": per_channel_quant,
            "routed_token_stats": route_stats,
        }

        output_path = os.environ.get(
            _MOE_SHAPE_INVENTORY_PATH_ENV, _MOE_SHAPE_INVENTORY_DEFAULT_PATH
        )
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as inventory_file:
            inventory_file.write(json.dumps(record, sort_keys=True) + "\n")
        _MOE_SHAPE_INVENTORY_RECORDS += 1
    except Exception as exc:
        if not _MOE_SHAPE_INVENTORY_WARNED:
            logger.warning("Failed to write DeepSeek-V4 MoE shape inventory: %s", exc)
            _MOE_SHAPE_INVENTORY_WARNED = True


def _can_use_moe_deepgemm_prefill(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    ocp_mx_scheme: str | None,
    per_channel_quant: bool,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    block_shape: list[int] | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> bool:
    if _env_flag_disabled(_DEEPGEMM_PREFILL_ENV):
        return False

    min_tokens = _env_int(_DEEPGEMM_PREFILL_MIN_TOKENS_ENV, 2500)
    if hidden_states.size(0) < min_tokens:
        return False

    if (
        not use_fp8_w8a8
        or use_int8_w8a8
        or use_int8_w8a16
        or use_int4_w4a16
        or ocp_mx_scheme is not None
        or per_channel_quant
        or expert_map is not None
        or w1_scale is None
        or w2_scale is None
        or a1_scale is not None
        or a2_scale is not None
        or w1_bias is not None
        or w2_bias is not None
    ):
        return False

    if activation != "silu" or apply_router_weight_on_input:
        return False

    if block_shape != [128, 128]:
        return False

    if topk_ids.size(1) > 16:
        return False

    if hidden_states.dtype != torch.bfloat16:
        return False

    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return False

    if w1_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        return False

    if topk_ids.dtype != torch.int32:
        return False

    if not (
        hidden_states.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()
    ):
        return False

    E, N, K = w1.shape
    return (
        E in (256, 257)
        and N % 256 == 0
        and K % 128 == 0
        and K == hidden_states.size(1)
        and w2.shape == (E, K, N // 2)
        and w1_scale.shape == (E, N // 128, K // 128)
        and w2_scale.shape == (E, K // 128, (N // 2) // 128)
    )


def _silu_mul_per_token_group_fp8_quant_musa_large(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert input_tensor.dim() == 2
    assert input_tensor.is_contiguous()
    assert input_tensor.shape[-1] % (2 * group_size) == 0
    tokens = input_tensor.shape[0]
    hidden = input_tensor.shape[-1] // 2
    output_s = torch.empty(
        (tokens, hidden // group_size),
        device=input_tensor.device,
        dtype=torch.float32,
    )
    fp8_min, fp8_max = get_fp8_min_max()
    torch.ops._C_musa_ops.silu_and_mul_per_token_group_fp8_quant(
        input_tensor,
        output,
        output_s,
        group_size,
        1e-10,
        fp8_min,
        fp8_max,
    )
    return output, output_s


def _moe_deepgemm_prefill_impl(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    inplace: bool,
) -> torch.Tensor:
    # MUSA: SGLang-style fused-glue DeepGEMM MoE prefill. One fused preprocess
    # kernel does quant + permute + contiguous group-layout + m_indices/src2dst
    # (replacing moe_kernel_quantize_input + deepgemm_moe_permute), the two
    # grouped FP8 GEMMs are unchanged, and one post_reorder kernel does the
    # weighted un-permute (replacing deepgemm_unpermute_and_reduce). Falls back
    # to the unfused path when the fused preprocess cannot handle the shape.
    E = w1.shape[0]
    try:
        from vllm_musa.jit_kernel.post_reorder import post_reorder_triton_kernel
        from vllm_musa.jit_kernel.tilelang.deep_gemm_contig_preprocess import (
            can_use_fp8_tilelang,
            deep_gemm_contig_preprocess_fp8_tilelang,
        )

        use_fused = can_use_fp8_tilelang(hidden_states, topk_ids, E, True)
    except Exception:
        use_fused = False

    if not use_fused:
        return _moe_deepgemm_prefill_impl_unfused(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            inplace=inplace,
        )

    from vllm.utils.deep_gemm import (
        get_mk_alignment_for_contiguous_layout,
        m_grouped_fp8_gemm_nt_contiguous,
        mk_alignment_scope,
    )

    logger.info_once("Using the MUSA fused-glue grouped DeepGEMM MoE prefill path.")

    _, N, K = w1.shape
    M, top_k = topk_ids.shape
    block_m = int(get_mk_alignment_for_contiguous_layout()[0])
    dev = hidden_states.device

    src2dst_numel = M * top_k
    all_tokens = (
        (src2dst_numel + E * (block_m - 1) + block_m - 1) // block_m
    ) * block_m

    qhidden_perm = torch.empty((all_tokens, K), device=dev, dtype=torch.float8_e4m3fn)
    qhidden_scale_perm = torch.empty(
        (all_tokens, K // 128), device=dev, dtype=torch.float32
    )
    m_indices = torch.empty(all_tokens, device=dev, dtype=torch.int32)
    src2dst = torch.empty(src2dst_numel, device=dev, dtype=torch.int32)
    topk_ids_for_combine = torch.empty_like(topk_ids)
    counts = torch.empty(E, device=dev, dtype=torch.int32)
    cursor = torch.empty(E, device=dev, dtype=torch.int32)

    deep_gemm_contig_preprocess_fp8_tilelang(
        hidden_states,
        topk_ids,
        qhidden_perm,
        qhidden_scale_perm,
        m_indices,
        src2dst,
        topk_ids_for_combine,
        counts,
        cursor,
        E,
        block_m,
    )

    with mk_alignment_scope(block_m):
        mm1_out = torch.empty((all_tokens, N), device=dev, dtype=hidden_states.dtype)
        m_grouped_fp8_gemm_nt_contiguous(
            (qhidden_perm, qhidden_scale_perm.contiguous()),
            (w1, w1_scale.contiguous()),
            mm1_out,
            m_indices,
        )

        a2q = torch.empty((all_tokens, N // 2), device=dev, dtype=torch.float8_e4m3fn)
        a2q, a2q_scale = _silu_mul_per_token_group_fp8_quant_musa_large(
            mm1_out.view(-1, N),
            a2q,
            group_size=128,
        )

        mm2_out = torch.empty((all_tokens, K), device=dev, dtype=hidden_states.dtype)
        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale.contiguous()),
            (w2, w2_scale.contiguous()),
            mm2_out,
            m_indices,
        )

    if inplace and not disable_inplace():
        output = hidden_states
    else:
        output = torch.empty_like(hidden_states)

    post_reorder_triton_kernel[(M,)](
        mm2_out,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        top_k,
        K,
        BLOCK_SIZE=1024,
    )
    return output


def _moe_deepgemm_prefill_impl_unfused(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    inplace: bool,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
        deepgemm_moe_permute,
        deepgemm_unpermute_and_reduce,
    )
    from vllm.utils.deep_gemm import (
        m_grouped_fp8_gemm_nt_contiguous,
        mk_alignment_scope,
    )

    logger.info_once("Using the MUSA grouped DeepGEMM MoE prefill path.")

    qhidden, a1_scale = moe_kernel_quantize_input(
        A=hidden_states,
        A_scale=None,
        quant_dtype=torch.float8_e4m3fn,
        per_act_token_quant=False,
        block_shape=[128, 128],
    )

    (
        qhidden_perm,
        qhidden_scale_perm,
        expert_ids,
        inv_perm,
        align_used,
    ) = deepgemm_moe_permute(
        aq=qhidden,
        aq_scale=a1_scale,
        topk_ids=topk_ids,
        local_num_experts=w1.shape[0],
        expert_map=None,
        expert_tokens_meta=None,
    )

    _, N, K = w1.shape
    with mk_alignment_scope(align_used):
        mm1_out = torch.empty(
            (qhidden_perm.shape[0], N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        m_grouped_fp8_gemm_nt_contiguous(
            (qhidden_perm, qhidden_scale_perm.contiguous()),
            (w1, w1_scale.contiguous()),
            mm1_out,
            expert_ids,
        )

        a2q = torch.empty(
            (qhidden_perm.shape[0], N // 2),
            device=hidden_states.device,
            dtype=torch.float8_e4m3fn,
        )
        a2q, a2q_scale = _silu_mul_per_token_group_fp8_quant_musa_large(
            mm1_out.view(-1, N),
            a2q,
            group_size=128,
        )

        mm2_out = torch.empty(
            (qhidden_perm.shape[0], K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale.contiguous()),
            (w2, w2_scale.contiguous()),
            mm2_out,
            expert_ids,
        )

    if inplace and not disable_inplace():
        output = hidden_states
    else:
        output = torch.empty_like(hidden_states)

    deepgemm_unpermute_and_reduce(
        a=mm2_out,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        inv_perm=inv_perm,
        expert_map=None,
        output=output,
    )
    return output


def _maybe_moe_deepgemm_prefill(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool,
    activation: str,
    apply_router_weight_on_input: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    ocp_mx_scheme: str | None,
    per_channel_quant: bool,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    block_shape: list[int] | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> torch.Tensor | None:
    global _DEEPGEMM_PREFILL_WARNED

    if not _can_use_moe_deepgemm_prefill(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_ids=topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        per_channel_quant=per_channel_quant,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    ):
        return None

    try:
        assert w1_scale is not None
        assert w2_scale is not None
        return _moe_deepgemm_prefill_impl(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            inplace=inplace,
        )
    except Exception as exc:
        if not _DEEPGEMM_PREFILL_WARNED:
            logger.warning(
                "Grouped DeepGEMM MoE prefill path failed; "
                "falling back to the upstream Triton path: %s",
                exc,
            )
            _DEEPGEMM_PREFILL_WARNED = True
        return None


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
        (kFp8StaticTensorSym, kFp8DynamicTensorSym),
    ]
    return (weight_key, activation_key) in SUPPORTED_W_A


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
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
    *,
    inplace: bool = False,
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

    _maybe_record_deepseek_v4_moe_shape_inventory(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
    )

    deepgemm_prefill_output = _maybe_moe_deepgemm_prefill(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        per_channel_quant=per_channel_quant,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
    if deepgemm_prefill_output is not None:
        return deepgemm_prefill_output

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

_upstream_fused_moe = vllm.model_executor.layers.fused_moe.fused_moe
if not hasattr(_upstream_fused_moe, "_musa_original_fused_experts_impl"):
    _upstream_fused_moe._musa_original_fused_experts_impl = (
        _upstream_fused_moe.fused_experts_impl
    )


def _try_deepgemm_prefill_dispatch(args, kwargs) -> torch.Tensor | None:
    try:
        bound = inspect.signature(fused_experts_impl).bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    a = bound.arguments
    if not a.get("use_fp8_w8a8"):
        return None
    w1 = a["w1"]
    w2 = a["w2"]
    return _maybe_moe_deepgemm_prefill(
        hidden_states=a["hidden_states"],
        w1=w1,
        w2=w2,
        topk_weights=a["topk_weights"],
        topk_ids=a["topk_ids"],
        inplace=a.get("inplace", False),
        activation=a.get("activation", "silu"),
        apply_router_weight_on_input=a.get("apply_router_weight_on_input", False),
        use_fp8_w8a8=a.get("use_fp8_w8a8", False),
        use_int8_w8a8=a.get("use_int8_w8a8", False),
        use_int8_w8a16=a.get("use_int8_w8a16", False),
        use_int4_w4a16=a.get("use_int4_w4a16", False),
        ocp_mx_scheme=a.get("ocp_mx_scheme"),
        per_channel_quant=a.get("per_channel_quant", False),
        expert_map=a.get("expert_map"),
        w1_scale=_maybe_expand_fp8_moe_per_tensor_scale(a.get("w1_scale"), w1),
        w2_scale=_maybe_expand_fp8_moe_per_tensor_scale(a.get("w2_scale"), w2),
        a1_scale=a.get("a1_scale"),
        a2_scale=a.get("a2_scale"),
        block_shape=a.get("block_shape"),
        w1_bias=a.get("w1_bias"),
        w2_bias=a.get("w2_bias"),
    )


def _musa_fused_experts_impl_dispatch(*args, **kwargs) -> torch.Tensor:
    # The native GEMV MoE path (fused_experts_impl) is opt-in: it can hurt some
    # prefill workloads, but DeepSeek-V4-Flash-Base TP8 graph+MTP decode depends
    # on it for its accepted baseline. Default to the upstream implementation for
    # other models; platform.py opts DeepSeek-V4 in per serving process.
    if os.environ.get("VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV", "0") == "1":
        return fused_experts_impl(*args, **kwargs)
    # Grouped DeepGEMM for large-M FP8 block-128 MoE prefill; the shape gate declines
    # decode and every non-matching shape, which then fall through to upstream Triton.
    dg_out = _try_deepgemm_prefill_dispatch(args, kwargs)
    if dg_out is not None:
        return dg_out
    return _upstream_fused_moe._musa_original_fused_experts_impl(*args, **kwargs)


_upstream_fused_moe.fused_experts_impl = _musa_fused_experts_impl_dispatch


def _patch_triton_experts_quant_scheme() -> None:
    """Patch the Triton MoE expert class across vLLM fused-MoE layouts."""
    try:
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )
    except ImportError:
        TritonExperts = getattr(_upstream_fused_moe, "TritonExperts", None)

    if TritonExperts is None:
        logger.warning(
            "Skipping MUSA TritonExperts quant-scheme patch: class not found."
        )
        return

    TritonExperts._supports_quant_scheme = _supports_quant_scheme


# The TritonExperts._supports_quant_scheme patch is independent; it expands
# MUSA's supported FP8 quant key list and stays in place for upstream dispatch.
_patch_triton_experts_quant_scheme()
