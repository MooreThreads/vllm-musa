# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import time
from functools import cache

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

from vllm_musa import _custom_ops as musa_ops
from vllm_musa.model_executor.layers.fused_moe.dispatch_policy import (
    MusaFusedMoeBackend,
    MusaFusedMoeShape,
    has_calibrated_dimensions,
    parse_dispatch_backend,
    select_fused_moe_backend,
    thresholds_for_shape,
)
from vllm_musa.jit_kernel.csrc.moe import maybe_fast_moe_sum

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
_DEEPGEMM_BF16_PREFILL_MIN_TOKENS = 1024
_DEEPGEMM_BF16_PREFILL_WARNED = False
_MUSA_GROUPED_GEMM_AVAILABLE = True
_MUSA_FUSED_MOE_REQUESTED_BACKEND = parse_dispatch_backend()


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _env_flag_disabled(name: str) -> bool:
    """Treat only an explicit falsy value as disabling a default-on path."""

    return os.environ.get(name, "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


_MOE_SHAPE_INVENTORY_ENABLED = _env_flag_enabled(_MOE_SHAPE_INVENTORY_ENV)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _tensors_share_device(
    reference: torch.Tensor,
    *tensors: torch.Tensor | None,
) -> bool:
    return all(
        tensor is None or tensor.device == reference.device for tensor in tensors
    )


def _musa_moe_routes_are_supported(
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> bool:
    return (
        topk_ids.dim() == 2
        and topk_weights.dim() == 2
        and topk_ids.shape == topk_weights.shape
        and topk_ids.shape[0] == hidden_states.shape[0]
        and 0 < topk_ids.shape[1] <= num_experts
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    )


@cache
def _musa_device_fingerprint(
    device_index: int,
) -> tuple[tuple[int, int], int]:
    try:
        capability = current_platform.get_device_capability(device_index)
        device_capability = (
            (int(capability[0]), int(capability[1]))
            if capability is not None
            else (-1, -1)
        )
        multiprocessor_count = int(
            torch.musa.get_device_properties(device_index).multi_processor_count
        )
        return device_capability, multiprocessor_count
    except Exception:
        return (-1, -1), -1


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

    if not _MOE_SHAPE_INVENTORY_ENABLED or _musa_stream_is_capturing():
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


def _can_use_musa_fp8_moe_grouped_gemm(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    ocp_mx_scheme: str | None,
    per_channel_quant: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    block_shape: list[int] | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> bool:
    if (
        not _MUSA_GROUPED_GEMM_AVAILABLE
        or not use_fp8_w8a8
        or use_int8_w8a8
        or use_int8_w8a16
        or use_int4_w4a16
        or ocp_mx_scheme is not None
        or per_channel_quant
        or expert_map is not None
        or w1_zp is not None
        or w2_zp is not None
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

    if hidden_states.dtype != torch.bfloat16:
        return False

    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return False

    if (
        w1_scale.dtype != torch.float32
        or w2_scale.dtype != torch.float32
        or not w1_scale.is_contiguous()
        or not w2_scale.is_contiguous()
    ):
        return False

    if w1.dim() != 3 or w2.dim() != 3 or hidden_states.dim() != 2:
        return False
    E, N, K = w1.shape
    if global_num_experts not in (-1, E):
        return False
    if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
        return False
    if not _musa_moe_routes_are_supported(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=E,
    ):
        return False
    if not (
        hidden_states.is_contiguous()
        and w1.is_contiguous()
        and w2.is_contiguous()
        and _tensors_share_device(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
        )
    ):
        return False

    if E <= 0 or N % 256 != 0 or K % 128 != 0:
        return False
    unpermute_block = min(K, 1024)
    if K % unpermute_block != 0 or unpermute_block & (unpermute_block - 1) != 0:
        return False
    return (
        K == hidden_states.size(1)
        and w2.shape == (E, K, N // 2)
        and w1_scale.shape == (E, N // 128, K // 128)
        and w2_scale.shape == (E, K // 128, (N // 2) // 128)
    )


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
    """Gate the established default-on large-M DeepGEMM prefill path."""

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
    if block_shape != [128, 128] or topk_ids.size(1) > 16:
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


def _can_use_moe_deepgemm_bf16_prefill(
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
    if hidden_states.size(0) < _DEEPGEMM_BF16_PREFILL_MIN_TOKENS:
        return False

    if (
        use_fp8_w8a8
        or use_int8_w8a8
        or use_int8_w8a16
        or use_int4_w4a16
        or ocp_mx_scheme is not None
        or per_channel_quant
        or expert_map is not None
        or w1_scale is not None
        or w2_scale is not None
        or a1_scale is not None
        or a2_scale is not None
        or block_shape is not None
        or w1_bias is not None
        or w2_bias is not None
    ):
        return False

    if activation != "silu" or apply_router_weight_on_input:
        return False
    if topk_ids.dtype != torch.int32 or topk_ids.size(1) > 16:
        return False
    if hidden_states.dtype != torch.bfloat16:
        return False
    if w1.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype:
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
    # Use the row-tiled kernel that fuses SiLU+Mul with the group-128 FP8
    # quantization used by the contiguous DeepGEMM path.
    from vllm_musa.jit_kernel.csrc.quant import per_token_group_quant_8bit

    fp8_min, fp8_max = get_fp8_min_max()
    per_token_group_quant_8bit(
        input_tensor,
        output,
        output_s,
        group_size=group_size,
        eps=1e-10,
        min_8bit=fp8_min,
        max_8bit=fp8_max,
        fuse_silu_and_mul=True,
    )
    return output, output_s


def _musa_fp8_moe_grouped_gemm_impl(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_map: torch.Tensor | None,
    inplace: bool,
    log_selection: bool = True,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
        deepgemm_moe_permute,
        deepgemm_unpermute_and_reduce,
    )
    from vllm.utils.deep_gemm import (
        m_grouped_fp8_gemm_nt_contiguous,
        mk_alignment_scope,
    )

    if log_selection:
        logger.info_once("MUSA fused-MoE diagnostic grouped DeepGEMM selected.")

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
        expert_map=expert_map,
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
        expert_map=expert_map,
        output=output,
    )
    return output


def _maybe_musa_fp8_moe_grouped_gemm(
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
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    block_shape: list[int] | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> torch.Tensor | None:
    global _DEEPGEMM_PREFILL_WARNED, _MUSA_GROUPED_GEMM_AVAILABLE

    if not _can_use_musa_fp8_moe_grouped_gemm(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
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
        return _musa_fp8_moe_grouped_gemm_impl(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            expert_map=expert_map,
            inplace=inplace,
        )
    except Exception as exc:
        _MUSA_GROUPED_GEMM_AVAILABLE = False
        if not _DEEPGEMM_PREFILL_WARNED:
            logger.warning(
                "MUSA grouped DeepGEMM MoE path failed; falling back to the "
                "established backend and disabling grouped DeepGEMM for this "
                "worker: %s",
                exc,
            )
            _DEEPGEMM_PREFILL_WARNED = True
        return None


def _musa_fp8_moe_deepgemm_prefill_impl(
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
    """Run the fused-glue contiguous DeepGEMM prefill implementation."""

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
        # The diagnostic grouped helper is the same contiguous DeepGEMM
        # operation with the unfused permute/reduce glue. Retain it as the
        # robust fallback when the fused glue declines a shape.
        return _musa_fp8_moe_grouped_gemm_impl(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            expert_map=None,
            inplace=inplace,
            log_selection=False,
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
    device = hidden_states.device

    src2dst_numel = M * top_k
    all_tokens = (
        (src2dst_numel + E * (block_m - 1) + block_m - 1) // block_m
    ) * block_m

    qhidden_perm = torch.empty(
        (all_tokens, K), device=device, dtype=torch.float8_e4m3fn
    )
    qhidden_scale_perm = torch.empty(
        (all_tokens, K // 128), device=device, dtype=torch.float32
    )
    m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)
    src2dst = torch.empty(src2dst_numel, device=device, dtype=torch.int32)
    topk_ids_for_combine = torch.empty_like(topk_ids)
    counts = torch.empty(E, device=device, dtype=torch.int32)
    cursor = torch.empty(E, device=device, dtype=torch.int32)

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
        mm1_out = torch.empty((all_tokens, N), device=device, dtype=hidden_states.dtype)
        m_grouped_fp8_gemm_nt_contiguous(
            (qhidden_perm, qhidden_scale_perm.contiguous()),
            (w1, w1_scale.contiguous()),
            mm1_out,
            m_indices,
        )

        a2q = torch.empty(
            (all_tokens, N // 2), device=device, dtype=torch.float8_e4m3fn
        )
        a2q, a2q_scale = _silu_mul_per_token_group_fp8_quant_musa_large(
            mm1_out.view(-1, N), a2q, group_size=128
        )

        mm2_out = torch.empty((all_tokens, K), device=device, dtype=hidden_states.dtype)
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


def _moe_deepgemm_bf16_prefill_impl(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool,
) -> torch.Tensor:
    import deep_gemm
    from vllm.utils.deep_gemm import (
        get_mk_alignment_for_contiguous_layout,
        mk_alignment_scope,
    )
    from vllm_musa.jit_kernel.post_reorder import post_reorder_triton_kernel
    from vllm_musa.jit_kernel.tilelang.deep_gemm_contig_preprocess import (
        can_use_bf16_tilelang,
        deep_gemm_contig_preprocess_bf16_tilelang,
    )

    E, N, K = w1.shape
    if not can_use_bf16_tilelang(hidden_states, topk_ids, E):
        raise RuntimeError(
            "TileLang BF16 DeepGEMM preprocess does not support this MoE shape"
        )

    logger.info_once("Using the MUSA grouped BF16 DeepGEMM MoE prefill path.")

    M, top_k = topk_ids.shape
    block_m = int(get_mk_alignment_for_contiguous_layout()[0])
    device = hidden_states.device
    src2dst_numel = M * top_k
    all_tokens = (
        (src2dst_numel + E * (block_m - 1) + block_m - 1) // block_m
    ) * block_m

    hidden_perm = torch.empty(
        (all_tokens, K), device=device, dtype=hidden_states.dtype
    )
    m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)
    src2dst = torch.empty(src2dst_numel, device=device, dtype=torch.int32)
    topk_ids_for_combine = torch.empty_like(topk_ids)
    counts = torch.empty(E, device=device, dtype=torch.int32)
    cursor = torch.empty(E, device=device, dtype=torch.int32)

    deep_gemm_contig_preprocess_bf16_tilelang(
        hidden_states,
        topk_ids,
        hidden_perm,
        m_indices,
        src2dst,
        topk_ids_for_combine,
        counts,
        cursor,
        E,
        block_m,
    )

    with mk_alignment_scope(block_m):
        mm1_out = torch.empty(
            (all_tokens, N), device=device, dtype=hidden_states.dtype
        )
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            hidden_perm,
            w1,
            mm1_out,
            m_indices,
            alignment_m=block_m,
        )

        act_out = torch.empty(
            (all_tokens, N // 2), device=device, dtype=hidden_states.dtype
        )
        torch.ops._C.silu_and_mul(act_out, mm1_out)

        mm2_out = torch.empty(
            (all_tokens, K), device=device, dtype=hidden_states.dtype
        )
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            act_out,
            w2,
            mm2_out,
            m_indices,
            alignment_m=block_m,
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
        return _musa_fp8_moe_deepgemm_prefill_impl(
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
                "Grouped DeepGEMM MoE prefill path failed; falling back to the "
                "established upstream backend: %s",
                exc,
            )
            _DEEPGEMM_PREFILL_WARNED = True
        return None


def _maybe_moe_deepgemm_bf16_prefill(
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
    global _DEEPGEMM_BF16_PREFILL_WARNED

    if not _can_use_moe_deepgemm_bf16_prefill(
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
        return _moe_deepgemm_bf16_prefill_impl(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
        )
    except Exception as exc:
        if not _DEEPGEMM_BF16_PREFILL_WARNED:
            logger.warning(
                "Grouped BF16 DeepGEMM MoE prefill path failed; "
                "falling back to the upstream Triton path: %s",
                exc,
            )
            _DEEPGEMM_BF16_PREFILL_WARNED = True
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
    if input_size // block_size == 1 and block_size != 128:
        raise ValueError(
            "MUSA native FP8 MoE GEMV cannot disambiguate a 64-wide "
            "single input scale block."
        )
    if output_size % block_size != 0:
        raise ValueError(
            "MUSA native FP8 MoE GEMV requires the weight output "
            f"dimension to be divisible by scale block {block_size}, got "
            f"{output_size}."
        )
    output_blocks = output_size // block_size
    input_blocks = input_size // block_size
    return (
        per_expert_scale.view(num_experts, 1, 1)
        .expand(num_experts, output_blocks, input_blocks)
        .contiguous()
    )


def _musa_fp8_moe_scale_layout_is_supported(
    scale: torch.Tensor | None,
    weight: torch.Tensor,
) -> bool:
    if scale is None or scale.dtype != torch.float32 or not scale.is_contiguous():
        return False
    num_experts, output_size, input_size = weight.shape
    per_tensor_layout = False
    if scale.dim() == 0:
        per_tensor_layout = True
    elif scale.dim() == 1:
        per_tensor_layout = scale.numel() in (1, num_experts)
    elif scale.dim() == 2:
        per_tensor_layout = scale.shape == (num_experts, 1)
    if per_tensor_layout:
        try:
            group_size = _musa_fp8_moe_scale_block_size(input_size)
        except ValueError:
            return False
        return output_size % group_size == 0 and not (
            input_size // group_size == 1 and group_size != 128
        )
    if scale.dim() != 3 or scale.shape[0] != num_experts:
        return False
    for group_size in (128, 64):
        if input_size % group_size != 0 or output_size % group_size != 0:
            continue
        input_blocks = input_size // group_size
        # The native op infers a 128-wide input scale block whenever the final
        # scale dimension is one; a 64-wide layout with one input block is
        # therefore ambiguous and unsafe.
        if input_blocks == 1 and group_size != 128:
            continue
        expected_shape = (
            num_experts,
            output_size // group_size,
            input_blocks,
        )
        if scale.shape == expected_shape:
            return True
    return False


def _can_use_musa_native_fp8_moe_gemv(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    ocp_mx_scheme: str | None,
    per_channel_quant: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> bool:
    if (
        not use_fp8_w8a8
        or use_int8_w8a8
        or use_int8_w8a16
        or use_int4_w4a16
        or ocp_mx_scheme is not None
        or per_channel_quant
        or expert_map is not None
        or w1_zp is not None
        or w2_zp is not None
        or a1_scale is not None
        or a2_scale is not None
        or w1_bias is not None
        or w2_bias is not None
        or activation != "silu"
    ):
        return False
    if (
        hidden_states.dtype != torch.bfloat16
        or w1.dtype != torch.float8_e4m3fn
        or w2.dtype != torch.float8_e4m3fn
        or topk_ids.dtype != torch.int32
        or topk_weights.dtype != torch.float32
    ):
        return False
    if not (
        hidden_states.is_contiguous()
        and w1.is_contiguous()
        and w2.is_contiguous()
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    ):
        return False
    if hidden_states.dim() != 2 or w1.dim() != 3 or w2.dim() != 3:
        return False

    num_experts, w1_output_size, hidden_size = w1.shape
    if w2.shape[0] != num_experts:
        return False
    if global_num_experts not in (-1, num_experts):
        return False
    if not _musa_moe_routes_are_supported(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=num_experts,
    ):
        return False
    if not _tensors_share_device(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
    ):
        return False
    if w1_output_size % 2 != 0 or hidden_size != hidden_states.shape[1]:
        return False
    if w2.shape != (num_experts, hidden_size, w1_output_size // 2):
        return False
    if hidden_size % 64 != 0 or (w1_output_size // 2) % 64 != 0:
        return False
    return _musa_fp8_moe_scale_layout_is_supported(
        w1_scale, w1
    ) and _musa_fp8_moe_scale_layout_is_supported(w2_scale, w2)


def _musa_fused_moe_shape(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    block_shape: list[int] | None,
    activation: str,
    expert_map: torch.Tensor | None,
    stream_is_capturing: bool,
    device_capability: tuple[int, int],
    multiprocessor_count: int,
) -> MusaFusedMoeShape:
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape and len(block_shape) > 1 else 0
    return MusaFusedMoeShape(
        device_capability=device_capability,
        multiprocessor_count=multiprocessor_count,
        local_experts=w1.shape[0],
        w1_output_size=w1.shape[1],
        w2_input_size=w2.shape[2],
        hidden_size=hidden_states.shape[1],
        top_k=topk_ids.shape[1],
        block_n=block_n,
        block_k=block_k,
        activation=activation,
        expert_parallel=expert_map is not None,
        hidden_dtype=str(hidden_states.dtype),
        weight_dtype=str(w1.dtype),
        scale_dtype=str(w1_scale.dtype) if w1_scale is not None else "none",
        w1_scale_shape=tuple(w1_scale.shape) if w1_scale is not None else (),
        w2_scale_shape=tuple(w2_scale.shape) if w2_scale is not None else (),
        gemv_block=os.environ.get("VLLM_MUSA_GEMV_MOE_BLOCK", "auto"),
        graph_mode="capture" if stream_is_capturing else "eager",
    )


def _musa_stream_is_capturing() -> bool:
    for module_name in ("musa", "cuda"):
        module = getattr(torch, module_name, None)
        is_capturing = getattr(module, "is_current_stream_capturing", None)
        if is_capturing is None:
            continue
        try:
            return bool(is_capturing())
        except Exception:
            continue
    # Grouped DeepGEMM is not capture-safe. An unavailable or broken capture
    # query must conservatively disable it instead of failing open.
    return True


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
    _allow_deepgemm_prefill: bool = True,
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

    # Preserve the base implementation's direct-call semantics.  The
    # dispatcher disables this only when it has explicitly selected native
    # GEMV, so a GEMV diagnostic cannot silently turn into DeepGEMM at a large
    # M.  The normal local implementation remains the established large-prefill
    # path for callers that bypass the dispatcher.
    if _allow_deepgemm_prefill:
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

    intermediate_cache3 = torch.empty(
        (M, top_k_num, K), device=hidden_states.device, dtype=hidden_states.dtype
    )

    # The first GEMV writes activation input to cache2; the second GEMV writes
    # top-k outputs to cache3 for moe_sum.
    intermediate_cache2 = torch.empty(
        (M * top_k_num, N // 2), device=hidden_states.device, dtype=hidden_states.dtype
    )

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
        moe_sum_input = curr_intermediate_cache3.view(*curr_intermediate_cache3.size())
        if not maybe_fast_moe_sum(moe_sum_input, curr_out_hidden_states):
            ops.moe_sum(moe_sum_input, curr_out_hidden_states)

    return out_hidden_states


import vllm.model_executor.layers.fused_moe.fused_moe  # noqa: E402

_upstream_fused_moe = vllm.model_executor.layers.fused_moe.fused_moe
if not hasattr(_upstream_fused_moe, "_musa_original_fused_experts_impl"):
    _upstream_fused_moe._musa_original_fused_experts_impl = (
        _upstream_fused_moe.fused_experts_impl
    )


def _musa_fused_experts_impl_dispatch(
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
) -> torch.Tensor:
    backend = MusaFusedMoeBackend.UPSTREAM
    policy = None
    shape = None

    # Skip all shape construction and capture queries for non-FP8 callers and
    # for the explicit rollback path. This wrapper sits on every MoE layer.
    if (
        _MUSA_FUSED_MOE_REQUESTED_BACKEND != MusaFusedMoeBackend.UPSTREAM
        and use_fp8_w8a8
        and not use_int8_w8a8
        and not use_int8_w8a16
        and not use_int4_w4a16
        and ocp_mx_scheme is None
        and (
            _MUSA_FUSED_MOE_REQUESTED_BACKEND != MusaFusedMoeBackend.AUTO
            or (
                hidden_states.dim() == 2
                and w1.dim() == 3
                and w2.dim() == 3
                and topk_ids.dim() == 2
                and has_calibrated_dimensions(
                    local_experts=w1.shape[0],
                    w1_output_size=w1.shape[1],
                    w2_input_size=w2.shape[2],
                    hidden_size=w1.shape[2],
                    top_k=topk_ids.shape[1],
                )
            )
        )
    ):
        device_index = hidden_states.device.index
        if device_index is None:
            try:
                device_index = torch.musa.current_device()
            except Exception:
                device_index = 0
        device_capability, multiprocessor_count = _musa_device_fingerprint(device_index)

        # The policy is calibrated only for S5000/MP31. Other MUSA
        # architectures retain the established upstream implementation.
        if device_capability == (3, 1):
            # The calibrated GEMV sweeps use router weights after the expert
            # projection.  Fail closed for the alternate input-weighting
            # contract until it has its own exact policy key and evidence.
            can_use_gemv = (
                not apply_router_weight_on_input
                and _can_use_musa_native_fp8_moe_gemv(
                    hidden_states=hidden_states,
                    w1=w1,
                    w2=w2,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    activation=activation,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    use_int4_w4a16=use_int4_w4a16,
                    ocp_mx_scheme=ocp_mx_scheme,
                    per_channel_quant=per_channel_quant,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    w1_zp=w1_zp,
                    w2_zp=w2_zp,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    w1_bias=w1_bias,
                    w2_bias=w2_bias,
                )
            )
            can_use_grouped_gemm = _can_use_musa_fp8_moe_grouped_gemm(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                ocp_mx_scheme=ocp_mx_scheme,
                per_channel_quant=per_channel_quant,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_zp=w1_zp,
                w2_zp=w2_zp,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_shape=block_shape,
                w1_bias=w1_bias,
                w2_bias=w2_bias,
            )
            if can_use_gemv or can_use_grouped_gemm:
                stream_is_capturing = _musa_stream_is_capturing()
                shape = _musa_fused_moe_shape(
                    hidden_states=hidden_states,
                    w1=w1,
                    w2=w2,
                    topk_ids=topk_ids,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    block_shape=block_shape,
                    activation=activation,
                    expert_map=expert_map,
                    stream_is_capturing=stream_is_capturing,
                    device_capability=device_capability,
                    multiprocessor_count=multiprocessor_count,
                )
                policy = thresholds_for_shape(shape)
                backend = select_fused_moe_backend(
                    shape=shape,
                    num_tokens=hidden_states.shape[0],
                    can_use_gemv=can_use_gemv,
                    can_use_grouped_gemm=can_use_grouped_gemm,
                    stream_is_capturing=stream_is_capturing,
                    requested=_MUSA_FUSED_MOE_REQUESTED_BACKEND,
                    thresholds=policy,
                )

    # Native GEMV records inside ``fused_experts_impl`` after expanding any
    # per-tensor scales.  Record every other dispatcher outcome here so the
    # opt-in inventory also observes unknown shapes that remain on upstream.
    if backend != MusaFusedMoeBackend.GEMV and _MOE_SHAPE_INVENTORY_ENABLED:
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

    if backend != MusaFusedMoeBackend.UPSTREAM:
        assert policy is not None
        assert shape is not None
        logger.info_once(
            "MUSA fused-MoE dispatcher selected backend=%s policy=%s for "
            "shape=(E=%d,N=%d,K=%d,topk=%d,graph=%s,mp=%d,gemv_block=%s).",
            backend.value,
            policy.source,
            shape.local_experts,
            shape.w1_output_size,
            shape.hidden_size,
            shape.top_k,
            shape.graph_mode,
            shape.multiprocessor_count,
            shape.gemv_block,
        )

    if backend == MusaFusedMoeBackend.GROUPED_GEMM:
        grouped_output = _maybe_musa_fp8_moe_grouped_gemm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            ocp_mx_scheme=ocp_mx_scheme,
            per_channel_quant=per_channel_quant,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )
        if grouped_output is not None:
            return grouped_output
    elif backend == MusaFusedMoeBackend.GEMV:
        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            ocp_mx_scheme,
            per_channel_quant,
            global_num_experts,
            expert_map,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            w1_bias,
            w2_bias,
            inplace=False,
            _allow_deepgemm_prefill=False,
        )

    bf16_prefill_candidate = (
        _MUSA_FUSED_MOE_REQUESTED_BACKEND == MusaFusedMoeBackend.AUTO
        and not use_fp8_w8a8
        and hidden_states.shape[0] >= _DEEPGEMM_BF16_PREFILL_MIN_TOKENS
        and w1.dim() == 3
        and w1.shape[0] in (256, 257)
    )
    if bf16_prefill_candidate and not _musa_stream_is_capturing():
        bf16_prefill_output = _maybe_moe_deepgemm_bf16_prefill(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
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
        if bf16_prefill_output is not None:
            return bf16_prefill_output

    # The default-on contiguous DeepGEMM is the large-M/prefill backend of the
    # established base path. Keep it after the small-M auto decision:
    # GEMV wins only in its calibrated decode window; large prefill must not
    # silently fall through to Triton because dispatch now owns this wrapper.
    deepgemm_prefill_candidate = (
        not _env_flag_disabled(_DEEPGEMM_PREFILL_ENV)
        and hidden_states.shape[0] >= _env_int(_DEEPGEMM_PREFILL_MIN_TOKENS_ENV, 2500)
        and w1.dim() == 3
        and w1.shape[0] in (256, 257)
    )
    if (
        _MUSA_FUSED_MOE_REQUESTED_BACKEND == MusaFusedMoeBackend.AUTO
        and deepgemm_prefill_candidate
        and not _musa_stream_is_capturing()
    ):
        try:
            deepgemm_w1_scale = _maybe_expand_fp8_moe_per_tensor_scale(w1_scale, w1)
            deepgemm_w2_scale = _maybe_expand_fp8_moe_per_tensor_scale(w2_scale, w2)
        except ValueError:
            # Unsupported expansion layouts may still be valid upstream.
            # Preserve their scales so the complete DeepGEMM gate declines.
            deepgemm_w1_scale = w1_scale
            deepgemm_w2_scale = w2_scale
        deepgemm_prefill_output = _maybe_moe_deepgemm_prefill(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            ocp_mx_scheme=ocp_mx_scheme,
            per_channel_quant=per_channel_quant,
            expert_map=expert_map,
            w1_scale=deepgemm_w1_scale,
            w2_scale=deepgemm_w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )
        if deepgemm_prefill_output is not None:
            return deepgemm_prefill_output

    return _upstream_fused_moe._musa_original_fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        ocp_mx_scheme,
        per_channel_quant,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        w1_bias,
        w2_bias,
    )


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
