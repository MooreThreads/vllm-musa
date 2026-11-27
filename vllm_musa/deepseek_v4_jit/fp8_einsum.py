# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 FP8 einsum helpers for MUSA provider experiments."""

from __future__ import annotations

import torch

_GROUP_SIZE = 128
# The fused GEMV re-reads the weight per token row while DeepGEMM stays
# weight-bound; keep the GEMV only for single-request verify shapes.
_DEEPGEMM_MIN_TOKENS = 8
_BLOCK_GEMV_MAX_TOKENS = 16


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return getattr(tensor, "device", None) is not None and tensor.device.type == "musa"


def _normalize_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if weight.dim() == 2:
        flat_out_dim, in_dim = weight.shape
        if flat_out_dim % groups != 0:
            raise ValueError(
                "DeepSeek-V4 FP8 GEMV provider expected 2D weight rows to be "
                f"divisible by groups={groups}, got {tuple(weight.shape)}"
            )
        out_dim = flat_out_dim // groups
        weight = weight.reshape(groups, out_dim, in_dim)
    elif weight.dim() == 3:
        weight_groups, out_dim, in_dim = weight.shape
        if weight_groups != groups:
            raise ValueError(
                "DeepSeek-V4 FP8 GEMV provider group mismatch: "
                f"activation groups={groups}, weight groups={weight_groups}"
            )
    else:
        raise ValueError(
            "DeepSeek-V4 FP8 GEMV provider expects a 2D or 3D weight tensor, "
            f"got {tuple(weight.shape)}"
        )

    if out_dim % _GROUP_SIZE != 0 or in_dim % _GROUP_SIZE != 0:
        raise ValueError(
            "DeepSeek-V4 FP8 GEMV provider requires 128-aligned dimensions, "
            f"got out_dim={out_dim}, in_dim={in_dim}"
        )
    out_blocks = out_dim // _GROUP_SIZE
    in_blocks = in_dim // _GROUP_SIZE

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and weight_scale.dtype == e8m0_dtype:
        exp_bits = weight_scale.view(torch.uint8).to(torch.int32)
        scales = (exp_bits << 23).view(torch.float32)
    else:
        scales = weight_scale.to(torch.float32)

    if scales.dim() == 2 and groups == 1 and scales.shape == (out_blocks, in_blocks):
        scales = scales.unsqueeze(0)
    elif scales.dim() == 2 and scales.numel() == groups * out_blocks * in_blocks:
        scales = scales.reshape(groups, out_blocks, in_blocks)
    elif scales.shape == (groups, in_blocks, out_blocks):
        scales = scales.transpose(-1, -2)

    if scales.shape != (groups, out_blocks, in_blocks):
        raise ValueError(
            "DeepSeek-V4 FP8 GEMV provider scale shape mismatch: "
            f"scale_shape={tuple(weight_scale.shape)}, "
            f"normalized={tuple(scales.shape)}, "
            f"expected={(groups, out_blocks, in_blocks)}"
        )

    return weight, scales, out_dim, in_dim


def try_musa_deepseek_v4_fp8_einsum_gemv(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> tuple[bool, str]:
    """Dispatch ``bhr,hdr->bhd`` to DeepGEMM or, for small M, the fused GEMV."""
    if equation != "bhr,hdr->bhd":
        return False, f"unsupported equation {equation!r}"
    if activation.dim() != 3 or out.dim() != 3:
        return False, (
            "expected activation/out shapes [tokens, groups, hidden] and "
            f"[tokens, groups, out], got {tuple(activation.shape)} -> "
            f"{tuple(out.shape)}"
        )
    if activation_scale.dim() != 3:
        return False, f"expected activation_scale dim 3, got {activation_scale.dim()}"
    if activation.dtype != torch.float8_e4m3fn:
        return False, f"expected fp8 activation, got {activation.dtype}"
    if weight.dtype != torch.float8_e4m3fn:
        return False, f"expected fp8 weight, got {weight.dtype}"
    tensors = (activation, activation_scale, weight, weight_scale, out)
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return False, "all tensors must be on MUSA"

    tokens, groups, in_dim = activation.shape
    if activation_scale.shape != (tokens, groups, in_dim // _GROUP_SIZE):
        return False, (
            "activation_scale shape mismatch: "
            f"{tuple(activation_scale.shape)} for activation={tuple(activation.shape)}"
        )

    try:
        weight, scales, out_dim, normalized_in_dim = _normalize_weight(
            weight,
            weight_scale,
            groups,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if normalized_in_dim != in_dim or out.shape != (tokens, groups, out_dim):
        return False, (
            "shape mismatch after weight normalization: "
            f"activation={tuple(activation.shape)}, weight={tuple(weight.shape)}, "
            f"out={tuple(out.shape)}"
        )

    deepgemm_error: str | None = None
    if tokens >= _DEEPGEMM_MIN_TOKENS and out.dtype == torch.bfloat16:
        try:
            from vllm.utils.deep_gemm import fp8_gemm_nt

            for group_idx in range(groups):
                group_out_view = out[:, group_idx, :]
                copy_group_out = not group_out_view.is_contiguous()
                group_out = (
                    torch.empty_like(
                        group_out_view,
                        memory_format=torch.contiguous_format,
                    )
                    if copy_group_out
                    else group_out_view
                )
                fp8_gemm_nt(
                    (
                        activation[:, group_idx, :].contiguous(),
                        activation_scale[:, group_idx, :].contiguous(),
                    ),
                    (
                        weight[group_idx].contiguous(),
                        scales[group_idx].contiguous(),
                    ),
                    group_out,
                    is_deep_gemm_e8m0_used=False,
                )
                if copy_group_out:
                    group_out_view.copy_(group_out)
        except Exception as exc:
            deepgemm_error = f"{type(exc).__name__}: {exc}"
        else:
            return True, "musa_deepgemm_fp8_o_proj"

    try:
        from vllm_musa import _custom_ops as musa_ops

        for group_idx in range(groups):
            group_out_view = out[:, group_idx, :]
            direct_group_out = (
                group_out_view if group_out_view.is_contiguous() else None
            )
            group_out = musa_ops.musa_fused_gemv(
                activation[:, group_idx, :].contiguous(),
                weight[group_idx].contiguous(),
                activation_scale[:, group_idx, :].contiguous(),
                scales[group_idx].contiguous(),
                output=direct_group_out,
            )
            if direct_group_out is None:
                group_out_view.copy_(group_out)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if deepgemm_error is not None:
        return True, f"musa_fused_gemv (DeepGEMM fallback: {deepgemm_error})"
    return True, "musa_fused_gemv"


def _validate_group_size_divisible(name: str, size: int) -> None:
    if size % _GROUP_SIZE != 0:
        raise ValueError(
            f"{name} dimension must be divisible by {_GROUP_SIZE}, got {size}"
        )


def _dequant_activation(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
) -> torch.Tensor:
    tokens, groups, hidden = activation.shape
    _validate_group_size_divisible("activation hidden", hidden)
    scale_blocks = hidden // _GROUP_SIZE
    activation_blocks = activation.to(torch.float32).reshape(
        tokens,
        groups,
        scale_blocks,
        _GROUP_SIZE,
    )
    return (
        activation_blocks * activation_scale.to(torch.float32).unsqueeze(-1)
    ).reshape(tokens, groups, hidden)


def _dequant_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    try:
        weight, scales, out_dim, in_dim = _normalize_weight(
            weight,
            weight_scale,
            groups,
        )
    except ValueError:
        # Not a 128x128-block weight: take the block shape from the scales.
        weight, scales, out_dim, in_dim = _normalize_blocked_weight(
            weight,
            weight_scale,
            groups,
        )
    else:
        _validate_group_size_divisible("weight output", out_dim)
        _validate_group_size_divisible("weight input", in_dim)
    out_blocks, in_blocks = scales.shape[1], scales.shape[2]
    weight_blocks = weight.to(torch.float32).reshape(
        groups,
        out_blocks,
        out_dim // out_blocks,
        in_blocks,
        in_dim // in_blocks,
    )
    return (weight_blocks * scales[:, :, None, :, None]).reshape(
        groups, out_dim, in_dim
    )


def _normalize_blocked_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if weight.dim() == 2:
        if weight.shape[0] % groups != 0:
            raise ValueError(
                f"weight rows {weight.shape[0]} not divisible by groups={groups}"
            )
        weight = weight.reshape(groups, weight.shape[0] // groups, weight.shape[1])
    _, out_dim, in_dim = weight.shape

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and weight_scale.dtype == e8m0_dtype:
        exp_bits = weight_scale.view(torch.uint8).to(torch.int32)
        scales = (exp_bits << 23).view(torch.float32)
    else:
        scales = weight_scale.to(torch.float32)
    if scales.dim() == 2:
        if scales.shape[0] % groups != 0:
            raise ValueError(
                f"scale rows {scales.shape[0]} not divisible by groups={groups}"
            )
        scales = scales.reshape(groups, scales.shape[0] // groups, scales.shape[1])
    if scales.dim() != 3 or scales.shape[0] != groups:
        raise ValueError(f"unsupported weight scale shape {tuple(weight_scale.shape)}")
    out_blocks, in_blocks = scales.shape[1], scales.shape[2]
    if out_dim % out_blocks or in_dim % in_blocks:
        raise ValueError(
            f"weight {tuple(weight.shape)} is not tiled by scales "
            f"{tuple(weight_scale.shape)}"
        )
    return weight, scales, out_dim, in_dim


def _try_musa_fp8_einsum_fine_blocks(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> tuple[bool, str]:
    """``bhr,hdr->bhd`` for weights whose scale blocks are finer than 128."""
    if (
        activation.dim() != 3
        or activation_scale.dim() != 3
        or out.dim() != 3
        or activation.dtype != torch.float8_e4m3fn
        or weight.dtype != torch.float8_e4m3fn
        or out.dtype != torch.bfloat16
    ):
        return False, "expected 3D FP8 activation and BF16 output"
    tokens, groups, in_dim = activation.shape
    try:
        weight, scales, out_dim, weight_in_dim = _normalize_blocked_weight(
            weight,
            weight_scale,
            groups,
        )
    except ValueError as exc:
        return False, f"ValueError: {exc}"
    if weight_in_dim != in_dim or out.shape != (tokens, groups, out_dim):
        return False, "shape mismatch after blocked weight normalization"
    if activation_scale.shape[:2] != (tokens, groups):
        return False, f"activation_scale shape {tuple(activation_scale.shape)}"
    act_scale = activation_scale.to(torch.float32)

    from vllm_musa.kernels.fp8_block_gemv import (
        fp8_block_gemv,
        fp8_block_gemv_supported,
    )
    from vllm_musa.kernels.fp8_block_splitk import fp8_block_splitk_config

    if all(
        (
            tokens <= _BLOCK_GEMV_MAX_TOKENS
            or fp8_block_splitk_config(weight[g], scales[g], tokens) is not None
        )
        and fp8_block_gemv_supported(act_scale[:, g, :], weight[g], scales[g])
        for g in range(groups)
    ):
        for group_idx in range(groups):
            fp8_block_gemv(
                activation[:, group_idx, :],
                act_scale[:, group_idx, :],
                weight[group_idx],
                scales[group_idx],
                out[:, group_idx, :],
            )
        return True, "musa_fp8_block_gemv"

    from vllm_musa.kernels.fp8_block_bf16 import (
        fp8_block_bf16_linear,
        fp8_block_bf16_supported,
    )

    if (
        activation.stride(-1) == 1
        and act_scale.stride(-1) == 1
        and in_dim % act_scale.shape[2] == 0
        and all(
            fp8_block_bf16_supported(weight[g], scales[g], tokens)
            for g in range(groups)
        )
    ):
        for group_idx in range(groups):
            out[:, group_idx, :].copy_(
                fp8_block_bf16_linear(
                    activation[:, group_idx, :],
                    weight[group_idx],
                    scales[group_idx],
                    act_scale[:, group_idx, :],
                )
            )
        return True, "musa_fp8_block_bf16_gemm"

    # The block GEMM takes one activation scale per weight K block; expand
    # coarser activation groups to the weight's K block.
    block_n = out_dim // scales.shape[1]
    block_k = in_dim // scales.shape[2]
    act_group = in_dim // act_scale.shape[2]
    if act_group % block_k != 0 or act_group < block_k:
        return False, f"activation group {act_group} vs weight block {block_k}"
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    from vllm_musa.kernels.fp8_block_gemm import (
        fp8_block_scaled_mm,
        fp8_block_scaled_mm_supported,
    )

    reason = "musa_fp8_block_scaled_mm"
    for group_idx in range(groups):
        group_activation = activation[:, group_idx, :].contiguous()
        group_scale = act_scale[:, group_idx, :]
        if act_group != block_k:
            group_scale = group_scale.repeat_interleave(act_group // block_k, dim=-1)
        group_scale = group_scale.contiguous()
        if fp8_block_scaled_mm_supported(
            group_activation, group_scale, weight[group_idx], scales[group_idx]
        ):
            result = fp8_block_scaled_mm(
                group_activation,
                group_scale,
                weight[group_idx],
                scales[group_idx],
                out.dtype,
            )
        else:
            reason = "triton_block_scaled_mm"
            result = w8a8_triton_block_scaled_mm(
                group_activation,
                weight[group_idx],
                group_scale,
                scales[group_idx],
                [block_n, block_k],
                out.dtype,
            )
        out[:, group_idx, :].copy_(result)
    return True, reason


def try_musa_deepseek_v4_fp8_einsum(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> tuple[bool, str]:
    """Try supported MUSA replacements for DeepSeek-V4 FP8 einsum."""
    handled, reason = try_musa_deepseek_v4_fp8_einsum_gemv(
        activation,
        activation_scale,
        weight,
        weight_scale,
        out,
        equation,
    )
    if handled:
        return True, reason
    if equation != "bhr,hdr->bhd":
        return False, f"unsupported equation {equation!r}"
    if all(_is_musa_tensor(t) for t in (activation, weight, out)):
        handled, reason = _try_musa_fp8_einsum_fine_blocks(
            activation,
            activation_scale,
            weight,
            weight_scale,
            out,
        )
        if handled:
            return True, reason

    activation_deq = _dequant_activation(activation, activation_scale)
    weight_deq = _dequant_weight(weight, weight_scale, activation.shape[1])
    out.copy_(torch.einsum(equation, activation_deq, weight_deq).to(out.dtype))
    return True, "torch_dequant_einsum_fallback"
