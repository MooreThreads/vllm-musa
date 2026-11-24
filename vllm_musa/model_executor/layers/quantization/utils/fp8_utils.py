# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import get_tma_aligned_size, is_deep_gemm_e8m0_used


def _upcast_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    exp_bits = scale.view(torch.uint8).to(torch.int32)
    return (exp_bits << 23).view(torch.float32)


def deepgemm_post_process_fp8_weight_block(
    wq: torch.Tensor,
    ws: torch.Tensor,
    quant_block_shape: tuple[int, ...],
    use_e8m0: bool,
    is_bmm: bool = False,
    bmm_batch_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del quant_block_shape, is_bmm, bmm_batch_size
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and ws.dtype == e8m0_dtype and not use_e8m0:
        return wq, _upcast_e8m0_to_fp32(ws)
    return wq, ws


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dtype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
        column_major_scales: Outputs scales in column major.
        tma_aligned_scales: Outputs scales in TMA-aligned layout.
        out_q: Optional output tensor. If not provided, function will create.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor.
    """
    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"
    if current_platform.is_musa() and x.dim() == 2 and not x.is_contiguous():
        x = x.contiguous()

    fp8_min, fp8_max = get_fp8_min_max()

    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = torch.empty(x.shape, device=x.device, dtype=dtype)

    # Allocate the scale tensor in either row- or column-major format.
    if column_major_scales:
        if tma_aligned_scales:
            m = x.shape[-2]
            sf_k = x.shape[-1] // group_size
            tma_aligned_m = get_tma_aligned_size(m, 4)
            shape = x.shape[:-2] + (m, sf_k)
            stride = (
                (1, tma_aligned_m)
                if x.dim() == 2
                else (tma_aligned_m * sf_k, 1, tma_aligned_m)
            )
            x_s = torch.empty_strided(
                shape, stride, device=x.device, dtype=torch.float32
            )
        else:
            shape = x.shape[:-2] + (x.shape[-1] // group_size, x.shape[-2])
            x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(
                -1, -2
            )
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    # Prefer the MUSA kernel when possible. MUSA Triton fallback is unavailable,
    # so materialize supported strided 2D activations before dispatching.
    if current_platform.is_musa():
        if x.dim() != 2:
            raise NotImplementedError(
                f"MUSA backend currently only supports 2D tensors for per_token_group_fp8_quant. "
                f"Got tensor with {x.dim()} dimensions, shape={x.shape}"
            )
        x_kernel = x if x.is_contiguous() else x.contiguous()

        # Register-resident vec kernel for the common group_size==128 bf16/fp16
        # activation-quant path (no shared-memory staging). Column-major and
        # TMA-aligned scale layouts are honored via output_s strides. Other
        # configs (ue8m0, non-128 groups, fp32 input) use the stable kernel; so
        # does a contiguous view whose base is not 16-byte aligned, which the
        # kernel's 128-bit vector load requires.
        if (
            group_size == 128
            and not use_ue8m0
            and x_kernel.dtype in (torch.bfloat16, torch.float16)
            and x_kernel.data_ptr() % 16 == 0
        ):
            torch.ops._C_musa_ops.per_token_group_quant_8bit_vec(
                x_kernel,
                x_q,
                x_s,
                group_size,
                eps,
                fp8_min,
                fp8_max,
            )
        else:
            torch.ops._C.per_token_group_fp8_quant(
                x_kernel,
                x_q,
                x_s,
                group_size,
                eps,
                fp8_min,
                fp8_max,
                use_ue8m0,
                column_major_scales,
                tma_aligned_scales,
            )
        return x_q, x_s.contiguous()

    # TRITON FALLBACK
    # musa currently does not support triton fallback.
    raise NotImplementedError(
        f"per_token_group_fp8_quant is not supported for platform: {current_platform} or input is not contiguous. "
        "MUSA Triton fallback is currently not supported."
    )


def _silu_fp8_fusion_enabled() -> bool:
    value = os.getenv("VLLM_MUSA_SILU_FP8_QUANT_FUSION", "1").lower()
    return value not in ("0", "false", "no", "off")


def _silu_fp8_fusion_max_tokens() -> int:
    try:
        return int(os.getenv("VLLM_MUSA_SILU_FP8_QUANT_MAX_TOKENS", "512"))
    except ValueError:
        return 512


def _sphere_silu_fp8_enabled() -> bool:
    value = os.getenv("VLLM_MUSA_SPHERE_SILU_FP8", "1").lower()
    return value not in ("0", "false", "no", "off")


def _sphere_silu_fp8_max_tokens() -> int:
    try:
        return int(os.getenv("VLLM_MUSA_SPHERE_SILU_FP8_MAX_TOKENS", "256"))
    except ValueError:
        return 256


_SPHERE_SILU_FP8_MOD = None
_SPHERE_SILU_FP8_PROBE_DONE = False


def _maybe_sphere_silu_fp8():
    """Lazy-load `sphere_silu_post_quant`; cache success / failure once per process."""
    global _SPHERE_SILU_FP8_MOD, _SPHERE_SILU_FP8_PROBE_DONE
    if _SPHERE_SILU_FP8_PROBE_DONE:
        return _SPHERE_SILU_FP8_MOD
    _SPHERE_SILU_FP8_PROBE_DONE = True
    try:
        import sphere_silu_post_quant  # noqa: F401

        _SPHERE_SILU_FP8_MOD = sphere_silu_post_quant
    except (ImportError, ModuleNotFoundError):
        _SPHERE_SILU_FP8_MOD = None
    return _SPHERE_SILU_FP8_MOD


def silu_mul_per_token_group_quant_fp8_musa(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MUSA small-M SiLU+mul plus FP8 group-quant fast path.

    This mirrors vLLM's DeepGEMM helper contract for gate-up tensors shaped
    ``[tokens, 2 * hidden]`` but only returns row-major contiguous FP32 scales.
    Unsupported shapes fall back to the upstream helper.
    """

    from vllm.model_executor.layers.quantization.utils import (
        fp8_utils as upstream_fp8_utils,
    )

    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()

    def fallback() -> tuple[torch.Tensor, torch.Tensor]:
        return upstream_fp8_utils.silu_mul_per_token_group_quant_fp8_colmajor(
            input=input,
            output=output,
            use_ue8m0=use_ue8m0,
            eps=eps,
        )

    group_size = 128
    if (
        not _silu_fp8_fusion_enabled()
        or not current_platform.is_musa()
        or use_ue8m0
        or input.dim() != 2
        or not input.is_contiguous()
        or input.dtype not in (torch.bfloat16, torch.float16)
        or input.shape[0] > _silu_fp8_fusion_max_tokens()
        or input.shape[-1] % (2 * group_size) != 0
    ):
        return fallback()

    tokens = input.shape[0]
    hidden = input.shape[-1] // 2

    fp8_dtype = current_platform.fp8_dtype()

    # sphere_silu_post_quant_v1 is ~1.6-1.9x faster than the
    # _C_musa_ops kernel for small-M (tokens <= 256, hidden_dim == 3072,
    # row-major non-UE8M0 scale). Bit-exact dequant match validated at
    # M=128. If the caller pre-allocated `output`, we must copy into it
    # (the copy overhead negates most of the win, so we still try it but
    # the bigger gain is when output is None).
    if (
        _sphere_silu_fp8_enabled()
        and tokens <= _sphere_silu_fp8_max_tokens()
        and hidden == 1536  # M2.5 gate_up: input [tokens, 3072] -> out [tokens, 1536]
    ):
        sphere = _maybe_sphere_silu_fp8()
        if sphere is not None:
            sq, ss = sphere.silu_post_quant_v1(input)
            expected_q_shape = (tokens, hidden)
            expected_s_shape = (tokens, hidden // group_size)
            if (
                sq.shape == expected_q_shape
                and ss.shape == expected_s_shape
                and sq.dtype == fp8_dtype
                and ss.dtype == torch.float32
            ):
                if output is None:
                    return sq, ss
                if (
                    output.shape == expected_q_shape
                    and output.is_contiguous()
                    and output.dtype == fp8_dtype
                ):
                    output.copy_(sq)
                    return output, ss

    if output is None:
        output = torch.empty(
            (tokens, hidden),
            device=input.device,
            dtype=fp8_dtype,
        )
    elif (
        output.shape != (tokens, hidden)
        or not output.is_contiguous()
        or output.dtype != fp8_dtype
    ):
        return fallback()

    output_s = torch.empty(
        (tokens, hidden // group_size),
        device=input.device,
        dtype=torch.float32,
    )
    fp8_min, fp8_max = get_fp8_min_max()
    torch.ops._C_musa_ops.silu_and_mul_per_token_group_fp8_quant(
        input,
        output,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
    )
    return output, output_s


_HOOKS_INSTALLED = False


def install_hooks() -> None:
    """Install MUSA ``deepgemm_post_process_fp8_weight_block`` override (idempotent)."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    import vllm.model_executor.layers.quantization.utils.fp8_utils as _upstream_fp8_utils

    _upstream_fp8_utils.deepgemm_post_process_fp8_weight_block = (
        deepgemm_post_process_fp8_weight_block
    )
    _HOOKS_INSTALLED = True
