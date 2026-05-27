import logging
from typing import Optional, Union

import torch

try:
    import vllm_musa._C  # noqa: F401
except ImportError as e:
    logging.error("Failed to import from vllm._C: %r", e)


def musa_fused_gemv_moe(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale,
    B_scale,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    topk: int,
    use_int4_w4a16: bool,
    use_swigelu: bool,
) -> None:
    return torch.ops._C_musa_ops.musa_fused_gemv_moe(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        topk_ids,
        mul_routed_weight,
        topk,
        use_int4_w4a16,
        use_swigelu,
    )


def musa_fused_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    x_scales: torch.Tensor = None,
    qweight_scales: torch.Tensor = None,
    use_swigelu: bool = False,
    use_rms_norm: bool = False,
    gamma: torch.Tensor = None,
    eps: float = 1e-6,
):
    use_int4_w4a16 = False
    out_shape = x.shape[:-1] + (
        qweight.shape[0] if not use_swigelu else qweight.shape[0] // 2,
    )
    assert not (
        use_swigelu and use_rms_norm
    ), "gemv only fused one activation (swigelu or rms_norm)!"

    if use_rms_norm:
        if gamma is None:
            assert False, "rms_norm gamm is None!"

    # fp8 grouped matmul
    if qweight.dtype == torch.float8_e4m3fn:
        # x: [m, k]
        # qweight: [n, k]
        # x_scales: [m, k / 128]
        # qweight: [n / 128, k / 128]
        # assert x_scales is not None, "FP8 grouped matmul x scales is None!"
        assert (
            qweight.dtype == torch.float8_e4m3fn
        ), "FP8 grouped matmul weight only support float8_e4m3fn!"
        assert qweight_scales is not None, "FP8 grouped matmul weight scales is None!"
        output = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
        torch.ops._C_musa_ops.musa_fused_gemv(
            x,
            qweight,
            output,
            x_scales,
            qweight_scales,
            use_int4_w4a16,
            use_swigelu,
            use_rms_norm,
            gamma,
            eps,
        )
        return output
    # w4a16 gemv
    elif qweight_scales is not None:
        # qweight: [out, in/8]
        # scales: [out, in / group_size]
        assert (
            x.dtype == torch.bfloat16 or x.dtype == torch.float16
        ), "W4A16 gemv only support bfloat16 or float16!"
        use_int4_w4a16 = True
        out_shape = x.shape[:-1] + (
            qweight.shape[0] if not use_swigelu else qweight.shape[0] // 2,
        )
        output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        torch.ops._C_musa_ops.musa_fused_gemv(
            x,
            qweight,
            output,
            None,
            qweight_scales,
            use_int4_w4a16,
            use_swigelu,
            use_rms_norm,
            gamma,
            eps,
        )
        return output
    # general gemv
    else:
        output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        torch.ops._C_musa_ops.musa_fused_gemv(
            x,
            qweight,
            output,
            None,
            None,
            use_int4_w4a16,
            use_swigelu,
            use_rms_norm,
            gamma,
            eps,
        )
        return output


def musa_fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    return torch.ops._C_musa_ops.musa_fused_add_rms_norm(
        input,
        residual,
        weight,
        eps,
    )


def musa_reshape_and_cache_flash_nhd(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return torch.ops._C_musa_ops.musa_reshape_and_cache_flash_nhd(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
    )


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


def _top_k_renorm_probs_internal(
    probs: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
) -> torch.Tensor:
    probs = probs.float()
    maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
    renorm_probs = torch.empty_like(probs)
    torch.ops._C_musa_ops.top_k_renorm_probs.default(
        probs, renorm_probs, maybe_top_k_arr, top_k_val
    )
    return renorm_probs


def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))


def _top_p_renorm_probs_internal(
    probs: torch.Tensor,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
) -> torch.Tensor:
    probs = probs.float()
    maybe_top_p_arr = maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
    renorm_probs = torch.empty_like(probs)
    torch.ops._C_musa_ops.top_p_renorm_probs.default(
        probs, renorm_probs, maybe_top_p_arr, top_p_val
    )
    return renorm_probs


def top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    return _top_p_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_p))


def _top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor],
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    device = probs.device
    probs = probs.float()
    maybe_top_p_arr = maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
    samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
    torch.ops._C_musa_ops.top_p_sampling_from_probs.default(
        probs,
        samples,
        indices,
        maybe_top_p_arr,
        top_p_val,
        deterministic,
        generator,
    )
    return samples


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    if check_nan and torch.any(torch.isnan(probs)):
        raise ValueError("Input probs contains NaN.")
    return _top_p_sampling_from_probs_internal(
        probs, indices, *_to_tensor_scalar_tuple(top_p), deterministic, generator
    )


def _top_k_top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor],
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    device = probs.device
    probs = probs.float()
    maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
    maybe_top_p_arr = maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
    samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
    torch.ops._C_musa_ops.musa_top_k_top_p_sampling_from_probs.default(
        probs,
        samples,
        indices,
        maybe_top_k_arr,
        top_k_val,
        maybe_top_p_arr,
        top_p_val,
        deterministic,
        generator,
    )
    return samples


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    if filter_apply_order == "top_k_first":
        renorm_probs = top_k_renorm_probs(probs, top_k)
        return top_p_sampling_from_probs(
            renorm_probs,
            top_p,
            indices,
            deterministic,
            generator=generator,
            check_nan=check_nan,
        )
    if filter_apply_order == "joint":
        if check_nan and torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
        return _top_k_top_p_sampling_from_probs_internal(
            probs,
            indices,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
            generator,
        )
    raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


def _min_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor],
    maybe_min_p_arr: Optional[torch.Tensor],
    min_p_val: float,
    deterministic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    device = probs.device
    probs = probs.float()
    maybe_min_p_arr = maybe_min_p_arr.float() if maybe_min_p_arr is not None else None
    samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
    torch.ops._C_musa_ops.min_p_sampling_from_probs.default(
        probs,
        samples,
        indices,
        maybe_min_p_arr,
        min_p_val,
        deterministic,
        generator,
    )
    return samples


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    if check_nan and torch.any(torch.isnan(probs)):
        raise ValueError("Input probs contains NaN.")
    return _min_p_sampling_from_probs_internal(
        probs, indices, *_to_tensor_scalar_tuple(min_p), deterministic, generator
    )
