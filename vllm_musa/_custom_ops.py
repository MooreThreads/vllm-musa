import logging
from functools import cache

import torch

_QWEN_MIN_P_VOCABS = (151936, 248320)
_QWEN_MIN_P_MAX_BATCH = 64
_QWEN_TOP_K_RADIX_K = 50
_QWEN_TOP_K_RADIX_MAX_BATCH = 256
_QWEN_TOP_K_RADIX_VOCABS = (151936, 248320)
_TOP_K_MASK_MAX_ROWS = 4096
_TOP_K_MASK_MAX_K = 1024
_TOP_K_MASK_VOCABS = (151936, 152064, 248320)

try:
    import vllm_musa._C  # noqa: F401
except ImportError as e:
    logging.error("Failed to import from vllm._C: %r", e)

try:
    # MUSA: register the _moe_C MoE ops (moe_align_block_size, topk_softmax, moe_sum).
    import vllm._moe_C  # noqa: F401
except ImportError as e:
    logging.error("Failed to import from vllm._moe_C: %r", e)


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
    block_n: int = 0,
    block_k: int = 0,
    swiglu_limit: float = 0.0,
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
        block_n,
        block_k,
        swiglu_limit,
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
    output: torch.Tensor | None = None,
):
    use_int4_w4a16 = False
    out_shape = x.shape[:-1] + (
        qweight.shape[0] if not use_swigelu else qweight.shape[0] // 2,
    )
    assert not (use_swigelu and use_rms_norm), (
        "gemv only fused one activation (swigelu or rms_norm)!"
    )

    if use_rms_norm and gamma is None:
        raise AssertionError("rms_norm gamm is None!")

    # fp8 grouped matmul
    if qweight.dtype == torch.float8_e4m3fn:
        # x: [m, k]
        # qweight: [n, k]
        # x_scales: [m, k / 128]
        # qweight: [n / 128, k / 128]
        # assert x_scales is not None, "FP8 grouped matmul x scales is None!"
        assert qweight.dtype == torch.float8_e4m3fn, (
            "FP8 grouped matmul weight only support float8_e4m3fn!"
        )
        assert qweight_scales is not None, "FP8 grouped matmul weight scales is None!"
        if output is None:
            output = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
        else:
            assert output.shape == out_shape
            assert output.device == x.device
            assert output.dtype == torch.bfloat16
            assert output.is_contiguous()
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
        assert x.dtype == torch.bfloat16 or x.dtype == torch.float16, (
            "W4A16 gemv only support bfloat16 or float16!"
        )
        use_int4_w4a16 = True
        out_shape = x.shape[:-1] + (
            qweight.shape[0] if not use_swigelu else qweight.shape[0] // 2,
        )
        assert output is None, "caller-owned output is only supported for FP8 GEMV"
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
        assert output is None, "caller-owned output is only supported for FP8 GEMV"
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
    block_x: int = 0,
) -> None:
    return torch.ops._C_musa_ops.musa_fused_add_rms_norm(
        input,
        residual,
        weight,
        eps,
        block_x,
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


def _musa_device_index(device: torch.device) -> int | None:
    """Resolve a logical MUSA device index without changing the current device."""
    try:
        device_id = device.index
        if device_id is None:
            device_id = int(torch.musa.current_device())
        if device_id < 0 or device_id >= torch.musa.device_count():
            return None
        return device_id
    except Exception:
        return None


@cache
def _get_musa_device_capability(device_id: int) -> tuple[int, int]:
    """Query one logical MUSA device and cache successful results."""
    return torch.musa.get_device_capability(device_id)


def _is_validated_musa_device(device: torch.device) -> bool:
    """Return whether the tensor's device is the validated S5000 target."""
    try:
        device_id = _musa_device_index(device)
        if device_id is None:
            return False
        # torch.musa accepts visible logical IDs, matching torch.device.index.
        return _get_musa_device_capability(device_id) == (3, 1)
    except Exception:
        # A device query can fail while the platform is still initializing.
        # Falling back is safer than selecting an architecture-specialized op.
        return False


def _is_supported_musa_generator(
    generator: torch.Generator | None, device: torch.device
) -> bool:
    """Check that a seeded generator belongs to the sampled MUSA device."""
    if generator is None:
        return True
    try:
        generator_device = generator.device
        if getattr(generator_device, "type", None) != "musa":
            return False
        tensor_device_id = _musa_device_index(device)
        generator_device_id = _musa_device_index(generator_device)
        return (
            tensor_device_id is not None
            and generator_device_id is not None
            and tensor_device_id == generator_device_id
        )
    except Exception:
        return False


def _can_use_chunked_min_p_sampler(
    probs: torch.Tensor,
    indices: torch.Tensor | None,
    maybe_min_p_arr: torch.Tensor | None,
    deterministic: bool,
    generator: torch.Generator | None,
) -> bool:
    """Gate the chunked min-p kernel to its validated production contract."""
    return (
        deterministic
        and probs.device.type == "musa"
        and _is_validated_musa_device(probs.device)
        and probs.dtype == torch.float32
        and probs.ndim == 2
        and 0 < probs.shape[0] <= _QWEN_MIN_P_MAX_BATCH
        and probs.shape[1] in _QWEN_MIN_P_VOCABS
        and probs.is_contiguous()
        and _is_supported_musa_generator(generator, probs.device)
        and (
            indices is None
            or (
                indices.dtype == torch.int32
                and indices.device == probs.device
                and indices.is_contiguous()
                and indices.numel() >= probs.shape[0]
            )
        )
        and (
            maybe_min_p_arr is None
            or (
                maybe_min_p_arr.device == probs.device
                and maybe_min_p_arr.dtype == torch.float32
                and maybe_min_p_arr.is_contiguous()
                and maybe_min_p_arr.numel() >= probs.shape[0]
            )
        )
    )


def _top_k_renorm_probs_internal(
    probs: torch.Tensor,
    maybe_top_k_arr: torch.Tensor | None,
    top_k_val: int,
) -> torch.Tensor:
    probs = probs.float()
    maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
    renorm_probs = torch.empty_like(probs)
    # tests/test_rubymine_top_k_renorm_source.py pins this vocab tuple as a
    # literal, so the default dispatch cannot drift behind an environment gate.
    use_rubymine = (
        maybe_top_k_arr is None
        and top_k_val == 50
        and probs.device.type == "musa"
        and _is_validated_musa_device(probs.device)
        and probs.ndim == 2
        and probs.is_contiguous()
        and 0 < probs.shape[0] <= _QWEN_TOP_K_RADIX_MAX_BATCH
        and probs.shape[1] in (151936, 248320)
    )
    if use_rubymine:
        torch.ops._C_musa_ops.musa_rubymine_top_k_renorm_probs.default(
            probs, renorm_probs, top_k_val
        )
        return renorm_probs
    torch.ops._C_musa_ops.top_k_renorm_probs.default(
        probs, renorm_probs, maybe_top_k_arr, top_k_val
    )
    return renorm_probs


def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: torch.Tensor | int,
) -> torch.Tensor:
    return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))


def _supports_native_mask(logits: torch.Tensor) -> bool:
    return (
        logits.device.type == "musa"
        and _is_validated_musa_device(logits.device)
        and logits.ndim == 2
        and logits.is_contiguous()
        and logits.dtype in (torch.bfloat16, torch.float32)
        and 0 < logits.shape[0] <= _TOP_K_MASK_MAX_ROWS
        and logits.shape[1] in _TOP_K_MASK_VOCABS
        and (logits.shape[1] % 4) == 0
    )


def _per_row_top_k_applies(logits: torch.Tensor, top_k: torch.Tensor) -> bool:
    # The kernel maps k < 1 to "no cut" and clamps k to the vocabulary, so only
    # the shape and dtype are checked here: reading the values back would sync
    # the host on every per-row-k decode step.
    return (
        top_k.dim() == 1
        and top_k.numel() == logits.shape[0]
        and top_k.dtype in (torch.int32, torch.int64)
    )


def _can_use_native_top_k_mask(logits: torch.Tensor, top_k: int | torch.Tensor) -> bool:
    """Whether the native logits-domain radix top-k mask applies."""
    if not _supports_native_mask(logits):
        return False
    if isinstance(top_k, torch.Tensor):
        return _per_row_top_k_applies(logits, top_k)
    return (
        isinstance(top_k, int)
        and 1 <= top_k <= _TOP_K_MASK_MAX_K
        and top_k < logits.shape[1]
    )


def top_k_mask_logits(
    logits: torch.Tensor, top_k: int | torch.Tensor
) -> torch.Tensor | None:
    """Mask ``logits`` to its top-k values in place, without a softmax pass.

    ``top_k`` may be a scalar or a per-row int tensor, so mixed k recipes in one
    batch stay on the radix kernel.  A per-row 0 means "no top-k" for that row.
    Returns ``None`` when the native kernel does not apply so callers can fall
    back.  Values tied with the k-th largest stay eligible.
    """
    if not _can_use_native_top_k_mask(logits, top_k):
        return None
    if isinstance(top_k, torch.Tensor):
        k_arr = top_k.to(torch.int32)
        if not k_arr.is_contiguous():
            k_arr = k_arr.contiguous()
        torch.ops._C_musa_ops.musa_top_k_mask_logits.default(logits, logits, k_arr, 0)
        return logits
    torch.ops._C_musa_ops.musa_top_k_mask_logits.default(
        logits, logits, None, int(top_k)
    )
    return logits


def _can_use_native_top_p_mask(
    logits: torch.Tensor,
    top_p: float | torch.Tensor,
    top_k: int | torch.Tensor | None,
) -> bool:
    """Whether the fused native logits-domain top-k/top-p mask applies."""
    if not _supports_native_mask(logits):
        return False
    if isinstance(top_p, torch.Tensor):
        if top_p.dim() != 1 or top_p.numel() != logits.shape[0]:
            return False
        if top_p.dtype != torch.float32:
            return False
    elif isinstance(top_p, (int, float)):
        if not 0.0 < float(top_p) <= 1.0:
            return False
    else:
        return False
    if top_k is None:
        return True
    if isinstance(top_k, torch.Tensor):
        return _per_row_top_k_applies(logits, top_k)
    return isinstance(top_k, int) and 0 <= top_k <= _TOP_K_MASK_MAX_K


def top_p_mask_logits(
    logits: torch.Tensor,
    top_p: float | torch.Tensor,
    top_k: int | torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Apply top-k then top-p to ``logits`` in place without sorting a row.

    The kernel resolves the k-th largest key by counting radix, then the top-p
    cut by radix selection over ``exp(value - max)`` mass, so neither a softmax
    tensor nor a vocabulary sort is materialized.  Values tied with either
    boundary stay eligible as a group.  Returns ``None`` when unsupported.
    """
    if not _can_use_native_top_p_mask(logits, top_p, top_k):
        return None
    k_arr = None
    k_val = 0
    if isinstance(top_k, torch.Tensor):
        k_arr = top_k.to(torch.int32)
        if not k_arr.is_contiguous():
            k_arr = k_arr.contiguous()
    elif isinstance(top_k, int) and top_k > 0:
        k_val = int(top_k)
    p_arr = None
    p_val = 1.0
    if isinstance(top_p, torch.Tensor):
        p_arr = top_p if top_p.is_contiguous() else top_p.contiguous()
    else:
        p_val = float(top_p)
    torch.ops._C_musa_ops.musa_top_p_mask_logits.default(
        logits, logits, k_arr, k_val, p_arr, p_val
    )
    return logits


def _can_mask_logits_with_radix_top_k(logits: torch.Tensor, top_k: int) -> bool:
    """Whether the radix top-k renorm kernel can define the logits mask."""
    return (
        top_k == _QWEN_TOP_K_RADIX_K
        and logits.device.type == "musa"
        and _is_validated_musa_device(logits.device)
        and logits.ndim == 2
        and logits.is_contiguous()
        and 0 < logits.shape[0] <= _QWEN_TOP_K_RADIX_MAX_BATCH
        and logits.shape[1] in _QWEN_TOP_K_RADIX_VOCABS
    )


def top_k_mask_logits_radix(logits: torch.Tensor, top_k: int) -> torch.Tensor | None:
    """Mask ``logits`` to its uniform top-k set without ``torch.topk``.

    The radix renormalization resolves the k-th largest probability, so the same
    threshold is recovered without a sort.  Values tied with the k-th largest
    stay eligible, matching ``apply_top_k_pytorch``.  The mask is applied in
    place, like ``masked_fill_`` in the pytorch fallback.  Returns ``None`` when
    the radix specialization does not apply, so callers can fall back.
    """
    if not _can_mask_logits_with_radix_top_k(logits, top_k):
        return None
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    renorm_probs = _top_k_renorm_probs_internal(probs, None, top_k)
    threshold = torch.where(renorm_probs > 0, probs, float("inf")).amin(
        dim=-1, keepdim=True
    )
    return logits.masked_fill_(probs < threshold, float("-inf"))


def _top_p_renorm_probs_internal(
    probs: torch.Tensor,
    maybe_top_p_arr: torch.Tensor | None,
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
    top_p: torch.Tensor | float,
) -> torch.Tensor:
    return _top_p_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_p))


def _top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    indices: torch.Tensor | None,
    maybe_top_p_arr: torch.Tensor | None,
    top_p_val: float,
    deterministic: bool,
    generator: torch.Generator | None,
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
    top_p: torch.Tensor | float,
    indices: torch.Tensor | None = None,
    deterministic: bool = True,
    generator: torch.Generator | None = None,
    check_nan: bool = False,
) -> torch.Tensor:
    if check_nan and torch.any(torch.isnan(probs)):
        raise ValueError("Input probs contains NaN.")
    return _top_p_sampling_from_probs_internal(
        probs, indices, *_to_tensor_scalar_tuple(top_p), deterministic, generator
    )


def _top_k_top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    indices: torch.Tensor | None,
    maybe_top_k_arr: torch.Tensor | None,
    top_k_val: int,
    maybe_top_p_arr: torch.Tensor | None,
    top_p_val: float,
    deterministic: bool,
    generator: torch.Generator | None,
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
    top_k: torch.Tensor | int,
    top_p: torch.Tensor | float,
    indices: torch.Tensor | None = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: torch.Generator | None = None,
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
    indices: torch.Tensor | None,
    maybe_min_p_arr: torch.Tensor | None,
    min_p_val: float,
    deterministic: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    device = probs.device
    input_probs = probs
    input_min_p_arr = maybe_min_p_arr
    probs = probs.float()
    maybe_min_p_arr = maybe_min_p_arr.float() if maybe_min_p_arr is not None else None
    samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
    use_chunked = _can_use_chunked_min_p_sampler(
        input_probs,
        indices,
        input_min_p_arr,
        deterministic,
        generator,
    )
    op = (
        torch.ops._C_musa_ops.musa_chunked_min_p_sampling_from_probs.default
        if use_chunked
        else torch.ops._C_musa_ops.min_p_sampling_from_probs.default
    )
    op(
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
    min_p: torch.Tensor | float,
    indices: torch.Tensor | None = None,
    deterministic: bool = True,
    generator: torch.Generator | None = None,
    check_nan: bool = False,
) -> torch.Tensor:
    if check_nan and torch.any(torch.isnan(probs)):
        raise ValueError("Input probs contains NaN.")
    return _min_p_sampling_from_probs_internal(
        probs, indices, *_to_tensor_scalar_tuple(min_p), deterministic, generator
    )


def deepseek_v4_store_sparse_kv(
    normed: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    write_mask: torch.Tensor,
) -> None:
    return torch.ops._C_musa_ops.deepseek_v4_store_sparse_kv(
        normed,
        kv_cache,
        slot_mapping,
        write_mask,
    )


def deepseek_v4_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    cache_block_size: int,
    q_head_padded: int = 0,
    apply_q_norm: bool = True,
) -> torch.Tensor:
    if slot_mapping.shape[0] > q.shape[0]:
        # Graph+MTP warmup can carry padded cache slots while q/kv only hold
        # active rows. The native op stores one KV row per q/kv row.
        slot_mapping = slot_mapping[: q.shape[0]].contiguous()
    from vllm_musa.kernels.deepseek_v4_rope_kv_insert import (
        rope_kv_insert,
        rope_kv_insert_supported,
    )

    if rope_kv_insert_supported(
        q,
        kv,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        cache_block_size,
        q_head_padded,
    ):
        return rope_kv_insert(
            q,
            kv,
            kv_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            eps,
            cache_block_size,
            q_head_padded,
            apply_q_norm,
        )
    return torch.ops._C_musa_ops.deepseek_v4_qnorm_rope_kv_insert(
        q,
        kv,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        cache_block_size,
        q_head_padded,
        apply_q_norm,
    )


def deepseek_v4_c4_indexer_compress_cache(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    rms_eps: float,
    state_block_size: int,
    state_width: int,
    kv_block_size: int,
) -> None:
    return torch.ops._C_musa_ops.deepseek_v4_c4_indexer_compress_cache(
        state_cache,
        token_to_req_indices,
        positions,
        state_slot_mapping,
        block_table,
        rms_norm_weight,
        cos_sin_cache,
        kv_cache,
        kv_slot_mapping,
        rms_eps,
        state_block_size,
        state_width,
        kv_block_size,
    )


def deepseek_v4_sparse_compress_cache(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    rms_eps: float,
    state_block_size: int,
    state_width: int,
    kv_block_size: int,
    compress_ratio: int,
    token_stride: int,
    scale_dim: int,
    quant_block: int,
    kv_states: torch.Tensor | None = None,
    score_states: torch.Tensor | None = None,
    ape: torch.Tensor | None = None,
) -> None:
    return torch.ops._C_musa_ops.deepseek_v4_sparse_compress_cache(
        state_cache,
        token_to_req_indices,
        positions,
        state_slot_mapping,
        block_table,
        rms_norm_weight,
        cos_sin_cache,
        kv_cache,
        kv_slot_mapping,
        rms_eps,
        state_block_size,
        state_width,
        kv_block_size,
        compress_ratio,
        token_stride,
        scale_dim,
        quant_block,
        kv_states,
        score_states,
        ape,
    )


def deepseek_v4_fused_q_kv_rmsnorm(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C_musa_ops.deepseek_v4_fused_q_kv_rmsnorm(
        q,
        kv,
        q_weight,
        kv_weight,
        eps,
    )


def deepseek_v4_dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    return torch.ops._C_musa_ops.deepseek_v4_dequantize_and_gather_k_cache(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
    )


def deepseek_v4_compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C_musa_ops.deepseek_v4_compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req_indices,
        block_table,
        block_size,
        is_valid_token,
    )


def deepseek_v4_combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C_musa_ops.deepseek_v4_combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )


def deepseek_v4_indexer_topk_decode(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
) -> None:
    return torch.ops._C_musa_ops.deepseek_v4_indexer_topk_decode(
        q_quant,
        kv_cache,
        weights,
        seq_lens,
        block_table,
        topk_indices,
        topk,
    )


def deepseek_v4_indexer_topk_prefill(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
) -> None:
    return torch.ops._C_musa_ops.deepseek_v4_indexer_topk_prefill(
        q_quant,
        kv_cache,
        weights,
        block_table,
        cu_seq_lens,
        token_to_seq,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
        topk,
    )


def deepseek_v4_indexer_rerank_prefill(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    candidate_abs_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
) -> None:
    return torch.ops._C_musa_ops.deepseek_v4_indexer_rerank_prefill(
        q_quant,
        kv_cache,
        weights,
        block_table,
        cu_seq_lens,
        token_to_seq,
        cu_seqlen_ks,
        cu_seqlen_ke,
        candidate_abs_indices,
        topk_indices,
        topk,
    )


def sparse_indexer_fill_all(
    lengths: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
) -> None:
    return torch.ops._C_musa_ops.sparse_indexer_fill_all(
        lengths,
        topk_indices,
        topk,
    )


def sparse_indexer_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
) -> None:
    return torch.ops._C_musa_ops.sparse_indexer_topk(
        logits,
        row_starts,
        row_ends,
        topk_indices,
        topk,
    )


def sparse_indexer_topk_decode(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
) -> None:
    return torch.ops._C_musa_ops.sparse_indexer_topk_decode(
        logits,
        seq_lens,
        topk_indices,
        topk,
    )


def deepseek_v4_sparse_indexer_topk_decode(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
) -> None:
    return torch.ops._C_musa_ops.deepseek_v4_sparse_indexer_topk_decode(
        logits,
        seq_lens,
        topk_indices,
        topk,
    )


def glm52_indexer_topk_decode(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
) -> None:
    return torch.ops._C_musa_ops.glm52_indexer_topk_decode(
        q_quant,
        kv_cache,
        weights,
        seq_lens,
        block_table,
        topk_indices,
        topk,
    )


def glm52_indexer_topk_prefill(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
) -> None:
    return torch.ops._C_musa_ops.glm52_indexer_topk_prefill(
        q_quant,
        kv_cache,
        weights,
        block_table,
        cu_seq_lens,
        token_to_seq,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
        topk,
    )


def deepseek_v4_sparse_flashmla_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    extra_k_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_topk_length: torch.Tensor | None,
    out: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C_musa_ops.deepseek_v4_sparse_flashmla_decode(
        q,
        k_cache,
        indices,
        topk_length,
        attn_sink,
        extra_k_cache,
        extra_indices,
        extra_topk_length,
        out,
        softmax_scale,
    )


def deepseek_v4_fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    tma_aligned_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_musa.kernels.deepseek_v4_inv_rope_fp8_quant import (
        inv_rope_fp8_quant,
        inv_rope_fp8_quant_supported,
    )

    if inv_rope_fp8_quant_supported(
        o,
        positions,
        cos_sin_cache,
        nope_dim,
        rope_dim,
        quant_group_size,
        tma_aligned_scales,
    ):
        return inv_rope_fp8_quant(
            o, positions, cos_sin_cache, n_groups, heads_per_group
        )
    return torch.ops._C_musa_ops.deepseek_v4_fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        quant_group_size,
        tma_aligned_scales,
    )


def deepseek_v4_topk_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
) -> None:
    return torch.ops._C_musa_ops.deepseek_v4_topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias,
        input_tokens,
        hash_indices_table,
    )


def deepseek_v4_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> None:
    return torch.ops._C_musa_ops.deepseek_v4_mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
