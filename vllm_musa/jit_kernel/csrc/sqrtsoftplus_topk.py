"""DeepSeek-V4 sqrt-softplus top-k routing with an optional folded shared-expert
slot."""

from __future__ import annotations

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
from vllm_musa.jit_kernel.utils import cache_once

MAX_EXPERTS = 512
MAX_TOP_K = 8
# Layout of the expert route map of the small-batch FP8 MoE (kernels/fp8_moe_mma.py).
MAP_TOKENS = 16
MAP_SLOTS = 64
MAP_WIDTH = 18


@cache_once
def _module():
    # The flags of the extension that builds topkGatingSoftplusSqrt, so both
    # kernels round identically.
    return load_musa_jit(
        "vllm_musa_sqrtsoftplus_topk",
        ("topk/sqrtsoftplus_topk.mu",),
        extra_musa_cflags=("-fmusa-flush-denormals-to-zero", "-fno-signed-zeros"),
    )


def _sqrtsoftplus_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    bias_vl: torch.Tensor,
    input_ids: torch.Tensor,
    image_sentinel_lo: int,
    is_padding: torch.Tensor,
    has_padding: bool,
    out_w: torch.Tensor,
    out_ids: torch.Tensor,
    route_map: torch.Tensor,
    has_route_map: bool,
    x: torch.Tensor,
    xq: torch.Tensor,
    xs: torch.Tensor,
    quant_x: bool,
    renormalize: bool,
    scaling: float,
    shared_id: int,
    shared_weight: float,
) -> None:
    _module().sqrtsoftplus_topk(
        logits,
        bias,
        bias_vl,
        input_ids,
        image_sentinel_lo,
        is_padding,
        has_padding,
        out_w,
        out_ids,
        route_map,
        has_route_map,
        x,
        xq,
        xs,
        quant_x,
        renormalize,
        scaling,
        shared_id,
        shared_weight,
    )


def _no_op(*args, **kwargs) -> None:
    return None


direct_register_custom_op(
    op_name="musa_sqrtsoftplus_topk",
    op_func=_sqrtsoftplus_topk,
    mutates_args=["out_w", "out_ids", "route_map", "xq", "xs"],
    fake_impl=_no_op,
)


def sqrtsoftplus_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    renormalize: bool,
    scaling: float,
    is_padding: torch.Tensor | None = None,
    shared_id: int = -1,
    shared_weight: float = 1.0,
    with_route_map: bool = False,
    bias_vl: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
    x: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    tuple[torch.Tensor, torch.Tensor] | None,
]:
    """``(weights, ids, route_map, x_quant)`` of ``topk_hash_softplus_sqrt`` without a
    hash table, for FP32 or BF16 ``logits``; with ``shared_id >= 0`` one more column
    routes every token to ``shared_id`` at ``shared_weight``. Padding rows get id -1
    in every column.
    With ``bias_vl`` and ``image_sentinel_lo > 0``, tokens whose ``input_ids`` fall
    in the five sentinel ids from ``image_sentinel_lo`` select with ``bias_vl``.

    ``with_route_map`` also writes the expert route map of the small-batch FP8 MoE
    when the batch fits it (at most ``MAP_TOKENS`` tokens and ``MAP_SLOTS`` ids), and
    then ``x`` (the BF16 MoE input) is also quantized like
    ``per_token_group_quant_fp8(x, 32)`` into ``x_quant = (xq, xs)``. Otherwise
    both are None."""
    rows = logits.shape[0]
    cols = top_k + (1 if shared_id >= 0 else 0)
    device = logits.device
    out_w = torch.empty((rows, cols), dtype=torch.float32, device=device)
    out_ids = torch.empty((rows, cols), dtype=torch.int32, device=device)
    route_map = None
    x_quant = None
    if with_route_map and 0 < rows <= MAP_TOKENS and rows * cols <= MAP_SLOTS:
        route_map = torch.empty(
            ((MAP_SLOTS + 1) * MAP_WIDTH,), dtype=torch.int32, device=device
        )
        if (
            x is not None
            and x.dim() == 2
            and x.shape[0] == rows
            and x.dtype == torch.bfloat16
            and x.stride(-1) == 1
            and x.shape[1] % 32 == 0
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        ):
            x_quant = (
                torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=device),
                torch.empty(
                    (rows, x.shape[1] // 32), dtype=torch.float32, device=device
                ),
            )
    image = bias_vl is not None and image_sentinel_lo > 0
    if image:
        assert input_ids is not None, "image-token routing needs input_ids"

    def none() -> torch.Tensor:
        return torch.empty((0,), dtype=torch.int32, device=device)

    torch.ops.vllm.musa_sqrtsoftplus_topk(
        logits,
        bias,
        bias_vl if image else bias,
        input_ids if image else bias,
        int(image_sentinel_lo) if image else 0,
        is_padding if is_padding is not None else bias,
        is_padding is not None,
        out_w,
        out_ids,
        route_map if route_map is not None else none(),
        route_map is not None,
        x if x_quant is not None else none(),
        x_quant[0] if x_quant is not None else none(),
        x_quant[1] if x_quant is not None else none(),
        x_quant is not None,
        bool(renormalize),
        float(scaling),
        int(shared_id),
        float(shared_weight),
    )
    return out_w, out_ids, route_map, x_quant
