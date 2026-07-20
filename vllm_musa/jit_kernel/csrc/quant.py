from __future__ import annotations

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
from vllm_musa.jit_kernel.utils import cache_once

_SILU_ACTIVATION_TYPE = 0


@cache_once
def _quant_v2_module():
    return load_musa_jit(
        "vllm_musa_quant_v2",
        ("quant/per_token_group_quant_8bit_v2.mu",),
        extra_musa_cflags=(
            "-fmusa-flush-denormals-to-zero",
            "-fno-signed-zeros",
            "-mllvm",
            "-mtgpu-opt-level=1",
            "-mllvm",
            "-mtgpu-load-store-opt=1",
            "-mllvm",
            "-mtgpu-fold-global-ldst=1",
        ),
    )


def per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
) -> None:
    """Per-token-group 8-bit quant, optionally fusing silu+mul over a gated input.

    With fuse_silu_and_mul, `input` is (tokens, 2 * hidden) and `output_q` is
    (tokens, hidden): the gate/up halves are combined before quantizing.
    """
    torch.ops.vllm.musa_csrc_per_token_group_quant_8bit(
        input,
        output_q,
        output_s,
        int(group_size),
        float(eps),
        float(min_8bit),
        float(max_8bit),
        bool(scale_ue8m0),
        bool(fuse_silu_and_mul),
    )


_DUMMY_MASKED_M: dict[torch.device, torch.Tensor] = {}


def _dummy_masked_m(device: torch.device) -> torch.Tensor:
    t = _DUMMY_MASKED_M.get(device)
    if t is None:
        t = torch.empty((1,), device=device, dtype=torch.int32)
        _DUMMY_MASKED_M[device] = t
    return t


def _per_token_group_quant_8bit_custom(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool,
    fuse_silu_and_mul: bool,
) -> None:
    # The kernel takes a masked_m tensor for the masked-layout callers; the contiguous
    # layout used here has no mask, so pass a cached dummy and flag it off.
    masked_m = _dummy_masked_m(input.device)
    _quant_v2_module().sgl_per_token_group_quant_8bit_v2(
        input,
        output_q,
        output_s,
        int(group_size),
        float(eps),
        float(min_8bit),
        float(max_8bit),
        bool(scale_ue8m0),
        bool(fuse_silu_and_mul),
        _SILU_ACTIVATION_TYPE,
        masked_m,
        False,
    )


def _per_token_group_quant_8bit_custom_fake(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool,
    fuse_silu_and_mul: bool,
) -> None:
    return


direct_register_custom_op(
    op_name="musa_csrc_per_token_group_quant_8bit",
    op_func=_per_token_group_quant_8bit_custom,
    mutates_args=["output_q", "output_s"],
    fake_impl=_per_token_group_quant_8bit_custom_fake,
)
