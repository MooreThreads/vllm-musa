# SPDX-License-Identifier: Apache-2.0
"""FP8 GEMM for block-scaled weights whose K tile spans several scale groups.

The generic Triton block GEMM applies one activation and one weight scale per K
step, so its K tile is one scale group. With 32-wide groups that is a 32-wide
K step, which leaves the GEMM latency bound at every M. This kernel loads a
wider K tile per step as consecutive scale groups and scales each group's
partial product before accumulating, which is the same arithmetic.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fp8_block_scaled_mm_kernel(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_asm,
    stride_bsn,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = ((pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M).to(tl.int64)
    offs_bn = ((pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N).to(tl.int64)
    offs_k = tl.arange(0, GROUP_K)
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :]
    b_ptrs = B + offs_bn[None, :] * stride_bn + offs_k[:, None]
    as_ptrs = As + offs_am * stride_asm
    bs_ptrs = Bs + (offs_bn // GROUP_N) * stride_bsn
    groups = tl.cdiv(K, GROUP_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        for j in tl.static_range(BLOCK_K // GROUP_K):
            kk = k0 + j * GROUP_K
            g = kk // GROUP_K
            a = tl.load(a_ptrs + kk, mask=offs_k[None, :] < K - kk, other=0.0)
            b = tl.load(b_ptrs + kk, mask=offs_k[:, None] < K - kk, other=0.0)
            a_s = tl.load(as_ptrs + g, mask=g < groups, other=0.0)
            b_s = tl.load(bs_ptrs + g, mask=g < groups, other=0.0)
            acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        C + offs_cm[:, None].to(tl.int64) * stride_cm + offs_cn[None, :],
        acc.to(C.dtype.element_ty),
        mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N),
    )


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) by M bucket, measured on MTT
# S5000 for the DeepSeek-V4.1 TP8 projections with 32x32 scale blocks. A bucket
# without a config keeps the generic Triton block GEMM, which measured as fast.
_K5120_CLASS = (
    (64, (32, 64, 256, 8, 2)),
    (512, (32, 64, 128, 4, 2)),
    (None, (64, 64, 64, 4, 2)),
)
_K1024_CLASS = (
    (64, (32, 64, 64, 8, 2)),
    (256, (64, 64, 128, 8, 3)),
    (1024, None),
    (None, (64, 64, 64, 4, 2)),
)
_MEASURED_CONFIGS = {
    (1792, 5120): _K5120_CLASS,
    (576, 5120): _K5120_CLASS,
    (1024, 4096): _K5120_CLASS,
    (4096, 1280): _K1024_CLASS,
    (5120, 1024): _K1024_CLASS,
    (5120, 288): ((64, (32, 64, 64, 8, 2)), (1024, None), (None, (64, 64, 32, 4, 3))),
    (25600, 6144): (
        (16, (32, 64, 32, 4, 3)),
        (128, (64, 64, 32, 4, 3)),
        (None, (64, 64, 64, 4, 2)),
    ),
}


def fp8_block_scaled_mm_config(
    m: int, n: int, k: int
) -> tuple[int, int, int, int, int] | None:
    """Measured launch config for an M x N x K problem, or None to keep the
    generic GEMM."""
    for max_m, config in _MEASURED_CONFIGS.get((n, k), ()):
        if max_m is None or m <= max_m:
            return config
    return None


def fp8_block_scaled_mm_supported(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> bool:
    """Whether the scales tile K and N in 16/32/64-wide groups and the shape has a
    config."""
    if (
        a.dim() != 2
        or weight.dim() != 2
        or a_scale.dim() != 2
        or weight_scale.dim() != 2
    ):
        return False
    if a.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
        return False
    if a_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        return False
    if a.stride(-1) != 1 or weight.stride(-1) != 1:
        return False
    if a_scale.stride(-1) != 1 or weight_scale.stride(-1) != 1:
        return False
    m, k = a.shape
    n = weight.shape[0]
    if weight.shape[1] != k or a_scale.shape[0] != m:
        return False
    group_k = triton.cdiv(k, weight_scale.shape[1])
    group_n = triton.cdiv(n, weight_scale.shape[0])
    return (
        group_k in (16, 32, 64)
        and group_n in (16, 32, 64, 128)
        and triton.cdiv(k, group_k) == weight_scale.shape[1] == a_scale.shape[1]
        and triton.cdiv(n, group_n) == weight_scale.shape[0]
        and fp8_block_scaled_mm_config(m, n, k) is not None
    )


def fp8_block_scaled_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``dequant(a) @ dequant(weight).T`` with per-group activation and weight
    scales."""
    m, k = a.shape
    n = weight.shape[0]
    group_k = triton.cdiv(k, weight_scale.shape[1])
    group_n = triton.cdiv(n, weight_scale.shape[0])
    block_m, block_n, block_k, num_warps, num_stages = fp8_block_scaled_mm_config(
        m, n, k
    )
    block_k = max(block_k, group_k)
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _fp8_block_scaled_mm_kernel[grid](
        a,
        weight,
        out,
        a_scale,
        weight_scale,
        m,
        n,
        k,
        a.stride(0),
        weight.stride(0),
        out.stride(0),
        a_scale.stride(0),
        weight_scale.stride(0),
        GROUP_N=group_n,
        GROUP_K=group_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
