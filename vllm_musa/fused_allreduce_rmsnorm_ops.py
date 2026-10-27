# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA CAR-RMSNorm fused custom-op bindings.

This module exposes opaque custom ops to Inductor and routes runtime execution
to the MUSA JIT fused custom-allreduce + RMSNorm implementations. The graph-level
ABIs return both the RMSNorm output and the tensor preserved for downstream graph
users: all-reduced tensor for the no-residual path; residual output plus raw
all-reduced tensor for the residual path.
"""

from __future__ import annotations

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm.logger import init_logger
from vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce import (
    get_musa_jit_custom_allreduce_comm,
)

logger = init_logger(__name__)


def _musa_fused_allreduce_rms_norm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runtime implementation: use the registered MUSA JIT fused kernel."""
    comm = get_musa_jit_custom_allreduce_comm(comm_id)
    kwargs = {"gemma": True} if gemma else {}
    result = comm.fused_allreduce_rmsnorm(input, weight, float(eps), **kwargs)
    if result is None:
        raise RuntimeError(
            "MUSA fused allreduce-rmsnorm requires the registered communicator "
            "to accept this tensor for fused custom all-reduce + RMSNorm"
        )
    return result


def _musa_fused_allreduce_rms_norm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, eps, comm_id, gemma
    return torch.empty_like(input), torch.empty_like(input)


def _musa_fused_allreduce_residual_rms_norm_impl(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runtime implementation: use the registered MUSA JIT residual fused kernel."""
    comm = get_musa_jit_custom_allreduce_comm(comm_id)
    kwargs = {"gemma": True} if gemma else {}
    result = comm.fused_allreduce_residual_rmsnorm(
        input, residual, weight, float(eps), **kwargs
    )
    if result is None:
        reason_fn = getattr(
            comm, "reject_fused_allreduce_residual_rmsnorm_reason", None
        )
        reason = (
            reason_fn(input, residual, weight)
            if callable(reason_fn)
            else "registered communicator did not provide a rejection reason"
        )
        raise RuntimeError(
            "MUSA fused allreduce-residual-rmsnorm requires the registered "
            "communicator to accept these tensors for fused custom all-reduce + "
            f"residual + RMSNorm; rejection reason: {reason}"
        )
    return result


def _musa_fused_allreduce_residual_rms_norm_no_raw_impl(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runtime implementation for residual fused kernel without raw CAR output."""
    comm = get_musa_jit_custom_allreduce_comm(comm_id)
    kwargs = {"gemma": True} if gemma else {}
    result = comm.fused_allreduce_residual_rmsnorm_no_raw(
        input, residual, weight, float(eps), **kwargs
    )
    if result is None:
        reason_fn = getattr(
            comm, "reject_fused_allreduce_residual_rmsnorm_reason", None
        )
        reason = (
            reason_fn(input, residual, weight)
            if callable(reason_fn)
            else "registered communicator did not provide a rejection reason"
        )
        raise RuntimeError(
            "MUSA fused allreduce-residual-rmsnorm no-raw requires the registered "
            "communicator to accept these tensors for fused custom all-reduce + "
            f"residual + RMSNorm; rejection reason: {reason}"
        )
    return result


def _musa_fused_allreduce_residual_rms_norm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del residual, weight, eps, comm_id, gemma
    return torch.empty_like(input), torch.empty_like(input), torch.empty_like(input)


def _musa_fused_allreduce_residual_rms_norm_no_raw_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del residual, weight, eps, comm_id, gemma
    return torch.empty_like(input), torch.empty_like(input)


try:
    direct_register_custom_op(
        op_name="musa_fused_allreduce_rms_norm",
        op_func=_musa_fused_allreduce_rms_norm_impl,
        fake_impl=_musa_fused_allreduce_rms_norm_fake,
    )
except RuntimeError as exc:
    # The module can be imported more than once in plugin-heavy processes.
    if "musa_fused_allreduce_rms_norm" not in str(exc):
        raise
    logger.debug("MUSA fused allreduce-rmsnorm custom op already registered")


try:
    direct_register_custom_op(
        op_name="musa_fused_allreduce_residual_rms_norm",
        op_func=_musa_fused_allreduce_residual_rms_norm_impl,
        fake_impl=_musa_fused_allreduce_residual_rms_norm_fake,
    )
except RuntimeError as exc:
    # The module can be imported more than once in plugin-heavy processes.
    if "musa_fused_allreduce_residual_rms_norm" not in str(exc):
        raise
    logger.debug("MUSA fused allreduce-residual-rmsnorm custom op already registered")


try:
    direct_register_custom_op(
        op_name="musa_fused_allreduce_residual_rms_norm_no_raw",
        op_func=_musa_fused_allreduce_residual_rms_norm_no_raw_impl,
        fake_impl=_musa_fused_allreduce_residual_rms_norm_no_raw_fake,
    )
except RuntimeError as exc:
    # The module can be imported more than once in plugin-heavy processes.
    if "musa_fused_allreduce_residual_rms_norm_no_raw" not in str(exc):
        raise
    logger.debug(
        "MUSA fused allreduce-residual-rmsnorm no-raw custom op already registered"
    )


def musa_fused_allreduce_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.musa_fused_allreduce_rms_norm(
        input,
        weight,
        eps,
        comm_id,
        gemma,
    )


def musa_fused_allreduce_residual_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.musa_fused_allreduce_residual_rms_norm(
        input,
        residual,
        weight,
        eps,
        comm_id,
        gemma,
    )


def musa_fused_allreduce_residual_rms_norm_no_raw(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    comm_id: int,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.musa_fused_allreduce_residual_rms_norm_no_raw(
        input,
        residual,
        weight,
        eps,
        comm_id,
        gemma,
    )
