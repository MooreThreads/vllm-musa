# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility exports for MUSA tuning thresholds.

The CAR-RMSNorm shape/TP policy lives in
``vllm_musa.optimization_contract.car_rmsnorm``.  This module retains the
existing import surface for downstream callers.
"""

from vllm_musa.optimization_contract.car_rmsnorm import (
    CAR_RMSNORM_POLICY_TABLE,
    FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY,
    FUSED_ALLREDUCE_RMSNORM_POLICY_VERSION,
    FUSED_ALLREDUCE_RMSNORM_TARGET_HIDDEN_SIZE,
    FUSED_ALLREDUCE_RMSNORM_TP2_MIN_ROWS,
    FUSED_ALLREDUCE_RMSNORM_TP4_HIDDEN_SIZE,
    FUSED_ALLREDUCE_RMSNORM_TP4_POLICY,
    FUSED_ALLREDUCE_RMSNORM_TP4_SINGLETON_ROWS,
    can_use_fused_allreduce_rmsnorm,
    fused_allreduce_rmsnorm_compile_endpoints,
    fused_allreduce_rmsnorm_compile_reject_reason,
    fused_allreduce_rmsnorm_config_reject_reason,
    can_enable_fused_allreduce_rmsnorm,
    infer_car_rmsnorm_model_family,
)

# Minimum rows for the MUSA JIT fused-add RMSNorm provider on contiguous BF16
# H5120 workloads.
FUSED_ADD_RMSNORM_MIN_ROWS = FUSED_ALLREDUCE_RMSNORM_TP2_MIN_ROWS

__all__ = [
    "CAR_RMSNORM_POLICY_TABLE",
    "FUSED_ADD_RMSNORM_MIN_ROWS",
    "FUSED_ALLREDUCE_RMSNORM_POLICY_VERSION",
    "FUSED_ALLREDUCE_RMSNORM_MODEL_FAMILY",
    "FUSED_ALLREDUCE_RMSNORM_TP2_MIN_ROWS",
    "FUSED_ALLREDUCE_RMSNORM_TP4_HIDDEN_SIZE",
    "FUSED_ALLREDUCE_RMSNORM_TP4_POLICY",
    "FUSED_ALLREDUCE_RMSNORM_TP4_SINGLETON_ROWS",
    "FUSED_ALLREDUCE_RMSNORM_TARGET_HIDDEN_SIZE",
    "can_use_fused_allreduce_rmsnorm",
    "fused_allreduce_rmsnorm_compile_endpoints",
    "fused_allreduce_rmsnorm_compile_reject_reason",
    "fused_allreduce_rmsnorm_config_reject_reason",
    "can_enable_fused_allreduce_rmsnorm",
    "infer_car_rmsnorm_model_family",
]
