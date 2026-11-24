# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible re-export of :mod:`vllm_musa.ir.musa_ops`."""

from vllm_musa.ir.musa_ops import (  # noqa: F401
    MUSA_RMS_NORM_SUPPORTED,
    _c_ext_fused_add_rms_norm_supports_args,
    _c_ext_rms_norm_supports_args,
    _fused_add_rms_norm_supports_args,
    _has_C_rms_norm,
    _has_jit_rms_norm,
    _jit_fused_add_rms_norm_supports_args,
    _jit_rms_norm_supports_args,
    _rms_norm_supports_args,
    _select_musa_fused_add_rms_norm_impl,
    _select_musa_rms_norm_impl,
    fused_add_rms_norm,
    rms_norm,
)
