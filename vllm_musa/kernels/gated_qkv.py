# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible re-export of :mod:`vllm_musa.ir.gated_qkv`."""

from vllm_musa.ir.gated_qkv import (  # noqa: F401
    _gated_qk_norm_rope_token_kernel,
    _supports_gated_qkv,
    gated_qkv_rms_norm_rope,
)

__all__ = ["gated_qkv_rms_norm_rope"]
