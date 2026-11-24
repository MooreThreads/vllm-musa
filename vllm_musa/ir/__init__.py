# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA providers for ``vllm.ir.ops.*``.

Loaded exclusively by ``vllm_musa.platform.MUSAPlatformBase.import_ir_kernels()``
after ``super().import_ir_kernels()`` so upstream providers (``native``,
``vllm_c``, ``oink``, ``aiter``, ``xpu_kernels``) register first and MUSA
providers appear alongside them.

Keep standalone MUSA/Triton kernel helper modules under ``vllm_musa/kernels/``;
place only ``@ir.ops.<op>.register_impl(...)`` provider modules here.
"""

from . import gated_qkv, musa_ops  # noqa: F401

__all__ = ["gated_qkv", "musa_ops"]
