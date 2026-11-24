# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone MUSA / Triton kernel helpers.

vLLM IR providers now live in :mod:`vllm_musa.ir` (loaded by
``MUSAPlatformBase.import_ir_kernels()``) so importing a standalone kernel
helper from ``vllm_musa.kernels.<name>`` does not trigger IR provider
registration as an unintended side effect.
"""
