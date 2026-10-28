# SPDX-License-Identifier: Apache-2.0
"""Kernel-specific selectors: table in, launch tuple out.

The generic machinery in :mod:`vllm_musa.tactics` answers "which tactic"; a
selector is the one place that knows how a particular kernel wants it phrased
(``fp8_block_splitk`` wants a 7-tuple, a Triton kernel wants keyword arguments).
One module per kernel, named after the kernel.

Torch-free by rule: a selector may shape data but may not touch a tensor, so the
tables stay testable on a machine with no GPU.  Everything that needs a tensor --
dtype checks, strides, the launch itself -- stays in the kernel module.
"""
