# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.utils.deep_gemm.
"""

PATCHES = [
    # Patch is_deep_gemm_supported to support musa device type
    (
        "is_supported_arch = current_platform.is_cuda()",
        "is_supported_arch = current_platform.is_musa()",
    ),
    # Patch is_device_capability to support musa device type
    (
        "current_platform.is_device_capability(90)",
        "current_platform.is_device_capability(31)",
    ),
    # Patch get_mk_alignment_for_contiguous_layout to support musa deepep
    (
        "mk_align_size = _get_mk_alignment_for_contiguous_layout_impl()",
        "mk_align_size = 128",
    ),
]
