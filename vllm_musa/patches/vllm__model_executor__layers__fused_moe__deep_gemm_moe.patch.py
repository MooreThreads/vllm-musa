# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.fused_moe.deep_gemm_moe.DeepGemmExperts
"""

PATCHES = [
    # Patch DeepGemmExperts.apply where m_grouped_fp8_gemm_nt_contiguous need a2q_scale is_contiguous
    (
        """        mm2_out = _resize_cache(workspace2, (M_sum, K))
        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids
        )
""",
        """        mm2_out = _resize_cache(workspace2, (M_sum, K))
        a2q_scale = a2q_scale.contiguous()
        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids
        )
""",
    ),
]
