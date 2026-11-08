# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA attention-backend registrations."""

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

MUSA_ATTENTION_BACKENDS: dict[AttentionBackendEnum, str] = {
    AttentionBackendEnum.FLASHMLA: (
        "vllm_musa.v1.attention.backends.mla.flashmla.MUSAFlashMLABackend"
    ),
    AttentionBackendEnum.FLASHMLA_SPARSE: (
        "vllm_musa.v1.attention.backends.mla.flashmla_sparse."
        "MUSAFlashMLASparseBackend"
    ),
    AttentionBackendEnum.FLASH_ATTN: (
        "vllm_musa.v1.attention.backends.flash_attn.MUSAFlashAttentionBackend"
    ),
    AttentionBackendEnum.TURBOQUANT: (
        "vllm_musa.v1.attention.backends.turboquant.MUSATurboQuantAttentionBackend"
    ),
}


def register_attention_backends() -> None:
    """Point vLLM's attention backends at the MUSA implementations."""
    for backend, class_path in MUSA_ATTENTION_BACKENDS.items():
        register_backend(backend, class_path=class_path)
    # Only registered when this vLLM defines TREE_ATTN.
    tree_attn_backend = getattr(AttentionBackendEnum, "TREE_ATTN", None)
    if tree_attn_backend is not None:
        register_backend(
            tree_attn_backend,
            class_path=(
                "vllm_musa.v1.attention.backends.tree_attn.MUSATreeAttentionBackend"
            ),
        )
