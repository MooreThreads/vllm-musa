# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA attention-backend registrations."""

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

# Keyed by enum member name so that importing this module never dereferences a
# member; a backend this vLLM does not define is skipped at registration.
_MUSA_ATTENTION_BACKENDS: dict[str, str] = {
    "FLASHMLA": "vllm_musa.v1.attention.backends.mla.flashmla.MUSAFlashMLABackend",
    "FLASHMLA_SPARSE": (
        "vllm_musa.v1.attention.backends.mla.flashmla_sparse.MUSAFlashMLASparseBackend"
    ),
    "FLASH_ATTN": (
        "vllm_musa.v1.attention.backends.flash_attn.MUSAFlashAttentionBackend"
    ),
    "TURBOQUANT": (
        "vllm_musa.v1.attention.backends.turboquant.MUSATurboQuantAttentionBackend"
    ),
    "TREE_ATTN": ("vllm_musa.v1.attention.backends.tree_attn.MUSATreeAttentionBackend"),
}


def register_attention_backends() -> None:
    """Point vLLM's attention backends at the MUSA implementations."""
    for name, class_path in _MUSA_ATTENTION_BACKENDS.items():
        backend = getattr(AttentionBackendEnum, name, None)
        if backend is not None:
            register_backend(backend, class_path=class_path)
