from __future__ import annotations

import logging

import torch

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
from vllm_musa.jit_kernel.utils import cache_once

logger = logging.getLogger(__name__)


@cache_once
def _module():
    return load_musa_jit(
        "vllm_musa_qsa_fast_topk",
        ("topk/qsa_fast_topk.mu",),
    )


def qsa_fast_topk(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_indices: torch.Tensor,
    block_topk: int,
    row_starts: torch.Tensor,
) -> None:
    """Run the fixed-width radix-select kernel used by SGLang's QSA path."""
    if logits.dtype != torch.float32:
        raise TypeError("QSA fast top-k expects fp32 logits")
    if block_topk not in (512, 2048):
        raise ValueError(f"unsupported QSA top-k width: {block_topk}")
    lengths = visible_blocks.to(device=logits.device, dtype=torch.int32)
    _module().vllm_musa_qsa_fast_topk(
        logits, row_starts, block_indices, lengths, block_topk
    )


def qsa_fast_topk_available() -> bool:
    try:
        _module()
        return True
    except Exception:
        logger.warning("MUSA QSA fast top-k unavailable", exc_info=True)
        return False
