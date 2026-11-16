# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The split-k scratch is only reserved for sizes capture can replay."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

# qsa imports the layer module back, so load the model module first.
import vllm.models.qwen4_exp.nvidia.model  # noqa: F401
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
from vllm.platforms import current_platform


class _Prewarm:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


@pytest.fixture
def prewarm(monkeypatch):
    module = types.ModuleType("vllm_musa.jit_kernel.tilelang.qsa_sparse_attn_v2")
    captured = _Prewarm()
    module.prewarm_qsa_kernels = captured
    monkeypatch.setitem(
        sys.modules, "vllm_musa.jit_kernel.tilelang.qsa_sparse_attn_v2", module
    )
    monkeypatch.setattr(current_platform, "is_musa", lambda: True)
    return captured


def _layer() -> SimpleNamespace:
    return SimpleNamespace(
        num_heads=32,
        num_kv_heads=2,
        head_dim=128,
        indexer=SimpleNamespace(packed_output_width=513),
        kv_cache_torch_dtype=torch.bfloat16,
    )


def _config(capture_sizes, max_num_seqs: int = 256) -> SimpleNamespace:
    return SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=capture_sizes),
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_triton_decode_reserves_no_scratch(monkeypatch, prewarm):
    monkeypatch.delenv("VLLM_MUSA_QSA_TILELANG_DECODE", raising=False)

    Qwen4ExpQSAAttention._musa_prewarm_qsa_tilelang(
        _layer(), _config([1, 2, 4, 8, 16, 24, 32])
    )

    assert len(prewarm.calls) == 1
    assert prewarm.calls[0][1]["decode_rows"] == ()


def test_tilelang_decode_reserves_capture_sizes_only(monkeypatch, prewarm):
    monkeypatch.setenv("VLLM_MUSA_QSA_TILELANG_DECODE", "1")

    Qwen4ExpQSAAttention._musa_prewarm_qsa_tilelang(_layer(), _config([8, 1, 32]))

    assert len(prewarm.calls) == 1
    assert prewarm.calls[0][1]["decode_rows"] == (1, 8, 32)
