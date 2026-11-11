# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MUSA QSA warmup must stay scoped to models that carry QSA layers."""

import importlib
import sys
import types

import pytest
import torch
from torch import nn
from vllm.model_executor.warmup import qwen4_exp_qsa_warmup
from vllm.platforms import current_platform


class _Probe:
    def __init__(self) -> None:
        self.calls = 0

    def qsa_fast_topk_available(self) -> bool:
        self.calls += 1
        return True


class _BlockTable:
    def get_device_tensor(self, max_num_reqs: int) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.int32)


class _Runner:
    def __init__(self) -> None:
        self.kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=[types.SimpleNamespace(layer_names=["idx", "attn"])]
        )
        self.input_batch = types.SimpleNamespace(block_table=[_BlockTable()])
        self.max_num_reqs = 1
        self.max_num_tokens = 8
        self.uniform_decode_query_len = 1


class _Worker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def get_model(self) -> "_Worker":
        return self


@pytest.fixture
def probe(monkeypatch):
    module = types.ModuleType("vllm_musa.jit_kernel.csrc.qsa_topk")
    probed = _Probe()
    module.qsa_fast_topk_available = probed.qsa_fast_topk_available
    monkeypatch.setitem(sys.modules, "vllm_musa.jit_kernel.csrc.qsa_topk", module)
    monkeypatch.setattr(current_platform, "is_musa", lambda: True)
    return probed


def _install_qsa_modules(monkeypatch):
    class QSAIndexer(nn.Module):
        pass

    class Qwen4ExpQSAAttention(nn.Module):
        pass

    qsa_module = types.ModuleType("vllm.models.qwen4_exp.nvidia.indexer_qsa")
    qsa_module.QSAIndexer = QSAIndexer
    attn_module = types.ModuleType("vllm.models.qwen4_exp.nvidia.qsa")
    attn_module.Qwen4ExpQSAAttention = Qwen4ExpQSAAttention
    monkeypatch.setitem(
        sys.modules, "vllm.models.qwen4_exp.nvidia.indexer_qsa", qsa_module
    )
    monkeypatch.setitem(sys.modules, "vllm.models.qwen4_exp.nvidia.qsa", attn_module)
    return QSAIndexer, Qwen4ExpQSAAttention


def _stub_triton_warmups(monkeypatch, calls):
    indexer_ops = importlib.import_module(
        "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"
    )
    qsa_ops = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa")

    def decode(*args, **kwargs):
        calls.append("decode")
        return ()

    def sparse(*args, **kwargs):
        calls.append("sparse")
        return ()

    monkeypatch.setattr(indexer_ops, "warmup_qsa_mqa_paged_decode", decode)
    monkeypatch.setattr(qsa_ops, "warmup_qsa_sparse_paged_attention", sparse)


def test_other_musa_models_do_not_build_the_qsa_topk(monkeypatch, probe):
    monkeypatch.delitem(
        sys.modules, "vllm.models.qwen4_exp.nvidia.indexer_qsa", raising=False
    )
    monkeypatch.delitem(sys.modules, "vllm.models.qwen4_exp.nvidia.qsa", raising=False)

    qwen4_exp_qsa_warmup.qwen4_exp_qsa_triton_warmup(_Worker())

    assert probe.calls == 0


def test_qwen4_exp_model_without_qsa_layers_does_not_build_the_qsa_topk(
    monkeypatch, probe
):
    _install_qsa_modules(monkeypatch)

    qwen4_exp_qsa_warmup.qwen4_exp_qsa_triton_warmup(_Worker())

    assert probe.calls == 0


def test_qwen4_exp_model_with_qsa_layers_warms_every_kernel(monkeypatch, probe):
    indexer_cls, attention_cls = _install_qsa_modules(monkeypatch)
    calls: list[str] = []
    _stub_triton_warmups(monkeypatch, calls)

    worker = _Worker()
    worker.use_v2_model_runner = False
    worker.model_runner = _Runner()
    worker.indexer = indexer_cls()
    worker.indexer.compressed_key_cache = types.SimpleNamespace(
        prefix="idx", kv_cache=torch.ones(1)
    )
    worker.indexer.index_n_heads = 2
    worker.indexer.index_head_dim = 128
    worker.indexer.output_width = 512
    worker.attention = attention_cls()
    worker.attention.layer_name = "attn"
    worker.attention.kv_cache = torch.ones(1)
    worker.attention.num_heads = 4

    qwen4_exp_qsa_warmup.qwen4_exp_qsa_triton_warmup(worker)

    assert probe.calls == 1
    assert calls == ["decode", "sparse"]
