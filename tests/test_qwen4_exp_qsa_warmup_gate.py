# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MUSA QSA preload must stay scoped to models that carry QSA layers."""

import sys
import types

import pytest
from torch import nn

from vllm.model_executor.warmup import qwen4_exp_qsa_warmup
from vllm.platforms import current_platform


class _Probe:
    def __init__(self) -> None:
        self.calls = 0

    def qsa_fast_topk_available(self) -> bool:
        self.calls += 1
        return True


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
    monkeypatch.setitem(
        sys.modules, "vllm.models.qwen4_exp.nvidia.qsa", attn_module
    )
    return QSAIndexer, Qwen4ExpQSAAttention


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


def test_qwen4_exp_model_with_qsa_layers_builds_the_qsa_topk(monkeypatch, probe):
    indexer_cls, attention_cls = _install_qsa_modules(monkeypatch)
    worker = _Worker()
    worker.indexer = indexer_cls()
    worker.attention = attention_cls()

    qwen4_exp_qsa_warmup.qwen4_exp_qsa_triton_warmup(worker)

    assert probe.calls == 1
