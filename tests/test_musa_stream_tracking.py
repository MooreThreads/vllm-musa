# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""current_stream() follows stream switches made through torch_musa."""

import sys
import types

import torch

from tests.moe_contract_utils import load_applied_contract

torch_utils = load_applied_contract(
    "vllm/utils/torch_utils.py",
    ["_track_musa_set_stream"],
    {"torch": torch, "_current_stream_tls": types.SimpleNamespace()},
)


def _fake_torch_musa(monkeypatch):
    calls = []
    fake = types.ModuleType("torch_musa")
    fake.set_stream = calls.append
    monkeypatch.setitem(sys.modules, "torch_musa", fake)
    monkeypatch.setattr(torch.cuda, "set_stream", torch.cuda.set_stream)
    return fake, calls


def test_torch_musa_set_stream_updates_current_stream(monkeypatch):
    fake, calls = _fake_torch_musa(monkeypatch)
    monkeypatch.setattr(
        torch_utils._current_stream_tls, "value", "before", raising=False
    )
    torch_utils._track_musa_set_stream()

    stream = object()
    fake.set_stream(stream)
    assert calls == [stream]
    assert torch_utils._current_stream_tls.value is stream
    assert torch.cuda.set_stream is fake.set_stream


def test_tracking_is_installed_once(monkeypatch):
    fake, calls = _fake_torch_musa(monkeypatch)
    torch_utils._track_musa_set_stream()
    wrapped = fake.set_stream
    torch_utils._track_musa_set_stream()

    assert fake.set_stream is wrapped
    fake.set_stream("stream")
    assert calls == ["stream"]
