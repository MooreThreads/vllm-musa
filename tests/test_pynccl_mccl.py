# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pynccl on MUSA: MCCL library selection and symbol resolution."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchada")
pytest.importorskip("vllm")

pytestmark = pytest.mark.skipif(
    getattr(torch.version, "musa", None) is None,
    reason="needs a MUSA build of torch",
)


def test_find_nccl_library_selects_mccl(monkeypatch):
    from vllm.utils.nccl import find_nccl_library

    monkeypatch.delenv("VLLM_NCCL_SO_PATH", raising=False)
    assert find_nccl_library() == "libmccl.so.2"


def test_explicit_library_path_wins(monkeypatch):
    from vllm.utils.nccl import find_nccl_library

    monkeypatch.setenv("VLLM_NCCL_SO_PATH", "/opt/mccl/lib/libmccl.so.2")
    assert find_nccl_library() == "/opt/mccl/lib/libmccl.so.2"


def test_mccl_exports_every_pynccl_symbol(monkeypatch):
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

    monkeypatch.delenv("VLLM_NCCL_SO_PATH", raising=False)
    try:
        lib = NCCLLibrary()
    except OSError as exc:
        pytest.skip(f"MCCL is not installed: {exc}")
    missing = [
        func.name
        for func in NCCLLibrary.exported_functions
        if not lib.has_symbol(func.name)
    ]
    assert missing == []
    assert lib.ncclGetVersion()
