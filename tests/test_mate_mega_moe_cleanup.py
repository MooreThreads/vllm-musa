# SPDX-License-Identifier: Apache-2.0
"""The applied communicator closes MATE workspaces before transport teardown."""

import sys
from types import SimpleNamespace

import pytest

from tests.moe_contract_utils import load_applied_methods

MODULE = "vllm_musa.model_executor.layers.fused_moe.mate_mega_moe"


def _comm(device_type, events):
    module = load_applied_methods(
        "vllm/distributed/device_communicators/cuda_communicator.py",
        "CudaCommunicator",
        ["destroy"],
        {"current_platform": SimpleNamespace(device_type=device_type)},
    )
    comm = module.CudaCommunicator()
    comm.device_group = object()
    comm.pynccl_comm = SimpleNamespace(destroy=lambda: events.append("transport"))
    for field in (
        "ca_comm",
        "aiter_ar_comm",
        "fi_ar_comm",
        "fi_pcie_ipc_ar_comm",
        "all2all_manager",
    ):
        setattr(comm, field, None)
    return comm


@pytest.mark.parametrize("device_type", ["musa", "cuda"])
def test_buffer_cleanup_precedes_transport_destroy(monkeypatch, device_type):
    events = []
    comm = _comm(device_type, events)

    def cleanup(group):
        assert group is comm.device_group
        events.append("buffer")

    monkeypatch.setitem(
        sys.modules, MODULE, SimpleNamespace(destroy_mate_mega_moe_buffers=cleanup)
    )
    comm.destroy()
    assert events == (
        ["buffer", "transport"] if device_type == "musa" else ["transport"]
    )
    assert comm.pynccl_comm is None


def test_non_mega_shutdown_does_not_import_backend(monkeypatch):
    events = []
    comm = _comm("musa", events)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    comm.destroy()
    assert events == ["transport"]
    assert MODULE not in sys.modules


def test_failed_buffer_cleanup_does_not_destroy_transport(monkeypatch):
    events = []
    comm = _comm("musa", events)

    def fail(group):
        raise RuntimeError("buffer teardown failed")

    monkeypatch.setitem(
        sys.modules, MODULE, SimpleNamespace(destroy_mate_mega_moe_buffers=fail)
    )
    with pytest.raises(RuntimeError, match="buffer teardown failed"):
        comm.destroy()
    assert events == []
    assert comm.pynccl_comm is not None
