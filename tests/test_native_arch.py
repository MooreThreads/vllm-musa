# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The build arch record and the worker's native arch check."""

import sys
from types import SimpleNamespace

import pytest

# isort: off
import torchada  # noqa: F401
import torch

# isort: on
from torchada import arch as musa_arch

from build_utils.build_info import write_build_info
from vllm_musa import native_arch

MP31 = musa_arch.MusaArch(3, 1)


def _read_record(path):
    namespace = {}
    exec(path.read_text(), namespace)
    return namespace["MUSA_ARCHS"]


def test_write_build_info_records_sorted_unique_archs(tmp_path):
    path = write_build_info(tmp_path, [(3, 1), (2, 2), (3, 1)])
    assert path == tmp_path / "_build_info.py"
    assert _read_record(path) == ((2, 2), (3, 1))


def test_write_build_info_rejects_no_arch(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        write_build_info(tmp_path, [])


def test_built_archs_reads_the_record(monkeypatch):
    record = SimpleNamespace(MUSA_ARCHS=((3, 1),))
    monkeypatch.setitem(sys.modules, "vllm_musa._build_info", record)
    assert native_arch.built_archs() == (MP31,)


def test_built_archs_without_a_record(monkeypatch):
    # A None entry makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "vllm_musa._build_info", None)
    assert native_arch.built_archs() is None


def _device_capability(capability):
    def get_device_capability(device):
        return capability

    return get_device_capability


def test_check_device_arch_accepts_a_built_arch(monkeypatch):
    monkeypatch.setattr(native_arch, "built_archs", lambda: (MP31,))
    monkeypatch.setattr(torch.musa, "get_device_capability", _device_capability((3, 1)))
    native_arch.check_device_arch(torch.device("musa:0"))


def test_check_device_arch_refuses_another_arch(monkeypatch):
    monkeypatch.setattr(native_arch, "built_archs", lambda: (MP31,))
    monkeypatch.setattr(torch.musa, "get_device_capability", _device_capability((2, 2)))
    with pytest.raises(RuntimeError) as excinfo:
        native_arch.check_device_arch(torch.device("musa:1"))
    message = str(excinfo.value)
    assert "built for [mp_31]" in message
    assert "musa:1 is mp_22" in message
    assert "builds only for [mp_31]" in message
    assert "MTGPU_TARGET" not in message


def test_check_device_arch_suggests_a_rebuild_for_a_table_arch(monkeypatch):
    monkeypatch.setattr(musa_arch, "ARCHS", (musa_arch.MusaArch(2, 2), MP31))
    monkeypatch.setattr(native_arch, "built_archs", lambda: (MP31,))
    monkeypatch.setattr(torch.musa, "get_device_capability", _device_capability((2, 2)))
    with pytest.raises(RuntimeError, match="Rebuild vllm-musa with MTGPU_TARGET=mp_22"):
        native_arch.check_device_arch(torch.device("musa:1"))


def test_check_device_arch_skips_without_a_record(monkeypatch):
    def unexpected(device):
        raise AssertionError("the device is not queried without a record")

    monkeypatch.setattr(native_arch, "built_archs", lambda: None)
    monkeypatch.setattr(torch.musa, "get_device_capability", unexpected)
    native_arch.check_device_arch(torch.device("musa:0"))


def test_worker_checks_its_device_after_init_device(monkeypatch):
    # The plugin imports vllm_musa.v1 before vLLM imports the worker class.
    # isort: off
    import vllm_musa.v1  # noqa: F401
    from vllm.v1.worker.gpu_worker import Worker

    # isort: on
    from vllm_musa import worker as musa_worker

    calls = []

    def init_device(self):
        calls.append("init_device")
        self.device = torch.device("musa:0")

    monkeypatch.setattr(Worker, "init_device", init_device)
    monkeypatch.setattr(
        musa_worker, "check_device_arch", lambda device: calls.append(("check", device))
    )
    worker = object.__new__(musa_worker.MTGPUWorker)
    worker.init_device()
    assert calls == ["init_device", ("check", torch.device("musa:0"))]
