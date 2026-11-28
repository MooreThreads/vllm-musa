# SPDX-License-Identifier: Apache-2.0
"""Exercise the patched library and stream helpers without loading Torch or MCCL."""

import ast
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

ROOT = Path(__file__).resolve().parents[1]
VLLM = ROOT / "third_party/vllm/vllm"
COMM = VLLM / "distributed/device_communicators"


def load_function(path, name, *, owner=None, **bindings):
    body = ast.parse(path.read_text()).body
    if owner:
        body = next(node.body for node in body if getattr(node, "name", None) == owner)
    node = next(node for node in body if getattr(node, "name", None) == name)
    exec("from __future__ import annotations\n" + ast.unparse(node), bindings)
    return bindings[name]


@pytest.mark.parametrize(
    ("backend", "override", "explicit", "expected"),
    [
        ("musa", None, None, "libmccl.so.2"),
        ("musa", "/env", None, "/env"),
        ("musa", "/env", "/explicit", "/explicit"),
        ("cuda", None, None, "libnccl.so.2"),
        ("hip", None, None, "librccl.so.1"),
        (None, "/env", None, "/env"),
        (None, None, None, None),
    ],
)
def test_library_selection(backend, override, explicit, expected):
    version = SimpleNamespace(cuda=None, hip=None, musa=None)
    if backend:
        setattr(version, backend, "test-version")
    resolve = load_function(
        VLLM / "utils/nccl.py",
        "find_nccl_library",
        envs=SimpleNamespace(VLLM_NCCL_SO_PATH=override),
        torch=SimpleNamespace(version=version),
        logger=Mock(),
    )
    library = SimpleNamespace(
        path_to_dict_mapping={}, path_to_library_cache={}, exported_functions=[]
    )
    open_library = Mock()
    init = load_function(
        COMM / "pynccl_wrapper.py",
        "__init__",
        owner="NCCLLibrary",
        find_nccl_library=resolve,
        NCCLLibrary=library,
        ctypes=SimpleNamespace(CDLL=open_library),
    )
    if expected is None:
        with pytest.raises(ValueError):
            init(SimpleNamespace(), explicit)
        open_library.assert_not_called()
    else:
        init(SimpleNamespace(), explicit)
        open_library.assert_called_once_with(expected)


@pytest.fixture
def streams(monkeypatch, request):
    device_type = getattr(request, "param", "musa")
    native = Mock(spec_set=[])
    platform = SimpleNamespace(
        device_type=device_type,
        is_rocm=lambda: device_type == "hip",
        is_cuda=lambda: device_type == "cuda",
        is_cuda_alike=lambda: device_type in ("cuda", "hip", "musa"),
        is_cpu=lambda: device_type == "cpu",
        current_stream=Mock(),
    )
    if device_type == "musa":
        is_cuda_alike = load_function(
            ROOT / "vllm_musa/platform.py", "is_cuda_alike", owner="MUSAPlatformBase"
        )
        platform.is_cuda_alike = lambda: is_cuda_alike(platform)
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", SimpleNamespace(current_platform=platform)
    )
    cuda = SimpleNamespace(
        set_stream=native,
        Stream=Mock(),
        current_stream=Mock(return_value=object()),
    )
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace())
    bindings = {"torch": torch, "threading": threading}
    path = VLLM / "utils/torch_utils.py"
    source = path.read_text()
    functions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    start = functions["make_tensor_with_pad"].end_lineno
    end = functions["current_stream"].end_lineno
    section = "\n".join(source.splitlines()[start:end])
    exec("from __future__ import annotations\n" + section, bindings)
    return native, platform, cuda, bindings["current_stream"]


@pytest.mark.parametrize("streams", ["musa", "cuda", "hip"], indirect=True)
def test_cuda_alike_cold_read_creates_dedicated_stream(streams):
    native, platform, cuda, current = streams
    stream = current()
    assert current() is stream is cuda.Stream.return_value
    cuda.Stream.assert_called_once_with()
    cuda.current_stream.assert_called_once_with()
    stream.wait_stream.assert_called_once_with(cuda.current_stream.return_value)
    native.assert_called_once_with(stream)
    platform.current_stream.assert_not_called()


def test_stream_tracking_is_thread_local(streams):
    native, _, cuda, current = streams
    home, worker_home, side = Mock(), Mock(), object()
    previous, worker_previous = object(), object()
    cuda.Stream.side_effect = [home, worker_home]
    cuda.current_stream.side_effect = [previous, worker_previous]
    assert current() is home
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(current).result() is worker_home
        worker.submit(cuda.set_stream, side).result()
        assert worker.submit(current).result() is side
        assert current() is home
    home.wait_stream.assert_called_once_with(previous)
    worker_home.wait_stream.assert_called_once_with(worker_previous)
    assert native.call_args_list == [call(home), call(worker_home), call(side)]


@pytest.mark.parametrize("streams", ["cpu"], indirect=True)
def test_cpu_cold_read_keeps_placeholder(streams):
    native, _, cuda, current = streams
    placeholder = current()
    assert current() is placeholder
    assert placeholder.synchronize() is None
    cuda.Stream.assert_not_called()
    native.assert_not_called()


@pytest.mark.parametrize("mode", ["cold", "cached", "explicit"])
def test_all_gather_uses_current_stream_unless_stream_is_explicit(streams, mode):
    native, platform, cuda, current = streams
    device = object()
    dedicated = cuda.Stream.return_value
    dedicated.cuda_stream = 31
    supplied = SimpleNamespace(cuda_stream=42)
    if mode == "cached":
        cuda.set_stream(supplied)
    gather = load_function(
        COMM / "pynccl.py",
        "all_gather",
        owner="PyNcclCommunicator",
        current_stream=current,
        buffer_type=int,
        cudaStream_t=int,
        ncclDataTypeEnum=SimpleNamespace(from_torch=int),
    )
    communicator = SimpleNamespace(
        device=device, disabled=False, comm=object(), nccl=Mock()
    )
    tensor = SimpleNamespace(
        device=device, data_ptr=lambda: 1, numel=lambda: 1, dtype=0
    )
    gather(
        communicator,
        tensor,
        tensor,
        stream=supplied if mode == "explicit" else None,
    )
    if mode == "cold":
        cuda.Stream.assert_called_once_with()
        dedicated.wait_stream.assert_called_once_with(cuda.current_stream.return_value)
        native.assert_called_once_with(dedicated)
    else:
        cuda.Stream.assert_not_called()
        cuda.current_stream.assert_not_called()
        assert native.call_args_list == ([call(supplied)] if mode == "cached" else [])
    platform.current_stream.assert_not_called()
    assert communicator.nccl.ncclAllGather.call_args.args[-1] == (
        31 if mode == "cold" else 42
    )
