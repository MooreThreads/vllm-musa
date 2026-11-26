# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the PP PyNccl patch's library and stream choices on CPU."""

import re
import sys
from pathlib import Path
from textwrap import dedent
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

SERIES = Path(__file__).resolve().parents[1] / "vllm_musa/patches/series"
PYNCCL_PATCH = SERIES / "0218-MUSA-discover-MCCL-in-the-shared-library-loader.patch"
STREAM_PATCH = SERIES / "0219-MUSA-query-the-active-native-MUSA-stream.patch"


def _patched_hunk(patch: Path, path: str, marker: str) -> str:
    section = patch.read_text(encoding="utf-8").split(f"\n+++ b/{path}\n", 1)[1]
    section = section.split("\ndiff --git ", 1)[0]
    hunk = next(
        hunk
        for hunk in re.split(r"^@@ .*\n", section, flags=re.MULTILINE)
        if marker in hunk
    )
    return "\n".join(
        line[1:] for line in hunk.splitlines() if line.startswith((" ", "+"))
    )


@pytest.mark.parametrize(
    ("is_musa", "unique_name", "override", "expected"),
    [
        (True, "pp:0", None, True),
        (True, "ep:1", "/opt/musa/lib/libmccl.so.2", True),
        (True, "tp:0", None, False),
        (True, "ep:0", None, False),
        (True, "dp:0", None, False),
        (True, "pp_aux:0", None, False),
        (False, "pp:0", None, True),
        (False, "pp:0", "/opt/nccl/lib/libnccl.so.2", True),
    ],
)
def test_musa_pynccl_default_enablement_is_limited_to_pp(
    is_musa: bool, unique_name: str, override: str | None, expected: bool
) -> None:
    source = _patched_hunk(
        PYNCCL_PATCH,
        "vllm/distributed/device_communicators/cuda_communicator.py",
        "self.pynccl_comm = PyNcclCommunicator(",
    )
    communicator = SimpleNamespace(world_size=2, cpu_group=object(), device=object())
    make_pynccl = Mock()
    namespace = {
        "self": communicator,
        "unique_name": unique_name,
        "tcp_store_group": None,
        "PyNcclCommunicator": make_pynccl,
        "is_symmetric_memory_enabled": lambda: False,
        "current_platform": SimpleNamespace(device_type="musa" if is_musa else "cuda"),
        "envs": SimpleNamespace(VLLM_NCCL_SO_PATH=override),
    }

    code = "from __future__ import annotations\n" + dedent(source)
    exec(compile(code, str(PYNCCL_PATCH), "exec"), namespace)

    make_pynccl.assert_called_once_with(
        group=communicator.cpu_group, device=communicator.device, enabled=expected
    )
    assert communicator.pynccl_comm is make_pynccl.return_value


@pytest.mark.parametrize(
    ("musa", "cuda", "hip", "override", "expected"),
    [
        ("5.2.0", None, None, None, "libmccl.so.2"),
        ("5.2.0", None, None, "/opt/mccl.so", "/opt/mccl.so"),
        (None, "13.0", None, None, "libnccl.so.2"),
        (None, None, "7.0", None, "librccl.so.1"),
        (None, None, None, None, None),
    ],
)
def test_nccl_library_is_selected_by_torch_backend(
    musa: str | None,
    cuda: str | None,
    hip: str | None,
    override: str | None,
    expected: str | None,
) -> None:
    source = _patched_hunk(PYNCCL_PATCH, "vllm/utils/nccl.py", "libmccl.so.2")
    tail = source[source.index("    else:") :]
    code = (
        "def find_nccl_library():\n    so_file = envs.VLLM_NCCL_SO_PATH\n    if so_file:\n        pass\n"
        + tail
    )
    namespace = {
        "torch": SimpleNamespace(
            version=SimpleNamespace(musa=musa, cuda=cuda, hip=hip)
        ),
        "envs": SimpleNamespace(VLLM_NCCL_SO_PATH=override),
        "logger": Mock(),
    }
    exec(compile(code, str(PYNCCL_PATCH), "exec"), namespace)
    find_library = namespace["find_nccl_library"]

    if expected is None:
        with pytest.raises(ValueError, match="require CUDA, ROCm, or MUSA"):
            find_library()
    else:
        assert find_library() == expected


def test_current_stream_follows_native_musa_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _patched_hunk(
        STREAM_PATCH, "vllm/utils/torch_utils.py", "torch.musa.current_stream("
    )
    prologue_start = source.index("    from vllm.platforms import current_platform")
    cache_start = source.index("    if not hasattr(_current_stream_tls")
    # Execute the patched prologue with the upstream populated-cache return.
    code = f"def current_stream():\n{source[prologue_start:cache_start]}    return _current_stream_tls.value\n"

    default_stream, capture_stream, cached_stream = object(), object(), object()
    active = SimpleNamespace(stream=default_stream)
    native_stream = Mock(side_effect=lambda: active.stream)
    platform = ModuleType("vllm.platforms")
    platform.current_platform = SimpleNamespace(device_type="musa")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform)
    cache = SimpleNamespace(value=cached_stream)
    namespace = {
        "torch": SimpleNamespace(musa=SimpleNamespace(current_stream=native_stream)),
        "_current_stream_tls": cache,
    }
    exec(compile(code, str(STREAM_PATCH), "exec"), namespace)
    current_stream = namespace["current_stream"]

    assert current_stream() is default_stream
    active.stream = capture_stream
    assert current_stream() is capture_stream
    active.stream = default_stream
    assert current_stream() is default_stream
    assert cache.value is cached_stream

    platform.current_platform.device_type = "cuda"
    assert current_stream() is cached_stream
    assert native_stream.call_count == 3
