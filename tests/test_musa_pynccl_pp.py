# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise MUSA PP library selection, stream selection, and ordering on CPU."""

import re
import sys
from pathlib import Path
from textwrap import dedent, indent
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest

SERIES = Path(__file__).resolve().parents[1] / "vllm_musa/patches/series"
PYNCCL_PATCH = SERIES / "0218-MUSA-enable-PyNccl-for-pipeline-parallelism.patch"
EAGER_PATCH = SERIES / "0219-MUSA-eagerly-initialize-MCCL-for-PP.patch"
STREAM_PATCH = SERIES / "0220-MUSA-query-the-active-native-MUSA-stream.patch"
PP_ORDER_PATCH = SERIES / "0221-MUSA-order-PP-sends-before-sampling-broadcasts.patch"


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
        (True, "pp:0", None, "libmccl.so.2"),
        (True, "pp:1", "/opt/musa/lib/libmccl.so.2", "/opt/musa/lib/libmccl.so.2"),
        (True, "tp:0", None, None),
        (True, "ep:0", None, None),
        (True, "dp:0", None, None),
        (True, "pp_aux:0", None, None),
        (False, "pp:0", None, None),
        (False, "pp:0", "/opt/nccl/lib/libnccl.so.2", None),
    ],
)
def test_pynccl_default_library_is_musa_pp_only(
    is_musa: bool, unique_name: str, override: str | None, expected: str | None
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
        group=communicator.cpu_group, device=communicator.device, library_path=expected
    )
    assert communicator.pynccl_comm is make_pynccl.return_value


@pytest.mark.parametrize(
    ("backend", "pp_size", "expected_device"),
    [
        ("mccl", 2, ("musa", 1)),
        ("mccl", 1, None),
        ("mccl", None, None),
        ("gloo", 2, None),
    ],
)
def test_eager_mccl_uses_the_workers_visible_device(
    backend: str,
    pp_size: int | None,
    expected_device: tuple[str, int] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _patched_hunk(
        EAGER_PATCH, "vllm/distributed/parallel_state.py", "Eagerly initialize MCCL"
    )
    start = source.index("            # Eagerly initialize MCCL")
    end = source.index("        if enable_elastic_ep:")
    map_device = Mock(return_value=1)
    platform = ModuleType("vllm.platforms")
    platform.current_platform = SimpleNamespace(
        logical_device_id_to_visible_device_id=map_device
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform)
    init_process_group = Mock()
    namespace = {
        "backend": backend,
        "config": (
            None
            if pp_size is None
            else SimpleNamespace(
                parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size)
            )
        ),
        "local_rank": 0,
        "init_kwargs": {"backend": backend},
        "torch": SimpleNamespace(
            version=SimpleNamespace(musa="5.2.0"),
            device=lambda *args: args,
            distributed=SimpleNamespace(init_process_group=init_process_group),
        ),
    }

    exec(compile(dedent(source[start:end]), str(EAGER_PATCH), "exec"), namespace)

    if expected_device is None:
        map_device.assert_not_called()
        init_process_group.assert_called_once_with(backend=backend)
    else:
        map_device.assert_called_once_with(0)
        init_process_group.assert_called_once_with(
            backend=backend, device_id=expected_device
        )


@pytest.mark.parametrize(
    ("device_type", "use_v2_model_runner", "waits"),
    [("musa", True, True), ("musa", False, False), ("cuda", True, False)],
)
def test_orders_pp_activation_sends_before_mrv2_sampling(
    device_type: str, use_v2_model_runner: bool, waits: bool
) -> None:
    source = _patched_hunk(
        PP_ORDER_PATCH, "vllm/v1/worker/gpu_worker.py", "sampled-token broadcast"
    )
    start = source.index("        if current_platform")
    end = source.index("    @torch.inference_mode()")
    calls = Mock()
    worker = SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        _pp_send_work=[calls.work],
        model_runner=calls.runner,
    )
    namespace = {"current_platform": SimpleNamespace(device_type=device_type)}
    body = dedent(source[start:end]).rstrip()
    exec(
        compile(
            "def sample(self, grammar_output):\n" + indent(body, "    "),
            str(PP_ORDER_PATCH),
            "exec",
        ),
        namespace,
    )

    output = namespace["sample"](worker, None)

    assert output is calls.runner.sample_tokens.return_value
    if waits:
        assert calls.mock_calls == [call.work.wait(), call.runner.sample_tokens(None)]
        assert worker._pp_send_work == []
    else:
        assert calls.mock_calls == [call.runner.sample_tokens(None)]
        assert worker._pp_send_work == [calls.work]


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
