from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
custom_ar = pytest.importorskip(
    "vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce"
)


def _impl() -> custom_ar._MusaJitCustomAllreduceImpl:
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl.disabled = False
    impl.world_size = 2
    impl.rank = 0
    impl.max_size = 1024 * 1024
    impl.device = torch.device("musa:0")
    impl.buffer_rank_data = torch.zeros(8, dtype=torch.int64)
    impl.signal_ptrs_cpu = torch.zeros(2, dtype=torch.int64)
    impl.meta_ptrs = [101, 102]
    impl.buffer_ptrs = [201, 202]
    impl._IS_CAPTURING = False
    impl._is_communicator_tensor = lambda input_tensor: True
    impl._is_default_stream = lambda input_tensor: True
    return impl


def test_custom_all_gather_gate_accepts_only_last_dim_packed_2d():
    impl = _impl()
    accepted = torch.empty((4, 32), dtype=torch.bfloat16)

    assert impl.should_custom_all_gather(accepted, -1)
    assert impl.should_custom_all_gather(accepted, -1, output_dtype=torch.float32)
    assert not impl.should_custom_all_gather(accepted, -1, output_dtype=torch.float16)
    assert impl.should_custom_all_gather(accepted, 1)
    assert not impl.should_custom_all_gather(accepted, 0)
    assert not impl.should_custom_all_gather(accepted[:, ::2], -1)
    assert not impl.should_custom_all_gather(
        torch.empty((4, 31), dtype=torch.bfloat16), -1
    )
    assert not impl.should_custom_all_gather(
        torch.empty((4, 32), dtype=torch.int32), -1
    )
    assert not impl.should_custom_all_gather(
        torch.empty((2, 4, 32), dtype=torch.bfloat16), -1
    )
    impl.max_size = accepted.numel() * accepted.element_size()
    assert not impl.should_custom_all_gather(accepted, -1)
    impl.max_size = accepted.numel() * accepted.element_size() * impl.world_size
    assert impl.should_custom_all_gather(accepted, -1)
    assert not impl.should_custom_all_gather(accepted, -1, output_dtype=torch.float32)
    impl.max_size = 1024 * 1024
    impl.world_size = 6
    assert not impl.should_custom_all_gather(accepted, -1)


def test_custom_all_gather_gate_rejects_non_musa_tensor(monkeypatch):
    impl = _impl()
    del impl._is_communicator_tensor

    assert not impl.should_custom_all_gather(
        torch.empty((4, 32), dtype=torch.bfloat16), -1
    )

    impl.device = torch.device("musa")
    monkeypatch.setattr(
        torch,
        "musa",
        SimpleNamespace(current_device=lambda: 0),
        raising=False,
    )
    fake_musa_inp = SimpleNamespace(device=torch.device("musa:0"))
    assert impl._is_communicator_tensor(fake_musa_inp)
    fake_other_inp = SimpleNamespace(device=torch.device("musa:1"))
    assert not impl._is_communicator_tensor(fake_other_inp)


def test_custom_all_gather_uses_eager_staging_reserve_when_arena_enabled(monkeypatch):
    impl = _impl()
    impl._use_graph_staging_arena = True
    impl._graph_staging_eager_reserve_bytes = 256 * 1024
    input_tensor = torch.arange(128, dtype=torch.bfloat16).view(4, 32)
    captured = {}

    def launch_all_gather(
        rank_data,
        signals,
        inp,
        out,
        self_signal_ptr,
        self_buffer_ptr,
        max_size_bytes,
        rank,
        world_size,
    ):
        captured["max_size_bytes"] = max_size_bytes

    monkeypatch.setattr(custom_ar.jit_ar, "launch_all_gather", launch_all_gather)
    impl._custom_all_gather_impl(input_tensor)

    assert captured["max_size_bytes"] == 256 * 1024


def test_custom_all_gather_launches_with_last_dim_concatenated(monkeypatch):
    impl = _impl()
    input_tensor = torch.arange(128, dtype=torch.bfloat16).view(4, 32)
    captured = {}

    def launch_all_gather(
        rank_data,
        signals,
        inp,
        out,
        self_signal_ptr,
        self_buffer_ptr,
        max_size_bytes,
        rank,
        world_size,
    ):
        captured.update(
            rank_data=rank_data,
            signals=signals,
            inp=inp,
            out=out,
            self_signal_ptr=self_signal_ptr,
            self_buffer_ptr=self_buffer_ptr,
            max_size_bytes=max_size_bytes,
            rank=rank,
            world_size=world_size,
        )

    monkeypatch.setattr(custom_ar.jit_ar, "launch_all_gather", launch_all_gather)
    output = impl._custom_all_gather_impl(input_tensor)

    assert output.shape == (4, 64)
    assert output.dtype == input_tensor.dtype
    assert captured["rank_data"] is impl.buffer_rank_data
    assert captured["signals"] is impl.signal_ptrs_cpu
    assert captured["inp"] is input_tensor
    assert captured["out"] is output
    assert captured["self_signal_ptr"] == 101
    assert captured["self_buffer_ptr"] == 201
    assert captured["max_size_bytes"] == impl.max_size
    assert captured["rank"] == 0
    assert captured["world_size"] == 2


def test_custom_all_gather_launches_bfloat16_to_float32(monkeypatch):
    impl = _impl()
    input_tensor = torch.arange(128, dtype=torch.bfloat16).view(4, 32)
    captured = {}

    def launch_all_gather(
        rank_data,
        signals,
        inp,
        out,
        self_signal_ptr,
        self_buffer_ptr,
        max_size_bytes,
        rank,
        world_size,
    ):
        captured.update(inp=inp, out=out)

    monkeypatch.setattr(custom_ar.jit_ar, "launch_all_gather", launch_all_gather)
    output = impl._custom_all_gather_impl(input_tensor, output_dtype=torch.float32)

    assert output.shape == (4, 64)
    assert output.dtype == torch.float32
    assert captured["inp"] is input_tensor
    assert captured["out"] is output


@pytest.mark.parametrize(
    "capturing,stream_capturing,compiling",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_custom_all_gather_fails_closed_outside_eager(
    monkeypatch, capturing, stream_capturing, compiling
):
    impl = _impl()
    impl._IS_CAPTURING = capturing
    monkeypatch.setattr(impl, "_is_current_stream_capturing", lambda: stream_capturing)
    monkeypatch.setattr(impl, "_is_torch_compiling", lambda: compiling)
    monkeypatch.setattr(
        impl,
        "_custom_all_gather_impl",
        lambda input_tensor: pytest.fail("eager launch must not run"),
    )

    assert impl.custom_all_gather(torch.empty((4, 32), dtype=torch.bfloat16)) is None


def test_custom_all_gather_fails_closed_on_non_default_stream(monkeypatch):
    impl = _impl()
    monkeypatch.setattr(impl, "_is_default_stream", lambda input_tensor: False)
    monkeypatch.setattr(
        impl,
        "_custom_all_gather_impl",
        lambda input_tensor: pytest.fail("non-default stream must not launch"),
    )

    assert impl.custom_all_gather(torch.empty((4, 32), dtype=torch.bfloat16)) is None


def test_logits_helper_uses_only_musa_jit_communicator(monkeypatch):
    expected = torch.empty((4, 64), dtype=torch.float32)
    communicator = object.__new__(custom_ar.MusaJitCustomAllreduce)
    communicator._jit_comm = SimpleNamespace(disabled=False)
    monkeypatch.setattr(
        communicator,
        "custom_all_gather",
        lambda input_tensor, dim, output_dtype=None: (
            expected if output_dtype == torch.float32 else None
        ),
    )

    import vllm.distributed.parallel_state as parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(ca_comm=communicator)
        ),
    )
    output = custom_ar.maybe_musa_jit_logits_all_gather(
        torch.empty((4, 32), dtype=torch.bfloat16),
        output_dtype=torch.float32,
    )
    assert output is expected

    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=object())),
    )
    assert (
        custom_ar.maybe_musa_jit_logits_all_gather(
            torch.empty((4, 32), dtype=torch.bfloat16)
        )
        is None
    )


def test_communicator_fails_closed_when_a_peer_rank_is_unavailable(monkeypatch):
    closed = []

    class FakeImpl:
        disabled = False

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        custom_ar,
        "_MusaJitCustomAllreduceImpl",
        lambda **kwargs: FakeImpl(),
    )

    def mark_peer_unavailable(availability, **kwargs):
        availability.zero_()

    monkeypatch.setattr(custom_ar.dist, "all_reduce", mark_peer_unavailable)
    communicator = custom_ar.MusaJitCustomAllreduce(group=object(), device="musa:0")

    assert communicator.disabled
    assert closed == [True]
