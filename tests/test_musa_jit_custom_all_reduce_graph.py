from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
custom_ar = pytest.importorskip(
    "vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce"
)


def _deepseek_v4_config(
    capture_sizes: tuple[object, ...], *, mtp: bool = False
) -> SimpleNamespace:
    text_config = SimpleNamespace(
        model_type="deepseek_v4",
        architectures=("DeepseekV4ForCausalLM",),
        hidden_size=4096,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        vocab_size=129280,
        n_routed_experts=256,
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=2048,
        expert_dtype="fp8",
        hidden_act="silu",
        swiglu_limit=10.0,
        index_topk=512,
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    )
    model_config = SimpleNamespace(
        architectures=("DeepseekV4ForCausalLM",),
        hf_config=text_config,
        hf_text_config=text_config,
        dtype="bfloat16",
        quantization="deepseek_v4_fp8",
        use_mla=True,
        is_hybrid=False,
        is_moe=True,
        enforce_eager=False,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8", block_size=64),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=8195,
        ),
        attention_config=SimpleNamespace(backend="FLASHMLA"),
        compilation_config=SimpleNamespace(
            mode="NONE",
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=capture_sizes,
        ),
        speculative_config=(
            SimpleNamespace(method="mtp", num_speculative_tokens=4) if mtp else None
        ),
        quant_config=SimpleNamespace(weight_block_size=[128, 128]),
    )


@pytest.mark.parametrize(
    ("capture_sizes", "expected"),
    [
        ((1,), True),
        ((1, 2, 4), False),
        ((), False),
        ((1, "invalid"), False),
    ],
)
def test_graph_registered_inputs_preserve_deepseek_capture_guard(
    monkeypatch: pytest.MonkeyPatch,
    capture_sizes: tuple[object, ...],
    expected: bool,
) -> None:
    vllm_config = _deepseek_v4_config(capture_sizes)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: vllm_config,
    )

    assert custom_ar._use_graph_registered_inputs_for_current_model() is expected


@pytest.mark.parametrize(
    ("model_type", "architectures", "expected"),
    [
        ("deepseek_v4", ("Qwen3ForCausalLM",), False),
        ("qwen3", ("DeepseekV4ForCausalLM",), False),
        ("qwen3", ("FakeDeepseekV4ForCausalLM",), True),
    ],
)
def test_graph_registered_inputs_fail_closed_for_partial_deepseek_identity(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    architectures: tuple[str, ...],
    expected: bool,
) -> None:
    vllm_config = _deepseek_v4_config((1, 2, 4))
    text_config = vllm_config.model_config.hf_text_config
    text_config.model_type = model_type
    text_config.architectures = architectures
    vllm_config.model_config.architectures = architectures
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: vllm_config,
    )

    assert custom_ar._use_graph_registered_inputs_for_current_model() is expected


def test_graph_registered_inputs_guard_all_exact_deepseek_v4_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_config = _deepseek_v4_config((1, 2, 4))
    vllm_config.model_config.hf_text_config.hidden_size = 7168
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: vllm_config,
    )

    assert custom_ar._use_graph_registered_inputs_for_current_model() is False


@pytest.mark.parametrize(
    "capture_sizes",
    [
        (5,),
        (5, 10, 20, 40, 80, 160, 320),
        (5, 6),
        (),
    ],
)
def test_mtp_graph_registered_inputs_are_replaced_by_staging_plan(
    monkeypatch: pytest.MonkeyPatch,
    capture_sizes: tuple[object, ...],
) -> None:
    vllm_config = _deepseek_v4_config(capture_sizes, mtp=True)
    vllm_config.scheduler_config.max_num_seqs = 64
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: vllm_config,
    )

    assert custom_ar._use_graph_registered_inputs_for_current_model() is False


def test_mtp4_bs1_keeps_registered_inputs_for_capture_size_five(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_config = _deepseek_v4_config((5,), mtp=True)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: vllm_config,
    )

    assert custom_ar._use_graph_registered_inputs_for_current_model() is True


def test_register_graph_buffers_populates_persistent_rank_data(monkeypatch):
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl.rank = 0
    impl.world_size = 2
    impl.group = object()
    impl.device = torch.device("cpu")
    impl.graph_rank_data = torch.zeros((4, 8), dtype=torch.int64)
    impl._pending_graph_inputs = [torch.empty(4), torch.empty(8)]
    impl._graph_input_refs = []
    impl._graph_peer_bases = {}
    impl._graph_opened_ptrs = []
    impl._next_graph_slot = 0

    handle_size = ctypes.sizeof(custom_ar.cudaIpcMemHandle_t)
    local_handles = [(b"a" * handle_size, 8), (b"b" * handle_size, 16)]
    peer_handles = [(b"c" * handle_size, 24), (b"d" * handle_size, 32)]
    monkeypatch.setattr(
        impl, "_graph_pointer_meta", lambda tensor: local_handles.pop(0)
    )

    ranks = [4, 9]
    broadcast_sources = []

    def broadcast_object_list(payload, src, group, device):
        assert group is impl.group
        assert device == "cpu"
        broadcast_sources.append(src)
        if src == ranks[0]:
            assert payload[0] == [
                (b"a" * handle_size, 8),
                (b"b" * handle_size, 16),
            ]
        else:
            assert payload[0] is None
            payload[0] = peer_handles

    peer_bases = {ord("c"): 10_000, ord("d"): 20_000}

    class FakeRuntime:
        def ipc_open_mem_handle(self, handle):
            first_byte = ctypes.string_at(ctypes.byref(handle), 1)[0]
            return ctypes.c_void_p(peer_bases[first_byte])

    monkeypatch.setattr(custom_ar.dist, "get_process_group_ranks", lambda group: ranks)
    monkeypatch.setattr(custom_ar.dist, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr(
        custom_ar.dist,
        "all_gather_object",
        lambda *_args, **_kwargs: pytest.fail("Gloo object gather must not be used"),
    )
    monkeypatch.setattr(custom_ar, "_MusaRTLibrary", FakeRuntime)
    monkeypatch.setattr(
        custom_ar.torch,
        "musa",
        SimpleNamespace(synchronize=lambda device: None),
    )

    inputs = list(impl._pending_graph_inputs)
    impl._register_graph_buffers()

    assert impl.graph_rank_data[0, 0].item() == inputs[0].data_ptr()
    assert impl.graph_rank_data[0, 1].item() == 10_024
    assert impl.graph_rank_data[1, 0].item() == inputs[1].data_ptr()
    assert impl.graph_rank_data[1, 1].item() == 20_032
    assert impl._next_graph_slot == 2
    assert impl._pending_graph_inputs == []
    assert impl._graph_input_refs == inputs
    assert sorted(impl._graph_opened_ptrs) == [10_000, 20_000]
    assert broadcast_sources == ranks


def test_capture_resets_state_when_registration_fails(monkeypatch):
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl._IS_CAPTURING = False
    impl._pending_graph_inputs = []

    def fail_registration():
        raise RuntimeError("registration failed")

    monkeypatch.setattr(impl, "_register_graph_buffers", fail_registration)
    with pytest.raises(RuntimeError, match="registration failed"):
        with impl.capture():
            assert impl._IS_CAPTURING is True
            impl._pending_graph_inputs.append(torch.empty(1))
    assert impl._IS_CAPTURING is False


def test_graph_launch_uses_next_persistent_slot(monkeypatch):
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl.graph_rank_data = torch.zeros((8, 8), dtype=torch.int64)
    impl.signal_ptrs_cpu = torch.zeros(2, dtype=torch.int64)
    impl._pending_graph_inputs = []
    impl._next_graph_slot = 3
    impl.rank = 0
    impl.world_size = 2
    captured = {}

    def launch(rank_data, signals, input_tensor, output, rank, world_size, shot):
        captured["rank_data"] = rank_data
        captured["signals"] = signals
        captured["input"] = input_tensor
        captured["output"] = output
        captured["rank"] = rank
        captured["world_size"] = world_size
        captured["shot"] = shot

    monkeypatch.setattr(custom_ar.jit_ar, "launch_graph_registered", launch)
    monkeypatch.setattr(custom_ar.jit_ar, "preferred_shot", lambda world, nbytes: 2)

    input_tensor = torch.randn(4, dtype=torch.bfloat16)
    output = impl._graph_custom_all_reduce_impl(input_tensor)

    assert captured["rank_data"].data_ptr() == impl.graph_rank_data[3].data_ptr()
    assert captured["input"] is input_tensor
    assert captured["output"] is output
    assert captured["rank"] == 0
    assert captured["world_size"] == 2
    assert captured["shot"] == 2
    assert impl._pending_graph_inputs == [input_tensor]


def test_mtp_graph_registration_accepts_bounded_multi_capture_inputs():
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl._graph_registered_input_enabled = True

    assert impl._graph_registered_input_eligible(
        SimpleNamespace(
            numel=lambda: 64 * 1024,
            element_size=lambda: 8,
        )
    )
    assert not impl._graph_registered_input_eligible(
        SimpleNamespace(
            numel=lambda: 64 * 1024 + 1,
            element_size=lambda: 8,
        )
    )


def test_graph_staging_uses_disjoint_preallocated_slots_per_ordinal(monkeypatch):
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl._use_graph_staging_arena = True
    impl._IS_CAPTURING = True
    impl._graph_staging_eager_reserve_bytes = 1024
    impl._graph_staging_data_offset = 1024
    impl._graph_staging_meta_offset = 1024
    impl._graph_staging_data_limit = 8192
    impl._graph_staging_meta_limit = 8192
    impl._graph_staging_ledger = []
    impl._graph_staging_cpu_refs = []
    impl.buffer_rank_data = torch.tensor([10_000, 20_000], dtype=torch.int64)
    impl.signal_ptrs_cpu = torch.tensor([30_000, 40_000], dtype=torch.int64)
    impl.meta_ptrs = [50_000, 60_000]
    impl.buffer_ptrs = [70_000, 80_000]
    impl.meta_size = 256
    impl.max_size = 8192
    impl.rank = 0
    monkeypatch.setattr(impl, "_graph_staging_capture_active", lambda: True)
    monkeypatch.setattr(
        impl,
        "_current_graph_staging_descriptor",
        lambda: SimpleNamespace(num_tokens=5, num_reqs=1),
    )

    input_tensor = torch.empty((5, 8), dtype=torch.bfloat16)
    first = impl._graph_staging_launch_args(input_tensor, "first")
    second = impl._graph_staging_launch_args(input_tensor, "second")

    assert first[3] == 70_000 + 1024
    assert second[3] == 70_000 + 1280
    assert first[2] == 50_000 + 1024
    assert second[2] == 50_000 + 1536
    assert first[4] == 256
    assert second[4] == 256
    assert len(impl._graph_staging_ledger) == 2


def test_graph_staging_meta_partition_starts_after_eager_signal_region() -> None:
    plan = SimpleNamespace(
        eager_reserve_bytes=1024,
        graph_data_capacity_bytes=2048,
        graph_meta_capacity_bytes=4096,
        max_meta_bytes_per_slot=512,
    )
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl._graph_staging_plan = plan
    impl._use_graph_staging_arena = True
    impl._use_graph_collective_fallback = False
    impl._dsv4_mtp_graph_guard = True
    impl._graph_staging_eager_reserve_bytes = plan.eager_reserve_bytes
    impl.meta_size = 320
    impl.max_size = 8192

    impl._configure_graph_staging_arena()

    assert impl._graph_staging_data_start == 1024
    assert impl._graph_staging_meta_start == 1536
    assert impl._graph_staging_data_offset == 1024
    assert impl._graph_staging_meta_offset == 1536
    assert impl._graph_staging_meta_start >= impl.meta_size + 1024
    assert impl._graph_staging_data_limit == 3072
    assert impl._graph_staging_meta_limit == 5632


@pytest.mark.parametrize(
    ("num_tokens", "num_reqs", "expected"),
    [
        (5, 1, True),
        (80, 16, True),
        (80, None, False),
        (160, 32, False),
        (320, 64, False),
    ],
)
def test_graph_staging_gate_uses_forward_context_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    num_reqs: int | None,
    expected: bool,
) -> None:
    from vllm import forward_context

    plan = custom_ar._graph_staging_plan_for_current_model()
    if plan is None:
        # Build the exact contract without depending on process-global config.
        from vllm_musa.runtime_plan.policy import (
            deepseek_v4_mtp_car_graph_staging_plan,
        )

        plan = deepseek_v4_mtp_car_graph_staging_plan(
            _deepseek_v4_config((5, 10, 20, 40, 80, 160, 320), mtp=True)
        )
    assert plan is not None
    descriptor = forward_context.BatchDescriptor(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        uniform=True,
    )
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(batch_descriptor=descriptor),
    )
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl._use_graph_staging_arena = True
    impl._IS_CAPTURING = True
    impl._graph_staging_plan = plan

    assert impl._graph_staging_descriptor_enabled() is expected


def test_mtp3_graph_fails_closed_to_standard_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _deepseek_v4_config((4, 8, 16, 32, 64, 128, 256), mtp=True)
    config.speculative_config.num_speculative_tokens = 3
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: config,
    )
    assert custom_ar._dsv4_mtp_graph_guard_for_current_model()
    assert custom_ar._graph_staging_plan_for_current_model() is None

    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl.disabled = False
    impl._IS_CAPTURING = True
    impl._use_graph_collective_fallback = True
    impl._use_graph_staging_arena = False
    impl.max_size = 1024
    inp = torch.empty(8, dtype=torch.bfloat16)

    assert not impl.should_custom_ar(inp)
    impl._IS_CAPTURING = False
    assert impl.should_custom_ar(inp)


@pytest.mark.parametrize(
    "impl_name",
    [
        "_custom_all_reduce_impl",
        "_graph_custom_all_reduce_impl",
        "_fused_allreduce_rmsnorm_impl",
        "_fused_allreduce_residual_rmsnorm_impl",
        "_fused_allreduce_residual_rmsnorm_no_raw_impl",
    ],
)
def test_low_level_custom_ops_cannot_bypass_graph_collective_fallback(
    impl_name: str,
) -> None:
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl._IS_CAPTURING = True
    impl._use_graph_collective_fallback = True
    impl._use_graph_staging_arena = False
    impl.buffer_rank_data = torch.zeros(8, dtype=torch.int64)
    impl.signal_ptrs_cpu = torch.zeros(8, dtype=torch.int64)
    inp = torch.empty((1, 8), dtype=torch.bfloat16)
    weight = torch.empty(8, dtype=torch.bfloat16)
    residual = torch.empty_like(inp)
    if impl_name in {"_custom_all_reduce_impl", "_graph_custom_all_reduce_impl"}:
        args = (inp,)
    elif impl_name == "_fused_allreduce_rmsnorm_impl":
        args = (inp, weight, 1e-5)
    else:
        args = (inp, residual, weight, 1e-5)

    with pytest.raises(RuntimeError, match="requires the standard collective"):
        getattr(impl, impl_name)(*args)


def test_graph_staging_multi_capture_preserves_disjoint_arena_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl._IS_CAPTURING = False
    impl._use_graph_staging_arena = True
    impl._graph_staging_plan = SimpleNamespace(
        capture_descriptors=frozenset({(5, 1), (10, 2)})
    )
    impl._graph_staging_data_offset = 1024
    impl._graph_staging_meta_offset = 2048
    impl._graph_staging_ledger = []
    impl._graph_staging_cpu_refs = []
    impl._graph_staging_captured_descriptors = set()
    impl._graph_staging_capture_sealed = False
    monkeypatch.setattr(impl, "_validate_graph_staging_capture", lambda *_: None)

    first_ref = object()
    with impl.capture():
        impl._graph_staging_data_offset += 256
        impl._graph_staging_meta_offset += 512
        impl._graph_staging_ledger.append((5, 1, "allreduce", 1, 2, 3, 4, 5))
        impl._graph_staging_cpu_refs.append(first_ref)

    assert impl._graph_staging_captured_descriptors == {(5, 1)}
    assert not impl._graph_staging_capture_sealed

    with impl.capture():
        assert impl._graph_staging_data_offset == 1280
        assert impl._graph_staging_meta_offset == 2560
        assert impl._graph_staging_ledger == [
            (5, 1, "allreduce", 1, 2, 3, 4, 5)
        ]
        assert impl._graph_staging_cpu_refs == [first_ref]
        impl._graph_staging_data_offset += 256
        impl._graph_staging_meta_offset += 512
        impl._graph_staging_ledger.append((10, 2, "allreduce", 1, 2, 3, 4, 5))

    assert impl._graph_staging_captured_descriptors == {(5, 1), (10, 2)}
    assert impl._graph_staging_capture_sealed


def _capture_consensus_impl() -> object:
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl.rank = 0
    impl.world_size = 2
    impl.group = object()
    impl._graph_staging_plan = SimpleNamespace(capture_descriptors=frozenset())
    impl._graph_staging_ledger = []
    impl._graph_staging_data_offset = 1024
    impl._graph_staging_data_start = 1024
    impl._graph_staging_meta_offset = 2048
    impl._graph_staging_meta_start = 2048
    return impl


def test_graph_staging_plan_mismatch_fails_before_buffer_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl.rank = 0
    impl.world_size = 2
    impl.group = object()
    monkeypatch.setattr(impl, "_graph_staging_plan_fingerprint", lambda: ("local",))
    monkeypatch.setattr(custom_ar.dist, "get_process_group_ranks", lambda group: [4, 9])

    def broadcast(payload, src, group, device):
        if src == 9:
            payload[0] = ("peer",)

    monkeypatch.setattr(custom_ar.dist, "broadcast_object_list", broadcast)

    with pytest.raises(RuntimeError, match="contract differs across ranks"):
        impl._validate_graph_staging_plan_consensus()


def test_empty_graph_staging_ledger_still_reaches_rank_consensus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _capture_consensus_impl()
    sources: list[int] = []

    def broadcast(payload, src, group, device):
        assert group is impl.group
        assert device == "cpu"
        sources.append(src)
        if src == 9:
            payload[0] = {"success": True, "error": None, "ledger": []}

    monkeypatch.setattr(custom_ar.dist, "get_process_group_ranks", lambda group: [4, 9])
    monkeypatch.setattr(custom_ar.dist, "broadcast_object_list", broadcast)

    impl._validate_graph_staging_capture(True, None)

    assert sources == [4, 9]


def test_graph_staging_capture_failure_is_shared_with_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _capture_consensus_impl()
    sources: list[int] = []

    def broadcast(payload, src, group, device):
        sources.append(src)
        if src == 9:
            payload[0] = {"success": True, "error": None, "ledger": []}

    monkeypatch.setattr(custom_ar.dist, "get_process_group_ranks", lambda group: [4, 9])
    monkeypatch.setattr(custom_ar.dist, "broadcast_object_list", broadcast)

    with pytest.raises(RuntimeError, match="capture failed across ranks"):
        impl._validate_graph_staging_capture(False, RuntimeError("rank 0 failed"))

    assert sources == [4, 9]


def test_graph_staging_rank_ledger_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _capture_consensus_impl()

    def broadcast(payload, src, group, device):
        if src == 9:
            payload[0] = {
                "success": True,
                "error": None,
                "ledger": [(5, 1, "allreduce", 1, 2, 3, 4, 5)],
            }

    monkeypatch.setattr(custom_ar.dist, "get_process_group_ranks", lambda group: [4, 9])
    monkeypatch.setattr(custom_ar.dist, "broadcast_object_list", broadcast)

    with pytest.raises(RuntimeError, match="ledger differs across ranks"):
        impl._validate_graph_staging_capture(True, None)


def test_graph_staging_eager_allgather_respects_partition(monkeypatch) -> None:
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl.disabled = False
    impl._use_graph_staging_arena = True
    impl._graph_staging_eager_reserve_bytes = 64
    impl._IS_CAPTURING = False
    impl.max_size = 1024
    impl.world_size = 2
    impl.group_size = 2
    monkeypatch.setattr(impl, "_is_communicator_tensor", lambda tensor: True)

    assert impl.should_custom_all_gather(
        torch.empty((1, 8), dtype=torch.bfloat16),
        dim=1,
    )
    assert not impl.should_custom_all_gather(
        torch.empty((4, 8), dtype=torch.bfloat16),
        dim=1,
    )


def test_deepseek_eager_custom_ar_retains_fixed_staging(monkeypatch):
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl._use_graph_registered_inputs = False
    impl._use_graph_staging_arena = False
    impl._IS_CAPTURING = False
    impl.buffer_rank_data = torch.zeros(8, dtype=torch.int64)
    impl.signal_ptrs_cpu = torch.zeros(2, dtype=torch.int64)
    impl.meta_ptrs = [10, 20]
    impl.buffer_ptrs = [30, 40]
    impl.max_size = 1024
    impl.rank = 0
    impl.world_size = 2

    monkeypatch.setattr(impl, "_is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        impl,
        "_graph_custom_all_reduce_impl",
        lambda _input: pytest.fail("DeepSeek must not register graph input pointers"),
    )
    monkeypatch.setattr(
        impl,
        "_graph_staging_launch_args",
        lambda *_args: pytest.fail("non-target eager CAR must keep the direct path"),
    )
    monkeypatch.setattr(custom_ar.jit_ar, "preferred_shot", lambda _world, _size: 1)
    captured = {}

    def launch_unregistered(*args):
        captured["args"] = args

    monkeypatch.setattr(custom_ar.jit_ar, "launch_unregistered", launch_unregistered)
    input_tensor = torch.randn(4, dtype=torch.bfloat16)

    output = impl._custom_all_reduce_impl(input_tensor)

    assert output.shape == input_tensor.shape
    assert captured["args"][2] is input_tensor
    assert captured["args"][3] is output
