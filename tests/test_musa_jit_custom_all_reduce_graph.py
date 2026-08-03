from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
custom_ar = pytest.importorskip(
    "vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce"
)


def _deepseek_v4_config(capture_sizes: tuple[object, ...]) -> SimpleNamespace:
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
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        attention_config=SimpleNamespace(backend="FLASHMLA"),
        compilation_config=SimpleNamespace(
            mode="NONE",
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=capture_sizes,
        ),
        speculative_config=None,
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


def test_deepseek_graph_capture_uses_fixed_staging(monkeypatch):
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl._use_graph_registered_inputs = False
    impl._IS_CAPTURING = True
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
