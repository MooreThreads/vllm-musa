# SPDX-License-Identifier: Apache-2.0
"""Runtime contracts for Qwen3.5/3.6 attention return-through."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
qwen3_5 = pytest.importorskip("vllm.model_executor.models.qwen3_5")
qwen3_next = pytest.importorskip("vllm.model_executor.models.qwen3_next")
qwen_gdn = pytest.importorskip(
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
)


class _Projection:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.last_output = None

    def __call__(self, hidden_states):
        self.last_output = hidden_states * self.scale
        return self.last_output, None


def test_full_attention_returns_projection_or_preserves_output_buffer() -> None:
    projection = _Projection(scale=2.0)
    attention = SimpleNamespace(
        qkv_proj=lambda hidden_states: (hidden_states, None),
        _project_qkv_gate=lambda qkv, positions: (qkv, qkv, qkv, None),
        attn=lambda query, key, value: query + key + value,
        o_proj=projection,
    )
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    positions = torch.arange(3)

    returned = qwen3_next.Qwen3NextAttention.forward(
        attention, positions, None, hidden_states
    )
    assert returned is projection.last_output
    assert torch.equal(returned, hidden_states * 6)

    output = torch.full_like(hidden_states, -1)
    output_ptr = output.data_ptr()
    result = qwen3_next.Qwen3NextAttention.forward(
        attention, positions, output, hidden_states
    )
    assert result is None
    assert output.data_ptr() == output_ptr
    assert torch.equal(output, hidden_states * 6)


def test_gdn_projection_returns_projection_or_preserves_output_buffer() -> None:
    projection = _Projection(scale=3.0)
    attention = SimpleNamespace(
        norm=lambda core, gate: core + gate,
        out_proj=projection,
    )
    core = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    gate = torch.ones_like(core)
    expected = (core + gate).flatten(-2) * 3

    returned = qwen_gdn.QwenGatedDeltaNetAttention._output_projection(
        attention, core, gate, None, 3
    )
    assert returned is projection.last_output
    assert torch.equal(returned, expected)

    output = torch.full((5, 8), -1.0)
    output_ptr = output.data_ptr()
    result = qwen_gdn.QwenGatedDeltaNetAttention._output_projection(
        attention, core, gate, output, 3
    )
    assert result is None
    assert output.data_ptr() == output_ptr
    assert torch.equal(output[:3], expected)
    assert torch.equal(output[3:], torch.full((2, 8), -1.0))


@pytest.mark.parametrize(
    ("layer_type", "attribute"),
    (("linear_attention", "linear_attn"), ("full_attention", "self_attn")),
)
@pytest.mark.parametrize(
    ("enabled", "num_tokens", "is_prefill", "expects_return"),
    (
        (True, 1024, True, True),
        (True, 1023, True, False),
        (True, 2048, False, False),
        (False, 1024, True, False),
    ),
)
def test_decoder_return_gate_preserves_parity_and_projection_lifetime(
    monkeypatch,
    layer_type: str,
    attribute: str,
    enabled: bool,
    num_tokens: int,
    is_prefill: bool,
    expects_return: bool,
) -> None:
    hidden_states = torch.arange(num_tokens * 2, dtype=torch.float32).reshape(
        num_tokens, 2
    )
    projected = hidden_states + 7

    class _Attention:
        def __init__(self) -> None:
            self.output = object()

        def __call__(self, *, output, **kwargs):
            self.output = output
            if output is None:
                return projected
            output.copy_(projected)
            return None

    attention = _Attention()
    layer = SimpleNamespace(
        layer_type=layer_type,
        _musa_return_attention_output=enabled,
        _musa_attention_metadata_prefix="model.layers.0.attention",
        input_layernorm=lambda states: states,
        post_attention_layernorm=lambda states, residual: (states, residual),
        mlp=lambda states: states,
        layer_scale=False,
    )
    setattr(layer, attribute, attention)
    monkeypatch.setattr(
        qwen3_next, "_is_musa_qwen_next_prefill", lambda prefix: is_prefill
    )

    returned, residual = qwen3_next.Qwen3NextDecoderLayer.forward(
        layer, hidden_states, None, positions=torch.arange(num_tokens)
    )
    assert residual is hidden_states
    assert torch.equal(returned, projected)
    assert (attention.output is None) is expects_return
    assert (returned is projected) is expects_return


def test_prefill_gate_uses_layer_phase_metadata(monkeypatch) -> None:
    prefix = "model.layers.0.self_attn.attn"
    monkeypatch.setattr(qwen3_next, "is_forward_context_available", lambda: True)

    def set_metadata(metadata) -> None:
        monkeypatch.setattr(
            qwen3_next,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=metadata),
        )

    set_metadata({prefix: SimpleNamespace(num_prefills=1)})
    assert qwen3_next._is_musa_qwen_next_prefill(prefix)

    for metadata in (
        {prefix: SimpleNamespace(num_prefills=0)},
        {"another.layer": SimpleNamespace(num_prefills=1)},
        [{prefix: SimpleNamespace(num_prefills=1)}],
    ):
        set_metadata(metadata)
        assert not qwen3_next._is_musa_qwen_next_prefill(prefix)

    monkeypatch.setattr(qwen3_next, "is_forward_context_available", lambda: False)
    assert not qwen3_next._is_musa_qwen_next_prefill(prefix)


@pytest.mark.parametrize("disabled_value", ("0", "false", "off", "no"))
def test_static_gate_honors_environment(monkeypatch, disabled_value: str) -> None:
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    vllm_config = SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        lora_config=None,
    )
    config = SimpleNamespace(hidden_size=5120)
    monkeypatch.setattr(qwen3_next.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(qwen3_next, "get_tensor_model_parallel_world_size", lambda: 2)

    monkeypatch.delenv(qwen3_next.MUSA_QWEN_NEXT_RETURN_ATTN_OUTPUT_ENV, raising=False)
    assert qwen3_next._use_musa_qwen_next_return_attention_output(vllm_config, config)

    monkeypatch.setenv(qwen3_next.MUSA_QWEN_NEXT_RETURN_ATTN_OUTPUT_ENV, disabled_value)
    assert not qwen3_next._use_musa_qwen_next_return_attention_output(
        vllm_config, config
    )


@pytest.mark.parametrize(
    ("layer_type", "expected_prefix"),
    (
        ("linear_attention", "model.layers.0.linear_attn"),
        ("full_attention", "model.layers.0.self_attn.attn"),
    ),
)
def test_qwen35_custom_init_sets_return_gate_and_metadata_prefix(
    monkeypatch, layer_type: str, expected_prefix: str
) -> None:
    config = SimpleNamespace(
        model_type="qwen3_5_text",
        hidden_size=5120,
        intermediate_size=1024,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        layer_scale=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=config,
            dtype=torch.bfloat16,
        ),
        cache_config=None,
        quant_config=None,
        lora_config=None,
    )
    seen = []
    monkeypatch.setattr(
        qwen3_5,
        "_use_musa_qwen_next_return_attention_output",
        lambda actual_vllm_config, actual_config: seen.append(
            (actual_vllm_config, actual_config)
        )
        or True,
    )
    monkeypatch.setattr(
        qwen3_5, "QwenGatedDeltaNetAttention", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(qwen3_5, "Qwen3NextAttention", lambda *args, **kwargs: object())
    monkeypatch.setattr(qwen3_5, "Qwen3NextMLP", lambda *args, **kwargs: object())
    monkeypatch.setattr(qwen3_5, "Qwen3_5RMSNorm", lambda *args, **kwargs: object())

    layer = qwen3_5.Qwen3_5DecoderLayer(
        vllm_config, layer_type, prefix="model.layers.0"
    )

    assert seen == [(vllm_config, config)]
    assert layer._musa_return_attention_output
    assert layer._musa_attention_metadata_prefix == expected_prefix
