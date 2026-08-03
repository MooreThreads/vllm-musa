import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")
causal_conv1d = pytest.importorskip("vllm_musa.jit_kernel.tilelang.causal_conv1d")


def _prefill_gate(**overrides):
    kwargs = {
        "width": 4,
        "dim": 10240,
        "dtype": torch.bfloat16,
        "max_seq_len": 4096,
        "batch_size": 2,
        "has_conv_states": True,
        "has_cache_indices": True,
        "cache_indices_stride": 1,
        "x_inner_stride": 1,
        "out_inner_stride": 1,
        "weight_inner_stride": 1,
    }
    kwargs.update(overrides)
    return causal_conv1d._should_use_width4_prefill_split(**kwargs)


def test_prefill_split_gate_accepts_validated_tp1_width(monkeypatch):
    monkeypatch.setattr(causal_conv1d, "_ENABLE_WIDTH4_PREFILL_SPLIT", True)
    assert _prefill_gate(dim=10240)


def test_prefill_split_gate_rejects_tp4_and_small_prefill(monkeypatch):
    monkeypatch.setattr(causal_conv1d, "_ENABLE_WIDTH4_PREFILL_SPLIT", True)
    assert not _prefill_gate(dim=2048)
    assert not _prefill_gate(dim=2560)
    assert not _prefill_gate(dim=12288)
    assert not _prefill_gate(dim=8192)
    assert not _prefill_gate(max_seq_len=2048)
    assert not _prefill_gate(batch_size=1)
    assert not _prefill_gate(cache_indices_stride=2)
    assert not _prefill_gate(dtype=torch.float16)


def test_prefill_split_gate_honors_kill_switch(monkeypatch):
    monkeypatch.setattr(causal_conv1d, "_ENABLE_WIDTH4_PREFILL_SPLIT", False)
    assert not _prefill_gate()


def test_decode_kernel_keeps_supported_mixed_dtypes(monkeypatch):
    captured = {}

    def fake_kernel(*kernel_config):
        captured["kernel_config"] = kernel_config

        def run(x, weight, bias, state, indices, mapping, has_init, out, *scalars):
            captured["x"] = x
            captured["weight"] = weight
            captured["state"] = state
            captured["out"] = out
            captured["scalars"] = scalars
            out.copy_(x)

        return run

    monkeypatch.setattr(
        causal_conv1d,
        "_causal_conv1d_decode_width4_batched_kernel",
        fake_kernel,
    )
    monkeypatch.setattr(causal_conv1d, "_DECODE_HAS_INIT_BUF", {})

    x = torch.randn(4, 16, dtype=torch.bfloat16)
    state = torch.randn(4, 16, 3, dtype=torch.float32)
    weight = torch.randn(16, 4, dtype=torch.float32)
    indices = torch.arange(4, dtype=torch.int32)

    output = causal_conv1d.musa_tilelang_causal_conv1d_update(
        x,
        state,
        weight,
        activation="silu",
        conv_state_indices=indices,
    )

    assert output is not None
    assert output.dtype == torch.bfloat16
    assert captured["x"].dtype == torch.bfloat16
    assert captured["weight"].dtype == torch.float32
    assert captured["state"].dtype == torch.float32
    assert captured["out"].dtype == torch.bfloat16
    assert captured["kernel_config"][:5] == (
        "bfloat16",
        "float32",
        "bfloat16",
        "float32",
        "bfloat16",
    )
    assert captured["kernel_config"][13:19] == (
        False,  # has_bias
        True,  # has_cache_indices
        False,  # has_cache_index_mapping
        True,  # has_initial_states
        True,  # use_pad_slot
        True,  # silu_activation
    )
    assert captured["scalars"][-1] == causal_conv1d.NULL_BLOCK_ID


def test_decode_kernel_preserves_same_dtype_path(monkeypatch):
    captured = {}

    def fake_kernel(*kernel_config):
        captured["kernel_config"] = kernel_config

        def run(x, weight, bias, state, indices, mapping, has_init, out, *scalars):
            captured["x"] = x
            captured["scalars"] = scalars
            out.copy_(x)

        return run

    monkeypatch.setattr(
        causal_conv1d,
        "_causal_conv1d_decode_width4_batched_kernel",
        fake_kernel,
    )
    monkeypatch.setattr(causal_conv1d, "_DECODE_HAS_INIT_BUF", {})

    x = torch.randn(2, 8, dtype=torch.float32)
    state = torch.randn(2, 8, 3, dtype=torch.float32)
    weight = torch.randn(8, 4, dtype=torch.float32)

    output = causal_conv1d.musa_tilelang_causal_conv1d_update(
        x,
        state,
        weight,
    )

    assert output is not None
    assert output.dtype == torch.float32
    assert captured["x"] is not None
    assert captured["x"].dtype == torch.float32
    assert captured["kernel_config"][:5] == ("float32",) * 5
    assert captured["kernel_config"][13:19] == (
        False,  # has_bias
        False,  # has_cache_indices
        False,  # has_cache_index_mapping
        True,  # has_initial_states
        False,  # use_pad_slot
        False,  # silu_activation
    )
    assert captured["scalars"][-1] == causal_conv1d.PAD_SLOT_ID


def test_decode_kernel_rejects_unverified_mixed_dtype_tuple():
    x = torch.randn(2, 8, dtype=torch.float32)
    state = torch.randn(2, 8, 3, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)

    assert causal_conv1d.musa_tilelang_causal_conv1d_update(x, state, weight) is None


def test_decode_kernel_rejects_noncontiguous_indices():
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    state = torch.randn(4, 8, 3, dtype=torch.float32)
    weight = torch.randn(8, 4, dtype=torch.float32)
    indices = torch.arange(4, dtype=torch.int32)[::2]

    assert not indices.is_contiguous()
    assert (
        causal_conv1d.musa_tilelang_causal_conv1d_update(
            x, state, weight, conv_state_indices=indices
        )
        is None
    )
