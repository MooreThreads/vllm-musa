# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MUSA fused MoE chunked execution."""

import json

import pytest
import torch


def _patch_fake_moe_kernels(monkeypatch, fused_moe, top_k):
    def fake_musa_fused_gemv_moe(
        input_tensor,
        weight,
        output,
        _bias,
        _scale,
        _topk_weights,
        current_topk_ids,
        *_args,
        use_swigelu,
        **_kwargs,
    ):
        tokens = current_topk_ids.shape[0]
        if use_swigelu:
            output[: tokens * top_k].fill_(1)
        else:
            output[:tokens].fill_(2)

    def fake_moe_sum(intermediate_cache3, output):
        output.copy_(intermediate_cache3.sum(dim=1))

    monkeypatch.setattr(
        fused_moe.musa_ops, "musa_fused_gemv_moe", fake_musa_fused_gemv_moe
    )
    monkeypatch.setattr(fused_moe.ops, "moe_sum", fake_moe_sum)


def test_musa_fused_experts_preserves_output_shape_across_chunks(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    chunk_size = 16384
    num_tokens = chunk_size + 3
    hidden_size = 4
    intermediate_size = 8
    num_experts = 2
    top_k = 1

    hidden_states = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)
    w1 = torch.zeros(num_experts, intermediate_size, hidden_size)
    w2 = torch.zeros(num_experts, hidden_size, intermediate_size // 2)
    topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32)
    topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int64)

    second_gemm_calls = 0

    def fake_musa_fused_gemv_moe(
        input_tensor,
        weight,
        output,
        _bias,
        _scale,
        _topk_weights,
        current_topk_ids,
        *_args,
        use_swigelu,
        **_kwargs,
    ):
        nonlocal second_gemm_calls
        tokens = current_topk_ids.shape[0]
        if use_swigelu:
            output[: tokens * top_k].fill_(1)
        else:
            fill_value = 11.0 if second_gemm_calls == 0 else 23.0
            output[:tokens].fill_(fill_value)
            second_gemm_calls += 1

    def fake_moe_sum(intermediate_cache3, output):
        required_shape = (
            intermediate_cache3.shape[0],
            intermediate_cache3.shape[-1],
        )
        if tuple(output.shape) != required_shape:
            output.resize_(required_shape)
        output.copy_(intermediate_cache3.sum(dim=1))

    monkeypatch.setattr(
        fused_moe.musa_ops, "musa_fused_gemv_moe", fake_musa_fused_gemv_moe
    )
    monkeypatch.setattr(fused_moe.ops, "moe_sum", fake_moe_sum)

    result = fused_moe.fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    assert result.shape == (num_tokens, hidden_size)
    assert torch.all(result[:chunk_size] == 11.0)
    assert torch.all(result[chunk_size:] == 23.0)


def test_musa_fused_moe_shape_inventory_is_disabled_by_default(monkeypatch, tmp_path):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    top_k = 1
    output_path = tmp_path / "inventory.jsonl"

    monkeypatch.delenv("VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY", raising=False)
    monkeypatch.setenv(
        "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_PATH", str(output_path)
    )
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_MIN_TOKENS", "1")
    monkeypatch.setattr(fused_moe, "_MOE_SHAPE_INVENTORY_RECORDS", 0)
    monkeypatch.setattr(fused_moe, "_MOE_SHAPE_INVENTORY_ENABLED", False)
    _patch_fake_moe_kernels(monkeypatch, fused_moe, top_k)

    fused_moe.fused_experts_impl(
        hidden_states=torch.zeros(2, 4),
        w1=torch.zeros(2, 8, 4),
        w2=torch.zeros(2, 4, 4),
        topk_weights=torch.ones(2, top_k),
        topk_ids=torch.zeros(2, top_k, dtype=torch.int64),
    )

    assert not output_path.exists()


def test_musa_fused_moe_shape_inventory_records_bridge_contract(monkeypatch, tmp_path):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    top_k = 2
    output_path = tmp_path / "inventory.jsonl"

    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY", "1")
    monkeypatch.setenv(
        "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_PATH", str(output_path)
    )
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_MIN_TOKENS", "1")
    monkeypatch.setattr(fused_moe, "_MOE_SHAPE_INVENTORY_RECORDS", 0)
    monkeypatch.setattr(fused_moe, "_MOE_SHAPE_INVENTORY_ENABLED", True)
    monkeypatch.setattr(fused_moe, "_musa_stream_is_capturing", lambda: False)
    _patch_fake_moe_kernels(monkeypatch, fused_moe, top_k)

    result = fused_moe.fused_experts_impl(
        hidden_states=torch.zeros(3, 4),
        w1=torch.zeros(3, 8, 4),
        w2=torch.zeros(3, 4, 4),
        topk_weights=torch.ones(3, top_k),
        topk_ids=torch.tensor([[0, 2], [1, 2], [2, 0]], dtype=torch.int64),
        w1_scale=torch.ones(3),
        w2_scale=torch.ones(3),
        block_shape=[128, 128],
        use_fp8_w8a8=True,
        apply_router_weight_on_input=True,
        global_num_experts=3,
    )

    assert result.shape == (3, 4)
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 1

    record = records[0]
    assert record["event"] == "deepseek_v4_moe_shape_inventory"
    assert record["num_tokens"] == 3
    assert record["top_k"] == 2
    assert record["num_local_experts"] == 3
    assert record["global_num_experts"] == 3
    assert record["w1"]["shape"] == [3, 8, 4]
    assert record["w2"]["shape"] == [3, 4, 4]
    assert record["topk_ids"]["dtype"] == "torch.int64"
    assert record["w1_scale"]["shape"] == [3]
    assert record["block_shape"] == [128, 128]
    assert record["apply_router_weight_on_input"] is True
    assert record["use_fp8_w8a8"] is True
    assert record["routed_token_stats"]["histogram"] == [2, 1, 3]
    assert record["routed_token_stats"]["slot_histograms"] == [
        [1, 1, 1],
        [1, 0, 2],
    ]


def test_musa_fused_moe_shape_inventory_skips_graph_capture(monkeypatch, tmp_path):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    output_path = tmp_path / "inventory.jsonl"
    monkeypatch.setenv(
        "VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_PATH", str(output_path)
    )
    monkeypatch.setenv("VLLM_MUSA_DEEPSEEK_V4_MOE_SHAPE_INVENTORY_MIN_TOKENS", "1")
    monkeypatch.setattr(fused_moe, "_MOE_SHAPE_INVENTORY_RECORDS", 0)
    monkeypatch.setattr(fused_moe, "_MOE_SHAPE_INVENTORY_ENABLED", True)
    monkeypatch.setattr(fused_moe, "_musa_stream_is_capturing", lambda: True)
    _patch_fake_moe_kernels(monkeypatch, fused_moe, top_k=1)

    fused_moe.fused_experts_impl(
        hidden_states=torch.zeros(2, 4),
        w1=torch.zeros(2, 8, 4),
        w2=torch.zeros(2, 4, 4),
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )

    assert not output_path.exists()


def _deepgemm_gate_kwargs(torch_module):
    return {
        "hidden_states": torch_module.empty(
            (4100, 4096), device="meta", dtype=torch_module.bfloat16
        ),
        "w1": torch_module.empty(
            (256, 512, 4096),
            device="meta",
            dtype=torch_module.float8_e4m3fn,
        ),
        "w2": torch_module.empty(
            (256, 4096, 256),
            device="meta",
            dtype=torch_module.float8_e4m3fn,
        ),
        "topk_ids": torch_module.empty(
            (4100, 6), device="meta", dtype=torch_module.int32
        ),
        "topk_weights": torch_module.empty(
            (4100, 6), device="meta", dtype=torch_module.float32
        ),
        "activation": "silu",
        "apply_router_weight_on_input": False,
        "use_fp8_w8a8": True,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "ocp_mx_scheme": None,
        "per_channel_quant": False,
        "global_num_experts": 256,
        "expert_map": None,
        "w1_scale": torch_module.empty(
            (256, 4, 32), device="meta", dtype=torch_module.float32
        ),
        "w2_scale": torch_module.empty(
            (256, 32, 2), device="meta", dtype=torch_module.float32
        ),
        "w1_zp": None,
        "w2_zp": None,
        "a1_scale": None,
        "a2_scale": None,
        "block_shape": [128, 128],
        "w1_bias": None,
        "w2_bias": None,
    }


def _gemv_gate_kwargs(torch_module):
    grouped_kwargs = _deepgemm_gate_kwargs(torch_module)
    return {
        key: value
        for key, value in grouped_kwargs.items()
        if key
        not in {
            "apply_router_weight_on_input",
            "block_shape",
        }
    } | {
        "global_num_experts": 256,
    }


def test_musa_native_gemv_capability_accepts_supported_fp8_shape():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    assert fused_moe._can_use_musa_native_fp8_moe_gemv(**_gemv_gate_kwargs(torch))


def test_musa_native_gemv_scale_layout_rejects_ambiguous_or_padded_blocks():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    weight = torch.empty((4, 2048, 64), device="meta", dtype=torch.float8_e4m3fn)
    ambiguous_group64 = torch.empty((4, 32, 1), device="meta", dtype=torch.float32)
    assert not fused_moe._musa_fp8_moe_scale_layout_is_supported(
        ambiguous_group64, weight
    )
    assert not fused_moe._musa_fp8_moe_scale_layout_is_supported(
        torch.ones(4, device="meta", dtype=torch.float32), weight
    )

    padded_output = torch.empty(
        (4, 4160, 256), device="meta", dtype=torch.float8_e4m3fn
    )
    ceil_block128 = torch.empty((4, 33, 2), device="meta", dtype=torch.float32)
    assert not fused_moe._musa_fp8_moe_scale_layout_is_supported(
        ceil_block128, padded_output
    )

    aligned_group64_weight = torch.empty(
        (4, 2048, 256), device="meta", dtype=torch.float8_e4m3fn
    )
    aligned_group64_scale = torch.empty((4, 32, 4), device="meta", dtype=torch.float32)
    assert fused_moe._musa_fp8_moe_scale_layout_is_supported(
        aligned_group64_scale, aligned_group64_weight
    )


def test_musa_native_gemv_capability_rejects_unsafe_inputs():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _gemv_gate_kwargs(torch)
    kwargs["expert_map"] = torch.arange(256, device="meta", dtype=torch.int32)
    assert not fused_moe._can_use_musa_native_fp8_moe_gemv(**kwargs)

    kwargs = _gemv_gate_kwargs(torch)
    kwargs["topk_ids"] = torch.empty((4100, 6), device="meta", dtype=torch.int64)
    assert not fused_moe._can_use_musa_native_fp8_moe_gemv(**kwargs)

    kwargs = _gemv_gate_kwargs(torch)
    kwargs["activation"] = "gelu"
    assert not fused_moe._can_use_musa_native_fp8_moe_gemv(**kwargs)

    kwargs = _gemv_gate_kwargs(torch)
    kwargs["w1_bias"] = torch.empty((512,), device="meta", dtype=torch.bfloat16)
    assert not fused_moe._can_use_musa_native_fp8_moe_gemv(**kwargs)

    kwargs = _gemv_gate_kwargs(torch)
    kwargs["global_num_experts"] = 512
    assert not fused_moe._can_use_musa_native_fp8_moe_gemv(**kwargs)

    kwargs = _gemv_gate_kwargs(torch)
    kwargs["topk_weights"] = torch.empty((4099, 6), device="meta", dtype=torch.float32)
    assert not fused_moe._can_use_musa_native_fp8_moe_gemv(**kwargs)

    kwargs = _gemv_gate_kwargs(torch)
    kwargs["w1_scale"] = torch.empty(
        (256, 32, 4), device="meta", dtype=torch.float32
    ).transpose(1, 2)
    assert kwargs["w1_scale"].shape == (256, 4, 32)
    assert not kwargs["w1_scale"].is_contiguous()
    assert not fused_moe._can_use_musa_native_fp8_moe_gemv(**kwargs)

    kwargs = _gemv_gate_kwargs(torch)
    kwargs["w1_scale"] = torch.empty((256, 4, 32), dtype=torch.float32)
    assert not fused_moe._can_use_musa_native_fp8_moe_gemv(**kwargs)


def test_musa_grouped_gemm_gate_is_shape_gated():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    assert fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)


def test_musa_deepgemm_prefill_gate_accepts_folded_qwen_shape():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    for key in ("topk_weights", "global_num_experts", "w1_zp", "w2_zp"):
        kwargs.pop(key)
    assert fused_moe._can_use_moe_deepgemm_prefill(**kwargs)

    # Shared-expert folding appends one expert and one routing column.  The
    # large-M DeepGEMM gate must retain that base-path shape even before the
    # small-M GEMV threshold is calibrated.
    kwargs.update(
        hidden_states=torch.empty((4100, 2048), device="meta", dtype=torch.bfloat16),
        w1=torch.empty((257, 256, 2048), device="meta", dtype=torch.float8_e4m3fn),
        w2=torch.empty((257, 2048, 128), device="meta", dtype=torch.float8_e4m3fn),
        topk_ids=torch.empty((4100, 9), device="meta", dtype=torch.int32),
        w1_scale=torch.empty((257, 2, 16), device="meta", dtype=torch.float32),
        w2_scale=torch.empty((257, 16, 1), device="meta", dtype=torch.float32),
    )
    assert fused_moe._can_use_moe_deepgemm_prefill(**kwargs)

    kwargs["hidden_states"] = torch.empty(
        (64, 2048), device="meta", dtype=torch.bfloat16
    )
    assert not fused_moe._can_use_moe_deepgemm_prefill(**kwargs)

    kwargs["hidden_states"] = torch.empty(
        (4100, 2048), device="meta", dtype=torch.bfloat16
    )
    kwargs["expert_map"] = torch.empty((257,), device="meta", dtype=torch.int32)
    assert not fused_moe._can_use_moe_deepgemm_prefill(**kwargs)


def test_musa_grouped_gemm_gate_rejects_nonmatching_shape():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["w1"] = torch.empty(
        (256, 640, 4096), device="meta", dtype=torch.float8_e4m3fn
    )

    assert not fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)


def test_musa_grouped_gemm_gate_rejects_nonmatching_dtype():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["w1_scale"] = torch.empty((256, 4, 32), device="meta", dtype=torch.bfloat16)
    assert not fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["w2_scale"] = torch.empty((256, 32, 2), device="meta", dtype=torch.bfloat16)
    assert not fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["topk_ids"] = torch.empty((4100, 6), device="meta", dtype=torch.int64)
    assert not fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)


def test_musa_grouped_gemm_gate_rejects_unsafe_routing_and_layout():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["global_num_experts"] = 512
    assert not fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["topk_ids"] = torch.empty((4099, 6), device="meta", dtype=torch.int32)
    assert not fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["w2_scale"] = torch.empty(
        (256, 2, 32), device="meta", dtype=torch.float32
    ).transpose(1, 2)
    assert kwargs["w2_scale"].shape == (256, 32, 2)
    assert not kwargs["w2_scale"].is_contiguous()
    assert not fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["w1_zp"] = torch.empty((1,), device="meta", dtype=torch.int32)
    assert not fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["apply_router_weight_on_input"] = True
    assert not fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)


def test_musa_grouped_gemm_gate_matches_unpermute_block_contract():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs.update(
        hidden_states=torch.empty((4100, 384), device="meta", dtype=torch.bfloat16),
        w1=torch.empty((256, 256, 384), device="meta", dtype=torch.float8_e4m3fn),
        w2=torch.empty((256, 384, 128), device="meta", dtype=torch.float8_e4m3fn),
        w1_scale=torch.empty((256, 2, 3), device="meta", dtype=torch.float32),
        w2_scale=torch.empty((256, 3, 1), device="meta", dtype=torch.float32),
    )
    assert not fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)

    kwargs.update(
        hidden_states=torch.empty((4100, 1024), device="meta", dtype=torch.bfloat16),
        w1=torch.empty((256, 256, 1024), device="meta", dtype=torch.float8_e4m3fn),
        w2=torch.empty((256, 1024, 128), device="meta", dtype=torch.float8_e4m3fn),
        w1_scale=torch.empty((256, 2, 8), device="meta", dtype=torch.float32),
        w2_scale=torch.empty((256, 8, 1), device="meta", dtype=torch.float32),
    )
    assert fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)


def test_musa_grouped_gemm_failure_disables_worker_path(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    monkeypatch.setattr(fused_moe, "_MUSA_GROUPED_GEMM_AVAILABLE", True)
    monkeypatch.setattr(
        fused_moe,
        "_musa_fp8_moe_grouped_gemm_impl",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("jit failed")),
    )
    assert fused_moe._maybe_musa_fp8_moe_grouped_gemm(**kwargs, inplace=False) is None
    assert fused_moe._MUSA_GROUPED_GEMM_AVAILABLE is False
    assert not fused_moe._can_use_musa_fp8_moe_grouped_gemm(**kwargs)


def _patch_s5000_dispatch(monkeypatch, fused_moe):
    monkeypatch.setattr(
        fused_moe,
        "_musa_device_fingerprint",
        lambda *_args, **_kwargs: ((3, 1), 64),
    )
    monkeypatch.setattr(fused_moe, "_musa_stream_is_capturing", lambda: False)


def test_musa_capture_probe_fails_safe(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    for module_name in ("musa", "cuda"):
        module = getattr(torch, module_name, None)
        if module is not None:
            monkeypatch.setattr(
                module,
                "is_current_stream_capturing",
                lambda: (_ for _ in ()).throw(RuntimeError("probe failed")),
                raising=False,
            )
    assert fused_moe._musa_stream_is_capturing() is True


def test_musa_dispatcher_routes_forced_backends(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    _patch_s5000_dispatch(monkeypatch, fused_moe)
    kwargs = _deepgemm_gate_kwargs(torch)
    sentinel = torch.empty((1,), device="meta")

    native_calls = []
    monkeypatch.setattr(
        fused_moe,
        "fused_experts_impl",
        lambda *args, **call_kwargs: native_calls.append((args, call_kwargs))
        or sentinel,
    )
    monkeypatch.setattr(
        fused_moe,
        "_MUSA_FUSED_MOE_REQUESTED_BACKEND",
        fused_moe.MusaFusedMoeBackend.GEMV,
    )
    assert fused_moe._musa_fused_experts_impl_dispatch(**kwargs) is sentinel
    assert len(native_calls) == 1
    assert native_calls[0][1] == {
        "inplace": False,
        "_allow_deepgemm_prefill": False,
    }

    grouped_calls = []
    monkeypatch.setattr(
        fused_moe,
        "_maybe_musa_fp8_moe_grouped_gemm",
        lambda **call_kwargs: grouped_calls.append(call_kwargs) or sentinel,
    )
    monkeypatch.setattr(
        fused_moe,
        "_MUSA_FUSED_MOE_REQUESTED_BACKEND",
        fused_moe.MusaFusedMoeBackend.GROUPED_GEMM,
    )
    assert fused_moe._musa_fused_experts_impl_dispatch(**kwargs) is sentinel
    assert len(grouped_calls) == 1
    assert grouped_calls[0]["global_num_experts"] == 256
    assert grouped_calls[0]["w1_zp"] is None
    assert grouped_calls[0]["inplace"] is False


def test_musa_dispatcher_rejects_gemv_input_router_weighting(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    _patch_s5000_dispatch(monkeypatch, fused_moe)
    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["apply_router_weight_on_input"] = True
    fallback = torch.empty((1,), device="meta")
    monkeypatch.setattr(
        fused_moe,
        "_MUSA_FUSED_MOE_REQUESTED_BACKEND",
        fused_moe.MusaFusedMoeBackend.GEMV,
    )
    monkeypatch.setattr(
        fused_moe,
        "fused_experts_impl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("input-router weighting must not dispatch GEMV")
        ),
    )
    monkeypatch.setattr(
        fused_moe._upstream_fused_moe,
        "_musa_original_fused_experts_impl",
        lambda *_args, **_kwargs: fallback,
    )

    assert fused_moe._musa_fused_experts_impl_dispatch(**kwargs) is fallback


def test_musa_dispatcher_grouped_failure_and_capture_fall_back(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    _patch_s5000_dispatch(monkeypatch, fused_moe)
    kwargs = _deepgemm_gate_kwargs(torch)
    fallback = torch.empty((1,), device="meta")
    upstream_calls = []
    monkeypatch.setattr(
        fused_moe._upstream_fused_moe,
        "_musa_original_fused_experts_impl",
        lambda *args, **call_kwargs: upstream_calls.append((args, call_kwargs))
        or fallback,
    )
    monkeypatch.setattr(
        fused_moe,
        "_MUSA_FUSED_MOE_REQUESTED_BACKEND",
        fused_moe.MusaFusedMoeBackend.GROUPED_GEMM,
    )
    monkeypatch.setattr(
        fused_moe, "_maybe_musa_fp8_moe_grouped_gemm", lambda **_kwargs: None
    )
    assert fused_moe._musa_fused_experts_impl_dispatch(**kwargs) is fallback
    assert len(upstream_calls) == 1
    assert upstream_calls[0][1] == {}

    monkeypatch.setattr(fused_moe, "_musa_stream_is_capturing", lambda: True)
    monkeypatch.setattr(
        fused_moe,
        "_maybe_musa_fp8_moe_grouped_gemm",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )
    assert fused_moe._musa_fused_experts_impl_dispatch(**kwargs) is fallback
    assert len(upstream_calls) == 2


def test_musa_dispatcher_unknown_small_shape_uses_upstream_path(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs["hidden_states"] = torch.empty(
        (64, 4096), device="meta", dtype=torch.bfloat16
    )
    kwargs["topk_ids"] = torch.empty((64, 6), device="meta", dtype=torch.int32)
    kwargs["topk_weights"] = torch.empty((64, 6), device="meta", dtype=torch.float32)
    fallback = torch.empty((1,), device="meta")
    monkeypatch.setattr(
        fused_moe,
        "_MUSA_FUSED_MOE_REQUESTED_BACKEND",
        fused_moe.MusaFusedMoeBackend.AUTO,
    )
    monkeypatch.setattr(fused_moe, "has_calibrated_dimensions", lambda **_kwargs: False)
    monkeypatch.setattr(
        fused_moe,
        "_musa_device_fingerprint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unknown shape must not query the device")
        ),
    )
    monkeypatch.setattr(
        fused_moe,
        "_musa_stream_is_capturing",
        lambda: (_ for _ in ()).throw(
            AssertionError("small unknown shape must not probe capture")
        ),
    )
    monkeypatch.setattr(
        fused_moe._upstream_fused_moe,
        "_musa_original_fused_experts_impl",
        lambda *_args, **_kwargs: fallback,
    )
    assert fused_moe._musa_fused_experts_impl_dispatch(**kwargs) is fallback


def test_musa_dispatcher_auto_preserves_base_deepgemm_for_unknown_large_shape(
    monkeypatch,
):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    deepgemm_output = torch.empty((1,), device="meta")
    monkeypatch.setattr(
        fused_moe,
        "_MUSA_FUSED_MOE_REQUESTED_BACKEND",
        fused_moe.MusaFusedMoeBackend.AUTO,
    )
    monkeypatch.setattr(fused_moe, "_musa_stream_is_capturing", lambda: False)
    monkeypatch.setattr(fused_moe, "has_calibrated_dimensions", lambda **_kwargs: False)
    monkeypatch.setattr(
        fused_moe,
        "_maybe_moe_deepgemm_prefill",
        lambda **_kwargs: deepgemm_output,
    )
    monkeypatch.setattr(
        fused_moe._upstream_fused_moe,
        "_musa_original_fused_experts_impl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("large auto fallback should use base DeepGEMM")
        ),
    )

    assert fused_moe._musa_fused_experts_impl_dispatch(**kwargs) is deepgemm_output


def test_musa_dispatcher_unaligned_per_tensor_scales_fall_back(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    kwargs.update(
        hidden_states=torch.empty((4100, 2048), device="meta", dtype=torch.bfloat16),
        w1=torch.empty((256, 4160, 2048), device="meta", dtype=torch.float8_e4m3fn),
        w2=torch.empty((256, 2048, 2080), device="meta", dtype=torch.float8_e4m3fn),
        w1_scale=torch.empty((), device="meta", dtype=torch.float32),
        w2_scale=torch.empty((), device="meta", dtype=torch.float32),
    )
    fallback = torch.empty((1,), device="meta")
    monkeypatch.setattr(
        fused_moe,
        "_MUSA_FUSED_MOE_REQUESTED_BACKEND",
        fused_moe.MusaFusedMoeBackend.AUTO,
    )
    monkeypatch.setattr(fused_moe, "_musa_stream_is_capturing", lambda: False)
    monkeypatch.setattr(
        fused_moe._upstream_fused_moe,
        "_musa_original_fused_experts_impl",
        lambda *_args, **_kwargs: fallback,
    )

    assert fused_moe._musa_fused_experts_impl_dispatch(**kwargs) is fallback


def test_musa_dispatcher_unknown_large_capture_bypasses_deepgemm(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    fallback = torch.empty((1,), device="meta")
    monkeypatch.setattr(
        fused_moe,
        "_MUSA_FUSED_MOE_REQUESTED_BACKEND",
        fused_moe.MusaFusedMoeBackend.AUTO,
    )
    monkeypatch.setattr(fused_moe, "_musa_stream_is_capturing", lambda: True)
    monkeypatch.setattr(fused_moe, "has_calibrated_dimensions", lambda **_kwargs: False)
    monkeypatch.setattr(
        fused_moe,
        "_maybe_moe_deepgemm_prefill",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("capture must bypass DeepGEMM")
        ),
    )
    monkeypatch.setattr(
        fused_moe._upstream_fused_moe,
        "_musa_original_fused_experts_impl",
        lambda *_args, **_kwargs: fallback,
    )

    assert fused_moe._musa_fused_experts_impl_dispatch(**kwargs) is fallback


def test_musa_dispatcher_explicit_upstream_bypasses_deepgemm(monkeypatch):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _deepgemm_gate_kwargs(torch)
    fallback = torch.empty((1,), device="meta")
    monkeypatch.setattr(
        fused_moe,
        "_MUSA_FUSED_MOE_REQUESTED_BACKEND",
        fused_moe.MusaFusedMoeBackend.UPSTREAM,
    )
    monkeypatch.setattr(
        fused_moe,
        "_maybe_moe_deepgemm_prefill",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit upstream must bypass DeepGEMM")
        ),
    )
    monkeypatch.setattr(
        fused_moe._upstream_fused_moe,
        "_musa_original_fused_experts_impl",
        lambda *_args, **_kwargs: fallback,
    )

    assert fused_moe._musa_fused_experts_impl_dispatch(**kwargs) is fallback


def _qwen36_bf16_moe_meta_inputs(tokens: int):
    return {
        "hidden_states": torch.empty(
            tokens, 4096, dtype=torch.bfloat16, device="meta"
        ),
        "w1": torch.empty(257, 2048, 4096, dtype=torch.bfloat16, device="meta"),
        "w2": torch.empty(257, 4096, 1024, dtype=torch.bfloat16, device="meta"),
    }


def _qwen36_bf16_prefill_gate_kwargs(tokens: int = 2500):
    inputs = _qwen36_bf16_moe_meta_inputs(tokens)
    return {
        **inputs,
        "topk_ids": torch.empty(tokens, 9, dtype=torch.int32, device="meta"),
        "activation": "silu",
        "apply_router_weight_on_input": False,
        "use_fp8_w8a8": False,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "ocp_mx_scheme": None,
        "per_channel_quant": False,
        "expert_map": None,
        "w1_scale": None,
        "w2_scale": None,
        "a1_scale": None,
        "a2_scale": None,
        "block_shape": None,
        "w1_bias": None,
        "w2_bias": None,
    }


def test_musa_bf16_deepgemm_prefill_uses_validated_threshold():
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    assert not fused_moe._can_use_moe_deepgemm_bf16_prefill(
        **_qwen36_bf16_prefill_gate_kwargs(1023)
    )
    assert fused_moe._can_use_moe_deepgemm_bf16_prefill(
        **_qwen36_bf16_prefill_gate_kwargs(1024)
    )


@pytest.mark.parametrize(
    "unsupported",
    [
        {"use_fp8_w8a8": True},
        {"block_shape": [128, 128]},
        {"expert_map": torch.empty(1, dtype=torch.int32, device="meta")},
        {"w1_bias": torch.empty(1, dtype=torch.bfloat16, device="meta")},
        {"activation": "gelu"},
    ],
)
def test_musa_bf16_deepgemm_prefill_gate_rejects_unsupported_variants(
    unsupported,
):
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    kwargs = _qwen36_bf16_prefill_gate_kwargs()
    kwargs.update(unsupported)
    assert not fused_moe._can_use_moe_deepgemm_bf16_prefill(**kwargs)
