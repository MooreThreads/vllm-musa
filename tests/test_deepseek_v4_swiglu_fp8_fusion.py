# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S5000 correctness coverage for the DeepSeek-V4 shared-MLP fusion."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("torchada")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_musa")


@pytest.fixture(scope="module", autouse=True)
def _musa_device() -> Iterator[None]:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.musa.set_device(0)

    from vllm.platforms import current_platform

    import vllm_musa

    if not current_platform.is_device_capability((3, 1)):
        pytest.skip("the fused clamp-SwiGLU group-quant kernel requires mp31")
    vllm_musa.register_custom_ops()
    yield


def _quant_outputs(rows: int, hidden: int = 256):
    q = torch.empty((rows, hidden), device="musa", dtype=torch.float8_e4m3fn)
    s = torch.empty((rows, hidden // 128), device="musa", dtype=torch.float32)
    return q, s


def _materialized_activation(gate_up, limit: float = 10.0):
    hidden = gate_up.shape[-1] // 2
    gate = torch.clamp(gate_up[..., :hidden], max=limit)
    up = torch.clamp(gate_up[..., hidden:], min=-limit, max=limit)
    return gate * torch.sigmoid(gate) * up


@pytest.mark.parametrize("rows", [1, 4, 16, 64, 4096, 4100])
def test_clamp_swiglu_group_quant_is_bit_exact(rows: int) -> None:
    torch.manual_seed(60156 + rows)
    gate_up = torch.randn((rows, 512), device="musa", dtype=torch.bfloat16)
    boundary = torch.tensor(
        [
            float("-inf"),
            -20.0,
            -10.0,
            -0.0,
            0.0,
            10.0,
            20.0,
            float("inf"),
        ],
        device="musa",
        dtype=torch.bfloat16,
    )
    gate_up[0, :8] = boundary
    gate_up[0, 256:264] = boundary.flip(0)

    reference_q, reference_s = _quant_outputs(rows)
    fused_q, fused_s = _quant_outputs(rows)
    activated = _materialized_activation(gate_up)
    torch.ops._C_musa_ops.per_token_group_quant_8bit_vec(
        activated,
        reference_q,
        reference_s,
        128,
        1e-10,
        -448.0,
        448.0,
    )
    torch.ops._C_musa_ops.silu_and_mul_clamp_per_token_group_fp8_quant(
        gate_up,
        fused_q,
        fused_s,
        128,
        1e-10,
        -448.0,
        448.0,
        10.0,
    )
    torch.musa.synchronize()

    assert torch.equal(
        reference_q.view(torch.uint8).cpu(), fused_q.view(torch.uint8).cpu()
    )
    assert torch.equal(
        reference_s.view(torch.int32).cpu(), fused_s.view(torch.int32).cpu()
    )


@pytest.mark.parametrize("rows", [1, 4, 64])
def test_bit_exact_quantization_preserves_bf16_down_projection(rows: int) -> None:
    torch.manual_seed(60256 + rows)
    gate_up = torch.randn((rows, 512), device="musa", dtype=torch.bfloat16)
    reference_q, reference_s = _quant_outputs(rows)
    fused_q, fused_s = _quant_outputs(rows)

    torch.ops._C_musa_ops.per_token_group_quant_8bit_vec(
        _materialized_activation(gate_up),
        reference_q,
        reference_s,
        128,
        1e-10,
        -448.0,
        448.0,
    )
    torch.ops._C_musa_ops.silu_and_mul_clamp_per_token_group_fp8_quant(
        gate_up,
        fused_q,
        fused_s,
        128,
        1e-10,
        -448.0,
        448.0,
        10.0,
    )

    reference_dequant = reference_q.float() * reference_s.repeat_interleave(128, -1)
    fused_dequant = fused_q.float() * fused_s.repeat_interleave(128, -1)
    weight = torch.randn((4096, 256), device="musa", dtype=torch.bfloat16)
    reference_out = (reference_dequant @ weight.float().T).to(torch.bfloat16)
    fused_out = (fused_dequant @ weight.float().T).to(torch.bfloat16)
    torch.musa.synchronize()

    assert torch.equal(reference_out, fused_out)


def test_clamp_swiglu_group_quant_replays_with_new_inputs() -> None:
    gate_up = torch.empty((1, 512), device="musa", dtype=torch.bfloat16)
    fused_q, fused_s = _quant_outputs(1)

    warmup_stream = torch.musa.Stream()
    warmup_stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(warmup_stream):
        for seed in (23, 29, 31):
            torch.manual_seed(seed)
            gate_up.copy_(torch.randn_like(gate_up))
            torch.ops._C_musa_ops.silu_and_mul_clamp_per_token_group_fp8_quant(
                gate_up,
                fused_q,
                fused_s,
                128,
                1e-10,
                -448.0,
                448.0,
                10.0,
            )
    torch.musa.current_stream().wait_stream(warmup_stream)

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        torch.ops._C_musa_ops.silu_and_mul_clamp_per_token_group_fp8_quant(
            gate_up,
            fused_q,
            fused_s,
            128,
            1e-10,
            -448.0,
            448.0,
            10.0,
        )

    for seed in (37, 41, 43):
        torch.manual_seed(seed)
        next_gate_up = torch.randn_like(gate_up)
        reference_q, reference_s = _quant_outputs(1)
        torch.ops._C_musa_ops.per_token_group_quant_8bit_vec(
            _materialized_activation(next_gate_up),
            reference_q,
            reference_s,
            128,
            1e-10,
            -448.0,
            448.0,
        )
        gate_up.copy_(next_gate_up)

        graph.replay()
        torch.musa.synchronize()

        assert torch.equal(
            reference_q.view(torch.uint8).cpu(), fused_q.view(torch.uint8).cpu()
        )
        assert torch.equal(
            reference_s.view(torch.int32).cpu(), fused_s.view(torch.int32).cpu()
        )
