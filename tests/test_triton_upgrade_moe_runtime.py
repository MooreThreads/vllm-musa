# SPDX-License-Identifier: Apache-2.0
"""Small routed-expert numerical checks before full MoE shape tuning."""

import pytest

pytest.importorskip("torchada")
import torch


@pytest.fixture(scope="module", autouse=True)
def musa_workspace():
    if not getattr(torch.version, "musa", None):
        pytest.skip("requires a MUSA PyTorch build")
    assert torch.musa.is_available()
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    import vllm_musa

    vllm_musa.register_custom_ops()
    init_workspace_manager(torch.device("musa:0"))
    yield
    reset_workspace_manager()


@pytest.mark.parametrize("rows", [1, 7, 33])
@pytest.mark.parametrize("fp8", [False, True])
def test_fused_experts_against_cpu(rows: int, fp8: bool) -> None:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl

    torch.manual_seed(18021)
    x = torch.randn(rows, 256, dtype=torch.bfloat16) / 10
    w1 = torch.randn(4, 256, 256, dtype=torch.bfloat16) / 10
    w2 = torch.randn(4, 256, 128, dtype=torch.bfloat16) / 10
    ids = torch.stack((torch.arange(rows) % 4, (torch.arange(rows) + 1) % 4), dim=1)
    weights = torch.tensor([0.3, 0.7]).expand(rows, 2).contiguous()
    kwargs = {}
    if fp8:
        w1 = (w1.float() / 0.002).to(torch.float8_e4m3fn)
        w2 = (w2.float() / 0.002).to(torch.float8_e4m3fn)
        w1_ref, w2_ref = w1.float() * 0.002, w2.float() * 0.002
        kwargs = dict(
            use_fp8_w8a8=True,
            w1_scale=torch.full((4, 2, 2), 0.002, device="musa"),
            w2_scale=torch.full((4, 2, 1), 0.002, device="musa"),
            block_shape=[128, 128],
        )
    else:
        w1_ref, w2_ref = w1.float(), w2.float()
    expected = torch.zeros(rows, 256)
    for row in range(rows):
        for slot in range(2):
            expert = ids[row, slot].item()
            gate, up = (x[row].float() @ w1_ref[expert].T).chunk(2)
            activation = torch.nn.functional.silu(gate) * up
            expected[row] += (activation @ w2_ref[expert].T) * weights[row, slot]
    actual = fused_experts_impl(
        x.to("musa"),
        w1.to("musa"),
        w2.to("musa"),
        weights.to("musa"),
        ids.to(device="musa", dtype=torch.int32),
        **kwargs,
    )
    torch.testing.assert_close(
        actual.cpu().float(),
        expected,
        atol=0.01 if fp8 else 0.002,
        rtol=0.05 if fp8 else 0.02,
    )
