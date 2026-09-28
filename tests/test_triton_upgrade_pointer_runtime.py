# SPDX-License-Identifier: Apache-2.0
"""Real kernels covering pointer loads, empty requests and mixed branch dtypes."""

import pytest

pytest.importorskip("torchada")
import torch


@pytest.fixture(scope="module", autouse=True)
def musa_device():
    if not getattr(torch.version, "musa", None):
        pytest.skip("requires MUSA PyTorch")
    assert torch.musa.is_available()
    torch.musa.set_device(0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_eagle_empty_and_rejected_requests(dtype):
    from vllm.v1.spec_decode.utils import eagle_prepare_inputs_padded_kernel

    def gpu(values):
        return torch.tensor(values, device="musa", dtype=dtype)

    indices = gpu([-99] * 6)
    rejected = gpu([-99] * 6)
    eagle_prepare_inputs_padded_kernel[(6,)](
        gpu([0, 2, 4, 4, 7]),
        gpu([0, 2, 1, 1, 4]),
        gpu([0, 0, 3, 6, 7, 11]),
        indices,
        rejected,
        5,
    )
    # Empty request, partial rejection, full rejection, no draft, all accepted,
    # and one excess program that must leave both output sentinels unchanged.
    assert indices.cpu().tolist() == [-1, 1, 3, 6, 10, -99]
    assert rejected.cpu().tolist() == [0, 1, 2, 0, 0, -99]


def test_block_table_pointer_gather_and_padding():
    from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel

    widths = [7, 19]
    sources = [
        torch.arange(4 * width, dtype=torch.int32).reshape(4, width).to("musa")
        for width in widths
    ]
    outputs = [
        torch.full((4, width), -77, device="musa", dtype=torch.int32)
        for width in widths
    ]
    counts = torch.tensor([[1, 3, 7, 0], [19, 4, 3, 0]], dtype=torch.int32)
    mapping = torch.tensor([2, 0, 3], device="musa", dtype=torch.int32)
    src_ptrs = torch.tensor(
        [x.data_ptr() for x in sources], device="musa", dtype=torch.uint64
    )
    dst_ptrs = torch.tensor(
        [x.data_ptr() for x in outputs], device="musa", dtype=torch.uint64
    )
    _gather_block_tables_kernel[(2, 4)](
        mapping,
        src_ptrs,
        dst_ptrs,
        torch.tensor(widths, device="musa", dtype=torch.int64),
        counts.to("musa"),
        4,
        3,
        BLOCK_SIZE=8,
    )
    for group, output in enumerate(outputs):
        expected = torch.full_like(output, -77, device="cpu")
        for batch, request in enumerate([2, 0, 3]):
            count = int(counts[group, request])
            expected[batch, :count] = sources[group].cpu()[request, :count]
        expected[3] = 0
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "cp_size,cp_rank,interleave", [(1, 0, 1), (2, 0, 1), (2, 1, 2)]
)
def test_slot_mapping_pointer_context_parallel_and_padding(
    cp_size, cp_rank, interleave
):
    from vllm.v1.worker.gpu.block_table import _compute_slot_mappings_kernel

    tables = [
        torch.tensor([[3, 5, 7, 9], [11, 13, 15, 17]], dtype=torch.int32, device="musa")
        for _ in range(2)
    ]
    positions = [0, 3, 4, 9, 1, 6]
    mapping = [1, 0]
    sizes = [4, 8]
    output = torch.full((2, 11), -77, dtype=torch.int64, device="musa")
    _compute_slot_mappings_kernel[(2, 3)](
        11,
        torch.tensor(mapping, device="musa", dtype=torch.int32),
        torch.tensor([0, 4, 6], device="musa", dtype=torch.int32),
        torch.tensor(positions, device="musa", dtype=torch.int64),
        torch.tensor([t.data_ptr() for t in tables], device="musa", dtype=torch.uint64),
        torch.tensor([4, 4], device="musa", dtype=torch.int64),
        torch.tensor(sizes, device="musa", dtype=torch.int32),
        output,
        output.stride(0),
        cp_rank,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=interleave,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=4,
    )
    expected = torch.full((2, 11), -1, dtype=torch.int64)
    for group, size in enumerate(sizes):
        table = tables[group].cpu()
        for token, position in enumerate(positions):
            request = mapping[0 if token < 4 else 1]
            block, offset = divmod(position, size * cp_size)
            if offset // interleave % cp_size == cp_rank:
                local_offset = (
                    offset // (interleave * cp_size) * interleave + offset % interleave
                )
                expected[group, token] = table[request, block] * size + local_offset
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden,hc_mult", [(127, 2), (256, 4)])
def test_mtp_rmsnorm_branch_dtypes(dtype, hidden, hc_mult):
    from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (
        fused_mtp_input_rmsnorm,
    )

    torch.manual_seed(18070)
    embeds = torch.randn(3, hidden).to(dtype)
    previous = torch.randn(3, hc_mult, hidden).to(dtype)
    ew, hw = torch.randn(2, hidden).to(dtype)
    actual = fused_mtp_input_rmsnorm(
        embeds.to("musa"),
        torch.tensor([0, 1, 17], device="musa"),
        previous.to("musa"),
        ew.to("musa"),
        hw.to("musa"),
        1e-6,
        hc_mult,
    )
    masked = embeds.float().clone()
    masked[0] = 0
    for out, x, weight in zip(actual, [masked, previous.float()], [ew, hw]):
        expected = (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight.float()
        ).to(dtype)
        torch.testing.assert_close(out.cpu(), expected, rtol=0.01, atol=0.005)
