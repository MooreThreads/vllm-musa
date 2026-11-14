# SPDX-License-Identifier: Apache-2.0
"""Keep one communicator API with backend-local NCCL/RCCL/MCCL selection."""

from types import SimpleNamespace

import pytest
import torch

from tests.moe_contract_utils import load_applied_contract, load_applied_methods


@pytest.mark.parametrize(
    "version,expected",
    [
        ({"cuda": "12.9", "hip": None}, "libnccl.so.2"),
        ({"cuda": None, "hip": "6.4"}, "librccl.so.1"),
        ({"cuda": None, "hip": None, "musa": "5.2.0"}, "libmccl.so.2"),
    ],
)
def test_library_selection_uses_the_same_upstream_entry(version, expected):
    module = load_applied_contract(
        "vllm/utils/nccl.py",
        ["find_nccl_library"],
        {
            "torch": SimpleNamespace(version=SimpleNamespace(**version)),
            "envs": SimpleNamespace(VLLM_NCCL_SO_PATH=None),
            "logger": SimpleNamespace(
                info=lambda *args: None, debug_once=lambda *args: None
            ),
        },
    )
    assert module.find_nccl_library() == expected


@pytest.mark.parametrize(
    "library", ["/opt/custom/libnccl.so", "/opt/mccl/libmccl.so.2"]
)
def test_library_override_precedes_backend_detection(library):
    module = load_applied_contract(
        "vllm/utils/nccl.py",
        ["find_nccl_library"],
        {
            "torch": SimpleNamespace(version=object()),
            "envs": SimpleNamespace(VLLM_NCCL_SO_PATH=library),
            "logger": SimpleNamespace(info=lambda *args: None),
        },
    )
    assert module.find_nccl_library() == library


def test_non_gpu_backend_still_fails_closed():
    module = load_applied_contract(
        "vllm/utils/nccl.py",
        ["find_nccl_library"],
        {
            "torch": SimpleNamespace(version=SimpleNamespace(cuda=None, hip=None)),
            "envs": SimpleNamespace(VLLM_NCCL_SO_PATH=None),
            "logger": SimpleNamespace(),
        },
    )
    with pytest.raises(ValueError, match="CUDA, ROCm and MUSA"):
        module.find_nccl_library()


@pytest.mark.parametrize("backend", ["cuda", "rocm", "musa"])
@pytest.mark.parametrize("collective", ["all_gatherv", "reduce_scatterv"])
def test_only_mccl_avoids_grouped_variable_collectives(backend, collective):
    calls = []
    stream = SimpleNamespace(cuda_stream=123)
    module = load_applied_methods(
        "vllm/distributed/device_communicators/pynccl.py",
        "PyNcclCommunicator",
        ["all_gatherv", "reduce_scatterv"],
        {
            "torch": torch,
            "ReduceOp": torch.distributed.ReduceOp,
            "current_platform": SimpleNamespace(
                device_type="musa" if backend == "musa" else "cuda"
            ),
            "current_stream": lambda: stream,
            "buffer_type": lambda pointer: pointer,
            "cudaStream_t": lambda pointer: pointer,
            "ncclDataTypeEnum": SimpleNamespace(from_torch=lambda dtype: dtype),
            "ncclRedOpTypeEnum": SimpleNamespace(from_torch=lambda op: op),
        },
    )
    comm = module.PyNcclCommunicator()
    comm.disabled = False
    comm.device = torch.device("cpu")
    comm.comm = object()
    comm.nccl = SimpleNamespace(
        ncclGroupStart=lambda: calls.append("start"),
        ncclGroupEnd=lambda: calls.append("end"),
        ncclBroadcast=lambda *args: calls.append(("broadcast", args[2], args[4])),
        ncclReduce=lambda *args: calls.append(("reduce", args[2], args[5])),
    )
    comm._all_gatherv_padded = lambda *args: calls.append(("padded-gather", args[2]))
    comm._reduce_scatterv_padded = lambda *args: calls.append(
        ("padded-scatter", args[2])
    )
    sizes = [1, 3, 2]
    if collective == "all_gatherv":
        comm.all_gatherv(torch.empty(6, 4), torch.empty(1, 4), sizes)
        expected = [("broadcast", size * 4, rank) for rank, size in enumerate(sizes)]
        padded = "padded-gather"
    else:
        comm.reduce_scatterv(torch.empty(1, 4), torch.empty(6, 4), sizes)
        expected = [("reduce", size * 4, rank) for rank, size in enumerate(sizes)]
        padded = "padded-scatter"
    assert calls == (
        [(padded, sizes)] if backend == "musa" else ["start", *expected, "end"]
    )


@pytest.mark.parametrize("backend", ["cuda", "rocm", "musa"])
@pytest.mark.parametrize("sizes", [None, [1, 2, 0]])
def test_batched_gather_keeps_native_grouping_off_musa(backend, sizes):
    events = []
    platform = SimpleNamespace(device_type="musa" if backend == "musa" else "cuda")
    namespace = {
        "torch": torch,
        "current_platform": platform,
        "should_nccl_symm_mem_ag_rs": lambda: False,
    }
    module = load_applied_contract(
        "vllm/distributed/device_communicators/cuda_communicator.py",
        ["_row_bytes", "_pack_rows", "_unpack_rows", "_all_gather_packed"],
        namespace,
    )
    module = load_applied_methods(
        "vllm/distributed/device_communicators/cuda_communicator.py",
        "CudaCommunicator",
        ["all_gatherv"],
        namespace,
    )

    def gather(output, input_, sizes=None):
        events.append(("gather", input_.dtype, tuple(input_.shape)))
        output.zero_()

    comm = module.CudaCommunicator()
    comm.world_size = 3
    comm.rank_in_group = 0
    comm._can_use_aiter_ag_rs = lambda sizes: False
    comm.pynccl_comm = SimpleNamespace(
        disabled=False,
        all_gather=gather,
        all_gatherv=gather,
        group_start=lambda: events.append("start"),
        group_end=lambda: events.append("end"),
    )
    a, b = torch.ones(1, 4), torch.ones(1, 2, dtype=torch.int32)
    result = comm.all_gatherv([a, b], sizes=sizes)
    assert [value.dtype for value in result] == [a.dtype, b.dtype]
    assert [tuple(value.shape) for value in result] == [(3, 4), (3, 2)]
    if backend == "musa":
        assert events == [("gather", torch.uint8, (1, 24))]
    else:
        assert events == [
            "start",
            ("gather", a.dtype, (1, 4)),
            ("gather", b.dtype, (1, 2)),
            "end",
        ]
