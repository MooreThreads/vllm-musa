# SPDX-License-Identifier: Apache-2.0
"""Validate the MATE adapter's multi-rank numerics, graph replay and cleanup.

Run with torchrun --standalone --nproc_per_node=N. This is a functional test,
not a throughput benchmark. The forward context and EP service refer to the
real test process group; no kernel implementation is mocked. Constructor and
oracle selection remain the responsibility of the separate server regression.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# isort: off
import torchada  # noqa: F401
import torch
import torch.distributed as dist

# isort: on

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.forward_context import DPMetadata, ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)

from vllm_musa.model_executor.layers.fused_moe import mate_mega_moe as adapter


def block_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = weight.shape
    blocks = weight.reshape(n // 128, 128, k // 128, 128)
    scale = blocks.float().abs().amax(dim=(1, 3)).clamp_min(1e-4) / 448.0
    quant = (blocks.float() / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return quant.reshape(n, k).contiguous(), scale.contiguous()


def weights(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    tables: list[list[torch.Tensor]] = [[], [], [], []]
    generator = torch.Generator(device="musa")
    for expert in range(args.experts):
        generator.manual_seed(8231 + expert)
        projections = []
        for shape in (
            (2 * args.intermediate, args.hidden),
            (args.hidden, args.intermediate),
        ):
            raw = torch.randn(
                shape, dtype=torch.bfloat16, device="musa", generator=generator
            )
            projections.extend(block_quantize(raw * 0.03))
        for table, value in zip(tables, projections):
            table.append(value)
    return tuple(torch.stack(table) for table in tables)


def dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)


def inputs(
    args: argparse.Namespace, rows: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="musa").manual_seed(seed)
    x = (
        torch.randn(
            rows, args.hidden, dtype=torch.bfloat16, device="musa", generator=generator
        )
        * 0.25
    )
    scores = torch.rand(rows, args.experts, device="musa", generator=generator)
    ids = scores.topk(args.topk, dim=1).indices.to(torch.int32)
    routing = torch.rand(rows, args.topk, device="musa", generator=generator)
    routing = routing / routing.sum(dim=1, keepdim=True)
    return x, ids, routing


def reference(
    x: torch.Tensor,
    ids: torch.Tensor,
    routing: torch.Tensor,
    logical: tuple[torch.Tensor, ...],
    intermediate: int,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.empty_like(x)
    q, scale = per_token_group_quant_fp8(
        x, 128, column_major_scales=False, use_ue8m0=False
    )
    x_dequant = q.float() * scale.repeat_interleave(128, 1)
    w1, s1, w2, s2 = logical
    result = torch.zeros_like(x, dtype=torch.float32)
    # This reference uses source tokens and all logical experts, not the
    # interleaved weights or symmetric dispatch output under test.
    for expert in sorted(set(ids.cpu().reshape(-1).tolist())):
        positions = (ids == expert).nonzero()
        # MUSA index_add_ requires contiguous index vectors.
        rows = positions[:, 0].contiguous()
        slots = positions[:, 1].contiguous()
        hidden = (x_dequant[rows] @ dequantize(w1[expert], s1[expert]).t()).to(
            torch.bfloat16
        )
        gate = hidden[:, :intermediate].float()
        activated = (gate * torch.sigmoid(gate)).to(torch.bfloat16)
        activated = activated * hidden[:, intermediate:]
        aq, af = per_token_group_quant_fp8(
            activated, 128, column_major_scales=False, use_ue8m0=False
        )
        down = (
            (aq.float() * af.repeat_interleave(128, 1))
            @ dequantize(w2[expert], s2[expert]).t()
        ).to(torch.bfloat16)
        result.index_add_(0, rows, down.float() * routing[rows, slots, None])
    return result.to(torch.bfloat16)


def check_output(output: torch.Tensor, expected: torch.Tensor) -> dict:
    if output.numel() == 0:
        return {"empty": True, "passed": True}
    actual, ref = output.float(), expected.float()
    finite = bool(torch.isfinite(actual).all().item())
    non_poison = bool(actual.abs().max().item() > 0 and actual.std().item() > 0)
    relative_l2 = float(
        (actual - ref).norm().item() / ref.norm().clamp_min(1e-12).item()
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual.flatten(), ref.flatten(), dim=0
        ).item()
    )
    return {
        "finite": finite,
        "non_poison": non_poison,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "max_abs": float((actual - ref).abs().max().item()),
        "passed": finite and non_poison and relative_l2 <= 0.06 and cosine >= 0.998,
    }


@contextmanager
def forward_context(counts: list[int]):
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        dp_metadata=DPMetadata(torch.tensor(counts, dtype=torch.int32)),
    )
    with override_forward_context(context):
        yield


def make_experts(
    args: argparse.Namespace, group, rank: int, world: int, logical
) -> tuple[adapter.MateMegaMoEExperts, torch.nn.Module]:
    local = args.experts // world
    start, end = rank * local, (rank + 1) * local
    layer = torch.nn.Module()
    names = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")
    for name, tensor in zip(names, logical):
        layer.register_parameter(
            name, torch.nn.Parameter(tensor[start:end].clone(), requires_grad=False)
        )
    layer.expert_map_manager = SimpleNamespace(placement_strategy="linear")
    experts = object.__new__(adapter.MateMegaMoEExperts)
    experts.moe_config = SimpleNamespace(
        num_experts=args.experts,
        max_num_tokens=256,
        experts_per_token=args.topk,
        hidden_dim=args.hidden,
        intermediate_size_per_partition=args.intermediate,
    )
    experts._activation_clamp = None
    experts._buffer = None
    with patch.object(
        adapter, "get_ep_group", return_value=SimpleNamespace(device_group=group)
    ):
        experts.process_weights_after_loading(layer)
    return experts, layer


def invoke(experts, x, ids, routing, counts, output) -> None:
    with forward_context(counts):
        experts.apply(
            output=output,
            hidden_states=x,
            w1=None,
            w2=None,
            topk_weights=routing,
            topk_ids=ids,
            activation=MoEActivation.SILU,
            global_num_experts=experts.moe_config.num_experts,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )


def collective_result(row: dict, host_group, rank: int, world: int) -> list:
    results = [None] * world
    dist.all_gather_object(results, row, group=host_group)
    if rank == 0:
        print(json.dumps({"ranks": results}), flush=True)
    if not all(item["passed"] for item in results):
        raise AssertionError(f"Multi-rank check failed: {results}")
    return results


def cleanup_group(experts, group) -> None:
    # Run the real communicator teardown hook without unrelated transports.
    comm = object.__new__(CudaCommunicator)
    comm.device_group = group
    for field in (
        "pynccl_comm",
        "ca_comm",
        "aiter_ar_comm",
        "fi_ar_comm",
        "fi_pcie_ipc_ar_comm",
        "all2all_manager",
    ):
        setattr(comm, field, None)
    comm.destroy()
    assert not any(key.group is group for key in adapter._SYMM_BUFFERS)
    assert experts._buffer.buffer is None
    comm.destroy()
    dist.barrier(group=group)
    dist.destroy_process_group(group)


def graph_checks(args, experts, group, host_group, rank: int, world: int) -> list:
    results = []
    mixed = [(0, 1, 128, 129)[index % 4] for index in range(world)]
    for counts in ([1] * world, [128] * world, [129] * world, mixed):
        x, ids, routing = inputs(args, counts[rank], 711 + rank)
        captured = torch.empty_like(x)
        invoke(experts, x, ids, routing, counts, captured)
        torch.musa.synchronize()
        dist.barrier(group=group)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph):
                invoke(experts, x, ids, routing, counts, captured)
        torch.cuda.current_stream().wait_stream(stream)
        torch.musa.synchronize()
        for replay in range(args.replays):
            replacements = inputs(args, counts[rank], 913 + replay * 97 + rank)
            for tensor, replacement in zip((x, ids, routing), replacements):
                tensor.copy_(replacement)
            graph.replay()
            torch.musa.synchronize()
            observed = captured.clone()
            eager = torch.empty_like(x)
            invoke(experts, x, ids, routing, counts, eager)
            torch.musa.synchronize()
            row = {
                "rank": rank,
                "counts": counts,
                "kind": "graph",
                "replay": replay,
                "passed": bool(torch.equal(observed, eager)),
            }
            results.append(collective_result(row, host_group, rank, world))
        del graph, stream, captured
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("/work/results/distributed.json")
    )
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.musa.set_device(local_rank)
    if args.experts % world:
        raise ValueError("experts must divide the rank count")
    dist.init_process_group("mccl", device_id=torch.device("musa", local_rank))
    host_group = dist.new_group(backend="gloo")
    logical = weights(args)
    all_results = []
    patterns = [[value] * world for value in (0, 1, 127, 128, 129, 256)]
    patterns += [[(0, 1, 128, 129)[index % 4] for index in range(world)]]
    patterns += [[256] + [0] * (world - 1), [1] * world]
    for cycle in range(args.cycles):
        group = dist.new_group(backend="mccl")
        experts, layer = make_experts(args, group, rank, world, logical)
        for case, counts in enumerate(patterns):
            x, ids, routing = inputs(
                args, counts[rank], 1733 + 1000 * cycle + 31 * rank + case
            )
            out = torch.empty_like(x)
            invoke(experts, x, ids, routing, counts, out)
            torch.musa.synchronize()
            expected = reference(x, ids, routing, logical, args.intermediate)
            row = check_output(out, expected)
            row.update(rank=rank, cycle=cycle, counts=counts, kind="reference")
            all_results.append(collective_result(row, host_group, rank, world))
        if cycle == 0:
            all_results.extend(
                graph_checks(args, experts, group, host_group, rank, world)
            )
        cleanup_group(experts, group)
        del experts, layer
        gc.collect()
        torch.musa.empty_cache()
    dist.destroy_process_group(host_group)
    dist.destroy_process_group()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "passed": True,
            "world_size": world,
            "shape": {
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "experts": args.experts,
                "topk": args.topk,
            },
            "cycles": args.cycles,
            "replays": args.replays,
            "results": all_results,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print("MATE_ADAPTER_DISTRIBUTED_PASS", flush=True)


if __name__ == "__main__":
    main()
