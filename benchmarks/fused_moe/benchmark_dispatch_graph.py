#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402, I001

"""Validate native FP8/BF16 MoE dispatcher paths under CUDAGraph replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# isort: off
import torchada  # noqa: F401  # must patch before torch ecosystem imports
import torch
# isort: on

KERNEL_TACTICS_DIR = Path(__file__).resolve().parents[1] / "kernel_tactics"
if str(KERNEL_TACTICS_DIR) not in sys.path:
    sys.path.insert(0, str(KERNEL_TACTICS_DIR))

from _benchmark_utils import source_identity, verify_lease_device_fence

from benchmark_dispatch_crossover import (
    backend_kwargs,
    block_scales,
    comparison_metrics,
    package_versions,
    quantized_weight,
    repo_provenance,
    routes,
    synchronize,
)

from vllm_musa.model_executor.layers.fused_moe import fused_moe
from vllm_musa.model_executor.layers.fused_moe.dispatch_policy import (
    MusaFusedMoeBackend,
)
from vllm_musa.tuning import prime_musa_kernel_hardware


def _make_bf16_weights(
    *, experts: int, intermediate_size: int, hidden_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic BF16 weights with the same layout as FP8 cases."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    w1 = torch.randn(
        (experts, 2 * intermediate_size, hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(device="musa", dtype=torch.bfloat16)
    w2 = torch.randn(
        (experts, hidden_size, intermediate_size),
        generator=generator,
        dtype=torch.float32,
    ).to(device="musa", dtype=torch.bfloat16)
    return w1, w2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--folded-shared-expert", action="store_true")
    parser.add_argument("--weight-dtype", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--expected-physical-device", type=int, required=True)
    parser.add_argument("--expected-device-uuid", required=True)
    parser.add_argument("--expected-multiprocessor-count", type=int, required=True)
    parser.add_argument(
        "--graph-bucket-num-reqs",
        type=int,
        help=(
            "Inject a validated FULL graph bucket for production-policy replay; "
            "omit to exercise the fail-closed untracked-capture path."
        ),
    )
    parser.add_argument(
        "--requested",
        choices=("auto", "gemv"),
        default="auto",
        help="Use auto to validate the calibrated production policy.",
    )
    parser.add_argument(
        "--expected-backend",
        choices=("gemv", "upstream"),
        default="gemv",
        help="Backend identity expected during graph capture.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.folded_shared_expert and (args.experts < 2 or args.top_k < 2):
        raise ValueError("folded shared expert requires experts >= 2 and top-k >= 2")
    if args.replays <= 0:
        raise ValueError("replays must be positive")
    if args.weight_dtype == "bf16" and args.experts == 257:
        if not args.folded_shared_expert or args.top_k != 9:
            raise ValueError(
                "the E=257 Qwen3.5 BF16 contract requires "
                "--folded-shared-expert and top-k=9"
            )
        if args.graph_bucket_num_reqs not in (1, 2, 4, 8):
            raise ValueError(
                "the MP48 E=257 BF16 graph gate requires an exact "
                "graph-bucket-num-reqs in {1,2,4,8}"
            )
        if args.graph_bucket_num_reqs != args.tokens:
            raise ValueError("FULL decode graph buckets require num_reqs == num_tokens")
    lease_device_fence = verify_lease_device_fence(
        expected_physical_device=args.expected_physical_device,
        expected_device_uuid=args.expected_device_uuid,
    )
    capability = tuple(int(value) for value in torch.musa.get_device_capability())
    device_name = torch.musa.get_device_name(0)
    multiprocessor_count = int(
        torch.musa.get_device_properties(0).multi_processor_count
    )
    multiprocessor_count_passed = (
        multiprocessor_count == args.expected_multiprocessor_count
    )
    if not multiprocessor_count_passed:
        raise RuntimeError(
            "graph validation MP mismatch: "
            f"expected={args.expected_multiprocessor_count}, "
            f"actual={multiprocessor_count}"
        )
    if capability != (3, 1):
        raise RuntimeError(
            "this graph validation requires the supported MUSA architecture, "
            f"got capability={capability}"
        )
    primed_hardware = prime_musa_kernel_hardware(0)
    if (
        primed_hardware.device_capability != capability
        or primed_hardware.multiprocessor_count != multiprocessor_count
    ):
        raise RuntimeError(
            "worker-style MUSA hardware priming disagrees with the runtime "
            f"device: primed={primed_hardware!r}, capability={capability}, "
            f"multiprocessor_count={multiprocessor_count}"
        )
    if args.weight_dtype == "fp8":
        w1 = quantized_weight(
            (args.experts, 2 * args.intermediate_size, args.hidden_size), args.seed
        )
        w2 = quantized_weight(
            (args.experts, args.hidden_size, args.intermediate_size), args.seed + 1
        )
        w1_scale = block_scales(w1, 128, args.seed + 2)
        w2_scale = block_scales(w2, 128, args.seed + 3)
        block_size: int | None = 128
    else:
        w1, w2 = _make_bf16_weights(
            experts=args.experts,
            intermediate_size=args.intermediate_size,
            hidden_size=args.hidden_size,
            seed=args.seed,
        )
        w1_scale = None
        w2_scale = None
        block_size = None
    generator = torch.Generator(device="musa")
    generator.manual_seed(args.seed + 4)
    hidden_states = torch.randn(
        (args.tokens, args.hidden_size),
        device="musa",
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids, topk_weights = routes(
        args.tokens,
        args.experts,
        args.top_k,
        "balanced",
        args.seed + 5,
        args.folded_shared_expert,
    )
    # The native BF16 GEMV contract is intentionally stricter than the FP8
    # path: route indices are int32. Keep the graph inputs in the exact
    # production dtype instead of letting torch.topk's int64 result force the
    # dispatcher to upstream.
    topk_ids = topk_ids.to(dtype=torch.int32)
    kwargs = backend_kwargs(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_size=block_size,
    )
    # The production dispatcher needs the global expert count to distinguish
    # Qwen3.5 folded (E=257) from ordinary MoE (E=256). The lower-level helper
    # intentionally defaults this metadata to None for generic crossover
    # experiments, so make it explicit in the graph harness.
    kwargs["global_num_experts"] = args.experts
    kwargs["apply_router_weight_on_input"] = False

    old_backend = fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND
    # Production callers enter through the upstream module slot patched by
    # vllm-musa at import time. Keep that slot intact and invoke it below;
    # replacing it with a counter would bypass the dispatcher we are trying to
    # validate.
    production_dispatch = fused_moe._upstream_fused_moe.fused_experts_impl
    production_dispatch_installed = (
        production_dispatch is fused_moe._musa_fused_experts_impl_dispatch
    )
    if not production_dispatch_installed:
        raise RuntimeError(
            "vllm-musa fused-MoE production dispatcher is not installed at "
            "vllm.model_executor.layers.fused_moe.fused_moe.fused_experts_impl"
        )
    original_gemv = fused_moe.fused_experts_impl
    original_upstream = fused_moe._upstream_fused_moe._musa_original_fused_experts_impl
    original_graph_bucket_query = fused_moe.query_musa_forward_graph_bucket
    original_shape_builder = fused_moe._musa_fused_moe_shape
    original_qwen35_gate = fused_moe._can_use_qwen35_bf16_moe_decode_gemv
    original_native_bf16_gate = fused_moe._can_use_musa_native_bf16_moe_gemv
    original_device_fingerprint = fused_moe._musa_device_fingerprint
    original_thresholds_for_shape = fused_moe.thresholds_for_shape
    calls = {"gemv": 0, "upstream": 0}
    selection_trace: list[dict[str, Any]] = []
    shape_trace: list[dict[str, Any]] = []
    capability_trace: list[dict[str, Any]] = []
    device_fingerprint_trace: list[dict[str, Any]] = []
    backend_call_trace: list[dict[str, Any]] = []
    policy_trace: list[dict[str, Any]] = []
    captured_gemv_launch_kwargs: dict[str, Any] = {}

    def counted_gemv(*inner_args, **inner_kwargs):
        calls["gemv"] += 1
        captured_gemv_launch_kwargs.clear()
        captured_gemv_launch_kwargs.update(
            {
                key: inner_kwargs[key]
                for key in (
                    "inplace",
                    "_allow_deepgemm_prefill",
                    "_gemv_block",
                    "_gemv_blocks",
                )
                if key in inner_kwargs
            }
        )
        backend_call_trace.append(
            {
                "backend": "gemv",
                "launch_kwargs": dict(captured_gemv_launch_kwargs),
            }
        )
        return original_gemv(*inner_args, **inner_kwargs)

    def counted_upstream(*inner_args, **inner_kwargs):
        calls["upstream"] += 1
        backend_call_trace.append({"backend": "upstream"})
        return original_upstream(*inner_args, **inner_kwargs)

    original_select_backend = fused_moe.select_fused_moe_backend

    def traced_select_backend(*inner_args, **inner_kwargs):
        selected = original_select_backend(*inner_args, **inner_kwargs)
        shape = inner_kwargs.get("shape")
        selection_trace.append(
            {
                "selected": selected.value,
                "shape": None if shape is None else repr(shape),
                "num_tokens": inner_kwargs.get("num_tokens"),
                "stream_is_capturing": inner_kwargs.get("stream_is_capturing"),
            }
        )
        return selected

    def traced_qwen35_gate(*inner_args, **inner_kwargs):
        supported = original_qwen35_gate(*inner_args, **inner_kwargs)
        capability_trace.append(
            {
                "gate": "qwen35_bf16_decode_gemv",
                "supported": bool(supported),
                "global_num_experts": inner_kwargs.get("global_num_experts"),
                "topk_ids_dtype": str(inner_kwargs["topk_ids"].dtype),
                "topk_weights_dtype": str(inner_kwargs["topk_weights"].dtype),
            }
        )
        return supported

    def traced_native_bf16_gate(*inner_args, **inner_kwargs):
        supported = original_native_bf16_gate(*inner_args, **inner_kwargs)
        capability_trace.append(
            {
                "gate": "native_bf16_moe_gemv",
                "supported": bool(supported),
                "global_num_experts": inner_kwargs.get("global_num_experts"),
                "apply_router_weight_on_input": inner_kwargs.get(
                    "apply_router_weight_on_input"
                ),
                "topk_ids_dtype": str(inner_kwargs["topk_ids"].dtype),
                "topk_weights_dtype": str(inner_kwargs["topk_weights"].dtype),
            }
        )
        return supported

    def traced_device_fingerprint(device_index: int):
        device_capability, traced_mp_count = original_device_fingerprint(device_index)
        device_fingerprint_trace.append(
            {
                "device_index": device_index,
                "device_capability": list(device_capability),
                "multiprocessor_count": traced_mp_count,
            }
        )
        return device_capability, traced_mp_count

    def traced_thresholds_for_shape(shape):
        thresholds = original_thresholds_for_shape(shape)
        policy_trace.append(
            {
                "source": thresholds.source,
                "gemv_max_tokens": thresholds.gemv_max_tokens,
                "grouped_gemm_min_tokens": thresholds.grouped_gemm_min_tokens,
            }
        )
        return thresholds

    def traced_shape_builder(*inner_args, **inner_kwargs):
        shape = original_shape_builder(*inner_args, **inner_kwargs)
        shape_trace.append({"shape": repr(shape)})
        return shape

    try:
        if args.graph_bucket_num_reqs is not None:
            if args.graph_bucket_num_reqs <= 0:
                raise ValueError("graph-bucket-num-reqs must be positive")
            from vllm_musa.tuning import MusaForwardGraphBucket

            graph_bucket = MusaForwardGraphBucket(
                num_tokens=args.tokens,
                num_reqs=args.graph_bucket_num_reqs,
                uniform=True,
                runtime_mode="FULL",
                has_lora=False,
                num_active_loras=0,
                present=True,
            )
            fused_moe.query_musa_forward_graph_bucket = lambda: graph_bucket
        fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND = MusaFusedMoeBackend(
            args.requested
        )
        fused_moe.fused_experts_impl = counted_gemv
        fused_moe.select_fused_moe_backend = traced_select_backend
        fused_moe._musa_fused_moe_shape = traced_shape_builder
        fused_moe._can_use_qwen35_bf16_moe_decode_gemv = traced_qwen35_gate
        fused_moe._can_use_musa_native_bf16_moe_gemv = traced_native_bf16_gate
        fused_moe._musa_device_fingerprint = traced_device_fingerprint
        fused_moe.thresholds_for_shape = traced_thresholds_for_shape
        fused_moe._upstream_fused_moe._musa_original_fused_experts_impl = (
            counted_upstream
        )
        if args.expected_backend == "gemv":
            original_gemv(
                **kwargs,
                inplace=False,
                _allow_deepgemm_prefill=False,
            )
        else:
            original_upstream(**kwargs)
        synchronize()

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph):
                captured_output = production_dispatch(**kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        synchronize()
        capture_calls = dict(calls)

        comparisons = []
        route_modes = ("balanced", "unique_random", "hot")
        for replay_index in range(args.replays):
            replacement = torch.randn(
                hidden_states.shape,
                device="musa",
                dtype=hidden_states.dtype,
                generator=generator,
            )
            hidden_states.copy_(replacement)
            route_mode = route_modes[replay_index % len(route_modes)]
            replacement_ids, replacement_weights = routes(
                args.tokens,
                args.experts,
                args.top_k,
                route_mode,
                args.seed + 100 + replay_index,
                args.folded_shared_expert,
            )
            replacement_ids = replacement_ids.to(dtype=torch.int32)
            topk_ids.copy_(replacement_ids)
            topk_weights.copy_(replacement_weights)
            if args.expected_backend == "gemv":
                expected_backend_output = (
                    original_gemv(
                        **kwargs,
                        **captured_gemv_launch_kwargs,
                    )
                    .detach()
                    .clone()
                )
            else:
                expected_backend_output = original_upstream(**kwargs).detach().clone()
            upstream_reference = original_upstream(**kwargs).detach().clone()
            graph.replay()
            synchronize()
            difference = (
                captured_output.float() - expected_backend_output.float()
            ).abs()
            reference_metrics = comparison_metrics(captured_output, upstream_reference)
            reference_passed = bool(
                args.weight_dtype != "bf16"
                or (
                    reference_metrics["relative_l2_error"] <= 0.01
                    and reference_metrics["cosine_similarity"] >= 0.9999
                    and reference_metrics["max_row_relative_l2"] <= 0.02
                    and reference_metrics["min_row_cosine"] >= 0.9998
                    and reference_metrics["max_abs_diff_over_reference_absmax"] <= 0.05
                )
            )
            comparisons.append(
                {
                    "replay": replay_index,
                    "route_mode": route_mode,
                    "bitwise_equal": bool(
                        torch.equal(captured_output, expected_backend_output)
                    ),
                    "max_abs_diff": float(difference.max().item()),
                    "upstream_reference_passed": reference_passed,
                    "upstream_reference_metrics": reference_metrics,
                }
            )
    finally:
        fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND = old_backend
        fused_moe.fused_experts_impl = original_gemv
        fused_moe._upstream_fused_moe._musa_original_fused_experts_impl = (
            original_upstream
        )
        fused_moe.query_musa_forward_graph_bucket = original_graph_bucket_query
        fused_moe.select_fused_moe_backend = original_select_backend
        fused_moe._musa_fused_moe_shape = original_shape_builder
        fused_moe._can_use_qwen35_bf16_moe_decode_gemv = original_qwen35_gate
        fused_moe._can_use_musa_native_bf16_moe_gemv = original_native_bf16_gate
        fused_moe._musa_device_fingerprint = original_device_fingerprint
        fused_moe.thresholds_for_shape = original_thresholds_for_shape

    expected_capture_calls = {
        "gemv": int(args.expected_backend == "gemv"),
        "upstream": int(args.expected_backend == "upstream"),
    }
    expected_gemv_stage_blocks = None
    if (
        args.expected_backend == "gemv"
        and args.weight_dtype == "bf16"
        and args.experts == 257
        and multiprocessor_count == 48
    ):
        expected_gemv_stage_blocks = ((32, 4), (32, 4))
    gemv_stage_tactic_passed = bool(
        expected_gemv_stage_blocks is None
        or captured_gemv_launch_kwargs.get("_gemv_blocks") == expected_gemv_stage_blocks
    )
    selection_observed = bool(
        args.expected_backend == "upstream"
        or (
            len(selection_trace) == 1
            and selection_trace[0]["selected"] == args.expected_backend
        )
    )
    capability_passed = bool(
        args.weight_dtype != "bf16"
        or args.expected_backend != "gemv"
        or (
            {item["gate"] for item in capability_trace}
            == {"qwen35_bf16_decode_gemv", "native_bf16_moe_gemv"}
            and all(item["supported"] for item in capability_trace)
        )
    )
    passed = bool(
        capture_calls == expected_capture_calls
        and selection_observed
        and capability_passed
        and multiprocessor_count_passed
        and gemv_stage_tactic_passed
        and all(
            item["bitwise_equal"] and item["upstream_reference_passed"]
            for item in comparisons
        )
    )
    payload = {
        "passed": passed,
        "requested": args.requested,
        "expected_backend": args.expected_backend,
        "weight_dtype": args.weight_dtype,
        "shape": {
            "experts": args.experts,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "top_k": args.top_k,
            "tokens": args.tokens,
            "block_size": block_size,
        },
        "folded_shared_expert": args.folded_shared_expert,
        "seed": args.seed,
        "replays": args.replays,
        "graph_bucket_num_reqs": args.graph_bucket_num_reqs,
        "device": {
            "name": device_name,
            "capability": capability,
            "multiprocessor_count": multiprocessor_count,
        },
        "primed_hardware": {
            "device_capability": primed_hardware.device_capability,
            "multiprocessor_count": primed_hardware.multiprocessor_count,
            "cache_key": primed_hardware.cache_key,
        },
        "lease_device_fence": lease_device_fence,
        "packages": package_versions(),
        "repo": repo_provenance(),
        "source": source_identity(Path(__file__)),
        "expected_capture_calls": expected_capture_calls,
        "capture_calls": capture_calls,
        "expected_gemv_stage_blocks": expected_gemv_stage_blocks,
        "gemv_stage_tactic_passed": gemv_stage_tactic_passed,
        "selection_observed": selection_observed,
        "capability_passed": capability_passed,
        "expected_multiprocessor_count": args.expected_multiprocessor_count,
        "multiprocessor_count_passed": multiprocessor_count_passed,
        "production_dispatch": {
            "installed": production_dispatch_installed,
            "module": production_dispatch.__module__,
            "name": production_dispatch.__name__,
        },
        "capability_trace": capability_trace,
        "device_fingerprint_trace": device_fingerprint_trace,
        "backend_call_trace": backend_call_trace,
        "policy_trace": policy_trace,
        "selection_trace": selection_trace,
        "shape_trace": shape_trace,
        "comparisons": comparisons,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
