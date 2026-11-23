#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate the native FP8 MoE GEMV dispatcher under CUDAGraph replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# isort: off
import torchada  # noqa: F401  # must patch before torch ecosystem imports
import torch
# isort: on

from benchmark_dispatch_crossover import (
    backend_kwargs,
    block_scales,
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
    capability = tuple(int(value) for value in torch.musa.get_device_capability())
    device_name = torch.musa.get_device_name(0)
    multiprocessor_count = int(
        torch.musa.get_device_properties(0).multi_processor_count
    )
    if capability != (3, 1) or "S5000" not in device_name.upper():
        raise RuntimeError(
            "this graph validation is valid only on MTT S5000/MP31, got "
            f"device={device_name!r} capability={capability}"
        )
    w1 = quantized_weight(
        (args.experts, 2 * args.intermediate_size, args.hidden_size), args.seed
    )
    w2 = quantized_weight(
        (args.experts, args.hidden_size, args.intermediate_size), args.seed + 1
    )
    w1_scale = block_scales(w1, 128, args.seed + 2)
    w2_scale = block_scales(w2, 128, args.seed + 3)
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
    kwargs = backend_kwargs(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_size=128,
    )

    old_backend = fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND
    original_gemv = fused_moe.fused_experts_impl
    original_upstream = fused_moe._upstream_fused_moe._musa_original_fused_experts_impl
    calls = {"gemv": 0, "upstream": 0}

    def counted_gemv(*inner_args, **inner_kwargs):
        calls["gemv"] += 1
        return original_gemv(*inner_args, **inner_kwargs)

    def counted_upstream(*inner_args, **inner_kwargs):
        calls["upstream"] += 1
        return original_upstream(*inner_args, **inner_kwargs)

    try:
        fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND = MusaFusedMoeBackend(
            args.requested
        )
        fused_moe.fused_experts_impl = counted_gemv
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
                captured_output = fused_moe._musa_fused_experts_impl_dispatch(**kwargs)
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
            topk_ids.copy_(replacement_ids)
            topk_weights.copy_(replacement_weights)
            if args.expected_backend == "gemv":
                expected = (
                    original_gemv(
                        **kwargs,
                        inplace=False,
                        _allow_deepgemm_prefill=False,
                    )
                    .detach()
                    .clone()
                )
            else:
                expected = original_upstream(**kwargs).detach().clone()
            graph.replay()
            synchronize()
            difference = (captured_output.float() - expected.float()).abs()
            comparisons.append(
                {
                    "replay": replay_index,
                    "route_mode": route_mode,
                    "bitwise_equal": bool(torch.equal(captured_output, expected)),
                    "max_abs_diff": float(difference.max().item()),
                }
            )
    finally:
        fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND = old_backend
        fused_moe.fused_experts_impl = original_gemv
        fused_moe._upstream_fused_moe._musa_original_fused_experts_impl = (
            original_upstream
        )

    expected_capture_calls = {
        "gemv": int(args.expected_backend == "gemv"),
        "upstream": int(args.expected_backend == "upstream"),
    }
    passed = bool(
        capture_calls == expected_capture_calls
        and all(item["bitwise_equal"] for item in comparisons)
    )
    payload = {
        "passed": passed,
        "requested": args.requested,
        "expected_backend": args.expected_backend,
        "shape": {
            "experts": args.experts,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "top_k": args.top_k,
            "tokens": args.tokens,
        },
        "folded_shared_expert": args.folded_shared_expert,
        "seed": args.seed,
        "replays": args.replays,
        "device": {
            "name": device_name,
            "capability": capability,
            "multiprocessor_count": multiprocessor_count,
        },
        "packages": package_versions(),
        "repo": repo_provenance(),
        "expected_capture_calls": expected_capture_calls,
        "capture_calls": capture_calls,
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
