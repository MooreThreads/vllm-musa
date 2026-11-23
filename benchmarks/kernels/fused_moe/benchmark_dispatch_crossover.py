#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cold-cache crossover sweep for MUSA FP8 fused-MoE backends.

All dimensions are the actual per-rank kernel dimensions. A coarse sweep only
produces a candidate interval; rerun densely around both crossings and confirm
the result with serving A/B before adding a policy entry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# isort: off
import torchada  # noqa: F401  # must patch before torch ecosystem imports
import torch
import torch.nn.functional as F
# isort: on

from mate.testing.utils import bench_gpu_time_with_musa_event

from vllm_musa.model_executor.layers.fused_moe import fused_moe
from vllm_musa.model_executor.layers.fused_moe.dispatch_policy import (
    MusaFusedMoeBackend,
    MusaFusedMoeShape,
)


@dataclass(frozen=True)
class Shape:
    name: str
    experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    block_size: int


@dataclass
class Result:
    shape: dict[str, Any]
    tokens: int
    backend: str
    samples_ms: list[float]
    median_ms: float | None
    p20_ms: float | None
    p95_ms: float | None
    iqr_ms: float | None
    min_ms: float | None
    max_ms: float | None
    max_abs_diff: float | None
    mean_abs_diff: float | None
    relative_l2_error: float | None
    cosine_similarity: float | None
    max_row_relative_l2: float | None
    min_row_cosine: float | None
    max_abs_diff_over_reference_absmax: float | None
    output_absmax: float | None
    output_std: float | None
    finite: bool | None
    non_poison: bool | None
    correctness_pass: bool | None
    correctness_basis: str
    reference_backend: str
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--block-size", type=int, choices=(128,), default=128)
    parser.add_argument(
        "--tokens",
        default=(
            "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32,48,64,"
            "96,128,192,256,384,512,"
            "768,1024,1536,2048,3072,4096,6144,8192,12288,16384"
        ),
    )
    parser.add_argument(
        "--backends",
        default="gemv,grouped_gemm,upstream",
        help=(
            "Comma-separated subset of gemv, grouped_gemm, "
            "deepgemm_prefill, and upstream; upstream is mandatory for "
            "qualification"
        ),
    )
    parser.add_argument(
        "--route-mode",
        choices=("balanced", "unique_random", "hot"),
        default="balanced",
    )
    parser.add_argument(
        "--folded-shared-expert",
        action="store_true",
        help=(
            "Treat the final expert and final routing slot as an always-selected "
            "folded shared expert"
        ),
    )
    parser.add_argument("--sweep-kind", choices=("coarse", "dense"), default="coarse")
    parser.add_argument(
        "--dense-prefix-max",
        type=int,
        default=16,
        help=(
            "For dense sweeps, materialize every token count from 1 through "
            "this bound so a <= GEMV threshold never spans unmeasured holes."
        ),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats-per-round", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--l2-flush-mb", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.08)
    parser.add_argument("--gemv-max-relative-l2", type=float, default=0.06)
    parser.add_argument("--gemv-min-cosine", type=float, default=0.998)
    parser.add_argument("--gemv-max-row-relative-l2", type=float, default=0.08)
    parser.add_argument("--gemv-min-row-cosine", type=float, default=0.995)
    parser.add_argument("--gemv-max-normalized-abs-diff", type=float, default=0.10)
    parser.add_argument("--gemv-oracle-max-relative-l2", type=float, default=0.01)
    parser.add_argument("--gemv-oracle-min-cosine", type=float, default=0.9999)
    parser.add_argument("--gemv-oracle-max-row-relative-l2", type=float, default=0.02)
    parser.add_argument("--gemv-oracle-min-row-cosine", type=float, default=0.9998)
    parser.add_argument(
        "--gemv-oracle-max-normalized-abs-diff", type=float, default=0.05
    )
    parser.add_argument("--oracle-probe-tokens", type=int, default=4)
    parser.add_argument("--grouped-max-relative-l2", type=float, default=0.01)
    parser.add_argument("--grouped-min-cosine", type=float, default=0.9999)
    parser.add_argument("--grouped-max-row-relative-l2", type=float, default=0.02)
    parser.add_argument("--grouped-min-row-cosine", type=float, default=0.999)
    parser.add_argument("--grouped-max-normalized-abs-diff", type=float, default=0.05)
    parser.add_argument("--regression-margin", type=float, default=0.03)
    parser.add_argument("--max-iqr-ratio", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def synchronize() -> None:
    torch.musa.synchronize()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        ).strip()[:4000]
    except Exception:
        return None


def repo_provenance() -> dict[str, Any]:
    module_path = Path(inspect.getfile(fused_moe)).resolve()
    repo = next(
        (
            parent
            for parent in (module_path.parent, *module_path.parents)
            if (parent / ".git").exists()
        ),
        None,
    )
    if repo is None:
        return {"module_path": str(module_path), "repo": None}
    head = command_output(["git", "-C", str(repo), "rev-parse", "HEAD"])
    status = command_output(["git", "-C", str(repo), "status", "--short"])
    try:
        diff = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--binary"],
            stderr=subprocess.DEVNULL,
        )
        diff_sha256 = hashlib.sha256(diff).hexdigest()
    except Exception:
        diff_sha256 = None
    return {
        "module_path": str(module_path),
        "repo": str(repo),
        "head": head,
        "status": status,
        "dirty_patch_sha256": diff_sha256,
        "custom_ops_module": str(Path(inspect.getfile(fused_moe.musa_ops)).resolve()),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "torch_musa", "torchada", "mate", "vllm", "vllm-musa"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def quantized_weight(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="musa")
    generator.manual_seed(seed)
    output = torch.empty(shape, device="musa", dtype=torch.float8_e4m3fn)
    chunk_experts = max(1, min(shape[0], 8))
    for start in range(0, shape[0], chunk_experts):
        end = min(start + chunk_experts, shape[0])
        source = torch.randn(
            (end - start, *shape[1:]),
            device="musa",
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(8.0)
        output[start:end].copy_(source.to(torch.float8_e4m3fn))
    return output


def block_scale_shape(weight: torch.Tensor, block_size: int) -> tuple[int, int, int]:
    experts, output_size, input_size = weight.shape
    if input_size % block_size != 0 or output_size % block_size != 0:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} is not aligned to block {block_size}"
        )
    return experts, output_size // block_size, input_size // block_size


def block_scales(weight: torch.Tensor, block_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="musa")
    generator.manual_seed(seed)
    scales = torch.rand(
        block_scale_shape(weight, block_size),
        device="musa",
        dtype=torch.float32,
        generator=generator,
    )
    return scales.mul_(0.0045).add_(0.0005).contiguous()


def routes(
    tokens: int,
    experts: int,
    top_k: int,
    mode: str,
    seed: int,
    folded_shared_expert: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="musa")
    generator.manual_seed(seed)
    routed_experts = experts - 1 if folded_shared_expert else experts
    routed_top_k = top_k - 1 if folded_shared_expert else top_k
    if mode == "balanced":
        rows = torch.arange(tokens, device="musa", dtype=torch.int32)[:, None]
        offsets = torch.arange(routed_top_k, device="musa", dtype=torch.int32)[None, :]
        ids = (rows * routed_top_k + offsets).remainder(routed_experts)
    elif mode == "hot":
        ids = torch.arange(routed_top_k, device="musa", dtype=torch.int32)[None, :]
        ids = ids.expand(tokens, routed_top_k).contiguous().remainder(routed_experts)
    else:
        scores = torch.rand(
            (tokens, routed_experts),
            device="musa",
            dtype=torch.float32,
            generator=generator,
        )
        ids = scores.topk(routed_top_k, dim=-1, sorted=False).indices.to(torch.int32)
    weights = torch.rand(
        (tokens, routed_top_k),
        device="musa",
        dtype=torch.float32,
        generator=generator,
    )
    weights.div_(weights.sum(dim=-1, keepdim=True))
    if folded_shared_expert:
        shared_ids = torch.full(
            (tokens, 1), experts - 1, device="musa", dtype=torch.int32
        )
        shared_logits = torch.randn(
            (tokens, 1),
            device="musa",
            dtype=torch.float32,
            generator=generator,
        )
        ids = torch.cat((ids, shared_ids), dim=1)
        weights = torch.cat((weights, shared_logits.sigmoid()), dim=1)
    return ids.contiguous(), weights.contiguous()


def oracle_coverage_routes(
    tokens: int,
    experts: int,
    top_k: int,
    folded_shared_expert: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a tiny deterministic probe covering first, middle, and tail experts."""

    routed_experts = experts - 1 if folded_shared_expert else experts
    routed_top_k = top_k - 1 if folded_shared_expert else top_k
    requested = tokens * routed_top_k
    stride = max(1, routed_experts // max(1, requested))
    ordered_candidates = [0, routed_experts // 2, routed_experts - 1]
    ordered_candidates.extend(
        (index * stride) % routed_experts for index in range(routed_experts)
    )
    ordered_candidates.extend(range(routed_experts))
    candidates = list(dict.fromkeys(ordered_candidates))

    rows: list[list[int]] = []
    cursor = 0
    for _ in range(tokens):
        row: list[int] = []
        while len(row) < routed_top_k:
            candidate = candidates[cursor % len(candidates)]
            cursor += 1
            if candidate not in row:
                row.append(candidate)
        if folded_shared_expert:
            row.append(experts - 1)
        rows.append(row)

    topk_ids = torch.tensor(rows, device="musa", dtype=torch.int32)
    routed_weights = torch.full(
        (tokens, routed_top_k),
        1.0 / routed_top_k,
        device="musa",
        dtype=torch.float32,
    )
    if folded_shared_expert:
        shared_weights = torch.full(
            (tokens, 1), 0.5, device="musa", dtype=torch.float32
        )
        topk_weights = torch.cat((routed_weights, shared_weights), dim=1)
    else:
        topk_weights = routed_weights
    return topk_ids.contiguous(), topk_weights.contiguous()


def backend_kwargs(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_size: int,
) -> dict[str, Any]:
    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "activation": "silu",
        "apply_router_weight_on_input": False,
        "use_fp8_w8a8": True,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "ocp_mx_scheme": None,
        "per_channel_quant": False,
        "global_num_experts": w1.shape[0],
        "expert_map": None,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "w1_zp": None,
        "w2_zp": None,
        "a1_scale": None,
        "a2_scale": None,
        "block_shape": [block_size, block_size],
        "w1_bias": None,
        "w2_bias": None,
    }


def build_backends(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    gemv_gate = {
        key: value
        for key, value in kwargs.items()
        if key not in {"apply_router_weight_on_input", "block_shape"}
    }
    grouped_gate = dict(kwargs)
    deepgemm_prefill_gate = {
        key: value
        for key, value in kwargs.items()
        if key not in {"topk_weights", "global_num_experts", "w1_zp", "w2_zp"}
    }
    capabilities = {
        "gemv": fused_moe._can_use_musa_native_fp8_moe_gemv(**gemv_gate),
        "grouped_gemm": fused_moe._can_use_musa_fp8_moe_grouped_gemm(**grouped_gate),
        "deepgemm_prefill": fused_moe._can_use_moe_deepgemm_prefill(
            **deepgemm_prefill_gate
        ),
        "upstream": True,
    }

    def gemv() -> torch.Tensor:
        return fused_moe.fused_experts_impl(
            **kwargs,
            inplace=False,
            _allow_deepgemm_prefill=False,
        )

    def grouped_gemm() -> torch.Tensor:
        return fused_moe._musa_fp8_moe_grouped_gemm_impl(
            hidden_states=kwargs["hidden_states"],
            w1=kwargs["w1"],
            w2=kwargs["w2"],
            topk_weights=kwargs["topk_weights"],
            topk_ids=kwargs["topk_ids"],
            w1_scale=kwargs["w1_scale"],
            w2_scale=kwargs["w2_scale"],
            expert_map=None,
            inplace=False,
        )

    def deepgemm_prefill() -> torch.Tensor:
        return fused_moe._musa_fp8_moe_deepgemm_prefill_impl(
            hidden_states=kwargs["hidden_states"],
            w1=kwargs["w1"],
            w2=kwargs["w2"],
            topk_weights=kwargs["topk_weights"],
            topk_ids=kwargs["topk_ids"],
            w1_scale=kwargs["w1_scale"],
            w2_scale=kwargs["w2_scale"],
            inplace=False,
        )

    def upstream() -> torch.Tensor:
        return fused_moe._upstream_fused_moe._musa_original_fused_experts_impl(**kwargs)

    return {
        "gemv": gemv,
        "grouped_gemm": grouped_gemm,
        "deepgemm_prefill": deepgemm_prefill,
        "upstream": upstream,
    }, capabilities


def dispatcher_smoke(
    kwargs: dict[str, Any],
    requested_backends: list[str],
    direct_outputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    dispatchable_backends = [
        backend
        for backend in requested_backends
        if backend in {"gemv", "grouped_gemm", "upstream"}
    ]
    old_backend = fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND
    original_gemv = fused_moe.fused_experts_impl
    original_grouped = fused_moe._maybe_musa_fp8_moe_grouped_gemm
    original_upstream = fused_moe._upstream_fused_moe._musa_original_fused_experts_impl
    outputs: dict[str, torch.Tensor] = {}
    errors: dict[str, str] = {}
    identities: dict[str, Any] = {}
    call_counts = {"gemv": 0, "grouped_gemm": 0, "upstream": 0}

    def counted_gemv(*args: Any, **inner_kwargs: Any) -> torch.Tensor:
        call_counts["gemv"] += 1
        return original_gemv(*args, **inner_kwargs)

    def counted_grouped(*args: Any, **inner_kwargs: Any) -> torch.Tensor | None:
        call_counts["grouped_gemm"] += 1
        return original_grouped(*args, **inner_kwargs)

    def counted_upstream(*args: Any, **inner_kwargs: Any) -> torch.Tensor:
        call_counts["upstream"] += 1
        return original_upstream(*args, **inner_kwargs)

    try:
        fused_moe.fused_experts_impl = counted_gemv
        fused_moe._maybe_musa_fp8_moe_grouped_gemm = counted_grouped
        fused_moe._upstream_fused_moe._musa_original_fused_experts_impl = (
            counted_upstream
        )
        for backend in dispatchable_backends:
            for name in call_counts:
                call_counts[name] = 0
            fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND = MusaFusedMoeBackend(backend)
            try:
                outputs[backend] = fused_moe._musa_fused_experts_impl_dispatch(
                    **kwargs
                ).detach()
                synchronize()
                identities[backend] = {
                    "call_counts": dict(call_counts),
                    "passed": call_counts[backend] == 1
                    and sum(call_counts.values()) == 1,
                }
            except Exception as exc:
                errors[backend] = f"{type(exc).__name__}: {exc}"
    finally:
        fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND = old_backend
        fused_moe.fused_experts_impl = original_gemv
        fused_moe._maybe_musa_fp8_moe_grouped_gemm = original_grouped
        fused_moe._upstream_fused_moe._musa_original_fused_experts_impl = (
            original_upstream
        )

    comparisons: dict[str, Any] = {}
    for backend, output in outputs.items():
        direct_output = direct_outputs.get(backend)
        if direct_output is None:
            comparisons[backend] = {"passed": False, "error": "direct output missing"}
            continue
        diff = (output.float() - direct_output.float()).abs()
        comparisons[backend] = {
            "passed": bool(torch.equal(output, direct_output)),
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
            **comparison_metrics(output, direct_output),
        }
    passed = bool(
        not errors
        and all(
            identities.get(name, {}).get("passed") for name in dispatchable_backends
        )
        and all(
            comparisons.get(name, {}).get("passed") for name in dispatchable_backends
        )
    )
    return {
        "passed": passed,
        "not_dispatchable": sorted(
            set(requested_backends) - set(dispatchable_backends)
        ),
        "errors": errors,
        "identities": identities,
        "comparisons": comparisons,
    }


def deepgemm_prefill_dispatch_smoke(
    kwargs: dict[str, Any], direct_output: torch.Tensor
) -> dict[str, Any]:
    """Prove that the default eager path reaches production DeepGEMM."""

    old_backend = fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND
    original_deepgemm = fused_moe._maybe_moe_deepgemm_prefill
    original_upstream = fused_moe._upstream_fused_moe._musa_original_fused_experts_impl
    calls = {"deepgemm_prefill": 0, "upstream": 0}

    def counted_deepgemm(**inner_kwargs: Any) -> torch.Tensor | None:
        calls["deepgemm_prefill"] += 1
        return original_deepgemm(**inner_kwargs)

    def counted_upstream(*args: Any, **inner_kwargs: Any) -> torch.Tensor:
        calls["upstream"] += 1
        return original_upstream(*args, **inner_kwargs)

    try:
        fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND = MusaFusedMoeBackend.AUTO
        fused_moe._maybe_moe_deepgemm_prefill = counted_deepgemm
        fused_moe._upstream_fused_moe._musa_original_fused_experts_impl = (
            counted_upstream
        )
        output = fused_moe._musa_fused_experts_impl_dispatch(**kwargs).detach()
        synchronize()
        difference = (output.float() - direct_output.float()).abs()
        comparison = {
            "bitwise_equal": bool(torch.equal(output, direct_output)),
            "max_abs_diff": float(difference.max().item()),
            "mean_abs_diff": float(difference.mean().item()),
            **comparison_metrics(output, direct_output),
        }
        expected_calls = {"deepgemm_prefill": 1, "upstream": 0}
        return {
            "passed": calls == expected_calls and comparison["bitwise_equal"],
            "calls": calls,
            "expected_calls": expected_calls,
            "comparison": comparison,
        }
    except Exception as exc:
        return {
            "passed": False,
            "calls": calls,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        fused_moe._MUSA_FUSED_MOE_REQUESTED_BACKEND = old_backend
        fused_moe._maybe_moe_deepgemm_prefill = original_deepgemm
        fused_moe._upstream_fused_moe._musa_original_fused_experts_impl = (
            original_upstream
        )


def deepgemm_prefill_implementation(kwargs: dict[str, Any]) -> str:
    try:
        from vllm_musa.jit_kernel.tilelang.deep_gemm_contig_preprocess import (
            can_use_fp8_tilelang,
        )

        use_fused_glue = can_use_fp8_tilelang(
            kwargs["hidden_states"],
            kwargs["topk_ids"],
            kwargs["w1"].shape[0],
            True,
        )
        return "fused_glue" if use_fused_glue else "grouped_glue_fallback"
    except Exception as exc:
        return f"probe_error:{type(exc).__name__}:{exc}"


def quantile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def comparison_metrics(
    output: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    output_flat = output.float().flatten()
    reference_flat = reference.float().flatten()
    difference = output_flat - reference_flat
    reference_norm = reference_flat.norm().clamp_min(torch.finfo(torch.float32).eps)
    relative_l2 = float((difference.norm() / reference_norm).item())
    cosine = float(F.cosine_similarity(output_flat, reference_flat, dim=0).item())
    output_rows = output.float().reshape(-1, output.shape[-1])
    reference_rows = reference.float().reshape(-1, reference.shape[-1])
    difference_rows = output_rows - reference_rows
    row_reference_norms = reference_rows.norm(dim=1).clamp_min(
        torch.finfo(torch.float32).eps
    )
    row_relative_l2 = difference_rows.norm(dim=1) / row_reference_norms
    row_cosine = F.cosine_similarity(output_rows, reference_rows, dim=1)
    reference_absmax = (
        reference_flat.abs().max().clamp_min(torch.finfo(torch.float32).eps)
    )
    return {
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "max_row_relative_l2": float(row_relative_l2.max().item()),
        "min_row_cosine": float(row_cosine.min().item()),
        "max_abs_diff_over_reference_absmax": float(
            (difference.abs().max() / reference_absmax).item()
        ),
    }


def dequantized_fp32_reference(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Compute a bounded correctness oracle without activation quantization.

    This is intentionally used only for a small, fixed-size coverage
    probe. Native GEMV consumes BF16 activations directly, while the
    established Triton and grouped paths dynamically quantize activations to
    FP8. Comparing GEMV to those paths with elementwise allclose can therefore
    reject the more accurate result.
    """

    tokens, hidden_size = hidden_states.shape
    intermediate_size = w2.shape[2]
    output = torch.zeros(
        (tokens, hidden_size), device=hidden_states.device, dtype=torch.float32
    )

    def dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        expanded_scale = scale.repeat_interleave(block_size, dim=0)
        expanded_scale = expanded_scale.repeat_interleave(block_size, dim=1)
        return weight.float() * expanded_scale

    for token_index in range(tokens):
        activation = hidden_states[token_index].float()
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token_index, slot].item())
            w1_dequantized = dequantize(w1[expert], w1_scale[expert])
            projected = F.linear(activation, w1_dequantized)
            del w1_dequantized
            activated = F.silu(projected[:intermediate_size])
            activated.mul_(projected[intermediate_size:])
            w2_dequantized = dequantize(w2[expert], w2_scale[expert])
            expert_output = F.linear(activated, w2_dequantized)
            del w2_dequantized
            output[token_index].add_(
                expert_output, alpha=float(topk_weights[token_index, slot].item())
            )
    return output


def recommend_thresholds(
    results: list[Result], tokens: list[int], margin: float, max_iqr_ratio: float
) -> dict[str, int | None]:
    points: dict[int, dict[str, Result]] = {}
    for result in results:
        if (
            result.error is None
            and result.correctness_pass is True
            and result.median_ms is not None
            and result.p95_ms is not None
            and result.iqr_ms is not None
        ):
            points.setdefault(result.tokens, {})[result.backend] = result

    def safely_faster(token_count: int, backend: str) -> bool:
        point = points.get(token_count, {})
        candidate = point.get(backend)
        alternatives = [result for name, result in point.items() if name != backend]
        if candidate is None or not alternatives:
            return False
        best_median = min(result.median_ms for result in alternatives)
        best_p95 = min(result.p95_ms for result in alternatives)
        best_alternative = min(alternatives, key=lambda result: result.median_ms)
        stable = (
            candidate.iqr_ms / candidate.median_ms <= max_iqr_ratio
            and best_alternative.iqr_ms / best_alternative.median_ms <= max_iqr_ratio
        )
        return bool(
            stable
            and candidate.median_ms <= best_median * (1 - margin)
            and candidate.p95_ms <= best_p95 * (1 + margin)
        )

    # A production ``<= threshold`` rule covers every integer token count.
    # Never extrapolate a continuous GEMV prefix across holes in a sparse
    # sweep: stop at the first unmeasured integer even if a later anchor wins.
    gemv_max = None
    expected_token = 1
    for token_count in tokens:
        if token_count != expected_token:
            break
        if not safely_faster(token_count, "gemv"):
            break
        gemv_max = token_count
        expected_token += 1

    grouped_min = None
    for index, token_count in enumerate(tokens):
        # A production ``>= threshold`` rule covers every integer above the
        # threshold.  Require the measured suffix itself to be contiguous;
        # otherwise a sparse sweep could silently skip an unsafe interval.
        suffix = tokens[index:]
        if suffix != list(range(token_count, suffix[-1] + 1)):
            continue
        stable = True
        for suffix_token in suffix:
            if not safely_faster(suffix_token, "grouped_gemm"):
                stable = False
                break
        if stable:
            grouped_min = token_count
            break
    return {
        "gemv_max_tokens": gemv_max,
        "grouped_gemm_min_tokens": grouped_min,
    }


def main() -> int:
    args = parse_args()
    shape = Shape(
        name=args.name,
        experts=args.experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        top_k=args.top_k,
        block_size=args.block_size,
    )
    tokens = sorted({int(value) for value in args.tokens.split(",") if value})
    if args.sweep_kind == "dense" and tokens:
        dense_prefix_max = min(max(tokens), args.dense_prefix_max)
        tokens = sorted(set(tokens).union(range(1, dense_prefix_max + 1)))
    requested_backends = [value for value in args.backends.split(",") if value]
    unknown = set(requested_backends) - {
        "gemv",
        "grouped_gemm",
        "deepgemm_prefill",
        "upstream",
    }
    if unknown:
        raise ValueError(f"unsupported backends: {sorted(unknown)}")
    if "upstream" not in requested_backends:
        raise ValueError("upstream is mandatory as the correctness reference")
    if not tokens or min(tokens) <= 0:
        raise ValueError("tokens must contain positive integers")
    if args.dense_prefix_max <= 0:
        raise ValueError("dense-prefix-max must be positive")
    if args.oracle_probe_tokens <= 0:
        raise ValueError("oracle-probe-tokens must be positive")
    if not 0 < args.regression_margin < 1:
        raise ValueError("regression-margin must be between zero and one")
    if args.max_iqr_ratio <= 0:
        raise ValueError("max-iqr-ratio must be positive")
    if shape.top_k <= 0 or shape.top_k > shape.experts:
        raise ValueError("top-k must be in [1, experts]")
    if args.folded_shared_expert and (shape.experts < 2 or shape.top_k < 2):
        raise ValueError("folded shared expert requires experts >= 2 and top-k >= 2")
    if args.rounds <= 0 or args.rounds % len(requested_backends) != 0:
        raise ValueError("rounds must be a positive multiple of backend count")

    w1 = quantized_weight(
        (shape.experts, 2 * shape.intermediate_size, shape.hidden_size), args.seed
    )
    w2 = quantized_weight(
        (shape.experts, shape.hidden_size, shape.intermediate_size), args.seed + 1
    )
    w1_scale = block_scales(w1, shape.block_size, args.seed + 2)
    w2_scale = block_scales(w2, shape.block_size, args.seed + 3)
    capability = tuple(int(value) for value in torch.musa.get_device_capability())
    device_name = torch.musa.get_device_name(0)
    multiprocessor_count = int(
        torch.musa.get_device_properties(0).multi_processor_count
    )
    if capability != (3, 1) or "S5000" not in device_name.upper():
        raise RuntimeError(
            "this policy sweep is valid only on MTT S5000/MP31, got "
            f"device={device_name!r} capability={capability}"
        )
    policy_key = MusaFusedMoeShape(
        device_capability=capability,
        multiprocessor_count=multiprocessor_count,
        local_experts=shape.experts,
        w1_output_size=2 * shape.intermediate_size,
        w2_input_size=shape.intermediate_size,
        hidden_size=shape.hidden_size,
        top_k=shape.top_k,
        block_n=shape.block_size,
        block_k=shape.block_size,
        activation="silu",
        expert_parallel=False,
        hidden_dtype=str(torch.bfloat16),
        weight_dtype=str(torch.float8_e4m3fn),
        scale_dtype=str(torch.float32),
        w1_scale_shape=tuple(w1_scale.shape),
        w2_scale_shape=tuple(w2_scale.shape),
        gemv_block=os.environ.get("VLLM_MUSA_GEMV_MOE_BLOCK", "auto"),
        graph_mode="eager",
    )
    metadata = {
        "schema": "musa-fused-moe-crossover.v6",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "shape": asdict(shape),
        "shape_dimensions_are_per_rank": True,
        "policy_key": asdict(policy_key),
        "route_mode": args.route_mode,
        "folded_shared_expert": args.folded_shared_expert,
        "sweep_kind": args.sweep_kind,
        "dense_prefix_max": args.dense_prefix_max,
        "seed": args.seed,
        "tokens": tokens,
        "gemv_recommendation_requires_contiguous_prefix": True,
        "grouped_recommendation_requires_contiguous_suffix": True,
        "backends": requested_backends,
        "warmup": args.warmup,
        "repeats_per_round": args.repeats_per_round,
        "rounds": args.rounds,
        "l2_flush_mb": args.l2_flush_mb,
        "cold_cache": True,
        "atol": args.atol,
        "rtol": args.rtol,
        "gemv_max_relative_l2": args.gemv_max_relative_l2,
        "gemv_min_cosine": args.gemv_min_cosine,
        "gemv_max_row_relative_l2": args.gemv_max_row_relative_l2,
        "gemv_min_row_cosine": args.gemv_min_row_cosine,
        "gemv_max_normalized_abs_diff": args.gemv_max_normalized_abs_diff,
        "gemv_oracle_max_relative_l2": args.gemv_oracle_max_relative_l2,
        "gemv_oracle_min_cosine": args.gemv_oracle_min_cosine,
        "gemv_oracle_max_row_relative_l2": (args.gemv_oracle_max_row_relative_l2),
        "gemv_oracle_min_row_cosine": args.gemv_oracle_min_row_cosine,
        "gemv_oracle_max_normalized_abs_diff": (
            args.gemv_oracle_max_normalized_abs_diff
        ),
        "oracle_probe_tokens": args.oracle_probe_tokens,
        "grouped_max_relative_l2": args.grouped_max_relative_l2,
        "grouped_min_cosine": args.grouped_min_cosine,
        "grouped_max_row_relative_l2": args.grouped_max_row_relative_l2,
        "grouped_min_row_cosine": args.grouped_min_row_cosine,
        "grouped_max_normalized_abs_diff": args.grouped_max_normalized_abs_diff,
        "regression_margin": args.regression_margin,
        "max_iqr_ratio": args.max_iqr_ratio,
        "gemv_block": os.environ.get("VLLM_MUSA_GEMV_MOE_BLOCK", "auto"),
        "deepgemm_prefill_env": os.environ.get(
            "VLLM_MUSA_MOE_DEEPGEMM_PREFILL", "default-on"
        ),
        "deepgemm_prefill_min_tokens": os.environ.get(
            "VLLM_MUSA_MOE_DEEPGEMM_PREFILL_MIN_TOKENS", "2500"
        ),
        "device_name": device_name,
        "device_capability": capability,
        "multiprocessor_count": multiprocessor_count,
        "torch_musa_runtime": getattr(torch.version, "musa", None),
        "packages": package_versions(),
        "repo": repo_provenance(),
        "mthreads_gmi": command_output(["mthreads-gmi", "-q"]),
        "mcc_version": command_output(["mcc", "--version"]),
        "estimated_static_bytes": (
            w1.numel() * w1.element_size()
            + w2.numel() * w2.element_size()
            + w1_scale.numel() * w1_scale.element_size()
            + w2_scale.numel() * w2_scale.element_size()
        ),
    }
    results: list[Result] = []
    dispatcher_evidence: dict[str, Any] = {}
    deepgemm_dispatch_evidence: dict[str, Any] = {}
    deepgemm_implementation: dict[str, str] = {}
    oracle_evidence: dict[str, Any] | None = None
    gemv_oracle_pass = "gemv" not in requested_backends

    if "gemv" in requested_backends:
        oracle_seed = args.seed + 104729
        oracle_generator = torch.Generator(device="musa")
        oracle_generator.manual_seed(oracle_seed)
        oracle_hidden_states = torch.randn(
            (args.oracle_probe_tokens, shape.hidden_size),
            device="musa",
            dtype=torch.bfloat16,
            generator=oracle_generator,
        )
        oracle_topk_ids, oracle_topk_weights = oracle_coverage_routes(
            args.oracle_probe_tokens,
            shape.experts,
            shape.top_k,
            args.folded_shared_expert,
        )
        oracle_kwargs = backend_kwargs(
            hidden_states=oracle_hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=oracle_topk_weights,
            topk_ids=oracle_topk_ids,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_size=shape.block_size,
        )
        oracle_backends, oracle_capabilities = build_backends(oracle_kwargs)
        oracle_outputs: dict[str, torch.Tensor] = {}
        oracle_errors: dict[str, str] = {}
        for backend in requested_backends:
            if not oracle_capabilities[backend]:
                oracle_errors[backend] = "production capability gate rejected"
                continue
            try:
                oracle_outputs[backend] = oracle_backends[backend]().detach().clone()
                synchronize()
            except Exception as exc:
                oracle_errors[backend] = f"{type(exc).__name__}: {exc}"
        try:
            oracle = dequantized_fp32_reference(
                hidden_states=oracle_hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=oracle_topk_weights,
                topk_ids=oracle_topk_ids,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                block_size=shape.block_size,
            )
            comparisons: dict[str, Any] = {}
            for backend, output in oracle_outputs.items():
                difference = (output.float() - oracle).abs()
                comparisons[backend] = {
                    "max_abs_diff": float(difference.max().item()),
                    "mean_abs_diff": float(difference.mean().item()),
                    **comparison_metrics(output, oracle),
                }
            gemv_comparison = comparisons.get("gemv")
            gemv_oracle_pass = bool(
                gemv_comparison is not None
                and not oracle_errors
                and gemv_comparison["relative_l2_error"]
                <= args.gemv_oracle_max_relative_l2
                and gemv_comparison["cosine_similarity"] >= args.gemv_oracle_min_cosine
                and gemv_comparison["max_row_relative_l2"]
                <= args.gemv_oracle_max_row_relative_l2
                and gemv_comparison["min_row_cosine"] >= args.gemv_oracle_min_row_cosine
                and gemv_comparison["max_abs_diff_over_reference_absmax"]
                <= args.gemv_oracle_max_normalized_abs_diff
            )
            oracle_evidence = {
                "tokens": args.oracle_probe_tokens,
                "route_mode": "deterministic_coverage",
                "seed": oracle_seed,
                "selected_experts": sorted(
                    int(value) for value in oracle_topk_ids.unique().cpu().tolist()
                ),
                "reference": "dequantized-fp32-no-activation-quantization",
                "gemv_passed": gemv_oracle_pass,
                "comparisons": comparisons,
                "errors": oracle_errors,
            }
            del oracle
        except Exception as exc:
            gemv_oracle_pass = False
            oracle_evidence = {
                "tokens": args.oracle_probe_tokens,
                "route_mode": "deterministic_coverage",
                "seed": oracle_seed,
                "reference": "dequantized-fp32-no-activation-quantization",
                "gemv_passed": False,
                "errors": {
                    **oracle_errors,
                    "oracle": f"{type(exc).__name__}: {exc}",
                },
            }
        del (
            oracle_hidden_states,
            oracle_topk_ids,
            oracle_topk_weights,
            oracle_outputs,
        )
        torch.musa.empty_cache()

    for token_count in tokens:
        generator = torch.Generator(device="musa")
        generator.manual_seed(args.seed + token_count)
        hidden_states = torch.randn(
            (token_count, shape.hidden_size),
            device="musa",
            dtype=torch.bfloat16,
            generator=generator,
        )
        topk_ids, topk_weights = routes(
            token_count,
            shape.experts,
            shape.top_k,
            args.route_mode,
            args.seed + token_count,
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
            block_size=shape.block_size,
        )
        backends, capabilities = build_backends(kwargs)
        unsupported = [name for name in requested_backends if not capabilities[name]]
        if unsupported:
            raise RuntimeError(f"production capability gate rejected: {unsupported}")
        outputs: dict[str, torch.Tensor] = {}
        errors: dict[str, str] = {}
        for backend in requested_backends:
            try:
                outputs[backend] = backends[backend]().detach().clone()
                synchronize()
            except Exception as exc:
                errors[backend] = f"{type(exc).__name__}: {exc}"

        dispatcher_evidence[str(token_count)] = dispatcher_smoke(
            kwargs, requested_backends, outputs
        )
        if "deepgemm_prefill" in requested_backends:
            deepgemm_implementation[str(token_count)] = deepgemm_prefill_implementation(
                kwargs
            )
            if "deepgemm_prefill" in outputs:
                deepgemm_dispatch_evidence[str(token_count)] = (
                    deepgemm_prefill_dispatch_smoke(kwargs, outputs["deepgemm_prefill"])
                )
            else:
                deepgemm_dispatch_evidence[str(token_count)] = {
                    "passed": False,
                    "error": errors.get("deepgemm_prefill", "direct output missing"),
                }

        reference = outputs.get("upstream")
        samples: dict[str, list[float]] = {name: [] for name in requested_backends}
        for round_index in range(args.rounds):
            offset = round_index % len(requested_backends)
            order = requested_backends[offset:] + requested_backends[:offset]
            for backend in order:
                if backend in errors:
                    continue
                try:
                    measurements = bench_gpu_time_with_musa_event(
                        backends[backend],
                        dry_run_iters=args.warmup,
                        repeat_iters=args.repeats_per_round,
                        l2_flush=True,
                        l2_flush_size_mb=args.l2_flush_mb,
                    )
                    samples[backend].extend(float(value) for value in measurements)
                except Exception as exc:
                    errors[backend] = f"{type(exc).__name__}: {exc}"
                finally:
                    torch.musa.empty_cache()

        for backend in requested_backends:
            diff_max = None
            diff_mean = None
            relative_l2 = None
            cosine = None
            max_row_relative_l2 = None
            min_row_cosine = None
            normalized_abs_diff = None
            output_absmax = None
            output_std = None
            finite = None
            non_poison = None
            correct = None
            correctness_basis = "unavailable"
            output = outputs.get(backend)
            if reference is not None and output is not None:
                output_float = output.float()
                diff = (output_float - reference.float()).abs()
                diff_max = float(diff.max().item())
                diff_mean = float(diff.mean().item())
                metrics = comparison_metrics(output, reference)
                relative_l2 = metrics["relative_l2_error"]
                cosine = metrics["cosine_similarity"]
                max_row_relative_l2 = metrics["max_row_relative_l2"]
                min_row_cosine = metrics["min_row_cosine"]
                normalized_abs_diff = metrics["max_abs_diff_over_reference_absmax"]
                output_absmax = float(output_float.abs().max().item())
                output_std = float(output_float.std().item())
                finite = bool(torch.isfinite(output_float).all().item())
                non_poison = finite and output_absmax > 1e-6 and output_std > 1e-7
                if backend == "gemv":
                    correctness_basis = (
                        "sampled-fp32-oracle-and-rowwise-normalized-envelope"
                    )
                    correct = bool(
                        non_poison
                        and gemv_oracle_pass
                        and relative_l2 <= args.gemv_max_relative_l2
                        and cosine >= args.gemv_min_cosine
                        and max_row_relative_l2 <= args.gemv_max_row_relative_l2
                        and min_row_cosine >= args.gemv_min_row_cosine
                        and normalized_abs_diff <= args.gemv_max_normalized_abs_diff
                    )
                elif backend == "upstream":
                    correctness_basis = "established-reference"
                    correct = bool(non_poison)
                else:
                    correctness_basis = "upstream-normalized-envelope"
                    correct = bool(
                        non_poison
                        and relative_l2 <= args.grouped_max_relative_l2
                        and cosine >= args.grouped_min_cosine
                        and max_row_relative_l2 <= args.grouped_max_row_relative_l2
                        and min_row_cosine >= args.grouped_min_row_cosine
                        and normalized_abs_diff <= args.grouped_max_normalized_abs_diff
                    )
            timings = samples[backend]
            q25 = quantile(timings, 0.25) if timings else None
            q75 = quantile(timings, 0.75) if timings else None
            results.append(
                Result(
                    shape=asdict(shape),
                    tokens=token_count,
                    backend=backend,
                    samples_ms=timings,
                    median_ms=statistics.median(timings) if timings else None,
                    p20_ms=quantile(timings, 0.2) if timings else None,
                    p95_ms=quantile(timings, 0.95) if timings else None,
                    iqr_ms=(q75 - q25) if q25 is not None and q75 is not None else None,
                    min_ms=min(timings) if timings else None,
                    max_ms=max(timings) if timings else None,
                    max_abs_diff=diff_max,
                    mean_abs_diff=diff_mean,
                    relative_l2_error=relative_l2,
                    cosine_similarity=cosine,
                    max_row_relative_l2=max_row_relative_l2,
                    min_row_cosine=min_row_cosine,
                    max_abs_diff_over_reference_absmax=normalized_abs_diff,
                    output_absmax=output_absmax,
                    output_std=output_std,
                    finite=finite,
                    non_poison=non_poison,
                    correctness_pass=correct,
                    correctness_basis=correctness_basis,
                    reference_backend="upstream",
                    error=errors.get(backend),
                )
            )
        del hidden_states, topk_ids, topk_weights, outputs, reference
        torch.musa.empty_cache()

    qualified = bool(
        dispatcher_evidence
        and all(item["passed"] for item in dispatcher_evidence.values())
        and (
            not deepgemm_dispatch_evidence
            or all(item["passed"] for item in deepgemm_dispatch_evidence.values())
        )
        and all(
            result.error is None and result.correctness_pass is True
            for result in results
        )
    )
    candidate = (
        recommend_thresholds(
            results, tokens, args.regression_margin, args.max_iqr_ratio
        )
        if qualified
        else {"gemv_max_tokens": None, "grouped_gemm_min_tokens": None}
    )
    payload = {
        "metadata": metadata,
        "qualification": {
            "passed": qualified,
            "reference_backend": {
                "gemv": "sampled-fp32-oracle-and-rowwise-normalized-envelope",
                "grouped_gemm": "upstream-normalized-envelope",
                "deepgemm_prefill": "upstream-normalized-envelope",
                "upstream": "established-reference",
            },
            "note": (
                "candidate only; dense crossover, route-mode, independent-seed, "
                "CUDAGraph, and serving A/B gates remain"
            ),
        },
        "oracle_evidence": oracle_evidence,
        "dispatcher_smoke_by_tokens": dispatcher_evidence,
        "deepgemm_dispatch_smoke_by_tokens": deepgemm_dispatch_evidence,
        "deepgemm_implementation_by_tokens": deepgemm_implementation,
        "candidate_recommendation": candidate,
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
