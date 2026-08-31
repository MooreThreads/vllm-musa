#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-L2 paired sweep for repo-owned native JIT top-k launch tactics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _benchmark_utils import (
    emit_payload,
    percentile,
    provenance,
    verify_lease_device_fence,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SHAPE_PATH = SCRIPT_DIR / "qwen_dsv4_fp8_quant_shapes.json"
CAMPAIGN_PATH = SCRIPT_DIR / "mp_tactic_campaign.json"
L2_FLUSH_BYTES = 8_000_000_000
PRODUCTION_WARPS_PER_CTA = 4
VALID_WARPS_PER_CTA = (1, 2, 4, 8)
BENCHMARK_MODULE_NAME = "vllm_musa_topk_gating_benchmark_tactics"
BENCHMARK_COMPILE_DEFINE = "-DVLLM_MUSA_TOPK_BENCHMARK_TACTICS=1"


@dataclass(frozen=True)
class Family:
    name: str
    scoring_func: str
    input_experts: int
    routed_experts: int
    routed_topk: int
    output_topk: int
    num_fused_shared_experts: int
    production_rows: tuple[int, ...]
    routed_rows: tuple[int, ...]
    campaign_family: str
    kernel_name: str
    production_reachable: bool
    scope_note: str
    renormalize: bool = True
    has_correction_bias: bool = False


def parse_warps_per_cta(value: str) -> int:
    try:
        warps = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"warps per CTA must be an integer: {value!r}"
        ) from exc
    if warps not in VALID_WARPS_PER_CTA:
        raise argparse.ArgumentTypeError(
            f"warps per CTA must be one of {VALID_WARPS_PER_CTA}"
        )
    return warps


def launch_geometry(rows: int, warps_per_cta: int) -> dict[str, int]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if warps_per_cta not in VALID_WARPS_PER_CTA:
        raise ValueError(f"warps per CTA must be one of {VALID_WARPS_PER_CTA}")
    return {
        "warps_per_cta": warps_per_cta,
        "threads_per_cta": warps_per_cta * 32,
        "grid_ctas": math.ceil(rows / warps_per_cta),
    }


def _campaign_families(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    families: set[str] = set()
    for cell in payload["cells"]:
        args = cell.get("base_args", [])
        for index, value in enumerate(args[:-1]):
            if value == "--family":
                families.add(str(args[index + 1]))
    return families


def _token_rows(routed_rows: Sequence[int], topk: int, label: str) -> tuple[int, ...]:
    if topk <= 0 or any(row <= 0 or row % topk for row in routed_rows):
        raise ValueError(
            f"{label} routed rows must be positive multiples of topk={topk}"
        )
    return tuple(int(row) // topk for row in routed_rows)


def load_production_families(
    shape_path: Path = SHAPE_PATH,
    campaign_path: Path = CAMPAIGN_PATH,
) -> dict[str, Family]:
    """Load row buckets from checked-in Qwen/DeepSeek campaign inputs."""
    shapes = json.loads(shape_path.read_text(encoding="utf-8"))
    routed_shapes = shapes["families"]["silu_routed_moe"]
    campaign_families = _campaign_families(campaign_path)
    required_campaign_families = {
        "dsv4_fp8",
        "qwen_bf16",
        "qwen35_folded_bf16",
    }
    missing = required_campaign_families - campaign_families
    if missing:
        raise ValueError(f"campaign is missing top-k shape families: {sorted(missing)}")

    qwen_routed_rows = tuple(routed_shapes["qwen_tp8_topk8"]["rows"])
    dsv4_routed_rows = tuple(routed_shapes["deepseek_v4_tp8_topk6"]["rows"])
    qwen_rows = _token_rows(qwen_routed_rows, 8, "Qwen")
    dsv4_rows = _token_rows(dsv4_routed_rows, 6, "DeepSeek-V4")
    return {
        "qwen_softmax_e256_k8": Family(
            name="qwen_softmax_e256_k8",
            scoring_func="softmax",
            input_experts=256,
            routed_experts=256,
            routed_topk=8,
            output_topk=8,
            num_fused_shared_experts=0,
            production_rows=qwen_rows,
            routed_rows=qwen_routed_rows,
            campaign_family="qwen_bf16",
            kernel_name="topk_softmax_no_bias_renorm_warp_kernel_fixed_k",
            production_reachable=True,
            scope_note="plain Qwen local softmax router",
        ),
        "qwen35_folded_softmax_e257_k9": Family(
            name="qwen35_folded_softmax_e257_k9",
            scoring_func="softmax",
            input_experts=257,
            routed_experts=256,
            routed_topk=8,
            output_topk=9,
            num_fused_shared_experts=1,
            production_rows=qwen_rows,
            routed_rows=qwen_routed_rows,
            campaign_family="qwen35_folded_bf16",
            kernel_name=(
                "topk_softmax_no_bias_renorm_warp_combined_shared1_kernel_fixed_k"
            ),
            production_reachable=True,
            scope_note="Qwen3.5 folded routed-plus-shared local softmax router",
        ),
        "deepseek_v4_sigmoid_e256_k6_local_no_bias": Family(
            name="deepseek_v4_sigmoid_e256_k6_local_no_bias",
            scoring_func="sigmoid",
            input_experts=256,
            routed_experts=256,
            routed_topk=6,
            output_topk=6,
            num_fused_shared_experts=0,
            production_rows=dsv4_rows,
            routed_rows=dsv4_routed_rows,
            campaign_family="dsv4_fp8",
            kernel_name="topk_sigmoid_no_bias_warp_kernel",
            production_reachable=False,
            scope_note=(
                "DSV4 production E/topk/rows on the local no-bias JIT arm only; "
                "the MATE grouped correction-bias router is excluded"
            ),
        ),
        "deepseek_v4_sigmoid_e256_k6_correction_bias": Family(
            name="deepseek_v4_sigmoid_e256_k6_correction_bias",
            scoring_func="sigmoid",
            input_experts=256,
            routed_experts=256,
            routed_topk=6,
            output_topk=6,
            num_fused_shared_experts=0,
            production_rows=dsv4_rows,
            routed_rows=dsv4_routed_rows,
            campaign_family="dsv4_fp8",
            kernel_name="topk_sigmoid_warp_kernel",
            production_reachable=True,
            scope_note="DSV4 local correction-bias sigmoid production router",
            has_correction_bias=True,
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    families = load_production_families()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        choices=tuple(families),
        nargs="+",
        default=list(families),
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        help="Optional subset; every row must be a production bucket for each family.",
    )
    parser.add_argument(
        "--warps-per-cta",
        type=parse_warps_per_cta,
        nargs="+",
        default=list(VALID_WARPS_PER_CTA),
    )
    parser.add_argument("--dry-runs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--expected-physical-device", type=int, required=True)
    parser.add_argument("--expected-device-uuid", required=True)
    parser.add_argument("--expected-multiprocessor-count", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def selected_rows(family: Family, requested: Sequence[int] | None) -> tuple[int, ...]:
    if requested is None:
        return family.production_rows
    rows = tuple(dict.fromkeys(int(row) for row in requested))
    invalid = sorted(set(rows) - set(family.production_rows))
    if invalid:
        raise ValueError(
            f"{family.name} rows must be production buckets {family.production_rows}; "
            f"got invalid {invalid}"
        )
    return rows


def validate_timing_args(dry_runs: int, repeats: int) -> None:
    if dry_runs <= 0:
        raise ValueError("dry runs must be positive")
    if repeats < 2 or repeats % 2:
        raise ValueError(
            "repeats must be an even integer of at least 2 for balanced AB/BA"
        )


def _load_runtime() -> tuple[Any, Any]:
    # Import order is part of the MUSA compatibility contract.
    # isort: off
    import torchada  # noqa: F401
    import torch
    from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
    # isort: on

    module = load_musa_jit(
        BENCHMARK_MODULE_NAME,
        ("topk/topk_gating.mu",),
        extra_musa_cflags=(BENCHMARK_COMPILE_DEFINE,),
    )
    return torch, module


def _allocate_outputs(
    torch: Any, family: Family, rows: int, device: Any
) -> tuple[Any, Any]:
    return (
        torch.empty((rows, family.output_topk), dtype=torch.float32, device=device),
        torch.empty((rows, family.output_topk), dtype=torch.int32, device=device),
    )


def _poison_outputs(outputs: tuple[Any, Any]) -> None:
    weights, ids = outputs
    weights.fill_(float("nan"))
    ids.fill_(-1)


def _launch(
    module: Any,
    family: Family,
    gating: Any,
    correction_bias: Any | None,
    outputs: tuple[Any, Any],
    warps_per_cta: int,
) -> None:
    weights, ids = outputs
    unused = weights.reshape(-1)
    if family.scoring_func == "softmax":
        module.sgl_musa_topk_softmax(
            weights,
            ids,
            gating,
            family.renormalize,
            0.0,
            unused,
            False,
            unused,
            family.num_fused_shared_experts,
            False,
            warps_per_cta,
        )
    else:
        module.sgl_musa_topk_sigmoid(
            weights,
            ids,
            gating,
            family.renormalize,
            correction_bias if correction_bias is not None else unused,
            family.has_correction_bias,
            unused,
            0,
            False,
            warps_per_cta,
        )


def _reference(
    torch: Any,
    family: Family,
    gating: Any,
    correction_bias: Any | None,
) -> tuple[Any, Any]:
    routed_logits = gating[:, : family.routed_experts].float()
    if family.scoring_func == "softmax":
        scores = torch.softmax(routed_logits, dim=-1)
    else:
        scores = torch.sigmoid(routed_logits)
    selection_scores = scores
    if family.has_correction_bias:
        if correction_bias is None:
            raise RuntimeError("correction-bias family requires a bias tensor")
        selection_scores = scores + correction_bias
    # Native warp_argmax resolves equal scores to the lower expert id. BF16
    # logits can tie, while torch.topk does not promise a stable tie order.
    ids = torch.argsort(
        selection_scores.detach().cpu(),
        dim=-1,
        descending=True,
        stable=True,
    )[:, : family.routed_topk].to(device=gating.device)
    weights = scores.gather(1, ids)
    if family.renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    if family.num_fused_shared_experts:
        shared = (
            torch.sigmoid(gating[:, family.routed_experts :]).to(gating.dtype).float()
        )
        shared_ids = torch.full_like(ids[:, :1], family.routed_experts)
        weights = torch.cat((weights, shared), dim=-1)
        ids = torch.cat((ids, shared_ids), dim=-1)
    return weights.float(), ids.to(torch.int32)


def _correctness(
    torch: Any,
    family: Family,
    outputs: tuple[Any, Any],
    production: tuple[Any, Any],
    reference: tuple[Any, Any],
) -> dict[str, Any]:
    weights, ids = outputs
    production_weights, production_ids = production
    reference_weights, reference_ids = reference
    finite = bool(torch.isfinite(weights).all().item())
    ids_in_range = bool(
        (ids >= 0).all().item()
        and (ids <= family.routed_experts).all().item()
        and (
            family.num_fused_shared_experts
            or (ids < family.routed_experts).all().item()
        )
    )
    nonzero = bool(weights.abs().max().item() > 1e-12)
    nonconstant = bool((weights.max() - weights.min()).abs().item() > 1e-12)
    poison_output = not (finite and ids_in_range and nonzero and nonconstant)
    ids_equal_production = bool(torch.equal(ids, production_ids))
    ids_equal_reference = bool(torch.equal(ids, reference_ids))
    weights_equal_production = bool(torch.equal(weights, production_weights))
    weights_close_reference = bool(
        torch.allclose(weights, reference_weights, atol=3e-3, rtol=3e-3)
    )
    max_abs_diff = float((weights - production_weights).abs().max().item())
    return {
        "finite": finite,
        "ids_in_range": ids_in_range,
        "nonzero": nonzero,
        "nonconstant": nonconstant,
        "poison_output": poison_output,
        "ids_equal_production": ids_equal_production,
        "ids_equal_reference": ids_equal_reference,
        "weights_equal_production": weights_equal_production,
        "weights_close_reference": weights_close_reference,
        "max_abs_diff_vs_production": max_abs_diff,
        "correctness_pass": bool(
            not poison_output
            and ids_equal_production
            and ids_equal_reference
            and weights_equal_production
            and weights_close_reference
        ),
    }


def _timed_launch(
    torch: Any,
    flush: Any,
    module: Any,
    family: Family,
    gating: Any,
    correction_bias: Any | None,
    outputs: tuple[Any, Any],
    warps_per_cta: int,
) -> float:
    _poison_outputs(outputs)
    flush.zero_()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    _launch(module, family, gating, correction_bias, outputs, warps_per_cta)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _effective_io_bytes(family: Family, rows: int, input_element_size: int) -> int:
    input_bytes = rows * family.input_experts * input_element_size
    output_bytes = rows * family.output_topk * (4 + 4)
    return input_bytes + output_bytes


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_timing_args(args.dry_runs, args.repeats)
    families = load_production_families()
    lease_device_fence = verify_lease_device_fence(
        args.expected_physical_device,
        args.expected_device_uuid,
        args.expected_multiprocessor_count,
    )
    torch, module = _load_runtime()
    if not torch.musa.is_available() or torch.musa.device_count() != 1:
        raise RuntimeError("JIT top-k benchmark requires exactly one visible MUSA GPU")

    device = torch.device("musa")
    properties = torch.musa.get_device_properties(0)
    flush = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int32, device=device)
    results: list[dict[str, Any]] = []

    for family_index, family_name in enumerate(args.families):
        family = families[family_name]
        for rows in selected_rows(family, args.rows):
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + family_index * 100_000 + rows
            )
            gating = torch.randn(
                (rows, family.input_experts),
                generator=generator,
                dtype=torch.float32,
            ).to(dtype=torch.bfloat16, device=device)
            correction_bias = None
            if family.has_correction_bias:
                correction_bias = (
                    torch.rand(
                        family.routed_experts,
                        generator=generator,
                        dtype=torch.float32,
                    ).to(device=device)
                    * 0.2
                    - 0.1
                )
            production = _allocate_outputs(torch, family, rows, device)
            candidate = _allocate_outputs(torch, family, rows, device)
            reference = _reference(torch, family, gating, correction_bias)

            _poison_outputs(production)
            _launch(module, family, gating, correction_bias, production, 0)
            torch.musa.synchronize()
            production_reference = _correctness(
                torch, family, production, production, reference
            )

            for warps_per_cta in dict.fromkeys(args.warps_per_cta):
                _poison_outputs(candidate)
                _launch(
                    module,
                    family,
                    gating,
                    correction_bias,
                    candidate,
                    warps_per_cta,
                )
                torch.musa.synchronize()
                correctness = _correctness(
                    torch, family, candidate, production, reference
                )
                correctness["production_reference_pass"] = production_reference[
                    "correctness_pass"
                ]
                correctness["correctness_pass"] = bool(
                    correctness["correctness_pass"]
                    and production_reference["correctness_pass"]
                )

                for dry_run in range(args.dry_runs):
                    tactic = warps_per_cta if dry_run % 2 else 0
                    outputs = candidate if tactic else production
                    _launch(
                        module,
                        family,
                        gating,
                        correction_bias,
                        outputs,
                        tactic,
                    )
                torch.musa.synchronize()

                baseline_samples: list[float] = []
                candidate_samples: list[float] = []
                paired_ratios: list[float] = []
                for repeat in range(args.repeats):
                    order = (0, warps_per_cta)
                    if repeat % 2:
                        order = (warps_per_cta, 0)
                    measured: dict[int, float] = {}
                    for tactic in order:
                        outputs = production if tactic == 0 else candidate
                        measured[tactic] = _timed_launch(
                            torch,
                            flush,
                            module,
                            family,
                            gating,
                            correction_bias,
                            outputs,
                            tactic,
                        )
                    baseline_ms = measured[0]
                    candidate_ms = measured[warps_per_cta]
                    baseline_samples.append(baseline_ms)
                    candidate_samples.append(candidate_ms)
                    paired_ratios.append(candidate_ms / baseline_ms)

                baseline_median = float(statistics.median(baseline_samples))
                candidate_median = float(statistics.median(candidate_samples))
                ratio_median = float(statistics.median(paired_ratios))
                io_bytes = _effective_io_bytes(family, rows, gating.element_size())
                results.append(
                    {
                        "family": family.name,
                        "campaign_family": family.campaign_family,
                        "scoring_func": family.scoring_func,
                        "rows": rows,
                        "input_experts": family.input_experts,
                        "routed_experts": family.routed_experts,
                        "routed_topk": family.routed_topk,
                        "output_topk": family.output_topk,
                        "num_fused_shared_experts": family.num_fused_shared_experts,
                        "dtype": "torch.bfloat16",
                        "kernel_name": family.kernel_name,
                        "has_correction_bias": family.has_correction_bias,
                        "production_reachable": family.production_reachable,
                        "scope_note": family.scope_note,
                        "tie_break": "lower-expert-id",
                        "baseline_tactic": (
                            "production"
                            if family.production_reachable
                            else "local-native-default"
                        ),
                        "baseline_resolved_warps_per_cta": PRODUCTION_WARPS_PER_CTA,
                        "candidate_warps_per_cta": warps_per_cta,
                        "baseline_launch": launch_geometry(
                            rows, PRODUCTION_WARPS_PER_CTA
                        ),
                        "candidate_launch": launch_geometry(rows, warps_per_cta),
                        "effective_io_bytes": io_bytes,
                        "flops_model": None,
                        "flops_model_note": (
                            "transcendental and iterative top-k comparisons are not "
                            "modeled as hardware FLOPs"
                        ),
                        "baseline_median_ms": baseline_median,
                        "candidate_median_ms": candidate_median,
                        "median_ratio": ratio_median,
                        "ratio_p95": percentile(paired_ratios, 0.95),
                        "ratio_p99": percentile(paired_ratios, 0.99),
                        "speedup_pct": (1.0 / ratio_median - 1.0) * 100.0,
                        "baseline_effective_gbps": io_bytes / baseline_median / 1e6,
                        "candidate_effective_gbps": io_bytes / candidate_median / 1e6,
                        "baseline_samples_ms": baseline_samples,
                        "candidate_samples_ms": candidate_samples,
                        "paired_ratios": paired_ratios,
                        **correctness,
                    }
                )
            del gating, production, candidate, reference, correction_bias
            torch.musa.empty_cache()

    payload = {
        "schema": "vllm-musa-jit-topk-warp-tactic-paired-ab.v1",
        "device_name": torch.musa.get_device_name(0),
        "device_capability": [int(properties.major), int(properties.minor)],
        "multiprocessor_count": int(properties.multi_processor_count),
        "lease_device_fence": lease_device_fence,
        "benchmark": {
            "dry_runs": args.dry_runs,
            "repeats": args.repeats,
            "inner_iters": 1,
            "l2_flush_bytes": flush.numel() * flush.element_size(),
            "flush_before_every_timed_launch": True,
            "cache_policy": "cold-l2-per-sample",
            "paired_alternating_order": True,
            "production_warps_per_cta": PRODUCTION_WARPS_PER_CTA,
        },
        "scope": {
            "kernel_source": "vllm_musa/jit_kernel/csrc/topk/topk_gating.mu",
            "benchmark_jit_module": BENCHMARK_MODULE_NAME,
            "benchmark_compile_define": BENCHMARK_COMPILE_DEFINE,
            "implementation": "vllm-musa local native JIT only",
            "excluded": [
                "MATE moe_fused_gate provider router",
                "TileLang grouped_topk router",
            ],
            "shape_sources": [
                str(SHAPE_PATH.relative_to(SCRIPT_DIR.parents[1])),
                str(CAMPAIGN_PATH.relative_to(SCRIPT_DIR.parents[1])),
            ],
        },
        "families": {
            name: {
                "campaign_family": family.campaign_family,
                "production_rows": list(family.production_rows),
                "routed_rows": list(family.routed_rows),
                "input_experts": family.input_experts,
                "routed_topk": family.routed_topk,
                "output_topk": family.output_topk,
                "production_reachable": family.production_reachable,
                "scope_note": family.scope_note,
            }
            for name, family in families.items()
            if name in args.families
        },
        "provenance": provenance(Path(__file__)),
        "rows": results,
        "skipped": [],
    }
    emit_payload(payload, args.output)
    return 0 if all(row["correctness_pass"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
