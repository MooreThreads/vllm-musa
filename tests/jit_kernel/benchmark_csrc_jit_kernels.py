"""Microbenchmark MUSA csrc JIT kernels against native/torch baselines.

Run on a MUSA host from the vllm-musa repository root:

    python tests/jit_kernel/benchmark_csrc_jit_kernels.py

The numbers are microbenchmarks for the direct kernel wrappers. They are useful
for checking whether the opt-in csrc JIT path is faster for representative
tensor shapes; model-level throughput still needs an end-to-end benchmark.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

# torchada redirects torch.cuda symbols to MUSA. Keep this import first.
import torchada  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

import vllm_musa  # noqa: F401
import vllm_musa._custom_ops  # noqa: F401
from vllm_musa.jit_kernel.csrc import rope as _rope_module  # noqa: F401
from vllm_musa.jit_kernel.csrc.norm import (
    gemma_rmsnorm,
    rmsnorm,
)
from vllm_musa.jit_kernel.csrc.topk import topk_sigmoid, topk_softmax

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("VLLM_MUSA_JIT_CACHE_DIR", "/tmp/vllm_musa_bench_jit_cache")
os.environ.setdefault("VLLM_MUSA_ARCH_LIST", "31")


@dataclass(frozen=True)
class BenchCase:
    kernel: str
    label: str
    baseline: str
    iters: int
    baseline_event_us: float
    jit_event_us: float
    baseline_wall_us: float
    jit_wall_us: float

    @property
    def event_speedup(self) -> float:
        if self.jit_event_us == 0:
            return math.nan
        return self.baseline_event_us / self.jit_event_us

    @property
    def wall_speedup(self) -> float:
        if self.jit_wall_us == 0:
            return math.nan
        return self.baseline_wall_us / self.jit_wall_us


def _sync() -> None:
    torch.cuda.synchronize()


def _measure(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    _sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    wall_end = time.perf_counter()
    return (
        start.elapsed_time(end) * 1000.0 / iters,
        (wall_end - wall_start) * 1_000_000.0 / iters,
    )


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> None:
    _sync()
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        max_abs = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"max_abs={max_abs}")


def _rmsnorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    gemma: bool = False,
) -> torch.Tensor:
    scale_weight = weight.float() + (1.0 if gemma else 0.0)
    normalized = x.float() * torch.rsqrt(
        x.float().pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * scale_weight).to(x.dtype)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _bench_rmsnorm(
    *,
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> list[BenchCase]:
    torch.manual_seed(123)
    device = torch.device("musa")
    eps = 1e-6
    x = torch.randn((rows, hidden), device=device, dtype=dtype)
    weight = torch.randn((hidden,), device=device, dtype=dtype)

    _assert_close(rmsnorm(x, weight, eps), _rmsnorm_ref(x, weight, eps))
    _assert_close(
        gemma_rmsnorm(x, weight, eps),
        _rmsnorm_ref(x, weight, eps, gemma=True),
    )

    cases: list[BenchCase] = []
    label = f"rows={rows},hidden={hidden},dtype={dtype}"

    def torch_rms() -> None:
        torch.nn.functional.rms_norm(x, (hidden,), weight, eps)

    def jit_rms() -> None:
        rmsnorm(x, weight, eps)

    baseline_event_us, baseline_wall_us = _measure(
        torch_rms, warmup=warmup, iters=iters
    )
    jit_event_us, jit_wall_us = _measure(jit_rms, warmup=warmup, iters=iters)
    cases.append(
        BenchCase(
            "rmsnorm",
            label,
            "torch.nn.functional.rms_norm",
            iters,
            baseline_event_us,
            jit_event_us,
            baseline_wall_us,
            jit_wall_us,
        )
    )

    def torch_gemma() -> None:
        _rmsnorm_ref(x, weight, eps, gemma=True)

    def jit_gemma() -> None:
        gemma_rmsnorm(x, weight, eps)

    baseline_event_us, baseline_wall_us = _measure(
        torch_gemma, warmup=warmup, iters=iters
    )
    jit_event_us, jit_wall_us = _measure(jit_gemma, warmup=warmup, iters=iters)
    cases.append(
        BenchCase(
            "gemma_rmsnorm",
            label,
            "torch_reference",
            iters,
            baseline_event_us,
            jit_event_us,
            baseline_wall_us,
            jit_wall_us,
        )
    )

    return cases


def _topk_softmax_ref(
    gating: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = gating.float()
    if correction_bias is not None:
        logits = logits + correction_bias.float().unsqueeze(0)
    scores = torch.softmax(logits, dim=-1)
    _, ids = torch.topk(logits, topk, dim=-1)
    values = scores.gather(1, ids)
    if renormalize:
        values = values / values.sum(dim=-1, keepdim=True)
    return values.float(), ids.int()


def _topk_sigmoid_ref(
    gating: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(gating.float())
    choice = scores
    if correction_bias is not None:
        choice = choice + correction_bias.float().unsqueeze(0)
    _, ids = torch.topk(choice, topk, dim=-1)
    values = scores.gather(1, ids)
    if renormalize:
        values = values / values.sum(dim=-1, keepdim=True)
    return values.float(), ids.int()


def _topk_inputs(
    rows: int,
    experts: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("musa")
    values = torch.linspace(-4.0, 4.0, experts, device=device, dtype=torch.float32)
    perm = torch.randperm(experts, device=device)
    base = torch.empty((experts,), device=device, dtype=torch.float32)
    base[perm] = values
    row_offsets = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    gating = (base.unsqueeze(0) + row_offsets * 0.001).to(dtype)
    bias = torch.randn((experts,), device=device, dtype=torch.float32) * 0.01
    return gating, bias


def _bench_topk(
    *,
    rows: int,
    experts: int,
    topk: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> list[BenchCase]:
    torch.manual_seed(789)
    device = torch.device("musa")
    gating, bias = _topk_inputs(rows, experts, dtype)
    weights = torch.empty((rows, topk), device=device, dtype=torch.float32)
    ids = torch.empty((rows, topk), device=device, dtype=torch.int32)
    label = f"rows={rows},experts={experts},topk={topk},dtype={dtype}"

    topk_softmax(weights, ids, gating, renormalize=True)
    ref_weights, ref_ids = _topk_softmax_ref(gating, topk, True)
    _assert_close(weights, ref_weights, atol=3e-3, rtol=3e-3)
    if not torch.equal(ids.cpu(), ref_ids.cpu()):
        raise AssertionError("topk_softmax ids differ")

    def torch_softmax() -> None:
        _topk_softmax_ref(gating, topk, True)

    def jit_softmax() -> None:
        topk_softmax(weights, ids, gating, renormalize=True)

    baseline_event_us, baseline_wall_us = _measure(
        torch_softmax, warmup=warmup, iters=iters
    )
    jit_event_us, jit_wall_us = _measure(jit_softmax, warmup=warmup, iters=iters)
    cases = [
        BenchCase(
            "topk_softmax",
            label,
            "torch.softmax+topk",
            iters,
            baseline_event_us,
            jit_event_us,
            baseline_wall_us,
            jit_wall_us,
        )
    ]

    topk_sigmoid(weights, ids, gating, renormalize=True, correction_bias=bias)
    ref_weights, ref_ids = _topk_sigmoid_ref(gating, topk, True, bias)
    _assert_close(weights, ref_weights, atol=3e-3, rtol=3e-3)
    if not torch.equal(ids.cpu(), ref_ids.cpu()):
        raise AssertionError("topk_sigmoid ids differ")

    def torch_sigmoid_bias() -> None:
        _topk_sigmoid_ref(gating, topk, True, bias)

    def jit_sigmoid_bias() -> None:
        topk_sigmoid(weights, ids, gating, renormalize=True, correction_bias=bias)

    baseline_event_us, baseline_wall_us = _measure(
        torch_sigmoid_bias, warmup=warmup, iters=iters
    )
    jit_event_us, jit_wall_us = _measure(jit_sigmoid_bias, warmup=warmup, iters=iters)
    cases.append(
        BenchCase(
            "topk_sigmoid_bias",
            label,
            "torch.sigmoid+topk",
            iters,
            baseline_event_us,
            jit_event_us,
            baseline_wall_us,
            jit_wall_us,
        )
    )
    return cases


@dataclass(frozen=True)
class RopeShape:
    label: str
    num_heads_per_rank: int
    num_kv_heads_per_rank: int
    head_dim: int
    rotary_dim: int
    num_tokens: int
    is_neox: bool = True
    dtype: torch.dtype = torch.bfloat16


ROPE_SHAPES = [
    RopeShape("eagle3_draft_tp8_decode", 3, 1, 128, 128, 1),
    RopeShape("eagle3_draft_tp8_chain8", 3, 1, 128, 128, 8),
    RopeShape("m25_target_tp8_decode", 6, 1, 128, 128, 1),
    RopeShape("m25_target_tp8_prefill", 6, 1, 128, 128, 4096),
    RopeShape("qwen3_8b_tp8_decode", 4, 1, 128, 128, 1),
    RopeShape("qwen3_8b_tp8_prefill", 4, 1, 128, 128, 1024),
    RopeShape("qwen3_30b_tp2_decode", 16, 4, 128, 128, 1),
    RopeShape("min_heads_1_decode", 1, 1, 128, 128, 1),
    RopeShape("min_heads_2_decode", 2, 1, 128, 128, 1),
]


def _make_rope_inputs(shape: RopeShape, max_pos: int, device: str):
    torch.manual_seed(0)
    positions = torch.arange(shape.num_tokens, dtype=torch.int64, device=device)
    query = torch.randn(
        shape.num_tokens,
        shape.num_heads_per_rank * shape.head_dim,
        dtype=shape.dtype,
        device=device,
    )
    key = torch.randn(
        shape.num_tokens,
        shape.num_kv_heads_per_rank * shape.head_dim,
        dtype=shape.dtype,
        device=device,
    )
    with set_current_vllm_config(VllmConfig()):
        rope = RotaryEmbedding(
            head_size=shape.head_dim,
            rotary_dim=shape.rotary_dim,
            max_position_embeddings=max(max_pos, shape.num_tokens),
            base=10000.0,
            is_neox_style=shape.is_neox,
            dtype=shape.dtype,
            init_cache=True,
        ).to(device)
    return positions, query, key, rope


def _native_rope(rope, positions, query, key):
    return RotaryEmbedding.forward_static(
        positions,
        query,
        key,
        rope.head_size,
        rope.rotary_dim,
        rope.cos_sin_cache,
        rope.is_neox_style,
    )


def _jit_rope(positions, query, key, shape: RopeShape, cos_sin_cache):
    torch.ops.vllm.musa_rotary_embedding(
        positions,
        query,
        key,
        shape.head_dim,
        cos_sin_cache,
        shape.is_neox,
    )
    return query, key


def _assert_rope_close(
    shape: RopeShape,
    rope,
    positions,
    query,
    key,
) -> None:
    q_native, k_native = _native_rope(rope, positions, query.clone(), key.clone())
    q_jit = query.clone()
    k_jit = key.clone()
    cos_sin_cache = rope.cos_sin_cache.to(query.device, dtype=query.dtype)
    _jit_rope(positions, q_jit, k_jit, shape, cos_sin_cache)
    _assert_close(q_jit, q_native, atol=1e-2, rtol=1e-2)
    _assert_close(k_jit, k_native, atol=1e-2, rtol=1e-2)


def _default_rope_iters(num_tokens: int, quick: bool) -> int:
    if quick:
        return 20 if num_tokens >= 1024 else 100
    if num_tokens >= 4096:
        return 50
    if num_tokens >= 1024:
        return 100
    if num_tokens >= 64:
        return 300
    return 1000


def _default_rope_warmup(num_tokens: int, quick: bool) -> int:
    if quick:
        return 5
    if num_tokens >= 1024:
        return 20
    return 50


def _parse_csv_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_kernel_filter(value: str) -> set[str]:
    if value.strip().lower() in {"", "all"}:
        return {"norm", "topk", "rope"}
    kernels = {item.strip().lower() for item in value.split(",") if item.strip()}
    unknown = kernels - {"norm", "topk", "rope"}
    if unknown:
        raise ValueError(f"unknown kernel group(s): {', '.join(sorted(unknown))}")
    return kernels


def _bench_rope_shape(
    shape: RopeShape,
    *,
    device: str,
    max_pos: int,
    warmup: int | None,
    iters: int | None,
    quick: bool,
    reset_inputs: bool,
    check_correctness: bool,
) -> BenchCase:
    positions, query, key, rope = _make_rope_inputs(
        shape, max_pos=max_pos, device=device
    )
    cos_sin_cache = rope.cos_sin_cache.to(query.device, dtype=query.dtype)

    _jit_rope(positions, query.clone(), key.clone(), shape, cos_sin_cache)
    _sync()

    if check_correctness:
        _assert_rope_close(shape, rope, positions, query, key)

    shape_warmup = (
        warmup if warmup is not None else _default_rope_warmup(shape.num_tokens, quick)
    )
    shape_iters = (
        iters if iters is not None else _default_rope_iters(shape.num_tokens, quick)
    )

    def run_native() -> None:
        _native_rope(rope, positions, query, key)

    if reset_inputs:
        q_buf = torch.empty_like(query)
        k_buf = torch.empty_like(key)

        def run_jit() -> None:
            q_buf.copy_(query)
            k_buf.copy_(key)
            _jit_rope(positions, q_buf, k_buf, shape, cos_sin_cache)

    else:
        q_buf = query.clone()
        k_buf = key.clone()

        def run_jit() -> None:
            _jit_rope(positions, q_buf, k_buf, shape, cos_sin_cache)

    baseline_event_us, baseline_wall_us = _measure(
        run_native, warmup=shape_warmup, iters=shape_iters
    )
    jit_event_us, jit_wall_us = _measure(
        run_jit, warmup=shape_warmup, iters=shape_iters
    )
    label = (
        f"{shape.label},tokens={shape.num_tokens},q_heads={shape.num_heads_per_rank},"
        f"kv_heads={shape.num_kv_heads_per_rank}"
    )
    return BenchCase(
        "rope",
        label,
        "vllm.forward_static",
        shape_iters,
        baseline_event_us,
        jit_event_us,
        baseline_wall_us,
        jit_wall_us,
    )


def _bench_rope(
    *,
    device: str,
    max_pos: int,
    warmup: int | None,
    iters: int | None,
    shape_filter: str | None,
    quick: bool,
    reset_inputs: bool,
    check_correctness: bool,
) -> list[BenchCase]:
    selected = _parse_csv_filter(shape_filter)
    unknown = selected - {shape.label for shape in ROPE_SHAPES} if selected else set()
    if unknown:
        raise ValueError(f"unknown rope shape label(s): {', '.join(sorted(unknown))}")

    cases: list[BenchCase] = []
    for shape in ROPE_SHAPES:
        if selected and shape.label not in selected:
            continue
        cases.append(
            _bench_rope_shape(
                shape,
                device=device,
                max_pos=max_pos,
                warmup=warmup,
                iters=iters,
                quick=quick,
                reset_inputs=reset_inputs,
                check_correctness=check_correctness,
            )
        )
    return cases


def _print_results(results: list[BenchCase]) -> None:
    print(
        "kernel                      label                                      "
        "baseline                         iters  baseline_event_us  jit_event_us  "
        "event_speedup  baseline_wall_us  jit_wall_us  wall_speedup"
    )
    for case in results:
        print(
            f"{case.kernel:<27} {case.label:<42} {case.baseline:<32} "
            f"{case.iters:>5} {case.baseline_event_us:>18.3f} "
            f"{case.jit_event_us:>12.3f} {case.event_speedup:>14.2f} "
            f"{case.baseline_wall_us:>17.3f} {case.jit_wall_us:>11.3f} "
            f"{case.wall_speedup:>12.2f}"
        )
        print(
            "BENCH "
            f"kernel={case.kernel} label={case.label.replace(' ', '_')} "
            f"baseline={case.baseline.replace(' ', '_')} iters={case.iters} "
            f"baseline_event_us={case.baseline_event_us:.3f} "
            f"jit_event_us={case.jit_event_us:.3f} "
            f"event_speedup={case.event_speedup:.3f} "
            f"baseline_wall_us={case.baseline_wall_us:.3f} "
            f"jit_wall_us={case.jit_wall_us:.3f} "
            f"wall_speedup={case.wall_speedup:.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark MUSA csrc JIT kernels against native/torch baselines."
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--kernels",
        default="norm,topk,rope",
        help="Comma-separated kernel groups to run: norm, topk, rope, or all.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rope-max-pos", type=int, default=8192)
    parser.add_argument("--rope-warmup", type=int, default=None)
    parser.add_argument("--rope-iters", type=int, default=None)
    parser.add_argument(
        "--rope-shapes",
        default=None,
        help="Comma-separated RoPE shape labels to run. Defaults to all.",
    )
    parser.add_argument(
        "--rope-quick",
        action="store_true",
        help="Use short warmup/iteration counts for RoPE smoke validation.",
    )
    parser.add_argument(
        "--rope-reset-inputs",
        action="store_true",
        help="Copy pristine inputs into the RoPE JIT buffers before every iteration.",
    )
    parser.add_argument(
        "--rope-skip-correctness",
        action="store_true",
        help="Skip the one-shot native-vs-JIT RoPE output check before timing.",
    )
    args = parser.parse_args()

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        print("FAIL no MUSA device available")
        return 1
    kernels = _parse_kernel_filter(args.kernels)
    torch.cuda.set_device(args.device)
    dtype = _dtype_from_name(args.dtype)

    results: list[BenchCase] = []
    if "norm" in kernels:
        results.extend(
            _bench_rmsnorm(
                rows=args.rows,
                hidden=args.hidden,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
            )
        )
    if "topk" in kernels:
        results.extend(
            _bench_topk(
                rows=args.rows,
                experts=args.experts,
                topk=args.topk,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
            )
        )
    if "rope" in kernels:
        results.extend(
            _bench_rope(
                device=args.device,
                max_pos=args.rope_max_pos,
                warmup=args.rope_warmup,
                iters=args.rope_iters,
                shape_filter=args.rope_shapes,
                quick=args.rope_quick,
                reset_inputs=args.rope_reset_inputs,
                check_correctness=not args.rope_skip_correctness,
            )
        )
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
