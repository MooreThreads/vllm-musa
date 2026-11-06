#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-cache three-arm benchmark of the sparse-indexer top-k on MUSA.

The question this answers: DeepSelect publishes speedups against ``torch.topk``, but
vLLM-MUSA does not call ``torch.topk`` for the indexer top-k — it calls its own radix
kernel. This script measures all three arms on the same tensors, in one process, with
the same cold-cache protocol, so the comparison is apples to apples:

  ``torch``      ``torch.topk`` — the baseline both other arms quote against.
  ``musa``       ``deepseek_v4_sparse_indexer_topk_decode`` (topk <= 512, staged radix)
                 or ``sparse_indexer_topk_decode`` above that, i.e. what vLLM-MUSA
                 runs today.
  ``deepselect`` ``deep_select.topk`` from the native MUSA build.

Protocol (copied from ``DeepSelect/benchmarks/musa_library_bench.py`` so the numbers
line up with the README): an 8 GiB buffer is zeroed before every timed launch, 10
warmup launches, 10 timed launches per round, 3 rounds interleaved A/B, median. The
metric is effective bandwidth — ``(input bytes + index output bytes) / kernel time`` —
because top-k does no floating-point math.

Every arm must select the same index *set* per row (order may differ; ties are broken
arbitrarily). A fast-but-wrong arm fails the run instead of printing a speedup.

Runs on a MUSA container with ``vllm_musa`` built (for the ``musa`` arm) and, for the
``deepselect`` arm, the native extension installed per ``third_party/PINS``.

Example:

    python tools/bench_indexer_topk_arms.py --batch 6 512 4096 \\
        --vocab 16384 65536 131072 262144 524288 1048576 --topk 512 \\
        --json-out /tmp/indexer_topk_arms.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Callable, Optional

import torchada
import torch


def _flush_buffer(gib: float) -> Optional[torch.Tensor]:
    if gib <= 0:
        return None
    elements = int(gib * (1 << 30)) // 4  # float32
    return torch.empty(elements, dtype=torch.float32, device="musa")


def _sample(fn: Callable[[], None], iters: int, flush: Optional[torch.Tensor]) -> float:
    starts = [torch.musa.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.musa.Event(enable_timing=True) for _ in range(iters)]
    for index in range(iters):
        if flush is not None:
            flush.zero_()
        starts[index].record()
        fn()
        ends[index].record()
    torch.musa.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def _median_ms(
    fn: Callable[[], None], *, warmup: int, iters: int, rounds: int,
    flush: Optional[torch.Tensor],
) -> float:
    for _ in range(warmup):
        if flush is not None:
            flush.zero_()
        fn()
    torch.musa.synchronize()
    samples = []
    for _ in range(rounds):
        samples.append(_sample(fn, iters, flush))
    return statistics.median(samples)


def _tbps(batch: int, vocab: int, topk: int, ms: float) -> float:
    if ms <= 0:
        return 0.0
    bytes_moved = batch * vocab * 2 + batch * topk * 4  # bf16 in, int32 index out
    return bytes_moved / (ms * 1e-3) / 1e12


def _selected_values(x: torch.Tensor, idx: torch.Tensor) -> list[torch.Tensor]:
    """Per-row selected values, sorted, so tie-breaking does not matter.

    Comparing index *sets* would abort on ties: bf16 has 8 mantissa bits and a 1M-wide
    row has thousands of tie pairs, so two correct kernels legitimately return
    different (equally valid) index sets.
    """
    flat = torch.gather(x, 1, idx.to(torch.int64).clamp_min(0))
    return [torch.sort(row).values for row in flat]


def _check_same_values(
    name: str, want: list[torch.Tensor], x: torch.Tensor, got: torch.Tensor
) -> None:
    for row, expected in enumerate(want):
        actual = torch.sort(x[row].gather(0, got[row].to(torch.int64))).values
        if not torch.equal(actual, expected):
            mismatch = int((actual != expected).sum().item())
            raise SystemExit(
                f"{name}: row {row} selected different values ({mismatch} of "
                f"{expected.numel()} differ) — refusing to time it"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[6, 512, 4096])
    parser.add_argument(
        "--vocab",
        type=int,
        nargs="+",
        default=[16384, 65536, 131072, 262144, 524288, 1048576],
    )
    parser.add_argument("--topk", type=int, nargs="+", default=[512])
    parser.add_argument("--flush-gb", type=float, default=8.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["torch", "musa", "deepselect"],
        choices=["torch", "musa", "deepselect"],
    )
    args = parser.parse_args(argv)

    import torch_musa  # noqa: F401  (registers the device)

    flush = _flush_buffer(args.flush_gb)

    deep_select = None
    if "deepselect" in args.arms:
        try:
            import deep_select  # noqa: F401
        except ImportError as error:
            print(f"skip deepselect arm: {error}", file=sys.stderr)
            args.arms = [a for a in args.arms if a != "deepselect"]

    musa_topk = None
    if "musa" in args.arms:
        try:
            import vllm_musa  # noqa: F401
            import vllm_musa._custom_ops as musa_ops
        except ImportError as error:
            print(f"skip musa arm: {error}", file=sys.stderr)
            args.arms = [a for a in args.arms if a != "musa"]
        else:
            def _musa_arm(topk: int):
                """What vLLM-MUSA itself calls for this topk (series patch 0032)."""
                if topk <= 512:
                    return musa_ops.deepseek_v4_sparse_indexer_topk_decode
                return musa_ops.sparse_indexer_topk_decode

    if not args.arms:
        print("no arm available", file=sys.stderr)
        return 1

    rows = []
    torch.manual_seed(args.seed)
    for batch in args.batch:
        for vocab in args.vocab:
            for topk in args.topk:
                if topk > vocab:
                    continue
                x = torch.randn(batch, vocab, dtype=torch.bfloat16, device="musa")
                seq_lens = torch.full((batch,), vocab, dtype=torch.int32, device="musa")
                want = _selected_values(
                    x, torch.topk(x, topk, dim=-1, sorted=False).indices
                )
                entry = {"batch": batch, "vocab": vocab, "topk": topk, "arms": {}}

                if "torch" in args.arms:
                    entry["arms"]["torch"] = _median_ms(
                        lambda: torch.topk(x, topk, dim=-1, sorted=False),
                        warmup=args.warmup, iters=args.iters, rounds=args.rounds,
                        flush=flush,
                    )

                if "musa" in args.arms:
                    musa_topk = _musa_arm(topk)
                    out = torch.full((batch, topk), -1, dtype=torch.int32, device="musa")

                    def _musa() -> None:
                        musa_topk(x, seq_lens, out, topk)

                    _musa()
                    torch.musa.synchronize()
                    _check_same_values("musa", want, x, out)
                    entry["arms"]["musa"] = _median_ms(
                        _musa, warmup=args.warmup, iters=args.iters,
                        rounds=args.rounds, flush=flush,
                    )

                if "deepselect" in args.arms:
                    import deep_select as ds

                    def _ds() -> None:
                        ds.topk(
                            x,
                            topk,
                            end=seq_lens,
                            indices_type=torch.int32,
                            return_value=False,
                        )

                    _, got = ds.topk(
                        x, topk, end=seq_lens,
                        indices_type=torch.int32, return_value=False,
                    )
                    torch.musa.synchronize()
                    _check_same_values("deepselect", want, x, got)
                    entry["arms"]["deepselect"] = _median_ms(
                        _ds, warmup=args.warmup, iters=args.iters,
                        rounds=args.rounds, flush=flush,
                    )

                base = entry["arms"].get("torch")
                for arm, ms in entry["arms"].items():
                    entry.setdefault("derived", {})[arm] = {
                        "ms": round(ms, 6),
                        "tbps": round(_tbps(batch, vocab, topk, ms), 4),
                        "vs_torch": round(base / ms, 3) if base else None,
                    }
                rows.append(entry)
                print(json.dumps(entry))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"wrote {args.json_out}", file=sys.stderr)

    print("\n| batch | vocab | topk | torch ms | musa ms | deepselect ms | DS/musa |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for entry in rows:
        arms = entry.get("derived", {})
        torch_ms = arms.get("torch", {}).get("ms")
        musa_ms = arms.get("musa", {}).get("ms")
        ds_ms = arms.get("deepselect", {}).get("ms")
        ratio = (
            round(musa_ms / ds_ms, 3) if musa_ms and ds_ms else None
        )
        fmt = lambda v: "—" if v is None else f"{v:.3f}"
        print(
            f"| {entry['batch']} | {entry['vocab']} | {entry['topk']} | "
            f"{fmt(torch_ms)} | {fmt(musa_ms)} | {fmt(ds_ms)} | {fmt(ratio)} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
