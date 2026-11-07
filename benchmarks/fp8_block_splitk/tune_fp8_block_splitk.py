#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-L2 tuner for the split-K FP8 GEMM rows of vllm_musa/tuning/fp8_block_splitk.json.

Every timing runs the production linear (MUSAFp8BlockGemvScaledMMKernel.apply_weights)
on a 32x32 block-FP8 weight of the per-rank shape. The split-K row for the shape is
swapped in the loaded table: a candidate config admitted for every row times the
split-K GEMM; no row times what the dispatch runs without it (GEMV, BF16 GEMM or
block GEMM). Both modes time direct calls with the dispatch sent down its graph-capture
or its eager branch; an L2 flush precedes every call, so the call is queued behind it
and its events time the kernels. The two sides alternate within each round. A row is
inside the band unless split-K is slower than the dispatch without it by more than the
tie margin.

Per shape the output also holds the table row this card gets, by the rule in
table_rows.py. ``--write`` merges the rows into the table when the run ends;
table_rows.py does the same from saved output and needs no torch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

# isort: off
import torchada  # noqa: F401  # must patch before torch ecosystem imports
import torch

# isort: on

from mate.testing.utils import bench_gpu_time_with_musa_event
from table_rows import TABLE, report_lines, table_row, write

# Coarse rows for the band sweeps; edges are bisected between a win and a loss.
COARSE_ROWS = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    80,
    96,
    128,
    160,
    192,
    224,
    256,
    320,
    384,
    448,
    511,
)
# Decode row counts that rank the candidate configs, in graph capture.
RANK_ROWS = (2, 4, 8, 16, 32, 48, 64)
SCREEN_ROWS = (4, 16, 48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", default="table", help="'table' (the default rows) or N:K[,N:K...]"
    )
    parser.add_argument(
        "--search",
        choices=("full", "none"),
        default="full",
        help="'none' measures only the shipped default config",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=6,
        help="configs that pass the screen into the ranking",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat-iters", type=int, default=20)
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument("--l2-flush-mb", type=int, default=8192)
    parser.add_argument("--max-capture-rows", type=int, default=256)
    parser.add_argument("--max-eager-rows", type=int, default=511)
    parser.add_argument(
        "--tie",
        type=float,
        default=0.03,
        help="split-K this much slower still counts as a tie",
    )
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument(
        "--output", type=Path, required=True, help="where the measurements go"
    )
    parser.add_argument(
        "--write",
        nargs="?",
        const=TABLE,
        type=Path,
        help="merge this card's rows into the table when the run ends",
    )
    return parser.parse_args()


def _git(cwd: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, text=True, capture_output=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# The code a measurement depends on, hashed so a record names it without git.
SOURCES = (
    "kernels/fp8_block_splitk.py",
    "kernels/fp8_block_gemv.py",
    "kernels/fp8_block_bf16.py",
    "model_executor/kernels/linear/scaled_mm/block_gemv.py",
    "tuning/fp8_block_splitk.json",
)


def provenance() -> dict:
    import vllm_musa

    package = Path(vllm_musa.__file__).resolve().parent
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    versions = {}
    for dist in ("torch", "torch_musa", "triton", "mate", "torchada", "vllm"):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    status = _git(package, "status", "--porcelain")
    return {
        "device": props.name,
        "capability": [props.major, props.minor],
        "multiprocessor_count": props.multi_processor_count,
        "musa": getattr(torch.version, "musa", None),
        "vllm_musa_path": str(package),
        "vllm_musa_commit": _git(package, "rev-parse", "HEAD"),
        "vllm_musa_dirty": None if status is None else bool(status),
        "sources_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
        "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_MUSA_")},
        "versions": versions,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def build_linear(n: int, k: int, seed: int):
    """The production block-FP8 linear and a layer holding an (n, k) 32x32 weight."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import FP8ScaledMMLinearLayerConfig
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        QuantKey,
        ScaleDesc,
    )

    from vllm_musa.model_executor.kernels.linear.scaled_mm.block_gemv import (
        MUSAFp8BlockGemvScaledMMKernel,
    )

    gen = torch.Generator(device="musa").manual_seed(seed)
    weight = (torch.randn((n, k), generator=gen, device="musa") * 0.5).to(
        torch.float8_e4m3fn
    )
    # Power-of-two block scales, as the checkpoints use.
    exponents = torch.randint(-8, -4, (n // 32, k // 32), generator=gen, device="musa")
    scale = torch.pow(2.0, exponents.float())
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=QuantKey(
            torch.float8_e4m3fn, ScaleDesc(torch.float32, True, GroupShape(32, 32))
        ),
        activation_quant_key=QuantKey(
            torch.float8_e4m3fn, ScaleDesc(torch.float32, False, GroupShape(1, 32))
        ),
        weight_shape=(n, k),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    with set_current_vllm_config(VllmConfig()):
        kernel = MUSAFp8BlockGemvScaledMMKernel(config)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
        kernel.process_weights_after_loading(layer)
    return kernel, layer


@contextmanager
def splitk_row(n: int, k: int, config: tuple | None):
    """Admit ``config`` for every row of (n, k), or no split-K row with None."""
    from vllm_musa.kernels import fp8_block_splitk as sk

    saved = (sk._ROWS, sk._CONFIGS, sk._EAGER_MAX_ROWS, sk._BINNED_SHAPES)
    rows = {key: row for key, row in sk._ROWS.items() if key[:2] != (n, k)}
    configs = {s: c for s, c in sk._CONFIGS.items() if s != (n, k)}
    eager = {s: e for s, e in sk._EAGER_MAX_ROWS.items() if s != (n, k)}
    if config is not None:
        wide = (*config[:4], 1, 1 << 20, config[6])
        rows[(n, k, None)] = (wide, 1 << 20)
        configs[(n, k)] = wide
        eager[(n, k)] = 1 << 20
    sk._ROWS, sk._CONFIGS, sk._EAGER_MAX_ROWS = rows, configs, eager
    sk._BINNED_SHAPES = frozenset()
    try:
        yield
    finally:
        sk._ROWS, sk._CONFIGS, sk._EAGER_MAX_ROWS, sk._BINNED_SHAPES = saved


class Bench:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def _times(self, fn) -> list[float]:
        return list(
            bench_gpu_time_with_musa_event(
                fn,
                dry_run_iters=self.args.dry_run_iters,
                repeat_iters=self.args.repeat_iters,
                l2_flush=True,
                l2_flush_size_mb=self.args.l2_flush_mb,
            )
        )

    @staticmethod
    @contextmanager
    def dispatch(capture: bool):
        """Send the dispatch down its graph-capture or its eager branch."""
        real = torch.cuda.is_current_stream_capturing
        torch.cuda.is_current_stream_capturing = lambda: capture
        try:
            yield
        finally:
            torch.cuda.is_current_stream_capturing = real

    def pair(
        self,
        kernel,
        layer,
        n: int,
        k: int,
        rows: int,
        config: tuple,
        capture: bool,
        rounds: int | None = None,
    ) -> dict:
        """Interleaved medians (ms) of split-K with ``config`` and of the dispatch without it."""
        x = torch.randn((rows, k), device="musa").to(torch.bfloat16)

        def splitk():
            with splitk_row(n, k, config):
                kernel.apply_weights(layer, x)

        def ref():
            with splitk_row(n, k, None):
                kernel.apply_weights(layer, x)

        sk_times, ref_times = [], []
        with self.dispatch(capture):
            for _ in range(rounds or self.args.rounds):
                sk_times += self._times(splitk)
                ref_times += self._times(ref)
        return {
            "rows": rows,
            "splitk_ms": statistics.median(sk_times),
            "ref_ms": statistics.median(ref_times),
            "splitk_iqr": _iqr_ratio(sk_times),
            "ref_iqr": _iqr_ratio(ref_times),
        }


def _iqr_ratio(times: list[float]) -> float:
    q = statistics.quantiles(times, n=4)
    return (q[2] - q[0]) / q[1] if q[1] else math.inf


def _launch_params(config: tuple) -> tuple:
    """``config`` with its row band zeroed, as the candidates carry it."""
    return (*config[:4], 0, 0, config[6])


def candidates(n: int, k: int) -> list[tuple]:
    out = []
    for split, block_n, warps, unroll, tile_m in itertools.product(
        (2, 4, 8, 16, 32), (32, 64, 128), (4, 8), (1, 2), (16, 32)
    ):
        if k % (split * 32 * unroll) == 0:
            out.append((split, block_n, warps, unroll, 0, 0, tile_m))
    return out


def rank(
    bench: Bench,
    kernel,
    layer,
    n: int,
    k: int,
    configs: list[tuple],
    default: tuple,
    log: list,
) -> list[dict]:
    """Geometric-mean split-K/ref ratio over the decode rows, best first."""

    def score(config, rows, rounds=None):
        ratios = []
        for r in rows:
            m = bench.pair(kernel, layer, n, k, r, config, capture=True, rounds=rounds)
            log.append({"stage": "rank", "config": list(config), **m})
            ratios.append(m["splitk_ms"] / m["ref_ms"])
        return math.exp(statistics.fmean(math.log(v) for v in ratios))

    screened = []
    for config in configs:
        try:
            screened.append((score(config, SCREEN_ROWS, rounds=1), config))
        except Exception as e:  # noqa: BLE001 - a config the kernel cannot run
            log.append({"stage": "screen", "config": list(config), "error": repr(e)})
    screened.sort()
    finalists = [c for _, c in screened[: bench.args.top]]
    default = _launch_params(default)
    if default not in finalists:
        finalists.append(default)
    ranked = [{"config": list(c), "score": score(c, RANK_ROWS)} for c in finalists]
    return sorted(ranked, key=lambda r: r["score"])


def band(
    bench: Bench,
    kernel,
    layer,
    n: int,
    k: int,
    config: tuple,
    log: list,
    capture_limit: int,
    eager_limit: int,
) -> dict:
    """MIN_ROWS and the last row of the winning run, per mode, for ``config``."""
    cache: dict = {}

    def wins(rows: int, capture: bool) -> bool:
        key = (rows, capture)
        if key not in cache:
            m = bench.pair(kernel, layer, n, k, rows, config, capture)
            log.append(
                {
                    "stage": "band",
                    "config": list(config),
                    "mode": "capture" if capture else "eager",
                    **m,
                }
            )
            cache[key] = m["splitk_ms"] <= m["ref_ms"] * (1 + bench.args.tie)
        return cache[key]

    # MIN_ROWS: the first row from 2 that wins with the next two rows, in capture.
    min_rows = next(
        (r for r in range(2, 17) if all(wins(r + d, True) for d in range(3))), None
    )
    if min_rows is None:
        return {"min_rows": None, "capture": None, "eager": None}
    result = {"min_rows": min_rows}
    for mode, capture, limit in (
        ("capture", True, capture_limit),
        ("eager", False, eager_limit),
    ):
        coarse = [r for r in COARSE_ROWS if min_rows <= r < limit] + [limit]
        last_win, first_loss = min_rows, None
        for r in coarse:
            if wins(r, capture):
                last_win = r
            else:
                first_loss = r
                break
        if first_loss is None:
            result[mode] = last_win
            continue
        lo, hi = last_win, first_loss
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if wins(mid, capture):
                lo = mid
            else:
                hi = mid
        result[mode] = lo
    return result


def check_outputs(kernel, layer, n: int, k: int, config: tuple) -> float:
    """Largest relative error of split-K against the dispatch without it."""
    worst = 0.0
    for rows in (2, 17, 64):
        x = torch.randn((rows, k), device="musa").to(torch.bfloat16)
        with splitk_row(n, k, config):
            got = kernel.apply_weights(layer, x).float()
        with splitk_row(n, k, None):
            ref = kernel.apply_weights(layer, x).float()
        error = ((got - ref).norm() / ref.norm()).item()
        if not math.isfinite(error):
            return math.inf
        worst = max(worst, error)
    return worst


def main() -> None:
    args = parse_args()
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm_musa.kernels import fp8_block_bf16 as bf
    from vllm_musa.kernels import fp8_block_splitk as sk

    if not sk._ENABLED:
        raise SystemExit(
            "VLLM_MUSA_FP8_SPLITK_MM=0 disables the split-K GEMM under test"
        )

    torch.manual_seed(args.seed)
    if args.shapes == "table":
        shapes = sorted(sk._CONFIGS)
    else:
        shapes = [tuple(int(v) for v in s.split(":")) for s in args.shapes.split(",")]
    prov = provenance()
    mp = prov["multiprocessor_count"]
    bench = Bench(args)
    report = {
        "provenance": prov,
        "args": {k: str(v) for k, v in vars(args).items()},
        "shapes": [],
    }
    for n, k in shapes:
        started = time.time()
        log: list = []
        kernel, layer = build_linear(n, k, args.seed)
        default = sk._CONFIGS.get((n, k))
        default_eager = sk._EAGER_MAX_ROWS.get((n, k))
        # The BF16 GEMM thresholds stay as they are: split-K rows end below them.
        capture_limit = min(
            args.max_capture_rows, bf._MIN_ROWS_BY_SHAPE.get((n, k), bf.MIN_ROWS) - 1
        )
        eager_limit = min(args.max_eager_rows, bf.MIN_ROWS - 1)
        shape = {
            "n": n,
            "k": k,
            "capture_limit": capture_limit,
            "eager_limit": eager_limit,
        }
        if default is not None:
            shape["default"] = {
                "config": list(default),
                "table": {
                    "min_rows": default[4],
                    "capture": default[5],
                    "eager": default_eager,
                },
                "measured": band(
                    bench, kernel, layer, n, k, default, log, capture_limit, eager_limit
                ),
            }
        if args.search == "full":
            ranked = rank(
                bench,
                kernel,
                layer,
                n,
                k,
                candidates(n, k),
                default or candidates(n, k)[0],
                log,
            )
            best = tuple(ranked[0]["config"])
            shape["ranked"] = ranked
            shape["best"] = {
                "config": list(best),
                "measured": band(
                    bench, kernel, layer, n, k, best, log, capture_limit, eager_limit
                ),
                "relative_error": check_outputs(kernel, layer, n, k, best),
            }
        shape["log"] = log
        shape["row"], shape["row_reasons"] = table_row(shape, mp)
        shape["seconds"] = round(time.time() - started, 1)
        report["shapes"].append(shape)
        args.output.write_text(json.dumps(report, indent=1) + "\n")
        print(
            json.dumps(
                {
                    key: shape.get(key)
                    for key in (
                        "n",
                        "k",
                        "default",
                        "best",
                        "row",
                        "row_reasons",
                        "seconds",
                    )
                }
            ),
            flush=True,
        )
    if args.write:
        result = write(args.write, [report])
        print("\n".join(report_lines(result["decisions"], result["proposals"])))
        print(json.dumps({"write": str(args.write), "changes": result["changes"]}))


if __name__ == "__main__":
    main()
