#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduce MP tactic run bundles into conservative promotion candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from _benchmark_utils import effective_gemv_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--prediction-input", type=Path)
    parser.add_argument(
        "--minimum-host-replicates",
        type=int,
        default=1,
        help="local qualification requirement; not stored in the public recipe",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def result_path(run_dir: Path, receipt: dict[str, Any]) -> Path:
    recorded = Path(receipt["result"])
    if recorded.is_file():
        return recorded
    local = run_dir / recorded.name
    if not local.is_file():
        raise FileNotFoundError(f"result missing for {receipt['run_id']}: {local}")
    return local


def row_key(schema: str, row: dict[str, Any]) -> tuple[Any, ...]:
    if schema == "musa-fused-gemv-moe-aot-paired-ab.v2":
        return (
            row["family"],
            int(row["tokens"]),
            row["route"],
            row["stage"],
        )
    if schema == "musa-fused-add-rmsnorm-aot-paired-ab.v2":
        return (int(row["hidden_size"]), int(row["rows"]))
    if schema == "musa-dsv4-mhc-jit-aot-paired-ab.v1":
        return (
            row.get("production_path", "standalone"),
            int(row["tokens"]),
            int(row["split_k"]),
        )
    raise ValueError(f"unsupported result schema: {schema}")


def tactic(schema: str, row: dict[str, Any]) -> str:
    if schema == "musa-fused-gemv-moe-aot-paired-ab.v2":
        requested = tuple(row["candidate_block"])
        effective, _applied, _reason = effective_gemv_block(
            row["family"], int(row["tokens"]), row["stage"], requested
        )
        return "x".join(str(value) for value in row.get("effective_block", effective))
    if schema == "musa-fused-add-rmsnorm-aot-paired-ab.v2":
        return str(row["candidate_block_x"])
    if schema == "musa-dsv4-mhc-jit-aot-paired-ab.v1":
        return "x".join(
            str(row[field]) for field in ("threads", "hidden_block", "pass_config")
        )
    raise ValueError(f"unsupported result schema: {schema}")


def key_dict(schema: str, key: tuple[Any, ...]) -> dict[str, Any]:
    if schema == "musa-fused-gemv-moe-aot-paired-ab.v2":
        family, tokens, route, stage = key
        return {
            "family": family,
            "tokens": tokens,
            "route": route,
            "stage": stage,
        }
    if schema == "musa-fused-add-rmsnorm-aot-paired-ab.v2":
        hidden_size, rows = key
        return {"hidden_size": hidden_size, "rows": rows}
    if schema == "musa-dsv4-mhc-jit-aot-paired-ab.v1":
        production_path, tokens, split_k = key
        return {
            "production_path": production_path,
            "tokens": tokens,
            "split_k": split_k,
        }
    raise ValueError(f"unsupported result schema: {schema}")


def mp_only_key(schema: str, key: tuple[Any, ...]) -> tuple[Any, ...]:
    if schema == "musa-fused-gemv-moe-aot-paired-ab.v2":
        family, tokens, _route, stage = key
        return (family, tokens, stage)
    return key


def promotion_pass(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    requested_applied = row.get("requested_block_applied", True)
    if (
        requested_applied
        and row.get("schema") == "musa-fused-gemv-moe-aot-paired-ab.v2"
    ):
        requested = tuple(row["candidate_block"])
        _effective, requested_applied, _reason = effective_gemv_block(
            row["family"], int(row["tokens"]), row["stage"], requested
        )
    return bool(
        row.get("correctness_pass")
        and not row.get("poison_output", False)
        and requested_applied
        and row.get("is_production_split", True)
        and float(row["median_ratio"]) <= float(gate["median_ratio_max"])
        and float(row["ratio_p95"]) <= float(gate["ratio_p95_max"])
    )


def amdahl_prediction(
    ratio: float, time_share: float | None, hit_rate: float | None
) -> float | None:
    if time_share is None or hit_rate is None:
        return None
    affected = time_share * hit_rate
    new_total = 1.0 - affected + affected * ratio
    return (1.0 / new_total - 1.0) * 100.0


def main() -> int:
    args = parse_args()
    predictions = (
        load_json(args.prediction_input) if args.prediction_input is not None else {}
    )
    observations: list[dict[str, Any]] = []
    campaign_sha256: str | None = None
    campaign: dict[str, Any] | None = None

    for run_dir_argument in args.run_dirs:
        run_dir = run_dir_argument.resolve()
        manifest = load_json(run_dir / "run-manifest.json")
        if manifest.get("schema") != "vllm-musa-mp-tactic-run.v1":
            raise ValueError(f"unsupported run manifest: {run_dir}")
        current_sha = manifest["campaign_sha256"]
        if campaign_sha256 is None:
            campaign_sha256 = current_sha
            campaign = load_json(run_dir / "campaign.json")
        elif current_sha != campaign_sha256:
            raise ValueError("cannot combine different campaign definitions")
        mp = int(manifest["expected_mp"])
        evidence_eligible = bool(manifest.get("lease_isolated", False)) and not bool(
            manifest.get("exploratory", True)
        )
        for receipt in manifest["runs"]:
            if receipt.get("error") is not None or receipt.get("returncode") != 0:
                continue
            result = load_json(result_path(run_dir, receipt))
            schema = result["schema"]
            grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in result["rows"]:
                grouped[row_key(schema, row)].append(row)
            for key, rows in grouped.items():
                winner = min(rows, key=lambda item: float(item["median_ratio"]))
                observations.append(
                    {
                        "mp": mp,
                        "cell": receipt["cell"],
                        "variant": receipt["variant"],
                        "schema": schema,
                        "family": row.get("family"),
                        "tokens": row.get("tokens"),
                        "stage": row.get("stage"),
                        "candidate_block": row.get("candidate_block"),
                        "key": key,
                        "key_fields": key_dict(schema, key),
                        "tactic": tactic(schema, winner),
                        "median_ratio": float(winner["median_ratio"]),
                        "ratio_p95": float(winner["ratio_p95"]),
                        "speedup_pct": float(winner["speedup_pct"]),
                        "correctness_pass": bool(winner["correctness_pass"]),
                        "poison_output": bool(winner.get("poison_output", False)),
                        "hostname": result.get("provenance", {}).get("hostname"),
                        "evidence_eligible": evidence_eligible,
                        "result": str(result_path(run_dir, receipt)),
                    }
                )

    assert campaign is not None and campaign_sha256 is not None
    gate = campaign["methodology"]["promotion_gate"]
    legacy_minimum_hosts = campaign.get("hardware", {}).get(
        "minimum_host_replicates", {}
    )
    if args.minimum_host_replicates < 1:
        raise ValueError("--minimum-host-replicates must be positive")
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_key[
            (
                observation["mp"],
                observation["cell"],
                observation["schema"],
                observation["key"],
            )
        ].append(observation)

    route_candidates: list[dict[str, Any]] = []
    for (mp, cell, schema, key), legs in sorted(by_key.items(), key=str):
        tactics = sorted({leg["tactic"] for leg in legs})
        hosts = sorted({leg["hostname"] for leg in legs if leg["hostname"]})
        # Historical v1 bundles may carry a per-bin host requirement.  New
        # public recipes deliberately do not: fleet-specific replication is a
        # local run policy supplied on the command line.
        required_hosts = int(
            legacy_minimum_hosts.get(str(mp), args.minimum_host_replicates)
        )
        stable = len(tactics) == 1
        all_pass = all(promotion_pass(leg, gate) for leg in legs)
        enough_hosts = len(hosts) >= required_hosts
        every_leg_eligible = all(leg["evidence_eligible"] for leg in legs)
        worst_median_ratio = max(leg["median_ratio"] for leg in legs)
        worst_p95_ratio = max(leg["ratio_p95"] for leg in legs)
        prediction = predictions.get(cell, {})
        time_share = prediction.get("op_time_share")
        hit_rate = prediction.get("production_hit_rate")
        route_candidates.append(
            {
                "mp": mp,
                "cell": cell,
                "schema": schema,
                "key": key_dict(schema, key),
                "winner": tactics[0] if stable else None,
                "observed_winners": tactics,
                "variants": sorted({leg["variant"] for leg in legs}),
                "hosts": hosts,
                "required_hosts": required_hosts,
                "worst_median_ratio": worst_median_ratio,
                "worst_p95_ratio": worst_p95_ratio,
                "stable_across_runs": stable,
                "all_legs_pass": all_pass,
                "host_replication_pass": enough_hosts,
                "evidence_eligible": every_leg_eligible,
                "route_candidate_pass": (
                    stable and all_pass and enough_hosts and every_leg_eligible
                ),
                "prediction": {
                    "op_time_share": time_share,
                    "production_hit_rate": hit_rate,
                    "predicted_e2e_speedup_pct": amdahl_prediction(
                        worst_median_ratio, time_share, hit_rate
                    ),
                    "status": (
                        "estimated"
                        if time_share is not None and hit_rate is not None
                        else "needs-current-profile"
                    ),
                },
            }
        )

    mp_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in route_candidates:
        key = tuple(candidate["key"].values())
        mp_groups[
            (
                candidate["mp"],
                candidate["cell"],
                candidate["schema"],
                mp_only_key(candidate["schema"], key),
            )
        ].append(candidate)

    mp_only_candidates: list[dict[str, Any]] = []
    for (mp, cell, schema, key), route_legs in sorted(mp_groups.items(), key=str):
        winners = sorted(
            {leg["winner"] for leg in route_legs if leg["winner"] is not None}
        )
        every_route_pass = all(leg["route_candidate_pass"] for leg in route_legs)
        same_winner = len(winners) == 1 and all(
            leg["winner"] == winners[0] for leg in route_legs
        )
        mp_only_candidates.append(
            {
                "mp": mp,
                "cell": cell,
                "schema": schema,
                "key": list(key),
                "winner": winners[0] if same_winner else None,
                "route_count": len(route_legs),
                "same_winner_across_routes": same_winner,
                "every_route_pass": every_route_pass,
                "promotion_ready": same_winner and every_route_pass,
            }
        )

    payload = {
        "schema": "vllm-musa-mp-tactic-summary.v1",
        "campaign_sha256": campaign_sha256,
        "gate": gate,
        "observation_count": len(observations),
        "route_candidates": route_candidates,
        "mp_only_candidates": mp_only_candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.markdown is not None:
        lines = [
            "# MP tactic campaign summary",
            "",
            f"Observations: {len(observations)}",
            "",
            "| MP | Cell | Key | Winner | Ready |",
            "|---:|---|---|---|---|",
        ]
        for candidate in mp_only_candidates:
            lines.append(
                "| {mp} | {cell} | `{key}` | {winner} | {ready} |".format(
                    mp=candidate["mp"],
                    cell=candidate["cell"],
                    key=json.dumps(candidate["key"], separators=(",", ":")),
                    winner=candidate["winner"] or "route/run dependent",
                    ready="yes" if candidate["promotion_ready"] else "no",
                )
            )
        lines.extend(
            [
                "",
                "`Ready=no` is expected until all required seeds and host replicas pass.",
            ]
        )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
