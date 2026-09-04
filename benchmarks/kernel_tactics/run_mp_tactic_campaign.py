#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a reviewed MP tactic campaign on one lease-scoped MUSA device."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _benchmark_utils import source_identity

DEFAULT_CAMPAIGN = Path(__file__).with_name("mp_tactic_campaign.json")
SUPPORTED_CAMPAIGN_SCHEMAS = frozenset(
    {
        "vllm-musa-mp-tactic-campaign.v1",
        "vllm-musa-mp-tactic-campaign.v2",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--expected-mp", type=int)
    parser.add_argument(
        "--allowed-mp",
        type=int,
        action="append",
        help=(
            "optional local qualification allow-list; keep fleet-specific "
            "values out of the checked-in campaign recipe"
        ),
    )
    parser.add_argument("--expected-physical-device", type=int)
    parser.add_argument("--expected-device-uuid")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--lease-isolated",
        action="store_true",
        help="attest that the SOL lease was isolated; omitted runs are exploratory",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def load_campaign(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") not in SUPPORTED_CAMPAIGN_SCHEMAS:
        raise ValueError(f"unsupported campaign schema in {path}")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("campaign must contain non-empty cells")
    ids = [cell.get("id") for cell in cells]
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in ids):
        raise ValueError("every campaign cell must have a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("campaign cell ids must be unique")
    return payload


def probe_device() -> dict[str, Any]:
    # Import order is part of the MUSA compatibility contract.
    # isort: off
    import torchada  # noqa: F401
    import torch
    # isort: on

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA device is not available")
    device_count = int(torch.musa.device_count())
    if device_count != 1:
        raise RuntimeError(
            "campaign requires exactly one lease-scoped visible MUSA device, "
            f"got {device_count}"
        )
    properties = torch.musa.get_device_properties(0)
    return {
        "device_count": device_count,
        "device_name": torch.musa.get_device_name(0),
        "device_capability": [int(properties.major), int(properties.minor)],
        "multiprocessor_count": int(properties.multi_processor_count),
    }


def verify_visible_device_contract() -> dict[str, str]:
    """Require explicit, identical MUSA/CUDA visibility for reported runs."""
    musa_visible = os.environ.get("MUSA_VISIBLE_DEVICES")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not musa_visible or cuda_visible != musa_visible:
        raise RuntimeError(
            "MUSA_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES must both be set to "
            "the same value for a campaign run"
        )
    return {
        "MUSA_VISIBLE_DEVICES": musa_visible,
        "CUDA_VISIBLE_DEVICES": cuda_visible,
    }


def add_device_fence_args(
    command: list[str],
    expected_physical_device: int | None,
    expected_device_uuid: str | None,
) -> list[str]:
    """Attach the lease-device fence required by every enabled benchmark."""
    if expected_physical_device is None or not expected_device_uuid:
        raise ValueError(
            "campaign runs require --expected-physical-device and "
            "--expected-device-uuid from the active SOL lease"
        )
    return [
        *command,
        "--expected-physical-device",
        str(expected_physical_device),
        "--expected-device-uuid",
        expected_device_uuid,
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def selected_cells(
    campaign: dict[str, Any], requested: list[str]
) -> list[dict[str, Any]]:
    cells = campaign["cells"]
    if not requested:
        return [cell for cell in cells if cell.get("enabled")]
    by_id = {cell["id"]: cell for cell in cells}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"unknown campaign cells: {', '.join(unknown)}")
    disabled = [cell_id for cell_id in requested if not by_id[cell_id].get("enabled")]
    if disabled:
        reasons = "; ".join(
            f"{cell_id}: {by_id[cell_id].get('blocked_reason', 'disabled')}"
            for cell_id in disabled
        )
        raise ValueError(f"requested disabled campaign cells: {reasons}")
    return [by_id[cell_id] for cell_id in requested]


def main() -> int:
    args = parse_args()
    campaign_path = args.campaign.resolve()
    campaign = load_campaign(campaign_path)
    cells = selected_cells(campaign, args.cell)
    if args.list:
        for cell in campaign["cells"]:
            state = "enabled" if cell.get("enabled") else "blocked"
            print(f"{cell['id']}\t{state}\tpriority={cell.get('priority')}")
        return 0
    if args.expected_mp is None or args.output_dir is None:
        raise ValueError("--expected-mp and --output-dir are required unless --list")
    if args.expected_physical_device is None or not args.expected_device_uuid:
        raise ValueError(
            "--expected-physical-device and --expected-device-uuid are required "
            "for a reported campaign run"
        )
    allowed_mps = args.allowed_mp or campaign.get("allowed_multiprocessor_counts")
    if allowed_mps is None:
        # Keep already-generated v1 evidence replayable.  New public recipes
        # intentionally omit this fleet-specific field.
        allowed_mps = campaign.get("hardware", {}).get("runtime_multiprocessor_counts")
    if allowed_mps is not None and args.expected_mp not in allowed_mps:
        raise ValueError(
            f"expected MP {args.expected_mp} is outside campaign bins {allowed_mps}"
        )

    visible_devices = verify_visible_device_contract()
    repo = Path(__file__).resolve().parents[2]
    source = source_identity(Path(__file__))
    if source["dirty"] and not args.allow_dirty:
        raise RuntimeError(
            "refusing reported campaign on dirty source; commit the harness or "
            "pass --allow-dirty for explicitly exploratory output"
        )
    device = probe_device()
    if device["multiprocessor_count"] != args.expected_mp:
        raise RuntimeError(
            f"runtime MP mismatch: expected {args.expected_mp}, "
            f"observed {device['multiprocessor_count']}"
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(campaign_path, output_dir / "campaign.json")
    campaign_sha256 = hashlib.sha256(campaign_path.read_bytes()).hexdigest()
    manifest: dict[str, Any] = {
        "schema": "vllm-musa-mp-tactic-run.v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "campaign_sha256": campaign_sha256,
        "source": source,
        "device": device,
        "expected_mp": args.expected_mp,
        "visible_devices": visible_devices,
        "lease_isolated": bool(args.lease_isolated),
        "exploratory": bool(source["dirty"] or not args.lease_isolated),
        "runs": [],
    }
    manifest_path = output_dir / "run-manifest.json"
    write_json(manifest_path, manifest)

    failures = 0
    for cell in cells:
        mode_args = cell["modes"][args.mode]
        variants = cell.get("variants") or [{"id": "default", "args": []}]
        for variant in variants:
            run_id = f"{cell['id']}__{variant['id']}"
            result_path = output_dir / f"{run_id}.json"
            log_path = output_dir / f"{run_id}.log"
            receipt_path = output_dir / f"{run_id}.receipt.json"
            command = [
                sys.executable,
                str(repo / cell["script"]),
                *cell.get("base_args", []),
                *mode_args,
                *variant.get("args", []),
                "--output",
                str(result_path),
            ]
            command = add_device_fence_args(
                command,
                args.expected_physical_device,
                args.expected_device_uuid,
            )
            started = datetime.now(timezone.utc)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            with log_path.open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            ended = datetime.now(timezone.utc)
            error = None
            output_sha256 = None
            if completed.returncode == 0:
                try:
                    result = json.loads(result_path.read_text())
                    observed_mp = result.get("multiprocessor_count")
                    if observed_mp != args.expected_mp:
                        raise ValueError(
                            f"result MP mismatch: expected {args.expected_mp}, "
                            f"got {observed_mp}"
                        )
                    output_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    error = str(exc)
            else:
                error = f"benchmark exited {completed.returncode}"
            if error is not None:
                failures += 1
            receipt = {
                "schema": "vllm-musa-mp-tactic-cell-receipt.v1",
                "run_id": run_id,
                "cell": cell["id"],
                "variant": variant["id"],
                "command": command,
                "started_at_utc": started.isoformat(),
                "ended_at_utc": ended.isoformat(),
                "duration_seconds": (ended - started).total_seconds(),
                "returncode": completed.returncode,
                "result": str(result_path),
                "result_sha256": output_sha256,
                "log": str(log_path),
                "error": error,
            }
            write_json(receipt_path, receipt)
            manifest["runs"].append(receipt)
            write_json(manifest_path, manifest)
            if error is not None and not args.continue_on_error:
                manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
                manifest["status"] = "failed"
                write_json(manifest_path, manifest)
                return 1

    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "passed" if failures == 0 else "partial"
    manifest["failure_count"] = failures
    write_json(manifest_path, manifest)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
