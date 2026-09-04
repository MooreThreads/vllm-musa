#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small provenance and reporting helpers shared by kernel tactic benches."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def command_output(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def verify_lease_device_fence(
    expected_physical_device: int,
    expected_device_uuid: str,
    expected_multiprocessor_count: int | None = None,
) -> dict[str, Any]:
    """Fail closed unless visibility, physical GPU, and optional MP match."""
    if expected_physical_device < 0:
        raise ValueError("expected physical device must be non-negative")
    expected_uuid = expected_device_uuid.strip().lower()
    if not expected_uuid:
        raise ValueError("expected device UUID must not be empty")

    visibility = {
        name: os.environ.get(name)
        for name in (
            "MTHREADS_VISIBLE_DEVICES",
            "MUSA_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        )
    }
    expected_visible = str(expected_physical_device)
    mismatched = {
        name: value for name, value in visibility.items() if value != expected_visible
    }
    if mismatched:
        raise RuntimeError(
            "lease device fence failed: visibility must all equal "
            f"{expected_visible}, got {visibility}"
        )

    query_mode = "csv-query"
    query = command_output(
        [
            "mthreads-gmi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader",
        ]
    )
    if query is None:
        query_mode = "legacy-full-query"
        query = command_output(["mthreads-gmi", "-q"])
        if query is None:
            raise RuntimeError(
                "lease device fence failed: both mthreads-gmi queries failed"
            )
        block = re.search(
            rf"^GPU{expected_physical_device}\b(?P<body>.*?)(?=^GPU\d+\b|\Z)",
            query,
            re.MULTILINE | re.DOTALL,
        )
        uuid_match = (
            re.search(r"GPU UUID\s*:\s*(\S+)", block.group("body"))
            if block is not None
            else None
        )
        name_match = (
            re.search(r"Product Name\s*:\s*(.+)", block.group("body"))
            if block is not None
            else None
        )
        if block is None or uuid_match is None or name_match is None:
            raise RuntimeError(
                "lease device fence failed: expected one matching legacy "
                f"mthreads-gmi block for GPU{expected_physical_device}"
            )
        actual_index = expected_visible
        actual_uuid = uuid_match.group(1).strip()
        device_name = name_match.group(1).strip()
        query_receipt = f"{actual_index}, {actual_uuid}, {device_name}"
    else:
        parsed_rows = [
            [field.strip() for field in line.split(",")]
            for line in query.splitlines()
            if line.strip()
        ]
        matching_rows = [
            fields
            for fields in parsed_rows
            if len(fields) == 3 and fields[0] == expected_visible
        ]
        if len(matching_rows) != 1:
            raise RuntimeError(
                "lease device fence failed: expected one matching "
                f"mthreads-gmi row {query!r}"
            )
        actual_index, actual_uuid, device_name = matching_rows[0]
        query_receipt = ", ".join(matching_rows[0])
    if actual_index != expected_visible or actual_uuid.lower() != expected_uuid:
        raise RuntimeError(
            "lease device fence failed: expected "
            f"index={expected_visible}, uuid={expected_uuid}; got "
            f"index={actual_index}, uuid={actual_uuid.lower()}"
        )
    if expected_multiprocessor_count is not None:
        if expected_multiprocessor_count <= 0:
            raise ValueError("expected multiprocessor count must be positive")
        try:
            import torch

            actual_multiprocessor_count = int(
                torch.musa.get_device_properties(
                    expected_physical_device
                ).multi_processor_count
            )
        except (AttributeError, RuntimeError, TypeError, ImportError) as exc:
            raise RuntimeError(
                "lease device fence failed: cannot query MUSA multiprocessor count"
            ) from exc
        if actual_multiprocessor_count != expected_multiprocessor_count:
            raise RuntimeError(
                "lease device fence failed: expected "
                f"multiprocessor_count={expected_multiprocessor_count}; got "
                f"{actual_multiprocessor_count}"
            )
    else:
        actual_multiprocessor_count = None
    return {
        "passed": True,
        "expected_physical_device": expected_physical_device,
        "expected_device_uuid": expected_uuid,
        "actual_physical_device": int(actual_index),
        "actual_device_uuid": actual_uuid.lower(),
        "device_name": device_name,
        "visibility": visibility,
        "mthreads_gmi_query_mode": query_mode,
        "mthreads_gmi_query": query_receipt,
        "expected_multiprocessor_count": expected_multiprocessor_count,
        "actual_multiprocessor_count": actual_multiprocessor_count,
    }


def effective_gemv_block(
    family_name: str,
    tokens: int,
    stage: str,
    requested: tuple[int, int],
) -> tuple[tuple[int, int], bool, str | None]:
    """Describe selector arms that supersede a requested GEMV block."""
    if family_name == "dsv4_fp8" and tokens == 1:
        selected = (4, 32) if stage == "w1" else (32, 4)
        return selected, requested == selected, "dsv4-one-token-split-tile"
    return requested, True, None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_identity(script: Path) -> dict[str, Any]:
    resolved_script = script.resolve()
    repo = next(
        (
            parent
            for parent in (resolved_script.parent, *resolved_script.parents)
            if (parent / ".git").exists() or (parent / ".source-revision").exists()
        ),
        resolved_script.parent,
    )
    has_git = (repo / ".git").exists()
    head = command_output(["git", "rev-parse", "HEAD"], cwd=repo) if has_git else None
    status = command_output(["git", "status", "--short"], cwd=repo) if has_git else None
    marker_path = repo / ".source-revision"
    archive_marker = None
    if marker_path.is_file():
        archive_marker = marker_path.read_text().strip() or None
        if head is None:
            head = archive_marker
    diff_sha256 = None
    if has_git:
        try:
            diff = subprocess.check_output(
                ["git", "diff", "--binary", "HEAD"],
                cwd=repo,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            diff_sha256 = hashlib.sha256(diff).hexdigest()
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "repo": str(repo),
        "head": head,
        "dirty": bool(status),
        "status": status or "",
        "diff_sha256": diff_sha256,
        "archive_marker": archive_marker,
    }


def provenance(script: Path) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "source": source_identity(script),
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "torch_musa",
                "torchada",
                "vllm",
                "vllm-musa",
                "mate",
            )
        },
        "mthreads_gmi": command_output(["mthreads-gmi", "-q"]),
        "mcc_version": command_output(["mcc", "--version"]),
    }


def emit_payload(payload: dict[str, Any], output: Path | None) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized)
    print(serialized, end="")
