# SPDX-License-Identifier: Apache-2.0
"""Validate advisory lineage and functional-group metadata for the patch series."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

DISPOSITIONS = frozenset(
    {
        "superseded",
        "folded",
        "upstream",
        "absent",
        "canceled",
        "unresolved",
        "performance_opportunity",
    }
)
CHECKS = frozenset(
    {
        "series",
        "replay",
        "round_trip",
        "verify",
        "source_contract",
        "native_build",
        "musa_runtime",
        "model_runtime",
        "mooncake_runtime",
    }
)
_NUMBER = re.compile(r"[0-9]{4}\Z")
_ENTRY = re.compile(r"([0-9]{4})-.*\.patch\Z")
_OLD_ROW = re.compile(r"^\| `([0-9]{4})` \|", re.MULTILINE)
_CURRENT_ROW = re.compile(r"^\| `([0-9]{4}-[^`]+\.patch)` \|", re.MULTILINE)
_LINEAGE_FIELDS = frozenset(
    {"legacy_id", "disposition", "owners", "rationale", "validation_required"}
)
_GROUP_FIELDS = frozenset({"id", "purpose", "patches", "depends_on", "required_checks"})


class Result(NamedTuple):
    errors: list[str]
    patch_count: int
    omission_count: int
    group_count: int
    lineage: list[dict]
    groups: list[dict]


def _load(path: Path, errors: list[str]) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing metadata: {path}")
        return {}
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"invalid JSON or unreadable metadata: {path}: {exc}")
        return {}
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
    ):
        errors.append(f"{path}: expected schema_version 1")
        return {}
    return data


def _records(data: dict, field: str, path: Path, errors: list[str]) -> list[dict]:
    records = data.get(field)
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        errors.append(f"{path}: {field} must be a list of objects")
        return []
    return records


def _text(record: dict, field: str, location: str, errors: list[str]) -> None:
    if not isinstance(record.get(field), str) or not record[field].strip():
        errors.append(f"{location}: {field} must be nonempty text")


def _list(record: dict, field: str, location: str, errors: list[str]) -> list:
    value = record.get(field)
    if not isinstance(value, list):
        errors.append(f"{location}: {field} must be a list")
        return []
    return value


def validate(series: Path, metadata: Path) -> Result:
    """Check disk-derived membership and the README's 24-row omission table."""
    errors: list[str] = []
    paths = sorted(series.glob("[0-9][0-9][0-9][0-9]-*.patch"))
    numbers = [path.name[:4] for path in paths if _ENTRY.fullmatch(path.name)]
    if len(numbers) != len(paths) or len(set(numbers)) != len(numbers):
        errors.append("series: invalid or duplicate patch numbers")
    present = set(numbers)
    try:
        readme = (series / "README.md").read_text(encoding="utf-8")
        legacy_rows = _OLD_ROW.findall(readme)
        current_rows = _CURRENT_ROW.findall(readme)
    except (OSError, UnicodeError) as exc:
        errors.append(f"series README missing/unreadable: {exc}")
        legacy_rows = []
        current_rows = []
    if len(legacy_rows) != len(set(legacy_rows)):
        errors.append("series README: duplicate legacy omission row")
    if len(current_rows) != len(set(current_rows)):
        errors.append("series README: duplicate current patch row")
    current_patch_names = {path.name for path in paths}
    missing_current_rows = sorted(current_patch_names - set(current_rows))
    extra_current_rows = sorted(set(current_rows) - current_patch_names)
    if missing_current_rows:
        errors.append(
            "missing current patch rows from README: " + ", ".join(missing_current_rows)
        )
    if extra_current_rows:
        errors.append(
            "unknown current patch rows in README: " + ", ".join(extra_current_rows)
        )

    lineage_path = metadata / "legacy-lineage.json"
    groups_path = metadata / "patch-groups.json"
    lineage = _records(_load(lineage_path, errors), "omissions", lineage_path, errors)
    groups = _records(_load(groups_path, errors), "groups", groups_path, errors)

    old_ids = []
    for i, row in enumerate(lineage):
        location = f"legacy-lineage omissions[{i}]"
        unexpected = sorted(row.keys() - _LINEAGE_FIELDS)
        if unexpected:
            errors.append(f"{location}: unknown fields {unexpected}")
        old = row.get("legacy_id")
        if not isinstance(old, str) or not _NUMBER.fullmatch(old):
            errors.append(f"{location}: invalid legacy_id {old!r}")
        else:
            old_ids.append(old)
        disposition = row.get("disposition")
        if not isinstance(disposition, str) or disposition not in DISPOSITIONS:
            errors.append(f"{location}: unknown disposition {disposition!r}")
        for field in ("rationale", "validation_required"):
            _text(row, field, location, errors)
        owners = _list(row, "owners", location, errors)
        if len(owners) != len(set(map(str, owners))):
            errors.append(f"{location}: duplicate owner")
        for owner in owners:
            if not isinstance(owner, str) or owner not in present:
                errors.append(f"{location}: unknown owner {owner!r}")
    for old, count in Counter(old_ids).items():
        if count > 1:
            errors.append(f"duplicate legacy omission {old}")
    missing = sorted(set(legacy_rows) - set(old_ids))
    extra = sorted(set(old_ids) - set(legacy_rows))
    if missing:
        errors.append(f"missing legacy omissions from metadata: {', '.join(missing)}")
    if extra:
        errors.append(f"unknown legacy omissions not in README: {', '.join(extra)}")

    ids = []
    owned = []
    for i, group in enumerate(groups):
        location = f"patch-groups groups[{i}]"
        unexpected = sorted(group.keys() - _GROUP_FIELDS)
        if unexpected:
            errors.append(f"{location}: unknown fields {unexpected}")
        ident = group.get("id")
        _text(group, "id", location, errors)
        _text(group, "purpose", location, errors)
        if isinstance(ident, str):
            ids.append(ident)
        patches = _list(group, "patches", location, errors)
        if not patches:
            errors.append(f"{location}: empty patch group")
        if all(isinstance(p, str) for p in patches) and patches != sorted(patches):
            errors.append(f"{location}: patch order must be ascending")
        for patch in patches:
            if not isinstance(patch, str) or patch not in present:
                errors.append(f"{location}: unknown patch {patch!r}")
            else:
                owned.append(patch)
        checks = _list(group, "required_checks", location, errors)
        if not checks:
            errors.append(f"{location}: no required checks")
        for check in checks:
            if not isinstance(check, str) or check not in CHECKS:
                errors.append(f"{location}: unknown check {check!r}")
        if len(checks) != len(set(map(str, checks))):
            errors.append(f"{location}: duplicate check")
        dependencies = _list(group, "depends_on", location, errors)
        if len(dependencies) != len(set(map(str, dependencies))):
            errors.append(f"{location}: duplicate dependency")
    group_starts = [
        group["patches"][0]
        for group in groups
        if isinstance(group.get("patches"), list)
        and group["patches"]
        and isinstance(group["patches"][0], str)
    ]
    if group_starts != sorted(group_starts):
        errors.append("patch-groups: group order must follow first owned patch")
    for ident, count in Counter(ids).items():
        if count > 1:
            errors.append(f"duplicate group id {ident!r}")
    for group in groups:
        for dependency in (
            group.get("depends_on", [])
            if isinstance(group.get("depends_on"), list)
            else []
        ):
            if dependency not in ids or dependency == group.get("id"):
                errors.append(
                    f"{group.get('id')}: unknown dependency or self-dependency {dependency!r}"
                )
    remaining = {
        group["id"]: set(group["depends_on"])
        for group in groups
        if isinstance(group.get("id"), str)
        and isinstance(group.get("depends_on"), list)
        and all(isinstance(dep, str) and dep in ids for dep in group["depends_on"])
    }
    while remaining:
        ready = {
            ident
            for ident, dependencies in remaining.items()
            if not dependencies.intersection(remaining)
        }
        if not ready:
            errors.append("patch-groups: dependency cycle")
            break
        for ident in ready:
            del remaining[ident]
    for number, count in Counter(owned).items():
        if count > 1:
            errors.append(f"duplicate patch ownership: {number}")
    missing_patches = sorted(present - set(owned))
    if missing_patches:
        errors.append(f"missing patch group ownership: {', '.join(missing_patches)}")
    return Result(errors, len(paths), len(lineage), len(groups), lineage, groups)
