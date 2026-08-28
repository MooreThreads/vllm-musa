# SPDX-License-Identifier: Apache-2.0

"""Bounded filesystem I/O for engine-plan JSON artifacts."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .json_utils import dumps as dump_json
from .json_utils import loads as load_json

DEFAULT_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


class ArtifactFileError(ValueError):
    """Raised when an artifact path or JSON payload is unsafe or malformed."""


def _regular_file_mode(path: Path) -> int | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ArtifactFileError(
            f"Unable to inspect artifact path {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ArtifactFileError(f"Artifact path must not be a symbolic link: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactFileError(f"Artifact path must be a regular file: {path}")
    return stat.S_IMODE(metadata.st_mode)


def load_json_object_file(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> dict[str, Any]:
    """Load one bounded regular-file JSON object without following symlinks."""

    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    source = Path(path)
    mode = _regular_file_mode(source)
    if mode is None:
        raise ArtifactFileError(f"Artifact file does not exist: {source}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ArtifactFileError(
            f"Unable to open artifact file {source}: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactFileError(f"Artifact path must be a regular file: {source}")
        if metadata.st_size > max_bytes:
            raise ArtifactFileError(
                f"Artifact file {source} exceeds the limit of {max_bytes} bytes"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(max_bytes + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > max_bytes:
        raise ArtifactFileError(
            f"Artifact file {source} exceeds the limit of {max_bytes} bytes"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactFileError(f"Unable to decode JSON from {source}: {exc}") from exc
    try:
        value = load_json(text, source=str(source))
    except ValueError as exc:
        raise ArtifactFileError(str(exc)) from exc
    if not isinstance(value, dict):
        raise ArtifactFileError(f"Artifact file {source} root must be a JSON object")
    return value


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_object_file(
    path: str | Path,
    document: Mapping[str, Any],
) -> None:
    """Atomically write one JSON object with durable same-directory replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = _regular_file_mode(destination)
    output_mode = existing_mode if existing_mode is not None else 0o600
    try:
        payload = (
            dump_json(
                dict(document),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactFileError(
            f"Unable to encode JSON for {destination}: {exc}"
        ) from exc

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, output_mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
