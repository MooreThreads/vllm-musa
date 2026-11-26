"""Repack the pinned Omni dependencies for a headless OpenCV installation."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from email.parser import BytesHeaderParser
from pathlib import Path

_WHEELS = (
    ("cosmos_guardrail", "0.3.2", "opencv-python>=4.10.0.84"),
    ("retinaface_py", "0.0.2", "opencv-python"),
)


def _replace_metadata(
    metadata: Path, distribution: str, version: str, requirement: str
) -> None:
    content = metadata.read_bytes()
    headers = BytesHeaderParser().parsebytes(content)
    if headers.get_all("Name") != [distribution.replace("_", "-")]:
        raise ValueError(f"{metadata}: expected metadata for {distribution}")
    if headers.get_all("Version") != [version]:
        raise ValueError(f"{metadata}: expected version {version}")

    opencv_requirements = [
        value
        for value in headers.get_all("Requires-Dist", [])
        if re.match(r"opencv[-_.]python\b", value, re.IGNORECASE)
    ]
    if opencv_requirements != [requirement]:
        raise ValueError(
            f"{metadata}: expected only {requirement!r}, found {opencv_requirements!r}"
        )

    # Update just the two headers and retain every other header and the body.
    header, separator, body = content.partition(b"\n\n")
    lines = header.split(b"\n")
    for original, replacement in (
        (f"Version: {version}", f"Version: {version}+musa"),
        (
            f"Requires-Dist: {requirement}",
            f"Requires-Dist: {requirement.replace('opencv-python', 'opencv-python-headless', 1)}",
        ),
    ):
        before = original.encode("ascii")
        if not separator or lines.count(before) != 1:
            raise ValueError(
                f"{metadata}: unsupported metadata layout for {original!r}"
            )
        lines[lines.index(before)] = replacement.encode("ascii")
    metadata.write_bytes(b"\n".join(lines) + separator + body)


def _repack(
    source: Path, output: Path, distribution: str, version: str, requirement: str
) -> None:
    name = f"{distribution}-{version}"
    with tempfile.TemporaryDirectory(prefix=f"{distribution}-") as tmp:
        subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", str(source), "-d", tmp],
            check=True,
        )
        unpacked = Path(tmp) / name
        info = unpacked / f"{name}.dist-info"
        if (unpacked / f"{name}.data").exists() or any(
            (info / filename).exists() for filename in ("RECORD.jws", "RECORD.p7s")
        ):
            raise ValueError(
                f"{source}: wheel data directories and signatures are unsupported"
            )
        _replace_metadata(info / "METADATA", distribution, version, requirement)
        info.rename(unpacked / f"{name}+musa.dist-info")

        # wheel pack regenerates RECORD. A fixed zip timestamp makes repeated
        # repacks independent of the extraction time.
        env = {"SOURCE_DATE_EPOCH": "315532800", **os.environ}
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", str(unpacked), "-d", str(output)],
            check=True,
            env=env,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for distribution, version, requirement in _WHEELS:
        source = args.input_dir / f"{distribution}-{version}-py3-none-any.whl"
        if not source.is_file():
            parser.error(f"missing pinned upstream wheel {source}")
        try:
            _repack(source, args.output_dir, distribution, version, requirement)
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            parser.error(str(exc))


if __name__ == "__main__":
    main()
