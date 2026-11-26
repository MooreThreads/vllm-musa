import subprocess
import sys
from email.parser import BytesHeaderParser
from pathlib import Path

from wheel.wheelfile import WheelFile

BUILDER = Path(__file__).resolve().parents[1] / "docker/omni/build_headless_wheels.py"


def _make_wheel(folder, name, version, requirement):
    stem = f"{name}-{version}"
    with WheelFile(folder / f"{stem}-py3-none-any.whl", "w") as archive:
        archive.writestr(f"{name}/__init__.py", b"# Upstream package\n")
        archive.writestr(
            f"{stem}.dist-info/METADATA",
            (
                f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\n"
                f"Version: {version}\nRequires-Dist: {requirement}\n"
                "Requires-Dist: numpy\n\nTest description.\n"
            ).encode("ascii"),
        )
        archive.writestr(
            f"{stem}.dist-info/WHEEL",
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )


def test_build_headless_wheels_preserves_payload_and_valid_record(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    output = tmp_path / "headless"
    wheels = (
        ("cosmos_guardrail", "0.3.2", ">=4.10.0.84"),
        ("retinaface_py", "0.0.2", ""),
    )
    for name, version, constraint in wheels:
        _make_wheel(source, name, version, f"opencv-python{constraint}")
    command = [
        sys.executable,
        str(BUILDER),
        "--input-dir",
        str(source),
        "--output-dir",
        str(output),
    ]
    subprocess.run(command, check=True)

    for name, version, constraint in wheels:
        stem = f"{name}-{version}+musa"
        with WheelFile(output / f"{stem}-py3-none-any.whl") as archive:
            # Reading with WheelFile verifies that the regenerated RECORD
            # matches the metadata and every file in the repacked wheel.
            for member in archive.namelist():
                archive.read(member)
            metadata = archive.read(f"{stem}.dist-info/METADATA")
            headers = BytesHeaderParser().parsebytes(metadata)
            assert headers["Version"] == f"{version}+musa"
            assert headers.get_all("Requires-Dist") == [
                f"opencv-python-headless{constraint}",
                "numpy",
            ]
            assert metadata.endswith(b"\n\nTest description.\n")
            assert archive.read(f"{name}/__init__.py") == b"# Upstream package\n"

    _make_wheel(source, "cosmos_guardrail", "0.3.2", "opencv-python>=5")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "expected only 'opencv-python>=4.10.0.84'" in result.stderr
