# SPDX-License-Identifier: Apache-2.0
"""Verify the vLLM-Omni image's Python installation."""

import argparse
import json
from importlib import metadata, util
from pathlib import Path
from sysconfig import get_path

from packaging.utils import canonicalize_name
from packaging.version import Version

WORKSPACE = Path("/vllm-workspace")
FOUNDATION = Path("/opt/vllm-omni-foundation.txt")
OMNI_REVISION = Path("/opt/vllm-omni-commit.txt")
OMNI_VERSION = Path("/opt/vllm-omni-version.txt")


def installed_versions() -> dict[str, str]:
    versions = {}
    paths = sorted({get_path("purelib"), get_path("platlib")})
    for dist in metadata.distributions(path=paths):
        name = canonicalize_name(dist.metadata["Name"])
        if name in versions:
            raise ValueError(f"multiple installed distributions for {name}")
        versions[name] = dist.version
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--snapshot", type=Path)
    modes.add_argument("--write-version", action="store_true")
    modes.add_argument("--mode", choices=("wheel", "editable"))
    args = parser.parse_args()

    if args.write_version:
        import vllm

        base = Version(vllm.__version__)
        revision = OMNI_REVISION.read_text().strip()
        OMNI_VERSION.write_text(f"{base.major}.{base.minor}.0+musa.g{revision[:9]}\n")
        return

    versions = installed_versions()
    if "vllm-musa" not in versions or "vllm" in versions:
        raise SystemExit("Omni expects a bundled vllm-musa base image")
    if "opencv-python-headless" not in versions or "opencv-python" in versions:
        raise SystemExit("Omni expects a single headless OpenCV distribution")

    if args.snapshot:
        if "vllm-omni" in versions:
            raise SystemExit("the base image already has vllm-omni installed")
        args.snapshot.write_text(
            "".join(
                f"{name}=={version}\n" for name, version in sorted(versions.items())
            )
        )
        return

    for line in FOUNDATION.read_text().splitlines():
        name, expected = line.split("==", 1)
        if versions.get(name) != expected:
            raise SystemExit(f"base dependency {line} changed to {versions.get(name)}")

    dist = metadata.distribution("vllm-omni")
    expected_version = OMNI_VERSION.read_text().strip()
    if dist.version != expected_version:
        raise SystemExit(f"expected vllm-omni {expected_version}, got {dist.version}")
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    spec = util.find_spec("vllm_omni")
    if spec is None or spec.origin is None:
        raise SystemExit("vllm_omni is not importable")
    origin = Path(spec.origin).resolve()
    checkout = WORKSPACE / "vllm-omni"
    if args.mode == "editable":
        if not direct_url.get("dir_info", {}).get(
            "editable"
        ) or not origin.is_relative_to(checkout):
            raise SystemExit(
                f"expected editable vllm_omni under {checkout}, found {origin}"
            )
    elif (
        direct_url.get("dir_info", {}).get("editable")
        or not origin.is_relative_to(Path(dist.locate_file("vllm_omni")).resolve())
        or checkout.exists()
        or Path("/src/vllm-omni").exists()
    ):
        raise SystemExit(
            f"expected an installed vllm_omni wheel without a checkout, found {origin}"
        )

    import cv2
    import numpy

    if numpy.__version__.split(".", 1)[0] != "1":
        raise SystemExit(f"expected NumPy 1.x, found {numpy.__version__}")
    if cv2.cvtColor(
        numpy.zeros((2, 2, 3), dtype=numpy.uint8), cv2.COLOR_BGR2RGB
    ).shape != (2, 2, 3):
        raise SystemExit("OpenCV conversion failed")
    print(
        f"PASS {args.mode} vllm-omni={dist.version} "
        f"import={origin} cv2={cv2.__version__}"
    )


if __name__ == "__main__":
    main()
