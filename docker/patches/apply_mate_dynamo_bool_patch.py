"""Apply the TorchDynamo compatibility patch to pre-0.2.6 MATE wheels."""

from __future__ import annotations

import argparse
import subprocess
from importlib.metadata import distribution
from pathlib import Path

from packaging.version import Version

MATE_FIX_VERSION = Version("0.2.6")
DEFAULT_PATCH = Path(__file__).with_name("mate-dynamo-bool-augassign.patch")


def apply_patch(patch_file: Path) -> None:
    mate_dist = distribution("mate")
    mate_version = Version(mate_dist.version)
    print(f"MATE version: {mate_version}", flush=True)

    if mate_version >= MATE_FIX_VERSION:
        print(
            f"SKIP mate_dynamo_bool_patch version={mate_version} "
            f"(fixed in >= {MATE_FIX_VERSION})",
            flush=True,
        )
        return

    site_packages = Path(mate_dist.locate_file(""))
    target = site_packages / "mate" / "mha_interface.py"
    if not target.is_file():
        raise RuntimeError(f"MATE source file not found: {target}")
    if not patch_file.is_file():
        raise RuntimeError(f"MATE patch file not found: {patch_file}")

    before = target.read_text()
    has_new_marker = "enable_mubin = enable_mubin & q.is_musa" in before
    has_old_marker = "enable_mubin &= " in before
    if has_new_marker and not has_old_marker:
        print(
            f"PASS mate_dynamo_bool_patch already_applied target={target}", flush=True
        )
        return
    if has_new_marker and has_old_marker:
        raise RuntimeError(f"MATE patch appears partially applied: {target}")
    if not has_old_marker:
        raise RuntimeError(
            f"MATE {mate_version} source does not match the expected pre-{MATE_FIX_VERSION} "
            f"layout: {target}"
        )

    patch_file = patch_file.resolve()
    subprocess.run(
        ["git", "apply", "--check", str(patch_file)],
        check=True,
        cwd=site_packages,
    )
    subprocess.run(
        ["git", "apply", str(patch_file)],
        check=True,
        cwd=site_packages,
    )

    after = target.read_text()
    if "enable_mubin &= " in after:
        raise RuntimeError(f"MATE patch was incomplete: {target}")
    if "enable_mubin = enable_mubin & q.is_musa" not in after:
        raise RuntimeError(f"MATE patch postcondition failed: {target}")
    print(
        f"PASS mate_dynamo_bool_patch version={mate_version} target={target}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-file", type=Path, default=DEFAULT_PATCH)
    args = parser.parse_args()
    apply_patch(args.patch_file)


if __name__ == "__main__":
    main()
