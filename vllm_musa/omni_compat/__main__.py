# SPDX-License-Identifier: Apache-2.0
"""``python -m vllm_musa.omni_compat`` -- inspect or activate the Omni shim.

status        print the shim state as JSON (exit 0)
install-pth   install the interpreter-startup hook into site-packages
remove-pth    remove it
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from vllm_musa import omni_compat

_PROBE = (
    "import json, sys;"
    "from vllm_musa.omni_compat import active_hooks;"
    "print(json.dumps(active_hooks()))"
)


def _startup_hook_comes_up() -> tuple[bool, str]:
    """Whether a fresh interpreter activates the hook from the file just written.

    The ``.pth`` goes into ``sysconfig``'s ``purelib``, which is where a site
    directory is for every layout this image has used. That is still an
    assumption about the interpreter, and a wrong one leaves a shim that is
    silently inert -- the hook never runs, every import resolves to the pin, and
    the first symptom is a model failing. So the installation proves itself in a
    subprocess instead of trusting the path.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()
    try:
        hooks = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, f"unreadable probe output: {proc.stdout!r}"
    if hooks.get("startup"):
        return True, ""
    return False, f"hooks={hooks} (stderr: {proc.stderr.strip()})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vllm_musa.omni_compat")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="print the shim state as JSON")
    install = sub.add_parser("install-pth", help="install the startup hook")
    install.add_argument(
        "--purelib",
        default=None,
        help="target site-packages (default: this interpreter's)",
    )
    remove = sub.add_parser("remove-pth", help="remove the startup hook")
    remove.add_argument("--purelib", default=None)
    args = parser.parse_args(argv)

    if args.command == "status":
        json.dump(omni_compat.status(), sys.stdout, indent=1, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.command == "install-pth":
        path = omni_compat.write_pth(args.purelib)
        print(f"installed {path}")
        ok, detail = _startup_hook_comes_up()
        if ok:
            print("verified: a fresh interpreter activates the startup hook")
            return 0
        trouble = (
            f"{path} did not activate the shim in a fresh interpreter. {detail}\n"
            "The file is not on a site path this interpreter processes, or the "
            "bootstrap module vllm_musa_omni_compat_pth is not importable from it."
        )
        if args.purelib is not None:
            # An explicit target is the operator's choice and may belong to a
            # different interpreter, so this
            # probe cannot be the verdict.
            print(f"warning: {trouble}", file=sys.stderr)
            return 0
        print(
            f"error: {trouble}\n"
            "Pass --purelib with the site directory this interpreter reads.",
            file=sys.stderr,
        )
        return 1
    path = omni_compat.pth_path(args.purelib)
    if path.exists():
        path.unlink()
        print(f"removed {path}")
    else:
        print(f"absent {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
