#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Omni drift shim against the installed vLLM.

What this proves, in a driverless build:

* the startup hook from the ``.pth`` is active -- not merely some hook, because
  only that one runs before Omni's CLI binds the symbols it moved;
* every ``RELOCATION`` and ``BACKPORT`` module the table declares really does
  import, and really does provide the names Omni imports from it;
* every ``ADAPTER`` is still applied after the MUSA sampler binds its function
  when the sampler is importable. In driverless mode, a device-dependent
  failure to import the sampler sets ``rebind_exercised`` to false;
* the unsupported paths are reported rather than silently absent.

What it deliberately does not prove: that ``import vllm_omni`` succeeds. That
import activates the MUSA platform plugin, which queries device capability
through ``torch_musa`` and raises without a driver, so it belongs to the
hardware smoke rather than the build. ``--require-omni`` runs it anyway, for use
on a leased GPU.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys

# These strings come from torch_musa, so they can move with a torch bump. An
# unrecognised message fails the gate rather than being skipped, which is the
# safe direction: a real shim failure must not be waved through as "no GPU". If a
# driverless build starts failing here right after a torch_musa upgrade, the
# message in the report is the first thing to read.
_DEVICE_MARKERS = (
    "MUSA driver initialization failed",
    "No MUSA runtime",
    "you might not have a Mthreads GPU",
)


def _stale_adapter(applied: dict) -> str | None:
    """Why an adapter that had nothing to do must still fail the build, or None.

    The import hook lets this through, because the pin already serves the call
    and a deployed interpreter is right to carry on rather than take every
    model down over a shim it does not need. An image is the other way round:
    it must not ship a table entry that does not describe its own pin, so the
    refusal lives here.
    """
    if not applied.get("stale"):
        return None
    return (
        "the adapter patched nothing because the pin already serves this call; "
        f"the entry is stale -- {applied['remove_when']}"
    )


def _needs_a_device(exc: BaseException) -> bool:
    """Whether an import failed for want of a GPU rather than for want of a shim."""
    return any(marker in str(exc) for marker in _DEVICE_MARKERS)


def _exercise_sampler_rebind(name: str) -> None:
    sampler = importlib.import_module("vllm.v1.sample.ops.topk_topp_sampler")
    musa_sampler = importlib.import_module(name)
    musa_impl = musa_sampler._apply_top_k_top_p
    # Bind the MUSA sampler explicitly if import left it unbound.
    if inspect.unwrap(sampler.apply_top_k_top_p) is not musa_impl:
        musa_sampler.install_hooks()
    if inspect.unwrap(sampler.apply_top_k_top_p) is not musa_impl:
        raise RuntimeError("sampler hook did not bind the MUSA apply_top_k_top_p")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-omni",
        action="store_true",
        help="also import vllm_omni (needs a MUSA driver)",
    )
    parser.add_argument(
        "--driverless",
        action="store_true",
        help="report a module that cannot import without a MUSA device as skipped, not failed",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    import vllm

    from vllm_musa import omni_compat
    from vllm_musa.omni_compat import table

    failures: list[str] = []
    result: dict = {
        "vllm_version": vllm.__version__,
        "hooks": omni_compat.active_hooks(),
        "installed": omni_compat.is_installed(),
        "served": {},
        "adapted": {},
        "unsupported": list(omni_compat.unsupported_paths()),
    }

    if not result["installed"]:
        failures.append(
            "no shim hook is installed; the .pth did not run and install() was not called"
        )
    elif not result["hooks"].get("startup"):
        # Any hook makes the imports resolve here, but only the .pth is early
        # enough for vllm_omni.entrypoints.cli.main, which binds moved vLLM
        # symbols before vLLM loads a plugin. A build that passed on an
        # in-process install() would still fail at the CLI.
        failures.append(
            f"the startup hook from the .pth is not active (hooks={result['hooks']}); "
            "only it runs early enough for Omni's CLI. Run "
            "'python -m vllm_musa.omni_compat install-pth', which verifies itself."
        )

    for entry in table.importable_entries():
        if not omni_compat.version_matches(entry.applies_to, vllm.__version__):
            result["served"][entry.module] = "inactive (pin outside applies_to)"
            continue
        try:
            module = importlib.import_module(entry.module)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if args.driverless and _needs_a_device(exc):
                result["served"][entry.module] = f"skipped, needs a device: {detail}"
                continue
            failures.append(f"{entry.module}: {detail}")
            result["served"][entry.module] = f"import failed: {detail}"
            continue
        missing = [name for name in entry.names if not hasattr(module, name)]
        if missing:
            failures.append(f"{entry.module}: served but missing {missing}")
        shim = getattr(module, "__vllm_musa_shim__", None)
        result["served"][entry.module] = {
            "kind": shim["kind"] if shim else "provided by the pin",
            "names": list(entry.names),
            "missing": missing,
        }

    # Check adapters after the production sampler installer binds its function.
    # The sampler may need a MUSA device at import time.
    sampler_name = "vllm_musa.v1.sample.topk_topp_sampler"
    try:
        _exercise_sampler_rebind(sampler_name)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        result["musa_hooks_installed"] = {sampler_name: detail}
        rebind_exercised = False
        if not (args.driverless and _needs_a_device(exc)):
            failures.append(f"could not exercise {sampler_name} rebind: {detail}")
    else:
        result["musa_hooks_installed"] = {sampler_name: True}
        rebind_exercised = True
    result["rebind_exercised"] = rebind_exercised

    for entry in table.adapter_entries():
        if not omni_compat.version_matches(entry.applies_to, vllm.__version__):
            result["adapted"][entry.module] = "inactive (pin outside applies_to)"
            continue
        try:
            module = importlib.import_module(entry.module)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if args.driverless and _needs_a_device(exc):
                # The module itself reaches into torch_musa at import. Nothing
                # about the adapter can be checked here; the hardware run with
                # --require-omni is where it is verified.
                result["adapted"][entry.module] = f"skipped, needs a device: {detail}"
                continue
            failures.append(f"{entry.module}: {detail}")
            result["adapted"][entry.module] = f"import failed: {detail}"
            continue
        applied = getattr(module, "__vllm_musa_adapted__", None)
        if applied is None:
            failures.append(
                f"{entry.module}: the adapter did not run; the hook was installed too late"
            )
            result["adapted"][entry.module] = "not adapted"
            continue
        stale = _stale_adapter(applied)
        if stale is not None:
            failures.append(f"{entry.module}: {stale}")
            result["adapted"][entry.module] = "stale (the pin provides it)"
            continue
        # The sampler can replace a binding after the adapter sets its marker.
        adapter = importlib.import_module(
            f"vllm_musa.omni_compat.adapters.{entry.adapter}"
        )
        live = adapter.is_applied(module)
        # Omni's v0.34 config module intentionally rebinds ModelConfig and
        # VllmConfig to subclasses after the adapter has added the inherited
        # method. The class identity changes, but the method contract survives
        # through inheritance and is_satisfied is the correct live check.
        if entry.adapter in {"config_model", "config_vllm"} and adapter.is_satisfied(
            module
        ):
            live = applied["names"]
        missing = [name for name in applied["names"] if name not in live]
        if missing:
            failures.append(
                f"{entry.module}: {', '.join(missing)} was adapted and is not any more; "
                "something rebound it after the hook ran"
            )
            result["adapted"][entry.module] = {
                "ran": applied["names"],
                "live": live,
            }
            continue
        result["adapted"][entry.module] = applied["names"]

    if args.require_omni:
        try:
            import vllm_omni

            result["vllm_omni_version"] = vllm_omni.__version__
        except Exception as exc:
            failures.append(f"import vllm_omni: {type(exc).__name__}: {exc}")
            result["vllm_omni_version"] = None

    result["failures"] = failures
    result["verdict"] = "pass" if not failures else "fail"

    text = json.dumps(result, indent=1, sort_keys=True)
    print(text)
    if args.report:
        with open(args.report, "w") as handle:
            handle.write(text + "\n")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
