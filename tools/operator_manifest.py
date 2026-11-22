#!/usr/bin/env python3
"""Validate and report the vLLM-MUSA operator integration manifest.

This tool loads the manifest file directly so source/CI diagnostics do not
initialize torch, torchada, or MUSA extensions.
"""

import argparse
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "vllm_musa" / "operator_manifest.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    namespace = runpy.run_path(str(MANIFEST))
    records = namespace["OPERATOR_RECORDS"]
    errors = namespace["validate_records"](ROOT, records)
    report = namespace["operator_report"](ROOT, records)
    if args.as_json:
        print(json.dumps({"errors": errors, "operators": report}, indent=2))
    else:
        print(f"operators: {len(report)}")
        print(f"preload: {len(namespace['preload_modules'](records))}")
        for row in report:
            print(f"- {row['name']}: {row['category']} ({row['provider']})")
        if errors:
            print("errors:")
            for error in errors:
                print(f"  - {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
