import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
PUBLIC_ARTIFACTS = (
    ROOT / "benchmarks/kernel_tactics/full_kernel_sweep.json",
    ROOT / "benchmarks/kernel_tactics/mp_tactic_campaign.json",
    ROOT / "benchmarks/kernel_tactics/qwen_dsv4_fp8_quant_shapes.json",
    ROOT / "benchmarks/kernel_tactics/qwen_dsv4_rmsnorm_shapes.json",
    ROOT / "docs/vllm_musa/README.md",
    ROOT / "docs/vllm_musa/hardware-aware-kernel-tactics.md",
)
FLEET_RECEIPT_MARKERS = (
    "10.20.",
    "archive_sha256",
    "generated/MUSA-",
    "GPU UUID :",
    "driver_version",
)


def test_public_kernel_artifacts_exclude_fleet_receipts() -> None:
    for path in PUBLIC_ARTIFACTS:
        content = path.read_text(encoding="utf-8")
        assert not any(marker in content for marker in FLEET_RECEIPT_MARKERS), path


def test_public_json_artifacts_are_parseable() -> None:
    for path in PUBLIC_ARTIFACTS[:4]:
        json.loads(path.read_text(encoding="utf-8"))
