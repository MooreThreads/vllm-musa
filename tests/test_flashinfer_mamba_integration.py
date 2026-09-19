"""Source contracts for the Nemotron FlashInfer Mamba consumer stack."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "third_party" / "PINS"
SERIES = ROOT / "vllm_musa" / "patches" / "series"


def _pins() -> dict[str, str]:
    values = {}
    for line in PINS.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_flashinfer_pin_targets_musa_nemotron_fork() -> None:
    pins = _pins()
    assert pins["FLASHINFER_REPOSITORY"] == "https://github.com/yeahdongcn/flashinfer.git"
    assert re.fullmatch(r"[0-9a-f]{40}", pins["FLASHINFER_COMMIT"])
    assert pins["FLASHINFER_COMMIT"] == "a1c69c8ebef07f2632707f96b4f2785f773ecf1c"


def test_setup_overlays_pinned_mamba_provider() -> None:
    setup = (ROOT / "setup.py").read_text()
    assert '"FLASHINFER_REPOSITORY"' in setup
    assert "def _install_flashinfer_mamba" in setup
    assert "MUSA_PROVIDER_COMMIT" in setup
    assert "SKIP_THIRD_PARTY" in setup


def test_consumer_patch_series_exposes_flashinfer_backend_controls() -> None:
    expected = {
        "0151-MUSA-enable-FlashInfer-Mamba-stochastic-rounding.patch": "is_musa_flashinfer",
        "0152-MUSA-route-Mamba2-SSD-to-FlashInfer.patch": "ssd_combined_fwd_varlen",
        "0153-MUSA-select-Mamba2-SSD-backend.patch": "backend=self.mamba_config.backend.value",
        "0154-MUSA-initialize-Mamba2-profiling-output.patch": "profiling",
    }
    for name, needle in expected.items():
        patch = SERIES / name
        assert patch.is_file(), name
        assert needle in patch.read_text(), name
