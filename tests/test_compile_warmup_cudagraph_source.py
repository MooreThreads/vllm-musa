# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa/patches/series/0109-MUSA-keep-compile-warmup-outside-CUDAGraph.patch"
)


def _added_lines() -> list[str]:
    return [
        line[1:]
        for line in PATCH.read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def test_compile_warmup_forces_no_graph_runtime_mode() -> None:
    added = "\n".join(_added_lines())

    assert "self.model_runner._dummy_run(" in added
    assert "cudagraph_runtime_mode=CUDAGraphMode.NONE" in added
    assert "skip_eplb=True" in added
    assert "remove_lora=False" in added


def test_compile_warmup_fix_is_independent_of_configured_graph_mode() -> None:
    source = PATCH.read_text()
    added = "\n".join(_added_lines())

    configured_modes = (
        "PIECEWISE",
        "FULL",
        "FULL_DECODE_ONLY",
        "FULL_AND_PIECEWISE",
    )
    assert all(f"CUDAGraphMode.{mode}" not in added for mode in configured_modes)
    assert "compilation_config.cudagraph_mode =" not in source
    assert "cudagraph_capture_sizes =" not in source
    assert "compile_ranges_endpoints =" not in source
    assert "capture_model(" not in source
