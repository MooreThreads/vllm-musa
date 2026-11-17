# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source contract of the MATE Mega MoE backend wiring in the vLLM series."""

from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PATCH = (
    ROOT
    / "vllm_musa/patches/series/0212-MUSA-add-the-MATE-Mega-MoE-FP8-experts-backend.patch"
)
MODULE = ROOT / "vllm_musa/model_executor/layers/fused_moe/mate_mega_moe.py"


@pytest.mark.parametrize(
    "needle",
    [
        '+    "mate_mega_moe",',
        "+SELF_DISPATCHING_MOE_BACKENDS = FLASHINFER_MOE_EP_BACKENDS | {MATE_MEGA_MOE}",
        "+    if moe_backend in SELF_DISPATCHING_MOE_BACKENDS:",
        '+    MATE_MEGA_MOE = "MATE_MEGA_MOE"',
        "+        return [MateMegaMoEExperts]",
        '+        "mate_mega_moe": Fp8MoeBackend.MATE_MEGA_MOE,',
        "+            Fp8MoeBackend.MATE_MEGA_MOE,",
        "+        if self.fp8_backend == Fp8MoeBackend.MATE_MEGA_MOE:",
        "+            self.moe_kernel.fused_experts.process_weights_after_loading(layer)",
    ],
)
def test_series_patch_wires_the_backend(needle):
    assert needle in PATCH.read_text()


def test_experts_module_matches_the_patch_import():
    source = MODULE.read_text()
    assert "class MateMegaMoEExperts(mk.FusedMoEExpertsModular):" in source
    assert "def process_weights_after_loading(self, layer" in source
    assert "TopKWeightAndReduceNoOP()" in source
