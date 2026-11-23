# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source checks for the MUSA QuantFP8 OOT registration."""

from pathlib import Path


def test_quant_fp8_oot_group_quant_uses_musa_helper():
    repo_root = Path(__file__).resolve().parents[1]
    model_executor_registration = (
        repo_root / "vllm_musa" / "model_executor" / "registration.py"
    )
    quant_fp8_oot = (
        repo_root
        / "vllm_musa"
        / "model_executor"
        / "layers"
        / "quantization"
        / "input_quant_fp8.py"
    )

    init_source = model_executor_registration.read_text()
    source = quant_fp8_oot.read_text()

    assert "vllm_musa.model_executor.layers.quantization.input_quant_fp8" in init_source
    assert "@QuantFP8.register_oot" in source
    assert "class MusaQuantFP8(QuantFP8)" in source
    assert (
        "self.is_group_quant and not self.static and current_platform.is_musa()"
        in source
    )
    assert "if x.dim() != 2:" in source
    assert "fp8_utils.per_token_group_quant_fp8" in source
    assert "return self.forward_native(x, scale, scale_ub, use_triton)" in source
