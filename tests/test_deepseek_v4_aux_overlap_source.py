# SPDX-License-Identifier: Apache-2.0
"""Source contract for the default DeepSeek-V4 auxiliary overlap path."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa/patches/series/0017-MUSA-vllm.models.deepseek_v4.nvidia.model.patch"
)


def test_deepseek_v4_aux_overlap_is_not_runtime_env_gated() -> None:
    patch = PATCH.read_text()

    assert "VLLM_MUSA_DEEPSEEK_V4_DISABLE_AUX_OVERLAP" not in patch
    assert "_musa_deepseek_v4_disable_aux_overlap" not in patch
    assert "current_platform.is_rocm()" in patch
    assert "current_platform.is_xpu()" in patch
    assert "else [torch.cuda.Stream() for _ in range(3)]" in patch
