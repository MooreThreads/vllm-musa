# SPDX-License-Identifier: Apache-2.0
"""Source contract for asynchronous MUSA MRoPE position uploads."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_musa_mrope_copy_uses_contiguous_rows_without_a_gate() -> None:
    source = (ROOT / "third_party/vllm/vllm/v1/worker/gpu_model_runner.py").read_text(
        encoding="utf-8"
    )

    assert "if self.uses_mrope:" in source
    assert "for row in range(self.mrope_positions.gpu.shape[0]):" in source
    assert "self.mrope_positions.gpu[row, :total_num_scheduled_tokens].copy_(" in source
    assert "self.mrope_positions.cpu[row, :total_num_scheduled_tokens]," in source
    assert "non_blocking=True," in source
    assert "VLLM_MUSA_MROPE_ROW_COPY" not in source
