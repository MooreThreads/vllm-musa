"""Source contract for model-local DeepSeek-OCR backend selection."""

from pathlib import Path

PATCH = (
    Path(__file__).parents[1]
    / "vllm_musa"
    / "patches"
    / "series"
    / "0194-MUSA-DeepSeek-OCR-model-backend-priority.patch"
)


def test_route_is_model_local() -> None:
    source = PATCH.read_text(encoding="utf-8")
    assert "vllm/model_executor/models/deepseek_ocr.py" in source
    assert "self.attn_backend = get_vit_attn_backend" in source
    assert "vllm/model_executor/models/config.py" not in source
    assert "vllm_musa/platform.py" not in source


def test_preserves_override_and_selects_triton_on_musa() -> None:
    source = PATCH.read_text(encoding="utf-8")
    assert "mm_encoder_attn_backend is None" in source
    assert "AttentionBackendEnum.TRITON_ATTN" in source
    assert "dtype=torch.float32" in source or "torch.float32" in source
