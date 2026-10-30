# SPDX-License-Identifier: Apache-2.0
"""Exercise DeepSeek-OCR's model-local backend configuration without a GPU."""

from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

PATCH = (
    Path(__file__).parents[1]
    / "vllm_musa"
    / "patches"
    / "series"
    / "0209-MUSA-default-DeepSeek-OCR-vision-attention-to-Triton.patch"
)


def test_route_is_model_local() -> None:
    source = PATCH.read_text(encoding="utf-8")
    assert "vllm/model_executor/models/deepseek_ocr.py" in source
    assert "get_vit_attn_backend" not in source
    assert "self.attn_backend" not in source
    assert "vllm/model_executor/models/config.py" not in source
    assert "vllm_musa/platform.py" not in source


@pytest.mark.parametrize("is_musa", [False, True])
@pytest.mark.parametrize("backend", [None, "TRITON_ATTN", "FLASH_ATTN", "CUSTOM"])
def test_model_backend_configuration(is_musa: bool, backend: str | None) -> None:
    source = PATCH.read_text(encoding="utf-8")
    # Execute the constructor block supplied by the patch, not a duplicate policy.
    block = dedent(
        "\n".join(
            line[1:] for line in source.splitlines() if line.startswith("+        ")
        )
    )
    config = SimpleNamespace(mm_encoder_attn_backend=backend)
    namespace = {
        "_is_musa_runtime": lambda: is_musa,
        "multimodal_config": config,
        "AttentionBackendEnum": SimpleNamespace(
            FLASH_ATTN="FLASH_ATTN", TRITON_ATTN="TRITON_ATTN"
        ),
    }
    exec(compile(block, str(PATCH), "exec"), namespace)
    expected = "TRITON_ATTN" if is_musa and backend is None else backend
    assert config.mm_encoder_attn_backend == expected
