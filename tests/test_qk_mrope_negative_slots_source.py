# SPDX-License-Identifier: Apache-2.0
"""Source contracts for negative qk_mrope cache slots."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "vllm_musa/jit_kernel/csrc/norm/qk_mrope.mu").read_text(
    encoding="utf-8"
)

SPECIALIZED_CACHE_KERNELS = (
    "fused_qk_rmsnorm_mrope_cache_32q4kv_h128_bf16_kernel",
    "fused_qk_rmsnorm_mrope_cache_h128_full_bf16_kernel",
    "fused_qk_rmsnorm_rope_cache_h128_full_bf16_kernel",
    "fused_qk_rmsnorm_rope_cache_h128r64_bf16_kernel",
    "fused_qk_rmsnorm_mrope_cache_h256r64_bf16_kernel",
)


def _braced_body(source: str, marker: str, start: int = 0) -> str:
    marker_offset = source.index(marker, start)
    body_start = source.index("{", marker_offset)
    depth = 0
    for offset in range(body_start, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start + 1 : offset]
    raise AssertionError(f"unclosed source block after {marker!r}")


def _all_braced_bodies(source: str, marker: str) -> list[str]:
    bodies = []
    offset = 0
    while marker in source[offset:]:
        marker_offset = source.index(marker, offset)
        body = _braced_body(source, marker, marker_offset)
        bodies.append(body)
        body_start = source.index("{", marker_offset)
        offset = body_start + len(body) + 2
    return bodies


@pytest.mark.parametrize("kernel_name", SPECIALIZED_CACHE_KERNELS)
def test_specialized_cache_kernel_guards_negative_slots(kernel_name: str) -> None:
    body = _braced_body(SOURCE, f"__global__ void {kernel_name}(")

    assert "const bool write_cache = cache_idx >= 0;" in body
    assert "if (group == 0 && write_cache)" in body
    assert "if (group == 0)" not in body

    value_cache_block = _braced_body(body, "if (group == 0 && write_cache)")
    assert "v_cache" in value_cache_block
    assert len(re.findall(r"\bv_cache\b", body)) == len(re.findall(r"\bv_cache\b", value_cache_block))

    cache_write_blocks = _all_braced_bodies(body, "if (write_cache)")
    assert cache_write_blocks
    assert any("k_cache" in block for block in cache_write_blocks)
    assert len(re.findall(r"\bk_cache\b", body)) == sum(
        len(re.findall(r"\bk_cache\b", block)) for block in cache_write_blocks
    )
    assert all("k_out" not in block for block in cache_write_blocks)
    assert "if constexpr (STORE_K_OUT)" in body


def test_generic_cache_kernel_guards_negative_slots_but_keeps_k_out() -> None:
    body = _braced_body(
        SOURCE, "__global__ void fused_qk_rmsnorm_mrope_generic_bf16_kernel("
    )

    assert "write_cache = cache_idx >= 0;" in body
    assert "if (group == 0 && write_cache)" in body
    assert "if (group == 0)" not in body

    value_cache_block = _braced_body(body, "if (group == 0 && write_cache)")
    assert "v_cache" in value_cache_block
    assert len(re.findall(r"\bv_cache\b", body)) == len(re.findall(r"\bv_cache\b", value_cache_block))

    cache_write_blocks = _all_braced_bodies(body, "if (write_cache)")
    assert len(cache_write_blocks) == 2
    assert all("k_cache" in block for block in cache_write_blocks)
    assert len(re.findall(r"\bk_cache\b", body)) == sum(
        len(re.findall(r"\bk_cache\b", block)) for block in cache_write_blocks
    )
    assert all("k_out" not in block for block in cache_write_blocks)
    assert body.count("if (k_out != nullptr)") == 2
