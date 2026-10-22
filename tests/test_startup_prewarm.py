# SPDX-License-Identifier: Apache-2.0
"""Startup prewarm wiring (no device required: the JIT layer is stubbed)."""

from __future__ import annotations

import pytest

from vllm_musa.jit_kernel.csrc import rope
from vllm_musa.model_executor.warmup import deep_gemm_warmup as warmup


@pytest.fixture
def built(monkeypatch):
    calls: list[dict] = []

    def fake_prewarm(rot_dim=None) -> list[str]:
        calls.append({"rot_dim": rot_dim})
        return ["mp31_stwb", "mp31_stcg"]

    monkeypatch.setattr(rope, "prewarm_rope_modules", fake_prewarm)
    return calls


def test_rope_prewarm_receives_the_models_rotary_width(built) -> None:
    """The warmup must tell the prewarm which variants are reachable."""
    warmup._warmup_rope_store_hints()

    assert built == [{"rot_dim": warmup._rotary_dim()}]


def test_auto_policy_variants_depend_on_the_rotary_width() -> None:
    """`stcg` is only selected at rot_dim 64, so only it may skip the build."""
    assert rope.reachable_store_hints(64) == ("stwb", "stcg")
    assert rope.reachable_store_hints(128) == ("stwb",)
    assert rope.reachable_store_hints(None) == ("stwb", "stcg")


def test_rope_prewarm_never_blocks_startup(monkeypatch) -> None:
    """A prewarm failure must degrade to the lazy path, not abort the plugin."""

    def boom() -> list[str]:
        raise RuntimeError("no device")

    monkeypatch.delenv("VLLM_MUSA_PREWARM_ROPE", raising=False)
    monkeypatch.setattr(rope, "prewarm_rope_modules", boom)

    warmup._warmup_rope_store_hints()  # must not raise


def test_store_hint_auto_policy_selects_two_variants(monkeypatch) -> None:
    """The prewarm list must cover every value the auto policy can return."""
    monkeypatch.delenv("VLLM_MUSA_ROPE_STORE_HINT", raising=False)

    produced = {
        rope._store_hint(num_tokens, rot_dim)
        for num_tokens in (1, 512, 2048, 4096)
        for rot_dim in (64, 128)
    }

    assert produced <= set(rope._ROPE_STORE_HINTS)
    # both variants are actually reachable, which is why both are prewarmed
    assert produced == {"stwb", "stcg"}
