# SPDX-License-Identifier: Apache-2.0
"""Startup prewarm wiring (no device required: the JIT layer is stubbed)."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_partial_rotary_models_use_the_rotated_width(monkeypatch) -> None:
    """Qwen3.5 reports head_dim 256 but rotates 64 lanes; stcg must survive."""
    hf_config = SimpleNamespace(head_dim=256, partial_rotary_factor=0.25)
    assert warmup._rotary_dim(hf_config) == 64

    full = SimpleNamespace(head_dim=128)
    assert warmup._rotary_dim(full) == 128


def test_auto_policy_variants_depend_on_the_rotary_width() -> None:
    """`stcg` is only selected at rot_dim 64, so only it may skip the build."""
    assert rope.reachable_store_hints(64) == ("stwb", "stcg")
    assert rope.reachable_store_hints(128) == ("stwb",)
    assert rope.reachable_store_hints(None) == ("stwb", "stcg")


def test_a_pinned_store_hint_is_the_only_variant_built(monkeypatch) -> None:
    """A pinned hint is what every request asks for, so the auto pair would mis-warm."""
    monkeypatch.setenv("VLLM_MUSA_ROPE_STORE_HINT", "stcs")
    assert rope.reachable_store_hints(64) == ("stcs",)

    monkeypatch.setenv("VLLM_MUSA_ROPE_STORE_HINT", "auto")
    assert rope.reachable_store_hints(64) == ("stwb", "stcg")


def test_the_arch_tag_accepts_an_arch_this_process_cannot_see(monkeypatch) -> None:
    """The failure message offers `VLLM_MUSA_ARCH_LIST`; it has to work."""
    monkeypatch.delenv("MATE_MUSA_ARCH_LIST", raising=False)
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "22")

    assert rope._musa_arch_tag() == "mp22"


def test_rope_prewarm_never_blocks_startup(monkeypatch) -> None:
    """A prewarm failure must degrade to the lazy path, not abort the plugin."""

    def boom(rot_dim=None) -> list[str]:
        raise RuntimeError("no device")

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
