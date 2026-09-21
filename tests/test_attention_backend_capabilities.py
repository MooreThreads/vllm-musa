# SPDX-License-Identifier: Apache-2.0
"""Guards the attention-backend capabilities MUSA shadows must keep declaring.

Several backend modules under ``vllm_musa/v1/attention/backends/`` are copies of
their upstream counterparts, and a copied ``AttentionBackend`` subclass must
re-declare every capability it actually supports: the abstract base answers
``False`` for all of them. That is a silent failure mode — nothing crashes at
import time, the backend is merely *rejected* during selection.

That is what happened to ``supports_sliding_window`` (MUSA-100051): the override
was dropped when the class was copied, so FLASH_ATTN was refused for any model
with a sliding window. A mixed sliding/full model then ran its sliding layers on
TRITON_ATTN and its full layers on FLASH_ATTN — two KV-cache layout families in
one step — and died in ``init_kv_cache`` with ``assert kv_cache.shape[1] == 2``.

The override is restored here, together with the tripwire regeneration and the
removal of the ``_INTENTIONAL_GAPS`` entry that #252 used to keep the gap
declared while the windowed MATE FA path had no numerical evidence. The tripwire
and the entry are a pair: ``test_known_intentional_gaps_still_exist`` fails if
one moves without the other.

The check is deliberately source-level (``ast``) so it runs anywhere, including
CPU-only CI, without importing torch or a MUSA build.
"""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "vllm_musa" / "patches" / "manifest.py"

# category "4a" entries are the module shadows that tripwire their delta against
# upstream; each one names the upstream file it shadows.
_SHADOW_CATEGORY = "4a"

# Upstream capabilities a MUSA shadow intentionally does NOT declare, keyed by
# entry id. Leave empty when the shadow is expected to keep full parity; every
# entry needs a reason, because it is exactly this list that hides a regression.
_INTENTIONAL_GAPS: dict[str, dict[str, str]] = {}


def _load_manifest():
    spec = importlib.util.spec_from_file_location("_musa_manifest", MANIFEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upstream_root() -> Path | None:
    """Root of the pinned vLLM checkout, or None when it is not present."""
    override = os.environ.get("VLLM_MUSA_UPSTREAM_ROOT")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_dir() else None
    candidate = ROOT / "third_party" / "vllm"
    return candidate if (candidate / "vllm").is_dir() else None


def _capabilities(path: Path) -> set[str]:
    """Names of every ``supports_*`` capability declared anywhere in a module."""
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("supports_")
    }


def _shadow_path(entry) -> Path:
    """MUSA file that shadows an upstream module path."""
    return ROOT / entry.upstream_path.replace("vllm/", "vllm_musa/", 1)


def _shadow_entries():
    return [e for e in _load_manifest().ENTRIES if e.category == _SHADOW_CATEGORY]


def test_there_is_something_to_check():
    entries = _shadow_entries()
    assert entries, "no cat-4a module shadows found; the manifest changed shape"
    assert any(
        _shadow_path(e).exists() for e in entries
    ), "no shadow file exists on disk; the id -> path mapping is wrong"


def test_every_musa_shadow_declares_every_upstream_capability():
    upstream_root = _upstream_root()
    if upstream_root is None:
        pytest.skip(
            "pinned vLLM checkout not available: set VLLM_MUSA_UPSTREAM_ROOT or "
            "run from a checkout with third_party/vllm populated"
        )

    missing_report = []
    for entry in _shadow_entries():
        shadow = _shadow_path(entry)
        upstream = upstream_root / entry.upstream_path
        if not shadow.exists() or not upstream.exists():
            continue
        gaps = _INTENTIONAL_GAPS.get(entry.id, {})
        missing = sorted(_capabilities(upstream) - _capabilities(shadow) - set(gaps))
        if missing:
            missing_report.append(f"{entry.id} ({shadow.name}): {missing}")
    assert not missing_report, (
        "a MUSA backend shadow no longer declares capabilities its upstream "
        "counterpart declares — the base AttentionBackend answers False for "
        "them, so backend selection will silently reject this backend:\n  "
        + "\n  ".join(missing_report)
        + "\nIf a gap is deliberate, add it to _INTENTIONAL_GAPS with a reason."
    )


def test_known_intentional_gaps_still_exist():
    """A documented gap must not outlive the code it excuses."""
    upstream_root = _upstream_root()
    if upstream_root is None:
        pytest.skip("pinned vLLM checkout not available")
    for entry_id, gaps in _INTENTIONAL_GAPS.items():
        entry = next((e for e in _shadow_entries() if e.id == entry_id), None)
        assert entry is not None, f"_INTENTIONAL_GAPS names unknown entry {entry_id}"
        shadow_caps = _capabilities(_shadow_path(entry))
        upstream_caps = _capabilities(upstream_root / entry.upstream_path)
        for name, reason in gaps.items():
            assert name in upstream_caps, f"{entry_id}.{name} is not upstream any more"
            assert name not in shadow_caps, (
                f"{entry_id} now declares {name}; remove it from _INTENTIONAL_GAPS "
                f"({reason})"
            )


def test_diffusion_model_is_pinned_to_triton_attn():
    """Diffusion passes a tensor `causal`; only TRITON_ATTN honours it (100051)."""
    platform = pytest.importorskip(
        "vllm_musa.platform",
        reason="vllm_musa.platform needs a MUSA build (torchada/torch_musa)",
    )
    force = platform.force_triton_attn_for_diffusion

    diffusion_config = SimpleNamespace(
        model_config=SimpleNamespace(is_diffusion=True),
        attention_config=SimpleNamespace(backend=None),
    )
    assert force(diffusion_config) is True
    assert (
        diffusion_config.attention_config.backend
        == platform.AttentionBackendEnum.TRITON_ATTN
    )

    explicit_fa = SimpleNamespace(
        model_config=SimpleNamespace(is_diffusion=True),
        attention_config=SimpleNamespace(
            backend=platform.AttentionBackendEnum.FLASH_ATTN
        ),
    )
    assert force(explicit_fa) is True
    assert (
        explicit_fa.attention_config.backend
        == platform.AttentionBackendEnum.TRITON_ATTN
    )

    other_backend = SimpleNamespace(
        model_config=SimpleNamespace(is_diffusion=True),
        attention_config=SimpleNamespace(
            backend=platform.AttentionBackendEnum.TURBOQUANT
        ),
    )
    assert force(other_backend) is False
    assert (
        other_backend.attention_config.backend
        == platform.AttentionBackendEnum.TURBOQUANT
    )

    already_triton = SimpleNamespace(
        model_config=SimpleNamespace(is_diffusion=True),
        attention_config=SimpleNamespace(
            backend=platform.AttentionBackendEnum.TRITON_ATTN
        ),
    )
    # the hook runs more than once per process; the second call must be a no-op
    assert force(already_triton) is False
    assert (
        already_triton.attention_config.backend
        == platform.AttentionBackendEnum.TRITON_ATTN
    )
    assert force(diffusion_config) is False, "repeated calls must stay idempotent"
    assert (
        diffusion_config.attention_config.backend
        == platform.AttentionBackendEnum.TRITON_ATTN
    )

    text_model = SimpleNamespace(
        model_config=SimpleNamespace(is_diffusion=False),
        attention_config=SimpleNamespace(backend=None),
    )
    assert force(text_model) is False
    assert text_model.attention_config.backend is None

    no_model = SimpleNamespace(
        model_config=None, attention_config=SimpleNamespace(backend=None)
    )
    assert force(no_model) is False
