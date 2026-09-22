# SPDX-License-Identifier: Apache-2.0
"""Guards the attention-backend capabilities MUSA shadows must keep declaring.

Several backend modules under ``vllm_musa/v1/attention/backends/`` are copies of
their upstream counterparts, and a copied ``AttentionBackend`` subclass must
re-declare every capability it actually supports: the abstract base answers
``False`` for all of them, nothing fails at import time, and the backend is merely
rejected during selection — so a dropped override is silent.

Two checks: every name the upstream class declares, and every name a shadow
answers a constant ``False`` from while upstream serves it. A deliberate refusal
belongs in ``_INTENTIONAL_GAPS`` with a reason.

Deliberately source-level (``ast``) so it runs anywhere, including CPU-only CI,
without importing torch or a MUSA build.
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


def _constant_capabilities(path: Path) -> dict[str, bool]:
    """``{name: value}`` for `supports_*` methods that just return a constant.

    The name-level check cannot see a shadow that declares the right method and
    answers ``False`` from it. This view catches that, and only that: methods
    with real logic are left to the reader.
    """
    tree = ast.parse(path.read_text())
    out: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("supports_")):
            continue
        body = [
            stmt
            for stmt in node.body
            if not (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )
        ]
        if (
            len(body) == 1
            and isinstance(body[0], ast.Return)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, bool)
        ):
            out[node.name] = body[0].value.value
    return out


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


def test_musa_fa_rejects_per_sequence_causal():
    """A per-request causal mask needs FA4; MATE is FA3-class, so refuse loudly.

    Without this the tensor travelled into ``causal=`` and surfaced as "Boolean
    value of Tensor with more than one element is ambiguous" from a branch
    condition, or silently selected the non-AOT scheduler path.
    """
    torch = pytest.importorskip("torch", reason="needs torch for the tensor check")
    fa_backend = pytest.importorskip(
        "vllm_musa.v1.attention.backends.flash_attn",
        reason="the MUSA FA shadow needs a MUSA build (torchada/torch_musa)",
    )
    reject = fa_backend.reject_per_sequence_causal

    # the profiling/dummy path: vLLM calls forward with attn_metadata=None before
    # its own early return, so the guard must be a no-op there (regression caught
    # on hardware: an unguarded `attn_metadata.causal` killed engine start for
    # every FA-served model).
    reject(None)
    reject(object())  # no `causal` attribute at all

    # scalar forms are exactly what mate's `causal: bool` accepts
    reject(SimpleNamespace(causal=True))
    reject(SimpleNamespace(causal=False))

    # any tensor is upstream-incompatible, including a single-request flag
    for tensor_mask in (torch.tensor([True, False]), torch.tensor([True])):
        with pytest.raises(NotImplementedError, match="requires FlashAttention v4"):
            reject(SimpleNamespace(causal=tensor_mask))


def test_no_shadow_constantly_refuses_a_capability_upstream_supports():
    """A shadow must not declare a capability and then constantly refuse it.

    The name-level parity check passes when a shadow declares every capability
    and answers `False` from the ones it cannot serve, which silently rejects it
    during selection. A deliberate refusal is declared in _INTENTIONAL_GAPS.
    """
    upstream_root = _upstream_root()
    if upstream_root is None:
        pytest.skip("pinned vLLM checkout not available")

    report = []
    for entry in _shadow_entries():
        shadow = _shadow_path(entry)
        upstream = upstream_root / entry.upstream_path
        if not shadow.exists() or not upstream.exists():
            continue
        gaps = _INTENTIONAL_GAPS.get(entry.id, {})
        upstream_values = _constant_capabilities(upstream)
        for name, value in _constant_capabilities(shadow).items():
            if (
                value is False
                and upstream_values.get(name) is True
                and name not in gaps
            ):
                report.append(
                    f"{entry.id} ({shadow.name}): {name}() returns False while "
                    f"upstream returns True"
                )
    assert not report, (
        "a MUSA shadow refuses a capability its upstream counterpart serves, so "
        "backend selection rejects the backend silently. Declare a deliberate gap "
        "in _INTENTIONAL_GAPS with a reason, or restore the capability:\n  "
        + "\n  ".join(report)
    )
