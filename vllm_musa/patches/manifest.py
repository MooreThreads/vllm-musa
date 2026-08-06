# SPDX-License-Identifier: Apache-2.0
"""MUSA Divergence Manifest (MDM) — the single declarative census of every way
vllm-musa diverges from upstream vLLM.

Six categories:

  1  PY-EDIT    python source edit to an upstream vLLM file        (build, pre-install)
  2  CSRC-EDIT  C++/CUDA edit to an upstream csrc file             (build, pre-compile)
  3  CSRC-FILE  MUSA-owned whole-file rewrite (diff vs upstream)   (build, pre-compile)
  4a MOD-COPY   genuine modified-copy module (drift-tripwire only) (runtime, none)
  4b MOD-COPY   single-method rebind → build-applied diff          (build, pre-install)
  5  NEW-MOD    genuinely-new MUSA module/csrc (tracked source)    (runtime / direct compile)
  6  RUNTIME-OBJ live-object monkey patch (def apply())            (runtime)

**Plain Python, stdlib-only** (``dataclasses`` + ``pathlib``) so it can be loaded
by file path during the build — before ``vllm_musa`` is installed — exactly like
``build_apply.py``. Importing this module must NOT trigger ``vllm_musa/__init__``.

This seed covers the 36 category-1 patches already in ``series/`` and
the 2 category-6 object patches. Categories 2/3/4/5 are populated by the later
MDM phases; ``musa_sync`` and the census doc both read ENTRIES.
"""

# NOTE: deliberately NO ``from __future__ import annotations`` here. With future
# annotations a @dataclass whose fields are string annotations makes dataclasses
# look up ``sys.modules[cls.__module__]`` — which is absent when this file is
# loaded BY PATH at build time (setup.py / musa_sync), raising AttributeError.
# Real annotations + ``typing.Optional`` (not ``X | None``) keep both the 3.9
# floor and the by-path load working.
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_SERIES_DIR = _HERE / "series"

# MUSA-owned whole-file csrc rewrites (category 3). Every other csrc/ edit is
# category 2; everything else (vllm/*.py) is category 1.
_CAT3_FILES = {
    "csrc/custom_all_reduce.cu",
    "csrc/custom_all_reduce.cuh",
    "csrc/mamba/mamba_ssm/selective_scan_fwd.cu",
    "csrc/quantization/activation_kernels.cu",
}
_DIFF_RE = re.compile(r"^diff --git a/(\S+) b/", re.M)

VALID_CATEGORIES = ("1", "2", "3", "4a", "4b", "5", "6")
VALID_PHASES = ("pre-install", "pre-compile", "runtime", "none")
# Categories whose form is a build-applied diff in series/ (consumed by build_apply).
BUILD_APPLIED_CATEGORIES = ("1", "2", "3", "4b")


@dataclass(frozen=True)
class DivSpec:
    """One declared divergence from upstream vLLM."""

    id: str
    category: str  # one of VALID_CATEGORIES
    path: (
        str  # repo-relative: the .patch (cat 1/2/3/4b), tripwire (4a), or source (5/6)
    )
    upstream_path: Optional[str] = None  # the vllm path this targets / the seam owner
    apply_phase: str = "pre-install"  # one of VALID_PHASES
    required: bool = True
    version_range: Optional[str] = None
    removal_condition: Optional[str] = None
    intent: str = ""
    after: tuple = ()  # ordering deps by id (e.g. eagle after parallel_state)

    def __post_init__(self):
        assert (
            self.category in VALID_CATEGORIES
        ), f"{self.id}: bad category {self.category!r}"
        assert (
            self.apply_phase in VALID_PHASES
        ), f"{self.id}: bad apply_phase {self.apply_phase!r}"


def _patch_target(patch_path: Path) -> str:
    """The first ``diff --git a/<path>`` target of a .patch (the file it edits)."""
    m = _DIFF_RE.search(patch_path.read_text(errors="replace"))
    return m.group(1) if m else ""


def _series_entries() -> list:
    """One DivSpec per ``series/*.patch``, classified by its target file:

    - ``csrc/`` whole-file rewrite (in ``_CAT3_FILES``) → category 3, pre-compile
    - other ``csrc/`` edit → category 2, pre-compile
    - everything else (``vllm/*.py``) → category 1, pre-install

    Derived from the series directory so the manifest stays in sync with the
    build-applied diffs by construction. Per-patch intent/version_range will be
    enriched from the PatchSpec commit-trailers in a later phase.
    """
    out: list = []
    if not _SERIES_DIR.is_dir():
        return out
    for p in sorted(_SERIES_DIR.glob("*.patch")):
        target = _patch_target(p)
        if target.startswith("csrc/"):
            cat = "3" if target in _CAT3_FILES else "2"
            phase = "pre-compile"
            intent = (
                "MUSA-owned whole-file csrc rewrite (diff vs upstream)"
                if cat == "3"
                else "csrc source edit to upstream vLLM"
            )
        else:
            cat = "1"
            phase = "pre-install"
            intent = (
                "python source edit to upstream vLLM (migrated runtime patch)"
                if target.endswith(".py")
                else f"build-applied edit to upstream {target}"
            )
        out.append(
            DivSpec(
                id=p.stem,
                category=cat,
                path=f"vllm_musa/patches/series/{p.name}",
                upstream_path=(target or None),
                apply_phase=phase,
                required=True,
                intent=intent,
            )
        )
    return out


# Category 6 — live-object monkey patches; applied by apply_object_patches(), not
# build_apply. No source-diff form. Ordering: eagle primes after parallel_state.
_CAT6: list[DivSpec] = [
    DivSpec(
        id="vllm__distributed__parallel_state",
        category="6",
        path="vllm_musa/patches/vllm__distributed__parallel_state.patch.py",
        upstream_path="vllm/distributed/parallel_state.py",
        apply_phase="runtime",
        required=False,
        intent="Eagle3 draft-at-TP=1 wiring (opt-in via VLLM_MUSA_DRAFT_TP1)",
    ),
    DivSpec(
        id="vllm__v1__spec_decode__eagle",
        category="6",
        path="vllm_musa/patches/vllm__v1__spec_decode__eagle.patch.py",
        upstream_path="vllm/v1/spec_decode/eagle.py",
        apply_phase="runtime",
        required=True,
        after=("vllm__distributed__parallel_state",),
        intent="Eagle3 draft kernel prime",
    ),
    # torch/vLLM config compat shims migrated from inline
    # vllm_musa/__init__.py functions to cat-6 object patches (def apply(),
    # loaded by apply_object_patches alongside the two above). torch.* targets
    # use upstream_path=None (not probeable from the vLLM clone).
    DivSpec(
        id="torch___functorch__config",
        category="6",
        path="vllm_musa/patches/torch___functorch__config.patch.py",
        upstream_path=None,
        apply_phase="runtime",
        required=False,
        intent="filter missing torch._functorch.config keys in vLLM compile contexts",
    ),
    DivSpec(
        id="torch___inductor__config",
        category="6",
        path="vllm_musa/patches/torch___inductor__config.patch.py",
        upstream_path=None,
        apply_phase="runtime",
        required=False,
        intent="filter missing torch._inductor.config keys in vLLM compile contexts",
    ),
    DivSpec(
        id="torch___inductor__aot_cache_safelist",
        category="6",
        path="vllm_musa/patches/torch___inductor__aot_cache_safelist.patch.py",
        upstream_path=None,
        apply_phase="runtime",
        required=False,
        intent="mark torch-wrap Tensor methods AOT-cache safe on torch 2.9 (compile cache)",
    ),
    DivSpec(
        id="vllm__compilation__backends",
        category="6",
        path="vllm_musa/patches/vllm__compilation__backends.patch.py",
        upstream_path="vllm/compilation/backends.py",
        apply_phase="runtime",
        required=False,
        intent=(
            "VllmBackend accepts unsupported torch.compile kwargs and applies the "
            "exact Qwen2 RoPE+KV raw-FX pre-split rewrite"
        ),
    ),
    DivSpec(
        id="vllm__compilation__compiler_interface",
        category="6",
        path="vllm_musa/patches/vllm__compilation__compiler_interface.patch.py",
        upstream_path="vllm/compilation/compiler_interface.py",
        apply_phase="runtime",
        required=False,
        intent="filter vLLM functorch config overrides for the installed Torch",
    ),
    DivSpec(
        id="vllm___custom_ops",
        category="6",
        path="vllm_musa/patches/vllm___custom_ops.patch.py",
        upstream_path="vllm/_custom_ops.py",
        apply_phase="runtime",
        required=False,
        intent="route vllm._custom_ops rms_norm/rotary to MUSA-safe dflash fallbacks",
    ),
]

# census of the runtime-override / vendored shadow modules (vllm_musa
# .py files that mirror a vllm path), classified by `musa_sync census` (difflib
# similarity vs upstream v0.22 + register-seam refinement). Categories:
#   4a = genuine modified COPY → drift tripwire, NEVER build-applied (apply_phase=none)
#   5  = tracked MUSA source: a new module, an OOT @register_oot/register_backend
#        subclass, or a low-sim rebind flagged for the 4b conversion.
# Tuples: (category, repo-path, upstream_path|None, intent). Regenerate with
# `musa_sync census` (difflib needs an upstream clone, so this is frozen data).
_SHADOW_MODULES = [
    (
        "5",
        "vllm_musa/compilation/passes/pass_manager.py",
        "vllm/compilation/passes/pass_manager.py",
        "OOT post-grad pass manager with MUSA-only fusion registration",
    ),
    (
        "5",
        "vllm_musa/compilation/passes/silu_deepgemm_fusion.py",
        "vllm/compilation/passes/fusion/act_quant_fusion.py",
        "MUSA dense SwiGLU plus FP8 DeepGEMM fusion pattern",
    ),
    (
        "5",
        "vllm_musa/compilation/passes/rms_deepgemm_fusion.py",
        "vllm/compilation/passes/fusion/rms_quant_fusion.py",
        "MUSA residual RMSNorm plus FP8 DeepGEMM fusion pattern",
    ),
    (
        "4a",
        "vllm_musa/v1/attention/backends/flash_attn.py",
        "vllm/v1/attention/backends/flash_attn.py",
        "modified copy (sim 0.70) — drift tripwire",
    ),
    (
        "4a",
        "vllm_musa/v1/attention/backends/mla/flashmla.py",
        "vllm/v1/attention/backends/mla/flashmla.py",
        "modified copy (sim 0.83) — drift tripwire",
    ),
    (
        "5",
        "vllm_musa/distributed/device_communicators/musa_jit_custom_all_reduce.py",
        None,
        "new MUSA JIT custom allreduce communicator",
    ),
    (
        "5",
        "vllm_musa/model_executor/layers/attention/mm_encoder_attention.py",
        "vllm/model_executor/layers/attention/mm_encoder_attention.py",
        "OOT MM encoder attention routing with MUSA FlashAttention",
    ),
    (
        "5",
        "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        "vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py",
        "OOT subclass/register seam",
    ),
    (
        "5",
        "vllm_musa/model_executor/kernels/linear/scaled_mm/torch_scaled_mm.py",
        None,
        "new MUSA module (no upstream)",
    ),
    (
        "5",
        "vllm_musa/model_executor/layers/activation.py",
        "vllm/model_executor/layers/activation.py",
        "OOT subclass/register seam",
    ),
    (
        "5",
        "vllm_musa/model_executor/layers/fused_moe/fused_moe.py",
        "vllm/model_executor/layers/fused_moe/fused_moe.py",
        "rebinds upstream (sim 0.08) — 4b-conversion candidate",
    ),
    (
        "5",
        "vllm_musa/model_executor/layers/fused_moe/router/grouped_topk_router.py",
        "vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py",
        "shadow (sim 0.25)",
    ),
    (
        "5",
        "vllm_musa/model_executor/layers/fused_moe/unquantized_fused_moe_method.py",
        "vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py",
        "OOT subclass/register seam",
    ),
    (
        "5",
        "vllm_musa/model_executor/layers/layernorm.py",
        "vllm/model_executor/layers/layernorm.py",
        "OOT subclass/register seam",
    ),
    (
        "5",
        "vllm_musa/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "OOT pluggable replacement for Qwen3.5 GDN MATE prefill/decode",
    ),
    (
        "5",
        "vllm_musa/model_executor/layers/quantization/fp8.py",
        "vllm/model_executor/layers/quantization/fp8.py",
        "rebinds upstream (sim 0.02) — 4b-conversion candidate",
    ),
    (
        "5",
        "vllm_musa/model_executor/layers/quantization/input_quant_fp8.py",
        "vllm/model_executor/layers/quantization/input_quant_fp8.py",
        "OOT subclass/register seam",
    ),
    (
        "5",
        "vllm_musa/model_executor/layers/quantization/utils/fp8_utils.py",
        "vllm/model_executor/layers/quantization/utils/fp8_utils.py",
        "shadow (sim 0.12)",
    ),
    (
        "5",
        "vllm_musa/model_executor/layers/rotary_embedding/base.py",
        "vllm/model_executor/layers/rotary_embedding/base.py",
        "OOT subclass/register seam",
    ),
    (
        "5",
        "vllm_musa/model_executor/warmup/deep_gemm_warmup.py",
        "vllm/model_executor/warmup/deep_gemm_warmup.py",
        "shadow (sim 0.02)",
    ),
    (
        "5",
        "vllm_musa/v1/attention/backends/fa_utils.py",
        "vllm/v1/attention/backends/fa_utils.py",
        "shadow (sim 0.03)",
    ),
    (
        "5",
        "vllm_musa/v1/attention/backends/mla/common.py",
        None,
        "new MUSA module (no upstream)",
    ),
    (
        "5",
        "vllm_musa/v1/attention/backends/mla/flashmla_sparse.py",
        "vllm/v1/attention/backends/mla/flashmla_sparse.py",
        "OOT subclass/register seam",
    ),
    (
        "5",
        "vllm_musa/v1/attention/backends/tree_attn.py",
        None,
        "new MUSA module (upstream deleted TREE_ATTN at v0.22)",
    ),
    (
        "5",
        "vllm_musa/v1/attention/backends/turboquant.py",
        None,
        "new MUSA module (no upstream)",
    ),
    (
        "5",
        "vllm_musa/v1/attention/ops/flashmla.py",
        "vllm/v1/attention/ops/flashmla.py",
        "shadow (sim 0.06)",
    ),
    (
        "5",
        "vllm_musa/v1/executor/multiproc_executor.py",
        "vllm/v1/executor/multiproc_executor.py",
        "rebinds upstream (sim 0.08) — 4b-conversion candidate",
    ),
    (
        "5",
        "vllm_musa/v1/sample/topk_topp_sampler.py",
        "vllm/v1/sample/ops/topk_topp_sampler.py",
        "shadow (sim 0.05)",
    ),
    (
        "5",
        "vllm_musa/v1/spec_decode/utils.py",
        "vllm/v1/spec_decode/utils.py",
        "shadow (sim 0.24)",
    ),
    (
        "5",
        "vllm_musa/worker.py",
        None,
        "new MUSA module (no upstream same-path; MTGPUWorker)",
    ),
]


def _module_entries() -> list:
    """DivSpec per shadow module (cat 4a / 5) from the frozen census."""
    out: list = []
    for cat, path, upstream, intent in _SHADOW_MODULES:
        out.append(
            DivSpec(
                id=path[len("vllm_musa/") :].rsplit(".py", 1)[0].replace("/", "."),
                category=cat,
                path=path,
                upstream_path=upstream,
                apply_phase="none" if cat == "4a" else "runtime",
                required=False,
                intent=intent,
            )
        )
    return out


ENTRIES: list[DivSpec] = _series_entries() + _module_entries() + _CAT6


def entries_for_phase(phase: str) -> list[DivSpec]:
    """All entries with the given ``apply_phase``, in declaration order."""
    return [e for e in ENTRIES if e.apply_phase == phase]


def series_apply_order(phase: Optional[str] = None) -> list[Path]:
    """Absolute ``.patch`` paths for the build-applied diff categories (1/2/3/4b),
    in apply order — the ``order`` list to hand to ``build_apply.apply_patch_series``.

    With ``phase`` set, restrict to that ``apply_phase`` (e.g. ``pre-install`` for
    cat-1/4b, ``pre-compile`` for cat-2/3).
    """
    root = _HERE.parent.parent  # vllm_musa/patches → repo root
    out: list[Path] = []
    for e in ENTRIES:
        if e.category not in BUILD_APPLIED_CATEGORIES:
            continue
        if phase is not None and e.apply_phase != phase:
            continue
        out.append(root / e.path)
    return out


def object_entries() -> list[DivSpec]:
    """Category-6 live-object patches (runtime, def apply())."""
    return [e for e in ENTRIES if e.category == "6"]
