# SPDX-License-Identifier: Apache-2.0
"""The declarative drift table: the one file you edit to support a new vLLM-Omni tree.

vLLM-Omni declares no ``vllm`` requirement, tracks vLLM main, and imports vLLM
internals directly, while vLLM-MUSA pins one immutable vLLM commit per release
branch. When Omni moves ahead of the pin its imports break. Each entry below
records one such break, which vLLM versions it applies to, how it is absorbed,
and the condition that makes the entry dead.

Four kinds, in increasing order of risk:

``RELOCATION``
    The module moved but the symbols are unchanged. The shim serves the new
    module name from the old location. Zero behaviour risk.

``ADAPTER``
    The module and symbol exist, but Omni calls the symbol with an argument the
    pinned signature cannot take, so the call raises ``TypeError`` at runtime
    rather than failing to import. A module in
    :mod:`vllm_musa.omni_compat.adapters` patches the pinned function when its
    module is first imported. Behaviour risk is whatever the adapter's docstring
    argues; when it cannot argue the translation is safe, the row belongs under
    ``CAPABILITY`` instead.

``BACKPORT``
    The symbol does not exist in the pin at all. The backport module implements
    it against pinned APIs. Highest risk: read its docstring before trusting it.

``CAPABILITY``
    Nothing is missing at import time, but a pinned API behaves differently
    under the same name and arity, so an Omni code path that looks fine is
    wrong. The shim cannot fix these by aliasing; the entry names the
    unsupported path so the audit can report it and a smoke can avoid it.

Entries are data: :mod:`vllm_musa.omni_compat` reads them, the audit in
``tools/check_omni_compat.py`` reports them, and ``tests/test_omni_compat.py``
fails when one is stale. Adding support for a new Omni commit is a new entry
plus a test, never a change to the installer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RELOCATION = "relocation"
ADAPTER = "adapter"
BACKPORT = "backport"
CAPABILITY = "capability"

KINDS = (RELOCATION, ADAPTER, BACKPORT, CAPABILITY)


@dataclass(frozen=True)
class ShimEntry:
    """One drift item between a vLLM-Omni tree and a pinned vLLM.

    ``applies_to`` is a PEP 440 specifier set matched against the installed
    ``vllm.__version__``; the shim is inert outside it, so a pin bump that
    removes the drift disables the entry without a code change.
    """

    module: str
    """The module name the Omni tree imports."""

    names: tuple[str, ...]
    """Symbols Omni imports from ``module``. Documented and verified, not enforced."""

    kind: str
    """One of :data:`KINDS`."""

    applies_to: str
    """PEP 440 specifier set over the installed vLLM version, e.g. ``"<0.33"``."""

    reason: str
    """Why the drift exists, in one line."""

    remove_when: str
    """The condition under which this entry is dead and must be deleted."""

    source: str | None = None
    """``RELOCATION``: the module in the pinned vLLM that holds ``names``."""

    backport: str | None = None
    """``BACKPORT``: module under ``omni_compat.backports`` implementing the missing
    module."""

    adapter: str | None = None
    """``ADAPTER``: module under ``omni_compat.adapters`` patching the pinned module."""

    target: str | None = None
    """``CAPABILITY``: the pinned attribute whose meaning changed."""

    unsupported: tuple[str, ...] = field(default_factory=tuple)
    """``CAPABILITY``: the Omni/vLLM feature paths that are not usable on the pin."""

    omni_refs: tuple[str, ...] = field(default_factory=tuple)
    """Omni source files that depend on this entry, for the audit report."""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(
                f"{self.module}: unknown kind {self.kind!r}, expected one of {KINDS}"
            )
        if self.kind == RELOCATION and not self.source:
            raise ValueError(f"{self.module}: {RELOCATION} needs source=")
        if self.kind == BACKPORT and not self.backport:
            raise ValueError(f"{self.module}: {BACKPORT} needs backport=")
        if self.kind == ADAPTER and not self.adapter:
            raise ValueError(f"{self.module}: {ADAPTER} needs adapter=")
        if self.kind == CAPABILITY:
            if not self.target:
                raise ValueError(f"{self.module}: {CAPABILITY} needs target=")
            if not self.unsupported:
                raise ValueError(f"{self.module}: {CAPABILITY} needs unsupported=")
        if self.kind != CAPABILITY and not self.names:
            raise ValueError(
                f"{self.module}: names= must list the symbols Omni imports"
            )


# --------------------------------------------------------------------------- #
# vLLM-Omni main (the vLLM 0.34 line) on the vLLM 0.32 pin.
#
# Measured with tools/check_omni_compat.py against Omni main on the vLLM 0.34
# line: 4307 imported vLLM symbols, 3 additional imports absent from 0.32, all
# covered below, plus 174 module attributes of which 2 are absent. The tool does
# not yet see constructor keywords or return arity, so runtime support remains
# path-specific. See docs/omni_compat.md.
# --------------------------------------------------------------------------- #
ENTRIES: tuple[ShimEntry, ...] = (
    ShimEntry(
        module="vllm.config.utils",
        names=("CompileFactors",),
        kind=ADAPTER,
        adapter="config_utils",
        applies_to="<0.34",
        reason=(
            "vLLM 0.34 introduced the CompileFactors type alias used by Omni's "
            "ModelConfig. The pin still exposes the equivalent legacy compute_hash "
            "API, so the alias is enough for imports and the config adapters below "
            "bridge the method called by Omni's subclass."
        ),
        remove_when=(
            "the pin is >= 0.34, where vllm.config.utils defines CompileFactors."
        ),
        omni_refs=("vllm_omni/config/model.py",),
    ),
    ShimEntry(
        module="vllm.config.model",
        names=("ModelConfig",),
        kind=ADAPTER,
        adapter="config_model",
        applies_to="<0.34",
        reason=(
            "Omni's v0.34 ModelConfig override calls super().compile_factors(), "
            "while the 0.32 pin still names the legacy method compute_hash. The "
            "adapter exposes a stable factor dictionary backed by that legacy hash."
        ),
        remove_when="the pin is >= 0.34, where ModelConfig.compile_factors is native.",
        omni_refs=("vllm_omni/config/model.py",),
    ),
    ShimEntry(
        module="vllm.config.vllm",
        names=("VllmConfig",),
        kind=ADAPTER,
        adapter="config_vllm",
        applies_to="<0.34",
        reason=(
            "The v0.34 compilation cache asks VllmConfig for compile_factors, "
            "where the pin exposes compute_hash. Returning that legacy hash as a "
            "single factor keeps the key deterministic on the pinned runtime."
        ),
        remove_when="the pin is >= 0.34, where VllmConfig.compile_factors is native.",
        omni_refs=("vllm_omni/config/model.py",),
    ),
    ShimEntry(
        module="vllm.v1.kv_cache_interface",
        names=("get_block_table_width",),
        kind=ADAPTER,
        adapter="kv_cache_interface",
        applies_to="<0.34",
        reason=(
            "vLLM 0.34 moved get_block_table_width from worker.block_table into "
            "kv_cache_interface. The implementation and signature are unchanged, "
            "so the adapter forwards the exact alignment calculation."
        ),
        remove_when="the pin is >= 0.34, where kv_cache_interface defines the helper.",
        omni_refs=("vllm_omni/diffusion/diffusion_kv/model_runner_backend.py",),
    ),
    ShimEntry(
        module="vllm.v1.worker.gpu.spec_decode.draft_logits",
        names=("DraftLogitsCache",),
        kind=BACKPORT,
        backport="draft_logits",
        applies_to="<0.34",
        reason=(
            "The v0.34 MiniCPM-o Talker imports DraftLogitsCache for its rejection "
            "sampler annotation, but the 0.32 pin has no matching module. The "
            "Talker delegates actual verification to the pinned RejectionSampler, "
            "so the backport is a type-only guard that fails explicitly if someone "
            "tries to construct it on an unsupported speculative path."
        ),
        remove_when="the pin is >= 0.34, where the draft_logits module is native.",
        omni_refs=(
            "vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py",
        ),
    ),
    ShimEntry(
        module="vllm.v1.worker.gpu.spec_decode.speculator_config",
        names=("SpeculatorConfig", "register_speculative_method"),
        kind=BACKPORT,
        backport="speculator_config",
        applies_to="<0.34",
        reason=(
            "vLLM 0.34 added speculator_config for custom speculative methods, and "
            "Omni's register_omni_models_to_vllm() registers "
            "HiggsAudioSpeculatorConfig unconditionally at startup. Recording the "
            "registration keeps Omni imports working while construction fails "
            "explicitly on the 0.32 pin."
        ),
        remove_when="the pin is >= 0.34, where speculator_config is native.",
        omni_refs=(
            "vllm_omni/engine/arg_utils.py",
            "vllm_omni/model_executor/models/higgs_audio_v3/spec_config.py",
        ),
    ),
    ShimEntry(
        module="vllm.v1.worker.extensible_kv_cache",
        names=("ensure_kv_cache_blocks", "num_committable_kv_blocks"),
        kind=BACKPORT,
        backport="extensible_kv_cache",
        applies_to="<0.34",
        reason=(
            "vLLM 0.34 extracted KV-cache warmup block budget helpers into "
            "extensible_kv_cache, which Omni's Triton attention prefill warmup "
            "imports. On the fixed-allocation 0.32 pin all allocated KV blocks are "
            "already committed."
        ),
        remove_when="the pin is >= 0.34, where extensible_kv_cache is native.",
        omni_refs=("vllm_omni/worker/triton_attention_warmup.py",),
    ),
    ShimEntry(
        module="vllm.v1.worker.gpu.warmup",
        names=("_run_step", "_warmup_token_ids"),
        kind=ADAPTER,
        adapter="worker_gpu_warmup",
        applies_to="<0.34",
        reason=(
            "vLLM 0.34 factored _run_step and _warmup_token_ids out of the GPU "
            "worker warmup helpers, and Omni's Triton attention prefill warmup "
            "imports both from vllm.v1.worker.gpu.warmup."
        ),
        remove_when=(
            "the pin is >= 0.34, where warmup defines _run_step and _warmup_token_ids."
        ),
        omni_refs=("vllm_omni/worker/triton_attention_warmup.py",),
    ),
    ShimEntry(
        module="vllm.entrypoints.common.cli_args",
        names=(
            "FrontendArgs",
            "make_arg_parser",
            "resolve_default_chat_template_kwargs",
            "validate_parsed_serve_args",
        ),
        kind=RELOCATION,
        source="vllm.entrypoints.launchers.cli_args",
        applies_to="<0.33",
        reason=(
            "0.33 moved the serve CLI argument surface from entrypoints.launchers to "
            "entrypoints.common."
        ),
        remove_when=(
            "the pin is >= 0.33, where vllm.entrypoints.common.cli_args exists."
        ),
        omni_refs=(
            "vllm_omni/config/omni_config.py",
            "vllm_omni/entrypoints/cli/serve.py",
            "vllm_omni/entrypoints/openai/api_server.py",
        ),
    ),
    ShimEntry(
        module="vllm.entrypoints.common.startup",
        names=("VLLM_SUBCMD_PARSER_EPILOG", "cli_env_setup"),
        kind=RELOCATION,
        source="vllm.entrypoints.serve.utils.api_utils",
        applies_to="<0.33",
        reason="0.33 split CLI startup helpers out of the serve API utilities.",
        remove_when="the pin is >= 0.33, where vllm.entrypoints.common.startup exists.",
        omni_refs=(
            "vllm_omni/entrypoints/cli/benchmark/main.py",
            "vllm_omni/entrypoints/cli/main.py",
            "vllm_omni/entrypoints/cli/serve.py",
        ),
    ),
    ShimEntry(
        module="vllm.v1.core.kv_cache_planning",
        names=("get_global_kv_cache_groups", "get_kv_cache_configs"),
        kind=BACKPORT,
        backport="kv_cache_planning",
        applies_to="<0.33",
        reason=(
            "vLLM moved the planning logic out of kv_cache_utils into "
            "kv_cache_planning in the v0.32.1 line, but get_global_kv_cache_groups "
            "only arrived in 0.33, and 0.33 also gave get_kv_cache_configs a fourth "
            "parameter. The range tracks the symbol, not the module: on a v0.32.1 pin "
            "the module exists, so this appended hook is never consulted and the "
            "missing symbol surfaces as a plain ImportError. That is a pin nobody "
            "builds; widening the range would not help, because serving a module the "
            "pin already has is not something an appended finder can do."
        ),
        remove_when=(
            "the pin reaches a vLLM whose kv_cache_planning defines "
            "get_global_kv_cache_groups (0.33 or later)."
        ),
        omni_refs=("vllm_omni/diffusion/diffusion_kv/initialization.py",),
    ),
    ShimEntry(
        module="vllm.v1.sample.ops.topk_topp_sampler",
        names=("apply_top_k_top_p",),
        kind=ADAPTER,
        adapter="topk_topp_sampler",
        applies_to="<0.32.1",
        reason=(
            "vLLM added has_topp_only_rows after v0.32.0, and Omni's Qwen3-Omni and "
            "MiniCPM-o 4.5 talker samplers pass it by keyword, so every talker "
            "sampling step raises TypeError on the pinned three-parameter function. "
            "Whichever implementation is bound takes the full top-p path, which is "
            "what both values of the hint select, so the adapter drops it. The adapter "
            "also has to survive vllm_musa's own sampler, which rebinds this name at "
            "module scope strictly later; see the adapter for how."
        ),
        remove_when=(
            "the pin reaches a vLLM whose apply_top_k_top_p takes the parameter "
            "(v0.32.1 or later)."
        ),
        omni_refs=(
            "vllm_omni/model_executor/models/minicpmo_4_5/talker_mrv2_sampling.py",
            "vllm_omni/model_executor/models/qwen3_omni/talker_sampler.py",
        ),
    ),
    ShimEntry(
        module="vllm.v1.attention.backends.utils",
        names=("compute_mm_prefix_ranges",),
        kind=ADAPTER,
        adapter="attention_backends_utils",
        applies_to="<0.33",
        reason=(
            "0.33 moved the PrefixLM range builder here from "
            "vllm.v1.worker.gpu.attn_utils and changed its calling convention. "
            "vllm_omni/patch.py reads the attribute at import time, so without it "
            "`import vllm_omni` raises AttributeError on the pin and no model runs. "
            "The bridge is exact at the default arguments and refuses the two newer "
            "ones, which the pin cannot express."
        ),
        remove_when="the pin is >= 0.33, which ships the function in this module.",
        omni_refs=("vllm_omni/patch.py", "vllm_omni/utils/mm_prefix.py"),
    ),
    ShimEntry(
        module="vllm.v1.worker.gpu.attn_utils",
        names=(),
        kind=CAPABILITY,
        target="compute_mm_prefix_ranges",
        applies_to="<0.33",
        reason=(
            "The adapter above installs 0.33's range builder in "
            "vllm.v1.attention.backends.utils, which is where 0.33 keeps it and where "
            "vllm_omni/patch.py reads it to alias img2img onto image. On the pin the "
            "builder lives here instead, and nothing on the pin reads the other "
            "module, so the aliasing does not reach the consumer. The v1 model runner "
            "is unaffected because it does not use a builder at all: it fills the "
            "ranges with an inline loop that filters only audio, so img2img is "
            "admitted. The current Omni main defaults to the v2 runner, so this "
            "limitation applies to the default img2img deployment."
        ),
        remove_when=(
            "the pin is >= 0.33, which keeps the builder in the module Omni patches."
        ),
        unsupported=(
            "img2img PrefixLM on a stage with model_runner: v2 -- the ranges come out "
            "empty rather than failing. Omni's stage default is v1, which builds them "
            "inline and is unaffected, and no shipped deploy sets v2 for an img2img "
            "model",
        ),
        omni_refs=("vllm_omni/patch.py", "vllm_omni/utils/mm_prefix.py"),
    ),
    ShimEntry(
        module="vllm.v1.worker.gpu.dp_utils",
        names=(),
        kind=CAPABILITY,
        target="dispatch_cg_and_sync_dp",
        applies_to="<0.33",
        reason=(
            "0.33 inserted allow_skip_drafts, which Omni's worker_v2 AR model runner "
            "passes by keyword, so the pinned function raises TypeError. Unlike the "
            "top-k hint there is no argument that dropping it preserves behaviour: it "
            "decides whether a data-parallel rank may skip draft execution, and "
            "guessing wrong risks a rank divergence or a hang."
        ),
        remove_when="the pin is >= 0.33, or an adapter can argue a safe translation.",
        unsupported=(
            "the worker_v2 autoregressive model runner under data parallelism "
            "(vllm_omni/worker_v2/omni_model_runner.py; diffusion pipelines do not use "
            "it)",
        ),
        omni_refs=("vllm_omni/worker_v2/omni_model_runner.py",),
    ),
    ShimEntry(
        module="vllm.config.vllm",
        names=(),
        kind=CAPABILITY,
        target="VllmConfig.adjust_dcp_kv_cache_interleave_size",
        applies_to="<0.33",
        reason=(
            "Same name and arity in both versions, different argument: 0.32 takes a "
            "KVCacheConfig and reads .kv_cache_groups off it, 0.33 takes the group "
            "list itself. Omni main passes the list, so the pinned method raises "
            "AttributeError -- but only after its dcp_size <= 1 and NixlConnector "
            "early returns, so an ordinary single-card run never reaches it."
        ),
        remove_when=(
            "the pin is >= 0.33, or the backport adapts the method under an object "
            "patch."
        ),
        unsupported=(
            "decode context parallelism (decode_context_parallel_size > 1) with a "
            "NixlConnector KV transfer",
        ),
        omni_refs=("vllm_omni/diffusion/diffusion_kv/initialization.py",),
    ),
)


_SERVED_KINDS = (RELOCATION, BACKPORT)


def entries_for_module(module: str) -> ShimEntry | None:
    """The entry whose module the import hook may *provide*, if any.

    Only ``RELOCATION`` and ``BACKPORT``: those modules are absent from the pin.
    An ``ADAPTER`` module exists in the pin and is patched after it loads, and a
    ``CAPABILITY`` module exists and is only reported, so serving either as a
    substitute would shadow working pinned code.
    """
    for entry in ENTRIES:
        if entry.module == module and entry.kind in _SERVED_KINDS:
            return entry
    return None


def importable_entries() -> tuple[ShimEntry, ...]:
    """Entries the import hook can serve as whole modules."""
    return tuple(entry for entry in ENTRIES if entry.kind in _SERVED_KINDS)


def adapter_entries() -> tuple[ShimEntry, ...]:
    """Entries that patch a pinned module after it is imported."""
    return tuple(entry for entry in ENTRIES if entry.kind == ADAPTER)


def capability_entries() -> tuple[ShimEntry, ...]:
    """Entries that only record an unsupported path."""
    return tuple(entry for entry in ENTRIES if entry.kind == CAPABILITY)
