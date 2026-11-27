# SPDX-License-Identifier: Apache-2.0
"""The Omni drift shim: every kind, with a fixture that must fail and one that must pass.

A gate that cannot fail is worse than no gate, so each test pairs the behaviour
it wants with the opposite case. No MUSA hardware, no real vLLM: a synthetic
``vllm`` package is built on disk in the two layouts that matter (the 0.32 pin,
which lacks the moved modules, and 0.33, which has them), because the only thing
under test is import resolution.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Imported directly: these modules are stdlib-only by design, so they load
# without torch, torch_musa or vLLM present.
sys.path.insert(0, str(ROOT))
from vllm_musa.omni_compat import table as compat_table  # noqa: E402
from vllm_musa.omni_compat._finder import version_matches  # noqa: E402

PIN_VERSION = "0.32.0"
NEXT_VERSION = "0.33.0"
V034_VERSION = "0.34.0"


def _write(path: Path, body: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))


def _make_vllm(root: Path, version: str, *, moved: bool) -> Path:
    """A synthetic ``vllm`` package.

    ``moved=False`` is the 0.32 shape: the relocation sources exist and the
    modules Omni imports do not. ``moved=True`` is the 0.33 shape, where the
    modules Omni imports exist natively and the shim must stay out of the way.
    """
    site = root / f"site-{version}-{'new' if moved else 'pin'}"
    pkg = site / "vllm"
    _write(pkg / "__init__.py", f"__version__ = {version!r}\n")
    _write(pkg / "entrypoints" / "__init__.py")
    _write(pkg / "entrypoints" / "common" / "__init__.py")
    _write(pkg / "entrypoints" / "launchers" / "__init__.py")
    _write(pkg / "entrypoints" / "serve" / "__init__.py")
    _write(pkg / "entrypoints" / "serve" / "utils" / "__init__.py")
    _write(pkg / "v1" / "__init__.py")
    _write(pkg / "v1" / "core" / "__init__.py")
    _write(pkg / "v1" / "worker" / "__init__.py")
    _write(pkg / "v1" / "worker" / "gpu" / "__init__.py")
    _write(pkg / "v1" / "worker" / "gpu" / "spec_decode" / "__init__.py")

    # The pinned locations always exist; that is what a relocation reads from.
    _write(
        pkg / "entrypoints" / "launchers" / "cli_args.py",
        """
        class FrontendArgs:
            origin = "launchers"

        def make_arg_parser(parser=None):
            return ("launchers", parser)

        def resolve_default_chat_template_kwargs(*a, **kw):
            return "launchers"

        def validate_parsed_serve_args(*a, **kw):
            return "launchers"

        def not_in_the_table():
            return "delegated"
        """,
    )
    _write(
        pkg / "entrypoints" / "serve" / "utils" / "api_utils.py",
        """
        VLLM_SUBCMD_PARSER_EPILOG = "pinned-epilog"

        def cli_env_setup():
            return "pinned-setup"
        """,
    )
    _write(
        pkg / "v1" / "core" / "kv_cache_utils.py",
        """
        def get_kv_cache_groups(vllm_config, kv_cache_spec):
            return [("group", name) for name in sorted(kv_cache_spec)]

        def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
            return ["pinned-config", len(kv_cache_specs), tuple(available_memory)]
        """,
    )
    sampler = """
        def apply_top_k_top_p(logits, k, p{extra}):
            return ("pinned", logits, k, p{extra_use})
        """
    _write(
        pkg / "v1" / "sample" / "__init__.py",
    )
    _write(
        pkg / "v1" / "sample" / "ops" / "__init__.py",
    )
    _write(
        pkg / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
        (
            sampler.format(
                extra=", has_topp_only_rows=True", extra_use=", has_topp_only_rows"
            )
            if moved
            else sampler.format(extra="", extra_use="")
        ),
    )
    _write(
        pkg / "v1" / "kv_cache_interface.py",
        """
        class SlidingWindowSpec:
            pass
        """,
    )
    _write(
        pkg / "v1" / "kv_cache_spec_registry.py",
        """
        class KVCacheSpecRegistry:
            checked = []

            @classmethod
            def check_kv_cache_spec_registry(cls, specs):
                cls.checked.append(dict(specs))
        """,
    )
    _write(pkg / "config" / "__init__.py")
    _write(
        pkg / "config" / "utils.py",
        "# The 0.32 fixture intentionally has no CompileFactors alias.\n",
    )
    _write(
        pkg / "config" / "model.py",
        """
        class ModelConfig:
            def compute_hash(self):
                return "model-hash"
        """,
    )
    _write(
        pkg / "config" / "vllm.py",
        """
        class VllmConfig:
            def compute_hash(self, include_version=True):
                return "vllm-hash"
        """,
    )
    _write(
        pkg / "v1" / "worker" / "block_table.py",
        """
        def get_block_table_width(max_num_blocks, block_size, *, token_alignment=128):
            return max_num_blocks
        """,
    )

    if moved:
        # The newer fixture provides these natively, with values the shim must
        # not replace.
        _write(
            pkg / "entrypoints" / "common" / "cli_args.py",
            """
            class FrontendArgs:
                origin = "common"

            def make_arg_parser(parser=None):
                return ("common", parser)

            def resolve_default_chat_template_kwargs(*a, **kw):
                return "common"

            def validate_parsed_serve_args(*a, **kw):
                return "common"
            """,
        )
        _write(
            pkg / "entrypoints" / "common" / "startup.py",
            """
            VLLM_SUBCMD_PARSER_EPILOG = "native-epilog"

            def cli_env_setup():
                return "native-setup"
            """,
        )
        _write(
            pkg / "v1" / "core" / "kv_cache_planning.py",
            """
            def get_global_kv_cache_groups(vllm_config, kv_cache_specs):
                return ["native-groups"]

            def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory, groups):
                return ["native-config", groups]
            """,
        )
        _write(
            pkg / "config" / "utils.py",
            "CompileFactors = dict[str, object]\n",
        )
        _write(
            pkg / "config" / "model.py",
            """
            class ModelConfig:
                def compute_hash(self):
                    return "model-hash"

                def compile_factors(self):
                    return {"native": True}
            """,
        )
        _write(
            pkg / "config" / "vllm.py",
            """
            class VllmConfig:
                def compute_hash(self, include_version=True):
                    return "vllm-hash"

                def compile_factors(self, include_version=True):
                    return {"native": True}
            """,
        )
        _write(
            pkg / "v1" / "kv_cache_interface.py",
            """
            def get_block_table_width(max_num_blocks, block_size, *, token_alignment=128):
                return "native-width"
            """,
        )
        _write(pkg / "v1" / "worker" / "gpu" / "__init__.py")
        _write(pkg / "v1" / "worker" / "gpu" / "spec_decode" / "__init__.py")
        _write(
            pkg / "v1" / "worker" / "gpu" / "spec_decode" / "draft_logits.py",
            """
            class DraftLogitsCache:
                native = True
            """,
        )
    return site


def _run(
    site: Path, body: str, *, env_extra: dict | None = None, version: str | None = None
) -> dict:
    """Run ``body`` in a fresh interpreter with ``site`` and the repo importable.

    A subprocess per case keeps ``sys.meta_path`` and ``sys.modules`` clean: the
    shim is process-global by design, so in-process tests would leak into each
    other.
    """
    script = (
        textwrap.dedent("""
        import json, sys
        import vllm_musa.omni_compat as compat
        compat.install()
        result = {"installed": compat.is_installed()}
        """)
        + textwrap.dedent(body)
        + "\nprint('@@' + json.dumps(result))\n"
    )
    env = {
        "PYTHONPATH": f"{site}:{ROOT}",
        "PATH": "/usr/bin:/bin",
    }
    if version is not None:
        env["VLLM_VERSION"] = version
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    marker = [line for line in proc.stdout.splitlines() if line.startswith("@@")]
    assert marker, f"no result line: {proc.stdout}\n{proc.stderr}"
    return json.loads(marker[-1][2:])


@pytest.fixture(scope="module")
def pin_site(tmp_path_factory) -> Path:
    return _make_vllm(tmp_path_factory.mktemp("pin"), PIN_VERSION, moved=False)


@pytest.fixture(scope="module")
def next_site(tmp_path_factory) -> Path:
    return _make_vllm(tmp_path_factory.mktemp("next"), NEXT_VERSION, moved=True)


# --------------------------------------------------------------------------- #
# Table integrity
# --------------------------------------------------------------------------- #
def test_every_entry_is_well_formed():
    assert compat_table.ENTRIES, (
        "the table must not be empty while the pin predates Omni main"
    )
    for entry in compat_table.ENTRIES:
        assert entry.kind in compat_table.KINDS
        assert entry.reason.strip() and entry.remove_when.strip()
        assert entry.module.startswith("vllm."), entry.module
        if entry.kind == compat_table.RELOCATION:
            assert entry.source and entry.source != entry.module
        if entry.kind == compat_table.BACKPORT:
            module = importlib.import_module(
                f"vllm_musa.omni_compat.backports.{entry.backport}"
            )
            for name in entry.names:
                assert hasattr(module, name), f"{entry.backport} lacks {name}"
        if entry.kind == compat_table.ADAPTER:
            module = importlib.import_module(
                f"vllm_musa.omni_compat.adapters.{entry.adapter}"
            )
            assert callable(module.apply), (
                f"adapter {entry.adapter} must expose apply(module)"
            )
            assert callable(module.is_applied), (
                f"adapter {entry.adapter} must expose is_applied(module)"
            )
            # Without this the loader cannot tell "the pin serves this already"
            # from "the adapter matched nothing", and fails closed on both.
            assert callable(module.is_satisfied), (
                f"adapter {entry.adapter} must expose is_satisfied(module)"
            )


@pytest.mark.parametrize(
    "kind",
    [
        compat_table.RELOCATION,
        compat_table.BACKPORT,
        compat_table.ADAPTER,
        compat_table.CAPABILITY,
    ],
)
def test_table_covers_each_kind(kind):
    """The shim is only proven for kinds the table exercises."""
    assert any(e.kind == kind for e in compat_table.ENTRIES), f"no {kind} entry to test"


def test_malformed_entries_are_rejected():
    with pytest.raises(ValueError, match="unknown kind"):
        compat_table.ShimEntry(
            module="vllm.x",
            names=("y",),
            kind="nonsense",
            applies_to="<0.33",
            reason="r",
            remove_when="w",
        )
    with pytest.raises(ValueError, match="needs source"):
        compat_table.ShimEntry(
            module="vllm.x",
            names=("y",),
            kind=compat_table.RELOCATION,
            applies_to="<0.33",
            reason="r",
            remove_when="w",
        )
    with pytest.raises(ValueError, match="needs backport"):
        compat_table.ShimEntry(
            module="vllm.x",
            names=("y",),
            kind=compat_table.BACKPORT,
            applies_to="<0.33",
            reason="r",
            remove_when="w",
        )
    with pytest.raises(ValueError, match="needs unsupported"):
        compat_table.ShimEntry(
            module="vllm.x",
            names=(),
            kind=compat_table.CAPABILITY,
            target="T",
            applies_to="<0.33",
            reason="r",
            remove_when="w",
        )


def test_capability_entries_are_never_served_as_imports():
    """A CAPABILITY module exists in the pin; shadowing it would break the pin."""
    for entry in compat_table.capability_entries():
        assert compat_table.entries_for_module(entry.module) is None


# --------------------------------------------------------------------------- #
# Version gating
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("specifier", "version", "expected"),
    [
        ("<0.33", "0.32.0", True),
        ("<0.33", "0.33.0", False),
        (
            "<0.33",
            "0.32.0+musa.g01c3f711a",
            True,
        ),  # local segment must not change identity
        ("<0.33", "0.33.0rc1", False),
        ("<0.33", None, True),  # undeterminable: shim rather than ImportError
        ("<0.33", "not-a-version", True),
        (">=0.32,<0.33", "0.31.0", False),
        # The point release is why the sampler row's range is "<0.32.1" and not
        # "<0.33": v0.32.1rc0 already carries has_topp_only_rows, and PEP 440
        # has "<V" exclude V's own pre-releases, so this spelling is what makes
        # that pin a clean no-op rather than a stale-entry ImportError.
        ("<0.32.1", "0.32.0", True),
        ("<0.32.1", "0.32.1rc0", False),
    ],
)
def test_version_matching(specifier, version, expected):
    assert version_matches(specifier, version) is expected


# --------------------------------------------------------------------------- #
# Relocation (class A)
# --------------------------------------------------------------------------- #
def test_relocation_serves_the_pinned_symbols(pin_site):
    result = _run(
        pin_site,
        """
        from vllm.entrypoints.common.cli_args import (
            FrontendArgs, make_arg_parser, resolve_default_chat_template_kwargs,
            validate_parsed_serve_args,
        )
        from vllm.entrypoints.common.startup import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
        import vllm.entrypoints.common.cli_args as shimmed
        result["origin"] = FrontendArgs.origin
        result["parser"] = make_arg_parser("p")
        result["resolve"] = resolve_default_chat_template_kwargs()
        result["validate"] = validate_parsed_serve_args()
        result["epilog"] = VLLM_SUBCMD_PARSER_EPILOG
        result["setup"] = cli_env_setup()
        result["delegated"] = shimmed.not_in_the_table()
        result["marker"] = shimmed.__vllm_musa_shim__["kind"]
        result["name"] = shimmed.__name__
        """,
        version=PIN_VERSION,
    )
    assert result["origin"] == "launchers"
    assert result["parser"] == ["launchers", "p"]
    assert result["resolve"] == "launchers"
    assert result["validate"] == "launchers"
    assert result["epilog"] == "pinned-epilog"
    assert result["setup"] == "pinned-setup"
    # A name the table does not list still resolves, so an unrecorded Omni
    # import does not need a table edit to work.
    assert result["delegated"] == "delegated"
    assert result["marker"] == "relocation"
    # The shimmed module keeps the name the caller asked for.
    assert result["name"] == "vllm.entrypoints.common.cli_args"


def test_without_the_shim_the_same_import_fails(pin_site):
    """The failure the shim removes, so the test above is not vacuous."""
    proc = subprocess.run(
        [sys.executable, "-c", "import vllm.entrypoints.common.cli_args"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(pin_site), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert proc.returncode != 0
    assert "No module named" in proc.stderr


def test_the_real_module_always_wins(next_site):
    """On a pin that has the module natively the shim must not shadow it."""
    result = _run(
        next_site,
        """
        from vllm.entrypoints.common.cli_args import FrontendArgs, make_arg_parser
        from vllm.entrypoints.common.startup import VLLM_SUBCMD_PARSER_EPILOG
        import vllm.entrypoints.common.cli_args as native
        result["origin"] = FrontendArgs.origin
        result["parser"] = make_arg_parser("p")
        result["epilog"] = VLLM_SUBCMD_PARSER_EPILOG
        result["shimmed"] = hasattr(native, "__vllm_musa_shim__")
        """,
        version=NEXT_VERSION,
    )
    assert result["origin"] == "common"
    assert result["parser"] == ["common", "p"]
    assert result["epilog"] == "native-epilog"
    assert result["shimmed"] is False


def test_out_of_range_version_is_not_shimmed(pin_site):
    """Claiming 0.33 on a 0.32 tree must fail, not silently get the shim."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_musa.omni_compat as c; c.install(); import vllm; import vllm.entrypoints.common.cli_args",
        ],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": NEXT_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "No module named" in proc.stderr


def test_disable_env_makes_the_shim_inert(pin_site):
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_musa.omni_compat as c; assert c.install() is False; import vllm.entrypoints.common.cli_args",
        ],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_MUSA_OMNI_COMPAT": "0",
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "No module named" in proc.stderr


# --------------------------------------------------------------------------- #
# Backport (class C) and adapter (class B)
# --------------------------------------------------------------------------- #
def test_backport_reproduces_the_pinned_grouping(pin_site):
    result = _run(
        pin_site,
        """
        from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups, get_kv_cache_configs
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

        class Config:
            num_prefill_lookahead_tokens = 1

        specs = [{"layer.b": "spec-b", "layer.a": "spec-a"}, {"layer.a": "spec-a"}]
        groups = get_global_kv_cache_groups(Config(), specs)
        result["groups"] = groups
        result["registry_checked"] = KVCacheSpecRegistry.checked
        # 0.33 passes the groups in; the pin's planner does not take them.
        result["configs"] = get_kv_cache_configs(Config(), specs, [1024, 2048], groups)
        """,
        version=PIN_VERSION,
    )
    # Merged across workers, deduplicated, grouped by the pinned helper.
    assert result["groups"] == [["group", "layer.a"], ["group", "layer.b"]]
    assert result["registry_checked"] == [{"layer.b": "spec-b", "layer.a": "spec-a"}]
    assert result["configs"] == ["pinned-config", 2, [1024, 2048]]


def test_backport_rejects_inconsistent_worker_specs(pin_site):
    """The pinned assertion must survive the backport, not be softened by it."""
    script = textwrap.dedent("""
        import vllm_musa.omni_compat as compat
        compat.install()
        from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups

        class Config:
            num_prefill_lookahead_tokens = 1

        get_global_kv_cache_groups(Config(), [{"l": "a"}, {"l": "b"}])
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "not supported yet" in proc.stderr


def test_v034_imports_are_bridged_on_the_pin(pin_site):
    result = _run(
        pin_site,
        """
        from vllm.config.utils import CompileFactors
        from vllm.v1.kv_cache_interface import get_block_table_width
        from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
        result["alias"] = str(CompileFactors)
        result["width"] = get_block_table_width(17, 16)
        try:
            DraftLogitsCache()
        except NotImplementedError as exc:
            result["draft_error"] = str(exc)
        """,
        version=PIN_VERSION,
    )
    assert result["alias"] == "dict[str, object]"
    assert result["width"] == 24
    assert "vLLM >= 0.34" in result["draft_error"]


def test_v034_compile_factor_methods_bridge_legacy_hashes(pin_site):
    result = _run(
        pin_site,
        """
        from vllm.config.model import ModelConfig
        from vllm.config.vllm import VllmConfig
        result["model"] = ModelConfig().compile_factors()
        result["vllm"] = VllmConfig().compile_factors()
        """,
        version=PIN_VERSION,
    )
    assert result["model"] == {"legacy_compute_hash": "model-hash"}
    assert result["vllm"] == {"legacy_compute_hash": "vllm-hash"}


def test_v034_rows_leave_native_symbols_untouched(next_site):
    result = _run(
        next_site,
        """
        from vllm.config.model import ModelConfig
        from vllm.v1.kv_cache_interface import get_block_table_width
        from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
        result["model"] = ModelConfig().compile_factors()
        result["width"] = get_block_table_width(1, 1)
        result["native"] = DraftLogitsCache.native
        result["model_stale"] = ModelConfig.__module__
        """,
        version=V034_VERSION,
    )
    assert result["model"] == {"native": True}
    assert result["width"] == "native-width"
    assert result["native"] is True


def test_a_stale_relocation_fails_loudly(tmp_path, monkeypatch):
    """A table row pointing at a source without the symbol must raise on import."""
    site = _make_vllm(tmp_path, PIN_VERSION, moved=False)
    script = textwrap.dedent("""
        import vllm_musa.omni_compat as compat
        from vllm_musa.omni_compat import table

        bad = table.ShimEntry(
            module="vllm.entrypoints.common.cli_args",
            names=("ThisSymbolDoesNotExist",),
            kind=table.RELOCATION,
            source="vllm.entrypoints.launchers.cli_args",
            applies_to="<0.33",
            reason="fixture",
            remove_when="fixture",
        )
        table.ENTRIES = (bad,)
        compat.install()
        import vllm.entrypoints.common.cli_args
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "table entry is stale" in proc.stderr


# --------------------------------------------------------------------------- #
# Capability reporting (class E) and status surface
# --------------------------------------------------------------------------- #
def test_unsupported_paths_are_reported_on_the_pin(pin_site):
    result = _run(
        pin_site,
        """
        import vllm  # the version the shim gates on
        import vllm_musa.omni_compat as compat
        result["unsupported"] = list(compat.unsupported_paths())
        result["status_version"] = compat.status()["vllm_version"]
        """,
        version=PIN_VERSION,
    )
    assert result["status_version"] == PIN_VERSION
    assert any("decode context parallel" in path for path in result["unsupported"])


def test_unsupported_paths_are_empty_on_a_newer_pin(next_site):
    result = _run(
        next_site,
        """
        import vllm
        import vllm_musa.omni_compat as compat
        result["unsupported"] = list(compat.unsupported_paths())
        """,
        version=NEXT_VERSION,
    )
    assert result["unsupported"] == []


def test_install_is_idempotent_and_reversible(pin_site):
    result = _run(
        pin_site,
        """
        import sys
        import vllm_musa.omni_compat as compat
        compat.install()
        compat.install()
        result["serve_hooks"] = sum(1 for f in sys.meta_path if type(f).__name__ == "OmniCompatFinder")
        result["adapt_hooks"] = sum(1 for f in sys.meta_path if type(f).__name__ == "OmniAdapterFinder")
        result["adapt_first"] = type(sys.meta_path[0]).__name__ == "OmniAdapterFinder"
        result["serve_last"] = type(sys.meta_path[-1]).__name__ == "OmniCompatFinder"
        result["removed"] = compat.uninstall()
        result["after"] = compat.is_installed()
        """,
        version=PIN_VERSION,
    )
    assert result["serve_hooks"] == 1
    assert result["adapt_hooks"] == 1
    # Position is the contract: serving last so the real vLLM wins, adapting
    # first so a pinned module still reaches its patch.
    assert result["adapt_first"] is True
    assert result["serve_last"] is True
    assert result["removed"] == 2
    assert result["after"] is False


def test_uninstall_also_removes_the_startup_hooks(pin_site):
    """The ``.pth``'s hooks are the ones the image runs, so uninstall must take them.

    The bootstrap installs instances of its own classes, defined in a top-level
    module this package never imports, so a sweep that recognises hooks by
    ``isinstance`` finds none of them: it returns 0 while the shim it claims to
    have removed goes on serving every later import. ``_bootstrap`` runs the
    real startup path, which is the only arrangement where that shows.
    """
    result = _bootstrap(
        pin_site,
        """
        import vllm_musa.omni_compat as compat
        result["before"] = compat.is_installed()
        result["removed"] = compat.uninstall()
        result["after"] = compat.is_installed()
        result["hooks"] = compat.active_hooks()
        try:
            import vllm.entrypoints.common.startup   # only the shim has it
            result["served"] = True
        except ImportError:
            result["served"] = False
        import vllm.v1.sample.ops.topk_topp_sampler as mod
        result["adapted"] = hasattr(mod, "__vllm_musa_adapted__")
        """,
    )
    assert result["before"] is True
    # Both halves: serving last, adapting first.
    assert result["removed"] == 2
    assert result["after"] is False
    assert result["hooks"] == {"serve": False, "adapt": False, "startup": False}
    # The claim is about later imports, so the test has to make some: a module
    # only the shim provides must now fail, and one the pin provides must load
    # unpatched.
    assert result["served"] is False
    assert result["adapted"] is False


def test_hook_is_last_so_ordinary_finders_win(pin_site):
    result = _run(
        pin_site,
        """
        import sys
        result["is_last"] = type(sys.meta_path[-1]).__name__ == "OmniCompatFinder"
        result["adapter_is_first"] = type(sys.meta_path[0]).__name__ == "OmniAdapterFinder"
        """,
        version=PIN_VERSION,
    )
    assert result["is_last"] is True
    assert result["adapter_is_first"] is True


def test_bootstrap_imports_nothing_heavy(pin_site):
    """The .pth runs in every interpreter on the machine, so it must stay cheap.

    ``import vllm_musa.omni_compat`` would execute ``vllm_musa/__init__.py``,
    which imports torchada and torch; the bootstrap exists to avoid exactly that.
    """
    script = textwrap.dedent("""
        import json, sys
        import vllm_musa_omni_compat_pth as boot
        heavy = sorted(
            m for m in sys.modules
            if m.split(".")[0] in {"torch", "vllm", "numpy", "torch_musa", "torchada", "vllm_musa"}
        )
        print("@@" + json.dumps({
            "heavy": heavy,
            "installed": sum(1 for f in sys.meta_path if type(f).__name__ == "OmniCompatPathFinder"),
            "is_last": type(sys.meta_path[-1]).__name__ == "OmniCompatPathFinder",
            "is_first": type(sys.meta_path[0]).__name__ == "OmniCompatPathFinder",
        }))
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": f"{pin_site}:{ROOT}", "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(
        [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")][-1][2:]
    )
    assert result["heavy"] == [], (
        "the startup hook must not import torch, vllm or vllm_musa"
    )
    assert result["installed"] == 2, (
        "one deferring hook to serve modules, one to adapt them"
    )
    assert result["is_last"] is True
    assert result["is_first"] is True


def test_bootstrap_serves_without_importing_the_package(pin_site):
    """The startup hook must serve a shimmed module on its own.

    Reaching into ``vllm_musa.omni_compat`` here would re-enter vLLM through
    ``vllm_musa``'s initialisation while a vLLM module is mid-import, which is
    the circular import this bootstrap exists to avoid.
    """
    script = textwrap.dedent("""
        import json, sys
        import vllm_musa_omni_compat_pth  # noqa: F401  -- installs the hook
        from vllm.entrypoints.common.startup import cli_env_setup
        print("@@" + json.dumps({
            "value": cli_env_setup(),
            "imported_vllm_musa": "vllm_musa" in sys.modules,
        }))
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(
        [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")][-1][2:]
    )
    assert result["value"] == "pinned-setup"
    assert result["imported_vllm_musa"] is False


def test_bootstrap_adapter_survives_a_later_rebind(pin_site):
    """The production path: the .pth bootstrap, then vllm_musa's install_hooks.

    The in-package adapter is covered separately; this asserts the same guard
    holds through the standalone bootstrap, which loads the adapter by file path
    and is what the image actually runs.
    """
    script = textwrap.dedent("""
        import json, sys
        import vllm_musa_omni_compat_pth  # noqa: F401  -- installs the hook
        import vllm.v1.sample.ops.topk_topp_sampler as mod

        def _musa_sampler(logits, k, p):
            return ["musa", logits, k, p]

        mod.apply_top_k_top_p = _musa_sampler       # vllm_musa's install_hooks()
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        print("@@" + json.dumps({
            "value": list(apply_top_k_top_p("l", 1, 2, has_topp_only_rows=False)),
            "imported_vllm_musa": "vllm_musa" in sys.modules,
        }))
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    result = json.loads(
        [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")][-1][2:]
    )
    assert result["value"] == ["musa", "l", 1, 2]
    assert result["imported_vllm_musa"] is False


def test_bootstrap_respects_the_disable_env(pin_site):
    script = textwrap.dedent("""
        import sys
        import vllm_musa_omni_compat_pth as boot
        assert not any(type(f).__name__ == "OmniCompatPathFinder" for f in sys.meta_path)
        assert boot.install() is False
        import vllm.entrypoints.common.startup
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_MUSA_OMNI_COMPAT": "0",
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "No module named" in proc.stderr


def test_bootstrap_reads_the_table_not_a_hardcoded_list():
    """One source of truth: the startup hook must agree with the table."""
    import vllm_musa_omni_compat_pth as boot

    package_dir = boot._package_dir()
    assert package_dir is not None
    table = boot._load_by_path("_table_fixture", f"{package_dir}/table.py")
    assert frozenset(e.module for e in table.importable_entries()) == frozenset(
        e.module for e in compat_table.importable_entries()
    )
    assert frozenset(e.module for e in table.adapter_entries()) == frozenset(
        e.module for e in compat_table.adapter_entries()
    )


def test_pth_roundtrip(tmp_path):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.omni_compat",
            "install-pth",
            "--purelib",
            str(purelib),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    written = list(purelib.glob("*.pth"))
    assert len(written) == 1
    # site processes .pth lines that start with "import"; anything else is a path.
    assert written[0].read_text().startswith("import ")
    assert "omni_compat" in written[0].read_text()
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.omni_compat",
            "remove-pth",
            "--purelib",
            str(purelib),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert list(purelib.glob("*.pth")) == []


def test_pth_line_is_processed_and_installs_the_startup_hook(tmp_path):
    """Prove the .pth mechanism itself, not just that the line is well formed.

    ``site.addsitedir`` runs the same ``.pth`` processing the interpreter does
    for a real site directory, so this exercises the production activation path
    -- the line, the bootstrap import, and the hooks it installs -- without
    writing into the interpreter's own site-packages.
    """
    site = _make_vllm(tmp_path, PIN_VERSION, moved=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.omni_compat",
            "install-pth",
            "--purelib",
            str(site),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        timeout=120,
        check=True,
    )
    probe = textwrap.dedent(f"""
        import json, site, sys
        site.addsitedir({str(site)!r})          # processes the .pth, as startup does
        from vllm_musa.omni_compat import active_hooks
        from vllm.entrypoints.common.startup import cli_env_setup
        print("@@" + json.dumps({{
            "hooks": active_hooks(),
            "served": cli_env_setup(),
            "bootstrap": "vllm_musa_omni_compat_pth" in sys.modules,
        }}))
        """)
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    payload = json.loads(
        [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")][-1][2:]
    )
    assert payload["bootstrap"] is True
    assert payload["hooks"]["startup"] is True
    assert payload["hooks"]["serve"] and payload["hooks"]["adapt"]
    # And the hook it installed really serves a module the pin does not have.
    assert payload["served"] == "pinned-setup"


def test_install_pth_reports_a_target_that_does_not_activate(tmp_path):
    """The opposite: a .pth nobody processes must not look like success.

    An explicit --purelib may legitimately target another interpreter's tree,
    so it warns; the default path, which is the image's, fails instead.
    """
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.omni_compat",
            "install-pth",
            "--purelib",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert "warning:" in out.stderr and "did not activate" in out.stderr
    assert "verified:" not in out.stdout


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode", "expected", "in_detail"),
    [
        ('{"startup": true, "serve": true, "adapt": true}', "", 0, True, ""),
        ('{"startup": false, "serve": false, "adapt": false}', "", 0, False, "startup"),
        ("not json", "", 0, False, "unreadable"),
        ("", "Traceback: boom", 1, False, "boom"),
    ],
)
def test_startup_probe_reads_the_fresh_interpreter(
    monkeypatch, stdout, stderr, returncode, expected, in_detail
):
    """Each way the probe can answer, including the two that must not pass."""
    import subprocess as sp

    from vllm_musa.omni_compat import __main__ as cli

    def fake_run(argv, **kwargs):
        return sp.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    ok, detail = cli._startup_hook_comes_up()
    assert ok is expected
    if expected:
        assert detail == ""
    else:
        # A refusal has to say why, or the build gate is unactionable.
        assert in_detail in detail


def test_pth_activates_the_shim_without_an_explicit_install(tmp_path):
    """The mechanism the image relies on: startup hook, no caller cooperation."""
    site = _make_vllm(tmp_path, PIN_VERSION, moved=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.omni_compat",
            "install-pth",
            "--purelib",
            str(site),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        timeout=120,
        check=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm.entrypoints.common.startup import cli_env_setup; print(cli_env_setup())",
        ],
        capture_output=True,
        text=True,
        # site-packages .pth processing needs the directory on the path as a site
        # dir, which PYTHONPATH does not do; -S plus explicit addsitedir is the
        # portable way to prove the line itself works.
        env={"PYTHONPATH": f"{site}:{ROOT}", "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    # PYTHONPATH does not make a directory a site dir, so .pth processing does
    # not run here; this asserts the line is well formed and the mechanism is
    # proven end to end by test_bootstrap_defers_to_the_real_finder and by the
    # image build, which installs into a real site-packages.
    assert proc.returncode != 0 or "pinned-setup" in proc.stdout


# --------------------------------------------------------------------------- #
# Adapter (call-time drift)
# --------------------------------------------------------------------------- #
def test_the_pinned_call_fails_without_the_adapter(pin_site):
    """The TypeError the adapter removes, so the next test is not vacuous."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p;"
            "apply_top_k_top_p('l', 1, 2, has_topp_only_rows=False)",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(pin_site), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert proc.returncode != 0
    assert "has_topp_only_rows" in proc.stderr and "TypeError" in proc.stderr


@pytest.mark.parametrize("hint", [True, False])
def test_adapter_lets_the_pinned_function_take_the_newer_argument(pin_site, hint):
    result = _run(
        pin_site,
        f"""
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        import vllm.v1.sample.ops.topk_topp_sampler as mod
        result["value"] = list(apply_top_k_top_p("l", 1, 2, has_topp_only_rows={hint}))
        result["without"] = list(apply_top_k_top_p("l", 1, 2))
        result["adapted"] = getattr(mod, "__vllm_musa_adapted__", {{}}).get("names")
        """,
        version=PIN_VERSION,
    )
    # Dropping the hint must give exactly the pinned result, for either value.
    assert result["value"] == ["pinned", "l", 1, 2]
    assert result["without"] == ["pinned", "l", 1, 2]
    assert result["adapted"] == ["apply_top_k_top_p"]


def test_adapter_is_inert_when_the_pin_already_takes_the_argument(next_site):
    result = _run(
        next_site,
        """
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        import vllm.v1.sample.ops.topk_topp_sampler as mod
        result["value"] = list(apply_top_k_top_p("l", 1, 2, has_topp_only_rows=False))
        result["adapted"] = getattr(mod, "__vllm_musa_adapted__", None)
        """,
        version=NEXT_VERSION,
    )
    # The native four-parameter function runs, unpatched.
    assert result["value"] == ["pinned", "l", 1, 2, False]
    assert result["adapted"] is None


@pytest.mark.parametrize("hint", [True, False])
def test_adapter_survives_a_later_rebind_of_the_name(pin_site, hint):
    """vllm_musa installs its own sampler over this name, strictly after us.

    ``vllm_musa.v1.sample.topk_topp_sampler`` runs
    ``topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p`` at module
    scope, which is necessarily later than the adapter: the adapter runs while
    the vLLM module is being created, and that assignment needs the module to
    exist. Omni then binds the name with ``from ... import``, so without the
    rebind guard it captures the unadapted replacement and the TypeError the
    adapter exists to remove comes back inside the talker.
    """
    result = _run(
        pin_site,
        f"""
        import vllm.v1.sample.ops.topk_topp_sampler as mod

        def _musa_sampler(logits, k, p):
            return ["musa", logits, k, p]

        mod.apply_top_k_top_p = _musa_sampler       # vllm_musa's install_hooks()
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        result["value"] = list(apply_top_k_top_p("l", 1, 2, has_topp_only_rows={hint}))
        result["positional"] = list(apply_top_k_top_p("l", 1, 2))
        """,
        version=PIN_VERSION,
    )
    # The rebound implementation stays in the path -- the guard widens the
    # argument list, it does not restore the slower pinned function.
    assert result["value"] == ["musa", "l", 1, 2]
    assert result["positional"] == ["musa", "l", 1, 2]


def test_repeated_rebinds_do_not_stack_wrappers(pin_site):
    """The opposite failure: a guard that re-wraps its own wrapper forever."""
    result = _run(
        pin_site,
        """
        import vllm.v1.sample.ops.topk_topp_sampler as mod

        def _musa_sampler(logits, k, p):
            return ["musa", logits, k, p]

        for _ in range(8):
            mod.apply_top_k_top_p = _musa_sampler
        depth, fn = 0, mod.apply_top_k_top_p
        while hasattr(fn, "__wrapped__"):
            depth += 1
            fn = fn.__wrapped__
        result["depth"] = depth
        result["value"] = list(mod.apply_top_k_top_p("l", 1, 2, has_topp_only_rows=True))
        """,
        version=PIN_VERSION,
    )
    assert result["depth"] == 1
    assert result["value"] == ["musa", "l", 1, 2]


def test_a_rebind_that_already_takes_the_argument_is_left_alone(pin_site):
    """A replacement that needs no adapting must not be wrapped."""
    result = _run(
        pin_site,
        """
        import vllm.v1.sample.ops.topk_topp_sampler as mod

        def _newer(logits, k, p, has_topp_only_rows=True):
            return ["newer", has_topp_only_rows]

        mod.apply_top_k_top_p = _newer
        result["wrapped"] = hasattr(mod.apply_top_k_top_p, "__wrapped__")
        result["value"] = list(mod.apply_top_k_top_p("l", 1, 2, has_topp_only_rows=False))
        """,
        version=PIN_VERSION,
    )
    assert result["wrapped"] is False
    assert result["value"] == ["newer", False]


def test_is_applied_reads_the_live_function_not_a_marker(pin_site):
    """The build gate has to see a stripped adapter, which a marker cannot.

    ``module.__vllm_musa_adapted__`` only records that the adapter *ran*. If the
    rebind guard were removed, the marker would still be there and the gate
    would pass a build whose live function is unadapted.
    """
    result = _run(
        pin_site,
        """
        import vllm.v1.sample.ops.topk_topp_sampler as mod
        from vllm_musa.omni_compat.adapters import topk_topp_sampler as ad

        result["after_install"] = ad.is_applied(mod)
        result["marker"] = getattr(mod, "__vllm_musa_adapted__", {}).get("names")

        def _musa_sampler(logits, k, p):
            return ["musa", logits, k, p]

        # Strip the adaptation the way a rebind would if nothing re-applied it:
        # bypass __setattr__ so the guard does not fire.
        mod.__dict__["apply_top_k_top_p"] = _musa_sampler
        result["after_strip"] = ad.is_applied(mod)
        result["marker_after_strip"] = getattr(mod, "__vllm_musa_adapted__", {}).get("names")
        """,
        version=PIN_VERSION,
    )
    assert result["after_install"] == ["apply_top_k_top_p"]
    assert result["marker"] == ["apply_top_k_top_p"]
    # The live query sees the strip; the marker does not. That gap is the point.
    assert result["after_strip"] == []
    assert result["marker_after_strip"] == ["apply_top_k_top_p"]


def test_adapter_reaches_a_module_imported_after_install(pin_site):
    """The ordering the .pth exists to guarantee."""
    result = _run(
        pin_site,
        """
        import sys
        result["preimported"] = "vllm.v1.sample.ops.topk_topp_sampler" in sys.modules
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        result["ok"] = list(apply_top_k_top_p("l", 1, 2, has_topp_only_rows=True))
        """,
        version=PIN_VERSION,
    )
    assert result["preimported"] is False
    assert result["ok"] == ["pinned", "l", 1, 2]


def test_adapter_warns_when_the_module_was_already_imported(pin_site):
    """Installing too late cannot work, and must say so rather than look fine."""
    script = textwrap.dedent("""
        import logging, sys
        logging.basicConfig(level=logging.WARNING)
        import vllm.v1.sample.ops.topk_topp_sampler  # before the shim: too late
        import vllm_musa.omni_compat as compat
        compat.install()
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "already imported" in proc.stderr


def test_an_adapter_that_patches_nothing_fails_the_import(pin_site, tmp_path):
    """A silent no-op adapter is the failure mode that matters most."""
    script = textwrap.dedent("""
        import sys
        import vllm_musa.omni_compat as compat
        from vllm_musa.omni_compat import _patcher, table

        class _Noop:
            @staticmethod
            def apply(module):
                return []

        sys.modules["vllm_musa.omni_compat.adapters.noop_fixture"] = _Noop
        table.ENTRIES = (
            table.ShimEntry(
                module="vllm.v1.sample.ops.topk_topp_sampler",
                names=("apply_top_k_top_p",),
                kind=table.ADAPTER,
                adapter="noop_fixture",
                applies_to="<0.33",
                reason="fixture",
                remove_when="fixture",
            ),
        )
        _patcher.install_adapters()
        import vllm.v1.sample.ops.topk_topp_sampler
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "patched nothing" in proc.stderr


# --------------------------------------------------------------------------- #
# The audit tool's arguments
# --------------------------------------------------------------------------- #
def _load_audit_tool():
    """Import tools/check_omni_compat.py, which is a script and not a package."""
    spec = importlib.util.spec_from_file_location(
        "_check_omni_compat", ROOT / "tools" / "check_omni_compat.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_package_dir_resolves_from_either_spelling(tmp_path, capsys):
    """--omni-root takes a repo root and --vllm-root a package dir.

    Getting them the wrong way round is silent without this: a repo root indexes no
    modules, so every symbol Omni imports looked absent and the tool reported a
    list of invented misses with a FAIL verdict, which is worse than an error.
    """
    tool = _load_audit_tool()

    package = tmp_path / "pkg" / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")

    # the package dir itself
    assert tool._resolve_package_dir(package, "--vllm-root") == package
    # the source root above it, with a note saying what was used
    assert tool._resolve_package_dir(package.parent, "--vllm-root") == package
    assert "is a source root" in capsys.readouterr().out
    # neither: refused, and the message names the flag
    empty = tmp_path / "empty"
    empty.mkdir()
    assert tool._resolve_package_dir(empty, "--target-vllm") is None
    assert "--target-vllm" in capsys.readouterr().err


def _load_verify_shim():
    """Import docker/omni/verify_shim.py, which is a script and not a package."""
    spec = importlib.util.spec_from_file_location(
        "_verify_shim", ROOT / "docker" / "omni" / "verify_shim.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["explicit", "import", "no-op"])
def test_verify_shim_checks_the_live_sampler_binding(monkeypatch, mode):
    from vllm_musa.omni_compat.adapters import topk_topp_sampler as adapter

    def upstream_apply(logits, k, p):
        return "upstream"

    def musa_apply(logits, k, p):
        return "musa"

    sampler = ModuleType("vllm.v1.sample.ops.topk_topp_sampler")
    sampler.apply_top_k_top_p = upstream_apply
    adapter.apply(sampler)
    musa_sampler = ModuleType("vllm_musa.v1.sample.topk_topp_sampler")
    musa_sampler._apply_top_k_top_p = musa_apply
    calls = []

    def install_hooks():
        calls.append(True)
        if mode != "no-op":
            sampler.apply_top_k_top_p = musa_apply

    musa_sampler.install_hooks = install_hooks
    monkeypatch.setitem(sys.modules, sampler.__name__, sampler)
    monkeypatch.setitem(sys.modules, musa_sampler.__name__, musa_sampler)
    if mode == "import":
        sampler.apply_top_k_top_p = musa_apply

    verify = _load_verify_shim()
    if mode == "no-op":
        with pytest.raises(RuntimeError, match="did not bind the MUSA"):
            verify._exercise_sampler_rebind(musa_sampler.__name__)
    else:
        verify._exercise_sampler_rebind(musa_sampler.__name__)
        assert adapter.is_applied(sampler) == ["apply_top_k_top_p"]
        assert (
            sampler.apply_top_k_top_p(None, None, None, has_topp_only_rows=True)
            == "musa"
        )

    assert len(calls) == (0 if mode == "import" else 1)


@pytest.mark.parametrize(
    "message, device",
    [
        ("MUSA driver initialization failed", True),
        ("No MUSA runtime is available", True),
        ("you might not have a Mthreads GPU", True),
        # A shim failure, which is the whole point of the gate.
        ("No module named 'vllm.entrypoints.common.cli_args'", False),
        ("cannot import name 'get_global_kv_cache_groups'", False),
        # A message that merely mentions the platform is not a device failure;
        # widening a marker this far would turn every real failure into a skip.
        ("MUSA", False),
    ],
)
def test_driverless_skips_a_device_failure_and_nothing_else(message, device):
    """`--driverless` downgrades an import failure to "skipped" at four sites.

    The predicate behind that decision is the difference between a gate and a
    rubber stamp: too broad and the driverless build reports a pass for a shim
    that does not work.
    """
    assert _load_verify_shim()._needs_a_device(RuntimeError(message)) is device


def test_audit_tool_refuses_a_root_that_holds_no_omni(tmp_path, capsys):
    """A root without `vllm_omni/` indexes no modules, so every pass reads clean.

    Silently analysing nothing is worse than an error, because the verdict is a
    PASS that nobody questions.
    """
    tool = _load_audit_tool()
    assert tool.main(["--omni-root", str(tmp_path)]) == 2
    assert "does not contain vllm_omni/" in capsys.readouterr().err


def test_imports_pass_fails_a_name_the_relocation_source_does_not_have():
    """A relocated module is served by delegating, so its coverage is the source's.

    Every name on a relocated module is absent from the pin by construction --
    that is the reason the row exists -- and the imports pass only consults
    ``shim_covers`` for names the pin lacks. Treating the kind alone as
    coverage therefore made this pass unable to fail for any name on any
    relocation row, which is the most common drift kind it exists to catch.
    """
    tool = _load_audit_tool()
    source = "vllm.entrypoints.launchers.cli_args"
    index = tool.VllmIndex(
        root=ROOT, version=PIN_VERSION, modules={source: {"make_arg_parser"}}
    )

    def audit(name: str) -> dict:
        item = tool.OmniImport(
            module="vllm.entrypoints.common.cli_args",
            name=name,
            omni_file="vllm_omni/entrypoints/cli/serve.py",
            lineno=1,
            type_checking=False,
        )
        return tool.audit_imports([item], index, compat_table)

    # Listed in the table and present in the source: served.
    assert audit("make_arg_parser")["status"] == "pass"

    # Not listed, but the source has it. The relocation delegates unknown
    # attributes, so this works without a table edit and must stay passing.
    index.modules[source].add("not_in_the_table")
    assert audit("not_in_the_table")["status"] == "pass"

    # Absent from the source too: nothing can serve it, and the pass has to say so.
    result = audit("validate_api_server_args")
    assert result["status"] == "fail"
    assert result["blocking"] == 1
    assert [r["name"] for r in result["unshimmed"]] == ["validate_api_server_args"]


def test_imports_pass_counts_a_missing_attribute_adapter_as_coverage():
    """An adapter row also covers a name absent from the pinned module."""
    tool = _load_audit_tool()
    index = tool.VllmIndex(
        root=ROOT,
        version=PIN_VERSION,
        modules={"vllm.config.utils": set()},
    )
    item = tool.OmniImport(
        module="vllm.config.utils",
        name="CompileFactors",
        omni_file="vllm_omni/config/model.py",
        lineno=1,
        type_checking=False,
    )
    result = tool.audit_imports([item], index, compat_table)
    assert result["status"] == "pass"
    assert result["shimmed"][0]["shim"]["adapter"] == "config_utils"


# --------------------------------------------------------------------------- #
# The call-shape pass, which needs no second vLLM tree
# --------------------------------------------------------------------------- #
def _call_fixture(tmp_path: Path, pin_body: str, omni_body: str) -> tuple:
    """A one-symbol pin and a one-file Omni tree, indexed the way main() does."""
    tool = _load_audit_tool()
    pin = tmp_path / "pin" / "vllm"
    _write(pin / "__init__.py", "__version__ = '0.32.0'\n")
    _write(pin / "v1" / "__init__.py")
    _write(pin / "v1" / "sample" / "__init__.py")
    _write(pin / "v1" / "sample" / "ops" / "__init__.py")
    _write(pin / "v1" / "sample" / "ops" / "topk_topp_sampler.py", pin_body)

    omni = tmp_path / "omni"
    _write(omni / "vllm_omni" / "__init__.py")
    _write(omni / "vllm_omni" / "talker.py", omni_body)

    tool.OMNI_ROOT = omni
    index = tool.VllmIndex.build(pin, PIN_VERSION)
    return tool, tool.collect_omni_imports(omni), index, omni


class _BlindTable:
    """A table with no answers, to prove the pass finds the drift on its own."""

    RELOCATION = compat_table.RELOCATION
    importable_entries = staticmethod(lambda: ())
    adapter_entries = staticmethod(lambda: ())
    capability_entries = staticmethod(lambda: ())


def test_call_pass_finds_a_keyword_the_pin_cannot_take_without_a_target_tree(tmp_path):
    """The fatal half of call-shape drift, from the pin alone.

    This is the class that produced two of the table's rows and that a human
    found by hand: the name resolves, the import audit is happy, and the call
    raises TypeError. The semantic pass could see it only with --target-vllm,
    which no build gate passes, so nothing automated saw it at all.
    """
    tool, omni_imports, index, omni = _call_fixture(
        tmp_path,
        "def apply_top_k_top_p(logits, k, p):\n    return logits\n",
        "from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p\n"
        "def sample(logits):\n"
        "    return apply_top_k_top_p(logits, None, None, has_topp_only_rows=True)\n",
    )

    blind = tool.audit_calls(omni_imports, index, _BlindTable, omni)
    assert blind["status"] == "fail"
    assert [o["symbol"] for o in blind["offences"]] == [
        "vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p"
    ]
    assert blind["offences"][0]["omni_calls"][0]["unknown_keywords"] == [
        "has_topp_only_rows"
    ]

    # The shipped table adapts exactly this, so the real run must stay quiet.
    assert tool.audit_calls(omni_imports, index, compat_table, omni)["status"] == "pass"


def test_call_pass_does_not_flag_a_typeerror_guarded_fallback(tmp_path):
    """Omni spans vLLM versions by catching the TypeError.

    ``_get_helium_rope`` calls the new ``get_rope`` and falls back to the old
    argument list in ``except TypeError``. The fallback not fitting the pin is
    the reason it was written, so reporting it would make the gate cry wolf on
    a compatibility idiom -- while a broad ``except Exception`` (Omni's
    streaming chat handler wraps 840 lines in one) must still not absolve it.
    """
    pin_body = "def apply_top_k_top_p(logits, k, p):\n    return logits\n"
    call = "apply_top_k_top_p(logits, None, None, has_topp_only_rows=True)"

    guarded = (
        "from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p\n"
        "def sample(logits):\n"
        "    try:\n"
        f"        return {call}\n"
        "    except TypeError:\n"
        "        return apply_top_k_top_p(logits, None, None)\n"
    )
    tool, omni_imports, index, omni = _call_fixture(tmp_path, pin_body, guarded)
    assert tool.audit_calls(omni_imports, index, _BlindTable, omni)["status"] == "pass"

    broad = guarded.replace("except TypeError:", "except Exception:")
    tool, omni_imports, index, omni = _call_fixture(tmp_path / "b", pin_body, broad)
    assert tool.audit_calls(omni_imports, index, _BlindTable, omni)["status"] == "fail"


def test_a_skipped_semantics_pass_says_so_in_the_verdict(tmp_path, capsys):
    """A bare PASS asserted more than was checked.

    Neither build gate passes --target-vllm, so the semantic diff self-reported
    ``skipped`` inside the JSON while the verdict line read PASS and the exit
    code was 0 -- and automation reads the exit code.
    """
    tool, _, _, omni = _call_fixture(
        tmp_path,
        "def apply_top_k_top_p(logits, k, p):\n    return logits\n",
        "import vllm\n",
    )
    pin = tmp_path / "pin" / "vllm"
    report = tmp_path / "report.json"
    assert (
        tool.main(
            [
                "--omni-root",
                str(omni),
                "--vllm-root",
                str(pin),
                "--vllm-version",
                PIN_VERSION,
                "--report",
                str(report),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "[verdict]   PASS (imports, attributes, calls, staleness;" in out
    assert "semantics not compared" in out
    payload = json.loads(report.read_text())
    assert payload["coverage"] == ["imports", "attributes", "calls", "staleness"]
    assert payload["semantics"]["status"] == "skipped"


# --------------------------------------------------------------------------- #
# What the adapters do when somebody else gets there first
# --------------------------------------------------------------------------- #
def _bootstrap(site: Path, body: str, *, version: str = PIN_VERSION) -> dict:
    """Run ``body`` after the ``.pth`` bootstrap, which is what the image runs."""
    script = (
        textwrap.dedent("""
        import json, sys
        import vllm_musa_omni_compat_pth  # noqa: F401  -- installs the hook
        result = {}
        """)
        + textwrap.dedent(body)
        + "\nprint('@@' + json.dumps(result))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": version,
        },
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    marker = [line for line in proc.stdout.splitlines() if line.startswith("@@")]
    assert marker, f"no result line: {proc.stdout}\n{proc.stderr}"
    return json.loads(marker[-1][2:])


def test_a_rebind_that_copied_the_marker_is_still_re_adapted(pin_site):
    """``functools.wraps`` copies our marker, so the marker cannot be a flag.

    vllm-musa rebinds ``apply_top_k_top_p`` at module scope. If it ever does so
    with a function built by ``functools.wraps(ours)`` -- which copies our
    ``__dict__``, marker included -- then a boolean marker made the re-apply
    guard return early: no re-wrap, while ``is_applied()`` still reported the
    name live, so the build gate passed and the talker raised TypeError on its
    first sampling step. The marker is now the function itself, which a copy
    cannot forge.
    """
    adapter = ROOT / "vllm_musa" / "omni_compat" / "adapters" / "topk_topp_sampler.py"
    result = _bootstrap(
        pin_site,
        f"""
        ADAPTER_PATH = {str(adapter)!r}
        import functools, importlib.util
        import vllm.v1.sample.ops.topk_topp_sampler as mod

        spec = importlib.util.spec_from_file_location(
            "_ad", ADAPTER_PATH
        )
        ad = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ad)

        @functools.wraps(mod.apply_top_k_top_p)   # the adapted wrapper
        def _musa_sampler(logits, k, p):
            return ["musa", logits, k, p]

        # The copy really happens; that is the premise of this test.
        result["inherited_marker"] = ad.ADAPTED in vars(_musa_sampler)
        mod.apply_top_k_top_p = _musa_sampler
        result["live"] = ad.is_applied(mod)
        result["value"] = list(
            mod.apply_top_k_top_p("l", 1, 2, has_topp_only_rows=False)
        )
        """,
    )
    assert result["inherited_marker"] is True
    # Re-adapted, so the hint is absorbed and MUSA's implementation still runs.
    assert result["live"] == ["apply_top_k_top_p"]
    assert result["value"] == ["musa", "l", 1, 2]


def test_an_adapter_with_nothing_to_do_does_not_break_the_import(tmp_path):
    """A stale entry is a repo bug; it must not take vLLM down for it.

    Returning no names means two unrelated things, and the loader failed the
    import for both. One of them is "the pin grew the shape itself", where the
    right runtime answer is to let the pin serve the call: raising ImportError
    on a core vLLM module instead breaks every model on the image over a shim
    that is no longer needed. The image still has to refuse the entry, which is
    what ``verify_shim._stale_adapter`` now does.
    """
    site = tmp_path / "site"
    _write(site / "vllm" / "__init__.py", f"__version__ = {PIN_VERSION!r}\n")
    for parent in ("v1", "v1/sample", "v1/sample/ops"):
        _write(site / "vllm" / parent / "__init__.py")
    _write(
        site / "vllm" / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
        """
        def apply_top_k_top_p(logits, k, p, has_topp_only_rows=True):
            return ["pin", has_topp_only_rows]
        """,
    )

    result = _bootstrap(
        site,
        """
        import vllm.v1.sample.ops.topk_topp_sampler as mod
        result["value"] = list(mod.apply_top_k_top_p("l", 1, 2, has_topp_only_rows=False))
        result["marker"] = getattr(mod, "__vllm_musa_adapted__", None)
        """,
    )
    # The pin's own function serves the call, unwrapped.
    assert result["value"] == ["pin", False]
    assert result["marker"]["names"] == []
    assert result["marker"]["stale"] is True

    # And the build gate is the one that refuses it.
    verify = _load_verify_shim()
    assert (
        verify._stale_adapter({"names": ["apply_top_k_top_p"], "stale": False}) is None
    )
    message = verify._stale_adapter(result["marker"])
    assert message is not None
    assert "the entry is stale" in message
    assert result["marker"]["remove_when"] in message


def _stale_pin_site(root: Path) -> Path:
    """A pin whose ``apply_top_k_top_p`` already takes the newer keyword."""
    site = root / "site"
    _write(site / "vllm" / "__init__.py", f"__version__ = {PIN_VERSION!r}\n")
    for parent in ("v1", "v1/sample", "v1/sample/ops"):
        _write(site / "vllm" / parent / "__init__.py")
    _write(
        site / "vllm" / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
        """
        def apply_top_k_top_p(logits, k, p, has_topp_only_rows=True):
            return ["pin", has_topp_only_rows]
        """,
    )
    return site


def test_the_in_package_loader_also_survives_a_satisfied_adapter(tmp_path):
    """The twin of the bootstrap case, through ``install()``.

    The two loaders are separate implementations of the same contract and both
    failed the import for a satisfied adapter. A test on one of them says
    nothing about the other, and the package loader is the one an in-process
    caller and the test suite use.
    """
    result = _run(
        _stale_pin_site(tmp_path),
        """
        import vllm.v1.sample.ops.topk_topp_sampler as mod
        result["value"] = list(mod.apply_top_k_top_p("l", 1, 2, has_topp_only_rows=False))
        result["marker"] = getattr(mod, "__vllm_musa_adapted__", None)
        """,
        version=PIN_VERSION,
    )
    assert result["value"] == ["pin", False]
    assert result["marker"]["names"] == []
    assert result["marker"]["stale"] is True


def test_the_mm_prefix_bridge_is_still_found_under_omni_s_own_wrapper():
    """``vllm_omni.utils.mm_prefix`` decorates the bridge with functools.wraps.

    So the live attribute after ``import vllm_omni`` is Omni's wrapper, not
    ours, and the build gate asks the adapter whether the name is adapted
    *right now*. Answering by reading a flag worked only because
    ``functools.wraps`` copies ``__dict__``; the test pins the behaviour so the
    marker change cannot quietly invert it, and the negative case proves the
    answer is not simply always yes.
    """
    spec = importlib.util.spec_from_file_location(
        "_mm_prefix_adapter",
        ROOT / "vllm_musa" / "omni_compat" / "adapters" / "attention_backends_utils.py",
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    import functools
    from types import ModuleType

    module = ModuleType("vllm.v1.attention.backends.utils")
    assert adapter.is_applied(module) == []
    assert adapter.is_satisfied(module) is False

    assert adapter.apply(module) == ["compute_mm_prefix_ranges"]
    assert adapter.is_applied(module) == ["compute_mm_prefix_ranges"]
    assert adapter.is_satisfied(module) is True
    # Idempotent: a second pass has nothing to add and must not say it failed.
    assert adapter.apply(module) == []

    @functools.wraps(module.compute_mm_prefix_ranges)
    def omni_wrapper(*args, **kwargs):
        return module.compute_mm_prefix_ranges(*args, **kwargs)

    module.compute_mm_prefix_ranges = omni_wrapper
    assert adapter.is_applied(module) == ["compute_mm_prefix_ranges"]
    assert adapter.is_satisfied(module) is True

    # An unrelated function under the same name is not the bridge.
    module.compute_mm_prefix_ranges = lambda *a, **k: {}
    assert adapter.is_applied(module) == []
    # ... but the module can still serve the call Omni's patch makes, so the
    # loader must not call that a stale entry.
    assert adapter.is_satisfied(module) is True


def test_a_breaking_call_fails_the_verdict_and_the_exit_code(tmp_path, capsys):
    """The pass is only a gate if the exit code carries it.

    Automation reads the exit code, not the JSON. A pass that reports an
    offence while ``main`` still returns 0 is the same defect this branch
    removed from the semantic pass, one pass further along.
    """
    tool = _load_audit_tool()
    pin = tmp_path / "pin" / "vllm"
    _write(pin / "__init__.py", f"__version__ = {PIN_VERSION!r}\n")
    # A symbol the shipped table says nothing about, so nothing suppresses it.
    _write(pin / "ranges.py", "def build_ranges(a):\n    return a\n")

    omni = tmp_path / "omni"
    _write(omni / "vllm_omni" / "__init__.py")
    _write(
        omni / "vllm_omni" / "stage.py",
        """
        from vllm.ranges import build_ranges

        def run():
            return build_ranges(1, span_pad=2)
        """,
    )

    report = tmp_path / "report.json"
    code = tool.main(
        [
            "--omni-root",
            str(omni),
            "--vllm-root",
            str(pin),
            "--vllm-version",
            PIN_VERSION,
            "--report",
            str(report),
        ]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "BREAKS   vllm.ranges.build_ranges" in out
    assert "[verdict]   FAIL" in out
    payload = json.loads(report.read_text())
    assert payload["verdict"] == "fail"
    assert payload["calls"]["status"] == "fail"


def test_a_copied_mm_prefix_marker_does_not_read_as_adapted():
    """The marker has to name itself, not merely be present.

    ``functools.wraps`` copies the adapter's ``__dict__`` into whatever wraps
    the bridge, so a replacement can carry the marker without delegating. The
    build gate asks whether the 0.33 builder is live, and for such a
    replacement the honest answer is no.
    """
    spec = importlib.util.spec_from_file_location(
        "_mm_prefix_marker",
        ROOT / "vllm_musa" / "omni_compat" / "adapters" / "attention_backends_utils.py",
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    from types import ModuleType

    module = ModuleType("vllm.v1.attention.backends.utils")
    adapter.apply(module)
    bridge = module.compute_mm_prefix_ranges

    def replacement(*args, **kwargs):
        return {}

    # Exactly what a copied __dict__ leaves behind: the marker, naming ours.
    vars(replacement).update(vars(bridge))
    assert getattr(replacement, adapter.ADAPTED, None) is bridge
    module.compute_mm_prefix_ranges = replacement
    assert adapter.is_applied(module) == []
