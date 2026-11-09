# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-MUSA plugin entry points: names, order, idempotency and VLLM_PLUGINS."""

import re
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
GENERAL_GROUP = "vllm.general_plugins"
PART_FUNCTIONS = {
    "kv_connector": "_configure_kv_connector",
    "patches": "_apply_patches",
    "custom_ops": "_register_custom_ops",
    "attn_backends": "_register_attn_backends",
    "runtime": "_register_runtime",
}


def _entry_points() -> dict[str, dict[str, str]]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    groups: dict[str, dict[str, str]] = {}
    for match in re.finditer(
        r'(?ms)^\[project\.entry-points\."(vllm\.[a-z_]+)"\]\n(.*?)(?=^\[|\Z)', text
    ):
        entries = groups.setdefault(match.group(1), {})
        for line in match.group(2).splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = (part.strip() for part in line.split("=", 1))
                entries[name] = value.strip('"')
    return groups


@pytest.fixture
def plugins(monkeypatch):
    from vllm_musa import plugins as module

    calls: list[str] = []
    monkeypatch.setattr(module, "_registered", set())
    monkeypatch.setattr(module, "_registering", set())
    for part, attr in PART_FUNCTIONS.items():
        monkeypatch.setattr(module, attr, lambda part=part: calls.append(part))
    return module, calls


def test_entry_points_name_every_part():
    groups = _entry_points()
    assert groups["vllm.platform_plugins"] == {"musa": "vllm_musa:musa_platform_plugin"}
    assert groups[GENERAL_GROUP] == {
        "musa": "vllm_musa:register_all",
        "musa_attn_backends": "vllm_musa:register_attn_backends",
        "musa_custom_ops": "vllm_musa:register_custom_ops",
        "musa_kv_connector": "vllm_musa:register_kv_connector",
        "musa_runtime": "vllm_musa:register_runtime",
    }
    import vllm_musa

    for value in groups[GENERAL_GROUP].values():
        assert callable(getattr(vllm_musa, value.split(":", 1)[1]))


def test_register_all_runs_every_part_once_in_order(plugins):
    module, calls = plugins
    module.register_all()
    module.register_all()
    assert calls == list(module.PARTS)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("register_kv_connector", ["kv_connector"]),
        ("register_custom_ops", ["patches", "custom_ops"]),
        ("register_attn_backends", ["patches", "attn_backends"]),
        ("register_runtime", ["patches", "runtime"]),
    ],
)
def test_each_part_applies_the_patches_first(plugins, entry, expected):
    module, calls = plugins
    getattr(module, entry)()
    getattr(module, entry)()
    assert calls == expected


def test_register_all_completes_the_parts_left_over(plugins):
    module, calls = plugins
    module.register_attn_backends()
    module.register_all()
    assert calls == [
        "patches",
        "attn_backends",
        "kv_connector",
        "custom_ops",
        "runtime",
    ]


def test_a_failed_part_is_registered_again(plugins, monkeypatch):
    module, calls = plugins
    attempts: list[int] = []

    def flaky() -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("first attempt fails")
        calls.append("custom_ops")

    monkeypatch.setattr(module, "_register_custom_ops", flaky)
    with pytest.raises(RuntimeError):
        module.register_custom_ops()
    module.register_custom_ops()
    module.register_custom_ops()
    assert len(attempts) == 2
    assert calls == ["patches", "custom_ops"]


def test_a_reentrant_call_is_skipped(plugins, monkeypatch):
    module, calls = plugins

    def reentrant() -> None:
        calls.append("runtime")
        module.register_all()

    monkeypatch.setattr(module, "_register_runtime", reentrant)
    module.register_all()
    assert calls == list(module.PARTS)


@pytest.mark.parametrize(
    ("allowlist", "expected"),
    [
        (None, ["kv_connector", "patches", "custom_ops", "attn_backends", "runtime"]),
        ("musa", ["kv_connector", "patches", "custom_ops", "attn_backends", "runtime"]),
        (
            "musa,musa_custom_ops",
            ["kv_connector", "patches", "custom_ops", "attn_backends", "runtime"],
        ),
        ("musa_custom_ops", ["patches", "custom_ops"]),
        ("", []),
    ],
)
def test_vllm_plugins_values(plugins, monkeypatch, allowlist, expected):
    vllm_plugins = pytest.importorskip("vllm.plugins")
    _, calls = plugins
    entries = metadata.EntryPoints(
        metadata.EntryPoint(name, value, GENERAL_GROUP)
        for name, value in sorted(_entry_points()[GENERAL_GROUP].items())
    )

    def entry_points(*, group=None, **_):
        return entries if group == GENERAL_GROUP else metadata.EntryPoints(())

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    monkeypatch.setattr(vllm_plugins, "plugins_loaded", False)
    if allowlist is None:
        monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    else:
        monkeypatch.setenv("VLLM_PLUGINS", allowlist)

    vllm_plugins.load_general_plugins()

    assert calls == expected


def test_attention_backends_have_one_registration():
    backends = ROOT / "vllm_musa" / "v1" / "attention" / "backends"
    for path in backends.rglob("*.py"):
        assert "@register_backend(" not in path.read_text(encoding="utf-8"), path
    platform = (ROOT / "vllm_musa" / "platform.py").read_text(encoding="utf-8")
    assert "def register_attention_backends(" not in platform
    assert (
        "from vllm_musa.v1.attention.registry import register_attention_backends"
        in platform
    )


def test_v1_package_has_no_import_side_effects():
    source = (ROOT / "vllm_musa" / "v1" / "__init__.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s", source, re.M)
