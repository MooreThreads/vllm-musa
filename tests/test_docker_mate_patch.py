"""Tests for the version-gated MATE Docker compatibility patch."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
HELPER_PATH = ROOT / "docker/patches/apply_mate_dynamo_bool_patch.py"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="Git is required by the Docker compatibility step",
)


def _load_helper():
    spec = importlib.util.spec_from_file_location("mate_docker_patch", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, source: str):
    site_packages = tmp_path / "site-packages"
    target = site_packages / "mate" / "mha_interface.py"
    target.parent.mkdir(parents=True)
    target.write_text(source)
    patch_file = tmp_path / "mate.patch"
    patch_file.write_text("""diff --git a/mate/mha_interface.py b/mate/mha_interface.py
--- a/mate/mha_interface.py
+++ b/mate/mha_interface.py
@@ -1 +1 @@
-enable_mubin &= q.is_musa
+enable_mubin = enable_mubin & q.is_musa
""")
    return site_packages, target, patch_file


def _fake_distribution(monkeypatch, helper, version: str, site_packages: Path):
    dist = SimpleNamespace(version=version, locate_file=lambda _: site_packages)
    monkeypatch.setattr(helper, "distribution", lambda _: dist)


def test_applies_to_mate_before_026(tmp_path, monkeypatch):
    helper = _load_helper()
    site_packages, target, patch_file = _fixture(
        tmp_path, "enable_mubin &= q.is_musa\n"
    )
    _fake_distribution(monkeypatch, helper, "0.2.4", site_packages)

    helper.apply_patch(patch_file)

    assert target.read_text() == "enable_mubin = enable_mubin & q.is_musa\n"


def test_skips_mate_026_and_newer(tmp_path, monkeypatch):
    helper = _load_helper()
    site_packages, target, patch_file = _fixture(
        tmp_path, "enable_mubin &= q.is_musa\n"
    )
    _fake_distribution(monkeypatch, helper, "0.2.6", site_packages)

    helper.apply_patch(patch_file)

    assert target.read_text() == "enable_mubin &= q.is_musa\n"


def test_rejects_partially_applied_patch(tmp_path, monkeypatch):
    helper = _load_helper()
    site_packages, _target, patch_file = _fixture(
        tmp_path,
        "enable_mubin = enable_mubin & q.is_musa\n" "enable_mubin &= k.is_musa\n",
    )
    _fake_distribution(monkeypatch, helper, "0.2.5", site_packages)

    with pytest.raises(RuntimeError, match="partially applied"):
        helper.apply_patch(patch_file)


def test_rejects_unknown_pre026_layout(tmp_path, monkeypatch):
    helper = _load_helper()
    site_packages, _target, patch_file = _fixture(tmp_path, "enable_mubin = True\n")
    _fake_distribution(monkeypatch, helper, "0.2.5", site_packages)

    with pytest.raises(RuntimeError, match="expected pre-0.2.6 layout"):
        helper.apply_patch(patch_file)
