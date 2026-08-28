# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
INIT_PATH = ROOT / "vllm_musa" / "__init__.py"
PYPROJECT_PATH = ROOT / "pyproject.toml"


def _source_version() -> str:
    module = ast.parse(INIT_PATH.read_text(encoding="utf-8"))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            continue
        value = ast.literal_eval(statement.value)
        assert isinstance(value, str)
        return value
    raise AssertionError("vllm_musa.__version__ must be a literal assignment")


def test_build_metadata_uses_the_source_version() -> None:
    pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")
    project_section = pyproject.split("[project]", maxsplit=1)[1].split(
        "\n[", maxsplit=1
    )[0]

    assert re.search(
        r'^dynamic = \["dependencies", "version"\]$',
        project_section,
        re.M,
    )
    assert re.search(
        r'^version = \{attr = "vllm_musa\.__version__"\}$',
        pyproject,
        re.M,
    )
    assert not re.search(r'^version = "[^"]+"$', project_section, re.M)
    assert _source_version() == "0.1.28"


def test_clean_install_and_engine_plan_provenance_share_version(tmp_path: Path) -> None:
    version = _source_version()
    site_packages = tmp_path / "site-packages"
    shutil.copytree(ROOT / "vllm_musa", site_packages / "vllm_musa")
    dist_info = site_packages / f"vllm_musa-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.4\n" "Name: vllm-musa\n" f"Version: {version}\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(site_packages)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from importlib.metadata import version; "
            "import vllm_musa; "
            "from vllm_musa.engine_plan.runtime import _distribution_version; "
            "installed = version('vllm-musa'); "
            "assert vllm_musa.__version__ == installed; "
            "assert _distribution_version('vllm_musa') == installed",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_uninstalled_source_exposes_the_canonical_version(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import vllm_musa; print(vllm_musa.__version__)",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == _source_version()
