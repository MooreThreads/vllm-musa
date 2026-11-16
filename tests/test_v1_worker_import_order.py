# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Import-order contract for the v1 worker hooks.

vLLM imports ``vllm_musa.v1.attention.ops.flashmla`` from inside
``vllm/v1/worker/gpu/model_runner.py``, so importing anything under
``vllm_musa.v1`` runs that package's import-time code while vLLM's worker tree
can still be initializing. The layout rules below are checked on the source
tree, so they hold in CI; the runtime rules need a MUSA device and are skipped
elsewhere. Every source check reads the module body only: imports and calls
inside a function are deferred and allowed, and that includes the deliberately
local ``GPUModelRunner`` imports in the installers.
"""

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_INIT = REPO_ROOT / "vllm_musa" / "__init__.py"
V1_INIT = REPO_ROOT / "vllm_musa" / "v1" / "__init__.py"
V1_PACKAGE = REPO_ROOT / "vllm_musa" / "v1"
MODEL_RUNNER = "vllm.v1.worker.gpu.model_runner"
WORKER_PACKAGE = "vllm_musa.v1.worker"
INSTALLERS = {
    "install_hooks",
    "install_qwen_sample_input_views",
}
RUNNER_HOOKS = (
    "_musa_select_uniform_decode_model_inputs",
    "_musa_select_next_input_ids_buffer",
    "_musa_select_sample_hidden_states",
)
WIRED_INSTALLERS = (
    "install_qwen_sample_input_views",
    "qwen_fused_next_input_ids.install_hooks",
    "qwen_identity_logits_view.install_hooks",
)


def package_of(path: Path) -> str:
    """Package that relative imports in ``path`` resolve against.

    For ``pkg/__init__.py`` that is ``pkg`` and for ``pkg/mod.py`` it is
    ``pkg``, so in both cases the last path component is the one to drop.
    """
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    return ".".join(parts[:-1])


def dotted(node: ast.expr) -> str | None:
    """``a.b.c`` for a Name/Attribute chain, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


class ModuleBody(ast.NodeVisitor):
    """Collects imports and calls executed at import time.

    Descends into classes, ``if``/``try``/``with`` blocks and comprehensions,
    because all of those still run during import. Function and lambda bodies are
    skipped: they run later, which is exactly what the installers rely on.
    """

    def __init__(self) -> None:
        self.imports: list[ast.Import | ast.ImportFrom] = []
        self.calls: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_header(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_header(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_arguments(node.args)

    def _visit_header(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Decorators, defaults and annotations run now; the body does not."""
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_arguments(node.args)
        if node.returns is not None:
            self.visit(node.returns)

    def _visit_arguments(self, args: ast.arguments) -> None:
        for default in args.defaults:
            self.visit(default)
        for default in args.kw_defaults:
            if default is not None:
                self.visit(default)
        every = [
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            args.vararg,
            args.kwarg,
        ]
        for arg in every:
            if arg is not None and arg.annotation is not None:
                self.visit(arg.annotation)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted(node.func)
        if name:
            self.calls.append(name)
        self.generic_visit(node)


def walk_module_body(tree: ast.Module) -> ModuleBody:
    visitor = ModuleBody()
    for node in tree.body:
        visitor.visit(node)
    return visitor


def resolve_import(node: ast.ImportFrom, package: str) -> str:
    """Absolute name for an import, which may be relative (``from . import x``)."""
    if not node.level:
        return node.module or ""
    parts = package.split(".") if package else []
    # level 1 is ``package`` itself, level 2 its parent, and so on.
    base = parts[: len(parts) - (node.level - 1)]
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base)


def imported_names(nodes: list[ast.Import | ast.ImportFrom], package: str) -> set[str]:
    found: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        else:
            base = resolve_import(node, package)
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def module_body(path: Path) -> ModuleBody:
    return walk_module_body(ast.parse(path.read_text()))


def module_scope_imports(path: Path) -> set[str]:
    return imported_names(module_body(path).imports, package_of(path))


def function_calls(path: Path, name: str) -> set[str]:
    """Names called inside ``name``, including nested bodies."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            calls: set[str] = set()
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    target = dotted(call.func)
                    if target:
                        calls.add(target)
            return calls
    raise AssertionError(f"{name} not found in {path}")


def test_walker_sees_control_flow_and_headers_but_not_bodies() -> None:
    """The blind spots, so the checks below cannot pass vacuously."""
    tree = ast.parse(
        "import a\n"
        "if True:\n"
        "    from . import worker\n"
        "try:\n"
        "    from vllm.v1.worker.gpu.model_runner import GPUModelRunner\n"
        "except ImportError:\n"
        "    install_qwen_sample_input_views()\n"
        "@decorate(install_hooks())\n"
        "def later(seen=install_hooks(), typed: install_hooks() = None):\n"
        "    from . import helper\n"
        "    install_qwen_sample_input_views()\n"
        "value = lambda arg=install_hooks(): arg\n"
    )
    body = walk_module_body(tree)
    imported = imported_names(body.imports, "vllm_musa.v1")
    assert WORKER_PACKAGE in imported
    assert MODEL_RUNNER in imported
    assert not any(name.endswith(".helper") for name in imported)
    called = {name.rsplit(".", 1)[-1] for name in body.calls}
    # One from the try block, then one each from the decorator, the default,
    # the annotation and the lambda default. None from the function body.
    assert called == {"decorate", "install_hooks", "install_qwen_sample_input_views"}
    assert body.calls.count("install_hooks") == 4


def test_package_of_handles_init_and_module_files() -> None:
    """A wrong base here resolves ``from . import x`` one level too deep."""
    assert package_of(V1_INIT) == "vllm_musa.v1"
    assert (
        package_of(V1_PACKAGE / "worker" / "qwen_identity_logits_view.py")
        == "vllm_musa.v1.worker"
    )


def test_v1_package_does_not_import_the_worker_package() -> None:
    """The trigger: any vllm_musa.v1.* import runs this package init."""
    offender = {
        name
        for name in module_scope_imports(V1_INIT)
        if name == WORKER_PACKAGE or name.startswith(f"{WORKER_PACKAGE}.")
    }
    assert not offender


def test_no_module_under_v1_imports_the_runner_at_module_scope() -> None:
    """A module-scope import re-enters the partially initialized module."""
    offenders = {}
    for path in sorted(V1_PACKAGE.rglob("*.py")):
        if MODEL_RUNNER in module_scope_imports(path):
            offenders[path.relative_to(REPO_ROOT).as_posix()] = sorted(
                module_scope_imports(path)
            )
    assert not offenders


def test_worker_modules_do_not_call_installers_at_import_time() -> None:
    """Calling one at module scope rebuilds the cycle the import checks forbid.

    Only ``vllm_musa/v1/worker`` is checked: the sampler modules elsewhere in
    this package install their own, unrelated ``install_hooks`` on import.
    """
    offenders = {}
    for path in sorted((V1_PACKAGE / "worker").rglob("*.py")):
        called = {name.rsplit(".", 1)[-1] for name in module_body(path).calls}
        if INSTALLERS & called:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = sorted(
                INSTALLERS & called
            )
    assert not offenders


def test_plugin_wires_every_hook_set() -> None:
    """Wiring only: CI has no MUSA device, so this reads the source."""
    assert function_calls(PACKAGE_INIT, "_register_modules") >= set(WIRED_INSTALLERS)


def test_general_plugin_entry_calls_the_registrars() -> None:
    assert "_register_modules" in function_calls(PACKAGE_INIT, "register_custom_ops")


def run_isolated(source: str) -> subprocess.CompletedProcess:
    """Run in a fresh interpreter at the repo root, so the checkout wins."""
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def require_musa_runtime() -> None:
    """Skip when absent, fail when installed but broken."""
    missing = [
        name
        for name in ("torchada", "torch", "vllm")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        pytest.skip("needs " + ", ".join(missing))
    probe = run_isolated(
        "import torchada, torch\n"
        "print('MUSA_DEVICES=' + str(torch.accelerator.device_count()))\n"
    )
    if probe.returncode != 0:
        pytest.fail(f"MUSA runtime probe failed: {probe.stderr[-1000:]}")
    # torchada logs to stdout during import, so parse a sentinel rather than
    # assuming the count is the only thing printed.
    match = re.search(r"MUSA_DEVICES=(\d+)", probe.stdout)
    if match is None:
        pytest.fail(f"no device count in probe output: {probe.stdout[-1000:]}")
    if int(match.group(1)) == 0:
        pytest.skip("no MUSA device visible")


def test_importing_gpu_worker_first_succeeds() -> None:
    """The reported failure: importing vLLM's worker before vllm_musa is used."""
    require_musa_runtime()
    result = run_isolated(
        "from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors\nprint('ok')\n"
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_general_plugin_installs_the_runner_hooks() -> None:
    """The hooks come from the general plugin, not from an import side effect."""
    require_musa_runtime()
    result = run_isolated(
        "from vllm.v1.worker.gpu.model_runner import GPUModelRunner\n"
        f"hooks = {RUNNER_HOOKS!r}\n"
        "early = [name for name in hooks if hasattr(GPUModelRunner, name)]\n"
        "assert not early, early\n"
        "from vllm.plugins import load_general_plugins\n"
        "load_general_plugins()\n"
        "missing = [name for name in hooks if not hasattr(GPUModelRunner, name)]\n"
        "assert not missing, missing\n"
        "import vllm_musa\n"
        f"assert vllm_musa.__file__ == {str(PACKAGE_INIT)!r}, vllm_musa.__file__\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr[-2000:]
