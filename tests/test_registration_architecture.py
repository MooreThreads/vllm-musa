# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration contract tests for the unified vLLM-MUSA plugin architecture.

Offline tests validate the static contract (entry points, package boundaries,
explicit idempotent installers, no import-time monkey-patches). Runtime tests
validate real plugin startup, attention backend registration, IR provider
selection, RMSNorm dispatch/numerical parity, and repeat-init idempotency on a
MUSA environment (e.g. inside the v0.32 pre-built image).
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_INIT = REPO_ROOT / "vllm_musa" / "__init__.py"
PLATFORM_PATH = REPO_ROOT / "vllm_musa" / "platform.py"
LAYERNORM_PATH = REPO_ROOT / "vllm_musa" / "model_executor" / "layers" / "layernorm.py"
MODEL_EXECUTOR_INIT = REPO_ROOT / "vllm_musa" / "model_executor" / "__init__.py"
V1_INIT = REPO_ROOT / "vllm_musa" / "v1" / "__init__.py"
IR_INIT_PATH = REPO_ROOT / "vllm_musa" / "ir" / "__init__.py"
IR_LAYERNORM_PATH = REPO_ROOT / "vllm_musa" / "ir" / "layernorm.py"
KERNELS_INIT_PATH = REPO_ROOT / "vllm_musa" / "kernels" / "__init__.py"

MODEL_EXECUTOR_ADAPTER_MODULES = (
    "vllm_musa/model_executor/layers/fused_moe/fused_moe.py",
    "vllm_musa/model_executor/layers/fused_moe/router/fused_topk_bias_router.py",
    "vllm_musa/model_executor/layers/fused_moe/router/fused_topk_router.py",
    "vllm_musa/model_executor/layers/fused_moe/router/grouped_topk_router.py",
    "vllm_musa/model_executor/layers/quantization/fp8.py",
    "vllm_musa/model_executor/layers/quantization/utils/fp8_utils.py",
    "vllm_musa/model_executor/warmup/deep_gemm_warmup.py",
)

V1_ADAPTER_MODULES = (
    "vllm_musa/v1/attention/backends/mla/common.py",
    "vllm_musa/v1/attention/backends/mla/flashmla_sparse.py",
    "vllm_musa/v1/attention/backends/turboquant.py",
    "vllm_musa/v1/executor/multiproc_executor.py",
    "vllm_musa/v1/sample/topk_topp_sampler.py",
    "vllm_musa/v1/sample/uniform_sample_counts.py",
    "vllm_musa/v1/spec_decode/utils.py",
)

EXPLICIT_INSTALLER_MODULES = MODEL_EXECUTOR_ADAPTER_MODULES + V1_ADAPTER_MODULES


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _run_isolated(
    code: str, env_overrides: dict[str, str | None] | None = None
) -> subprocess.CompletedProcess[str]:
    import os

    env = os.environ.copy()
    if env_overrides:
        for key, value in env_overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _require_vllm_musa_env() -> None:
    missing = [
        name
        for name in ("torch", "torchada", "vllm")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        pytest.skip("requires installed " + ", ".join(missing))


def _require_musa_device() -> None:
    _require_vllm_musa_env()
    import torch

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")


# ---------------------------------------------------------------------------
# 1. Offline / static architecture contract tests
# ---------------------------------------------------------------------------


def test_pyproject_declares_native_vllm_plugin_entrypoints() -> None:
    data = _load_pyproject()
    entry_points = data["project"]["entry-points"]

    assert entry_points["vllm.platform_plugins"] == {
        "musa": "vllm_musa:musa_platform_plugin",
    }
    assert entry_points["vllm.general_plugins"] == {
        "musa_out_of_tree": "vllm_musa:register_out_of_tree",
        "musa_attn_backends": "vllm_musa:register_attn_backends",
    }


def test_ir_providers_are_separated_from_standalone_kernels() -> None:
    platform_src = PLATFORM_PATH.read_text(encoding="utf-8")
    ir_init_src = IR_INIT_PATH.read_text(encoding="utf-8")
    kernels_init_src = KERNELS_INIT_PATH.read_text(encoding="utf-8")

    assert "import vllm_musa.ir" in platform_src
    assert "from . import gated_qkv, layernorm" in ir_init_src
    # vllm_musa/kernels/ must not hold IR provider modules or re-export shims.
    assert "layernorm" not in kernels_init_src
    assert "musa_ops" not in kernels_init_src
    assert "gated_qkv" not in kernels_init_src
    assert not (REPO_ROOT / "vllm_musa" / "kernels" / "musa_ops.py").exists()
    assert not (REPO_ROOT / "vllm_musa" / "kernels" / "gated_qkv.py").exists()


def test_rmsnorm_oot_layer_delegates_non_residual_path_to_ir() -> None:
    source = LAYERNORM_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    musa_rmsnorm = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MusaRMSNorm"
    )
    forward_oot = next(
        node
        for node in musa_rmsnorm.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_oot"
    )
    body_src = ast.get_source_segment(source, forward_oot)
    assert body_src is not None
    assert "musa_jit_norm.rmsnorm" not in body_src
    assert "nn.functional.rms_norm" not in body_src
    assert "return self.forward_native(x, residual)" in body_src


def test_musa_ir_rms_norm_provider_wires_jit_kernel_before_c_ext() -> None:
    source = IR_LAYERNORM_PATH.read_text(encoding="utf-8")

    assert "def _has_jit_rms_norm()" in source
    assert "def _jit_rms_norm_supports_args(" in source
    assert "def _select_musa_rms_norm_impl(" in source
    assert "musa_jit_norm.rmsnorm(x, weight, epsilon)" in source


def test_package_init_defers_inductor_heuristics_out_of_import_time() -> None:
    tree = ast.parse(PACKAGE_INIT.read_text(encoding="utf-8"))
    top_level_calls = [
        node
        for stmt in tree.body
        if not isinstance(stmt, ast.FunctionDef)
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "maybe_register_musa_template_heuristics"
    ]
    assert not top_level_calls


def test_platform_import_kernels_covers_all_musa_aot_extensions() -> None:
    source = PLATFORM_PATH.read_text(encoding="utf-8")
    start = source.index("    def import_kernels(cls) -> None:")
    end = source.index("    @classmethod", start + 10)
    body = source[start:end]
    for ext in ("vllm._C_stable_libtorch", "vllm._moe_C", "vllm_musa._C"):
        assert f'"{ext}"' in body


@pytest.mark.parametrize("rel_path", EXPLICIT_INSTALLER_MODULES)
def test_compat_adapter_uses_explicit_idempotent_install_hooks(rel_path: str) -> None:
    path = REPO_ROOT / rel_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)

    func_names = {stmt.name for stmt in tree.body if isinstance(stmt, ast.FunctionDef)}
    assert "install_hooks" in func_names, f"{rel_path} must define install_hooks()"

    # No top-level attribute mutation (monkey-patch) or top-level install_hooks() call.
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                assert not isinstance(
                    target, ast.Attribute
                ), f"{rel_path}:{stmt.lineno} mutates {ast.unparse(target)} at import time"
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            callee = ast.unparse(stmt.value.func)
            assert not callee.endswith(
                (
                    "install_hooks",
                    "_patch_triton_experts_quant_scheme",
                    "_install_prefill_backend_hook",
                )
            ), f"{rel_path}:{stmt.lineno} calls {callee}() at import time"


def test_all_compat_adapters_are_wired_into_package_registrars() -> None:
    me_src = MODEL_EXECUTOR_INIT.read_text(encoding="utf-8")
    v1_src = V1_INIT.read_text(encoding="utf-8")

    for rel_path in MODEL_EXECUTOR_ADAPTER_MODULES:
        mod_stem = Path(rel_path).stem
        assert (
            f", {mod_stem})" in me_src
        ), f"{rel_path} is not wired into register_model_executor_adapters()"

    v1_expected_entries = (
        '("mla.common", mla_common)',
        '("mla.flashmla_sparse", flashmla_sparse)',
        '("turboquant", turboquant)',
        '("multiproc_executor", multiproc_executor)',
        '("topk_topp_sampler", topk_topp_sampler)',
        '("uniform_sample_counts", uniform_sample_counts)',
        '("spec_decode.utils", spec_decode_utils)',
    )
    for entry in v1_expected_entries:
        assert entry in v1_src, f"{entry} is not wired into register_v1_adapters()"


def test_general_plugin_registrars_are_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_musa

    orig_oot = vllm_musa._OUT_OF_TREE_REGISTERED
    orig_attn = vllm_musa._ATTN_BACKENDS_REGISTERED
    mock_attn = MagicMock()
    fake_platform_mod = ModuleType("vllm_musa.platform")
    fake_platform_mod.register_attention_backends = mock_attn  # type: ignore[attr-defined]

    try:
        vllm_musa._OUT_OF_TREE_REGISTERED = False
        vllm_musa._ATTN_BACKENDS_REGISTERED = False
        monkeypatch.setitem(sys.modules, "vllm_musa.platform", fake_platform_mod)

        with (
            patch(
                "vllm_musa.distributed.mooncake_compat.configure_legacy_device_filter"
            ) as mock_mooncake,
            patch.object(vllm_musa, "_register_patches") as mock_patches,
            patch.object(vllm_musa, "_register_ops") as mock_ops,
            patch.object(vllm_musa, "_register_modules") as mock_modules,
        ):
            vllm_musa.register_out_of_tree()
            vllm_musa.register_out_of_tree()
            assert mock_mooncake.call_count == 1
            assert mock_patches.call_count == 1
            assert mock_ops.call_count == 1
            assert mock_modules.call_count == 1

            vllm_musa.register_attn_backends()
            vllm_musa.register_attn_backends()
            assert mock_attn.call_count == 1
    finally:
        vllm_musa._OUT_OF_TREE_REGISTERED = orig_oot
        vllm_musa._ATTN_BACKENDS_REGISTERED = orig_attn


# ---------------------------------------------------------------------------
# 2. Runtime / MUSA integration tests (run in pre-built MUSA v0.32 image)
# ---------------------------------------------------------------------------


def test_rms_norm_ir_priority_honors_eager_inductor_and_native_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_vllm_musa_env()
    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationMode

    from vllm_musa.platform import MUSAPlatformBase
    from vllm_musa.utils.environ import envs

    monkeypatch.delenv(envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.name, raising=False)

    eager_cfg = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        quant_config=None,
        compilation_config=SimpleNamespace(
            backend="eager",
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
    )
    assert MUSAPlatformBase.get_default_ir_op_priority(eager_cfg).rms_norm == [
        "musa",
        "native",
    ]

    with envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.override(True):
        assert MUSAPlatformBase.get_default_ir_op_priority(eager_cfg).rms_norm == [
            "native"
        ]

    inductor_cfg = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        quant_config=None,
        compilation_config=SimpleNamespace(
            backend="inductor",
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        ),
    )
    assert MUSAPlatformBase.get_default_ir_op_priority(inductor_cfg).rms_norm == [
        "native"
    ]


def test_runtime_plugin_startup_idempotency_and_preload_order() -> None:
    """Verify full vLLM plugin startup without VLLM_PLUGINS set: automatic
    platform & general plugin discovery, custom ops (AOT + JIT) auto-loading
    and invocation, attention registration, IR isolation, and idempotency.
    """
    _require_musa_device()
    result = _run_isolated(
        """
import os
import sys
import torch
import vllm_musa
from vllm.ir.op import IrOp
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# 0. VLLM_PLUGINS must be unset to prove zero-env auto-discovery.
assert "VLLM_PLUGINS" not in os.environ

# 1. Importing standalone kernel helpers must not register IR providers.
import vllm_musa.kernels  # noqa: F401
assert "rms_norm" not in IrOp.registry or "musa" not in IrOp.registry["rms_norm"].impls

# 2. General plugins auto-discover and register OOT layers, compat adapters,
#    custom-op schemas, and attention backends; repeat calls must be idempotent.
load_general_plugins()
vllm_musa.register_out_of_tree()
vllm_musa.register_attn_backends()

assert (
    AttentionBackendEnum.FLASH_ATTN.get_path()
    == "vllm_musa.v1.attention.backends.flash_attn.MUSAFlashAttentionBackend"
)
assert (
    AttentionBackendEnum.FLASHMLA.get_path()
    == "vllm_musa.v1.attention.backends.mla.flashmla.MUSAFlashMLABackend"
)

# 3. Platform import_kernels loads all 4 AOT extensions, and both AOT
#    (_C_musa_ops) and JIT (torch.ops.vllm.*) custom ops are callable.
current_platform.import_kernels()
for ext in ("vllm._C", "vllm._C_stable_libtorch", "vllm._moe_C", "vllm_musa._C"):
    assert ext in sys.modules, f"missing AOT extension: {ext}"

assert hasattr(torch.ops, "_C_musa_ops"), "AOT namespace _C_musa_ops not loaded"
assert hasattr(torch.ops._C_musa_ops, "rms_norm_musa"), (
    "AOT op _C_musa_ops::rms_norm_musa missing"
)
assert hasattr(torch.ops.vllm, "musa_csrc_rmsnorm"), (
    "JIT op vllm::musa_csrc_rmsnorm not preloaded by general plugin"
)
assert hasattr(torch.ops.vllm, "musa_csrc_fused_add_rmsnorm"), (
    "JIT op vllm::musa_csrc_fused_add_rmsnorm not preloaded by general plugin"
)

# Invoke representative AOT and JIT custom ops directly on MUSA tensors.
x = torch.randn((4, 256), device="musa", dtype=torch.bfloat16)
w = torch.ones((256,), device="musa", dtype=torch.bfloat16)
out_aot = torch.empty_like(x)
torch.ops._C_musa_ops.rms_norm_musa(out_aot, x, w, 1e-6)
out_jit = torch.ops.vllm.musa_csrc_rmsnorm(x, w, 1e-6)
torch.testing.assert_close(out_aot, out_jit, atol=2e-2, rtol=2e-2)

# 4. import_ir_kernels registers MUSA IR providers from vllm_musa.ir.
current_platform.import_ir_kernels()
assert IrOp.registry["rms_norm"].impls["musa"].supported is True
assert IrOp.registry["fused_add_rms_norm"].impls["musa"].supported is True
assert IrOp.registry["gated_qkv_rms_norm_rope"].impls["musa_inductor"].supported is True

# 5. All 14 compat adapters must have been installed by load_general_plugins().
for mod_name in (
    "vllm_musa.model_executor.layers.fused_moe.fused_moe",
    "vllm_musa.model_executor.layers.fused_moe.router.fused_topk_bias_router",
    "vllm_musa.model_executor.layers.fused_moe.router.fused_topk_router",
    "vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router",
    "vllm_musa.model_executor.layers.quantization.fp8",
    "vllm_musa.model_executor.layers.quantization.utils.fp8_utils",
    "vllm_musa.model_executor.warmup.deep_gemm_warmup",
    "vllm_musa.v1.attention.backends.mla.common",
    "vllm_musa.v1.attention.backends.mla.flashmla_sparse",
    "vllm_musa.v1.attention.backends.turboquant",
    "vllm_musa.v1.executor.multiproc_executor",
    "vllm_musa.v1.sample.uniform_sample_counts",
    "vllm_musa.v1.spec_decode.utils",
):
    mod = __import__(mod_name, fromlist=["_HOOKS_INSTALLED"])
    assert getattr(mod, "_HOOKS_INSTALLED", False) is True, (
        f"{mod_name}._HOOKS_INSTALLED is not True"
    )
import vllm.v1.sample.ops.topk_topp_sampler as _vllm_topk_mod
assert getattr(_vllm_topk_mod.TopKTopPSampler, "_musa_sampling_hooks_installed", False) is True
print("PASS_RUNTIME_PLUGIN_STARTUP")
""",
        env_overrides={"VLLM_PLUGINS": None},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "PASS_RUNTIME_PLUGIN_STARTUP" in result.stdout


def test_runtime_rmsnorm_oot_layer_dispatches_through_ir_on_musa() -> None:
    """Verify RMSNorm constructs MusaRMSNorm and routes non-residual forward
    through vllm.ir.ops.rms_norm (JIT musa provider + native fallback).
    """
    _require_musa_device()
    import torch
    from vllm import ir
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins

    from vllm_musa.model_executor.layers.layernorm import MusaRMSNorm

    load_general_plugins()
    current_platform.import_ir_kernels()

    with set_current_vllm_config(VllmConfig()):
        layer = RMSNorm(hidden_size=2048, eps=1e-6).to(
            device="musa", dtype=torch.bfloat16
        )
    assert isinstance(layer, MusaRMSNorm)

    torch.manual_seed(0)
    x_bf16 = torch.randn((8, 2048), device="musa", dtype=torch.bfloat16)
    x_fp32 = x_bf16.float()
    w_fp32 = layer.weight.data.float()

    with ir.ops.rms_norm.set_priority(["musa", "native"]):
        assert (
            ir.ops.rms_norm.dispatch(x_bf16, layer.weight.data, 1e-6).provider == "musa"
        )
        assert ir.ops.rms_norm.dispatch(x_fp32, w_fp32, 1e-6).provider == "native"
        out_musa = layer(x_bf16)

    with ir.ops.rms_norm.set_priority(["native"]):
        out_native = layer(x_bf16)

    torch.testing.assert_close(out_musa, out_native, atol=2e-2, rtol=2e-2)
