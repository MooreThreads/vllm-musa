"""CPU checks for real v2 fused factory configuration and source ABI routing.

The JIT loader is stubbed so these tests exercise factory/cache behavior without
claiming MUSA compilation, numerical accuracy, or graph replay correctness.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALLREDUCE = ROOT / "vllm_musa/jit_kernel/csrc/allreduce.py"
POLICY = ROOT / "vllm_musa/optimization_contract/Fuse_CAR_RMSnorm.py"
V2_SOURCE = ROOT / "vllm_musa/jit_kernel/csrc/distributed/custom_all_reduce_v2.mu"


def _load(path, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def jit(monkeypatch):
    for name in (
        "vllm_musa", "vllm_musa.jit_kernel", "vllm_musa.jit_kernel.csrc",
        "vllm_musa.optimization_contract",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    policy = _load(POLICY, "vllm_musa.optimization_contract.Fuse_CAR_RMSnorm", monkeypatch)
    torch = types.ModuleType("torch")
    torch.float16, torch.bfloat16, torch.float32 = "fp16", "bf16", "fp32"
    monkeypatch.setitem(sys.modules, "torch", torch)
    builds, calls = [], []

    def load_jit(name, sources, *, extra_musa_cflags):
        builds.append((name, sources, extra_musa_cflags))
        result = types.SimpleNamespace()
        for suffix in ("unregistered", "registered", "row_unregistered", "row_registered"):
            symbol = f"vllm_musa_custom_ar_v2_fused_allreduce_rmsnorm_{suffix}"
            def invoke(*args, _symbol=symbol):
                calls.append((_symbol, args))
            setattr(result, symbol, invoke)
        return result

    loader = types.ModuleType("vllm_musa.jit_kernel.csrc.jit")
    loader.load_musa_jit = load_jit
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    module = _load(ALLREDUCE, "vllm_musa.jit_kernel.csrc.allreduce", monkeypatch)
    return module, policy, builds, calls


@pytest.mark.parametrize("row,registered", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("has_residual,need_raw", [(True, False), (True, True), (False, True)])
@pytest.mark.parametrize("gemma", [False, True])
def test_fixed_abi_routes_all_four_source_entries(jit, row, registered, has_residual, need_raw, gemma):
    module, _, builds, calls = jit
    launcher = module.v2_fused_allreduce_rmsnorm_func(
        4, 2048, 1, mp_count=60, row_kernel=row, graph_rank_data=registered,
        has_residual=has_residual, need_raw=need_raw,
        gemma=gemma,
    )
    tensors = [object() for _ in range(7)]
    rd, sg, inp, residual, rout, norm, weight = tensors
    raw = object() if need_raw else None
    launcher(rd, sg, inp, residual if has_residual else None, rout, norm, weight, raw,
             11, 12, 1024, 0, 4, 1e-5)
    residual_arg = residual if has_residual else inp
    raw_arg = raw if need_raw else norm
    suffix = ("row_" if row else "") + ("registered" if registered else "unregistered")
    symbol, args = calls.pop()
    assert symbol.endswith(suffix)
    if registered:
        expected = (rd, sg, residual_arg, rout, norm, raw_arg, weight, 11, 0, 4)
    elif row:
        expected = (rd, sg, inp, residual_arg, rout, norm, raw_arg, weight, 11, 12, 1024, 0, 4)
    else:
        expected = (rd, sg, inp, norm, residual_arg, rout, raw_arg, weight, 11, 12, 1024, 0, 4)
    assert args == expected + ((2048,) if row else ()) + (1e-5,)
    assert builds[0][1] == ("distributed/custom_all_reduce_v2.mu",)
    assert "-DSGL_CUSTOM_AR_FUSED_RMSNORM=1" in builds[0][2]
    assert launcher.kernel_symbol == symbol
    assert launcher.jit_cflags == builds[0][2]
    assert f"-DSGL_CUSTOM_AR_THREADS={launcher.launch_config.threads}" in launcher.jit_cflags
    assert f"-DVLLM_MUSA_FUSED_GEMMA={int(gemma)}" in launcher.jit_cflags
    assert launcher.gemma is gemma


def test_gemma_modules_are_distinct_from_default_ordinary_modules(jit):
    module, _, builds, _ = jit
    factory = module.v2_fused_allreduce_rmsnorm_func
    ordinary = factory(2, 5120, 1, mp_count=56)
    explicit_ordinary = factory(2, 5120, 1, mp_count=56, gemma=False)
    gemma = factory(2, 5120, 1, mp_count=56, gemma=True)
    factory(2, 5120, 1, mp_count=56, gemma=True)
    assert len(builds) == 2
    assert builds[0][0] != builds[1][0]
    assert ordinary.gemma is explicit_ordinary.gemma is False
    assert gemma.gemma is True
    ordinary_flags = set(ordinary.jit_cflags)
    gemma_flags = set(gemma.jit_cflags)
    assert ordinary_flags - gemma_flags == {"-DVLLM_MUSA_FUSED_GEMMA=0"}
    assert gemma_flags - ordinary_flags == {"-DVLLM_MUSA_FUSED_GEMMA=1"}


@pytest.mark.parametrize("suffix", ["unregistered", "registered", "row_unregistered", "row_registered"])
def test_source_factories_forward_gemma_mode(jit, suffix):
    module, _, builds, _ = jit
    factory = getattr(module, f"v2_launch_fused_allreduce_rmsnorm_{suffix}_func")
    launcher = factory(2, 5120, row_count=1, mp_count=56, gemma=True)
    assert launcher.gemma is True
    assert "-DVLLM_MUSA_FUSED_GEMMA=1" in builds[0][2]


def test_policy_launch_fields_reach_compiler(jit):
    module, _, builds, _ = jit
    launcher = module.v2_fused_allreduce_rmsnorm_func(4, 2048, 1, mp_count=60)
    cfg = launcher.launch_config
    flags = builds[0][2]
    assert f"-DSGL_CUSTOM_AR_THREADS={cfg.threads}" in flags
    assert f"-DSGL_CUSTOM_AR_BLOCKS={cfg.blocks}" in flags
    assert f"-DSGL_CUSTOM_AR_MAX_BLOCKS={cfg.max_blocks}" in flags
    assert f"-DSGL_CUSTOM_AR_DYNAMIC_BLOCKS={cfg.dynamic_blocks}" in flags
    assert set(cfg.cflags()).issubset(flags)


def test_complete_effective_config_changes_memory_and_disk_cache(jit, monkeypatch):
    module, _, builds, _ = jit
    cfg = module.v2_fused_compile_config(4, 2048, 1, mp_count=60)
    monkeypatch.setattr(module, "v2_fused_compile_config", lambda *args, **kwargs: cfg)
    factory = module.v2_fused_allreduce_rmsnorm_func
    factory(4, 2048, 1, mp_count=60)
    factory(4, 2048, 1, mp_count=60)
    assert len(builds) == 1
    cfg = replace(cfg, max_blocks=cfg.max_blocks + 1)
    factory(4, 2048, 1, mp_count=60)
    assert len(builds) == 2 and builds[0][0] != builds[1][0]
    factory(4, 2048, 1, mp_count=60, cache_policy=1)
    assert len(builds) == 3 and builds[1][0] != builds[2][0]
    monkeypatch.setenv("VLLM_MUSA_CUSTOM_AR_VECTOR_LOAD", "1")
    factory(4, 2048, 1, mp_count=60, cache_policy=1)
    assert len(builds) == 4 and builds[2][0] != builds[3][0]


def test_native_fused_add_preserves_fp32_residual_in_final_compile_config(jit):
    module, _, _, _ = jit
    launcher = module.v2_fused_allreduce_rmsnorm_func(2, 5120, 1024, mp_count=60)
    assert launcher.launch_config.cache_hidden_limit >= 5120
    assert launcher.launch_config.typed_cache_hidden_limit == 0
    assert launcher.launch_config.tmp_with_residual == 0


def test_raw_output_retains_pre_residual_scratch(jit, monkeypatch):
    module, _, builds, calls = jit
    cfg = module.v2_fused_compile_config(4, 2048, 1, mp_count=60)
    cfg = replace(cfg, tmp_with_residual=1)
    monkeypatch.setattr(module, "v2_fused_compile_config", lambda *args, **kwargs: cfg)
    launcher = module.v2_fused_allreduce_rmsnorm_func(4, 2048, 1, mp_count=60, need_raw=True)
    assert launcher.launch_config.tmp_with_residual == 0
    assert "-DSGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL=0" in builds[0][2]
    assert "-DVLLM_MUSA_FUSED_WRITE_RAW=1" in builds[0][2]
    assert not calls


def test_output_contract_changes_compiled_module_and_rejects_mismatch(jit):
    module, _, builds, calls = jit
    factory = module.v2_fused_allreduce_rmsnorm_func
    for residual, raw in ((True, False), (True, True), (False, True)):
        factory(4, 2048, 1, mp_count=60, need_raw=raw, has_residual=residual)
    assert len({build[0] for build in builds}) == 3
    launcher = factory(4, 2048, 1, mp_count=60, need_raw=True)
    with pytest.raises(ValueError, match="compiled residual/raw contract"):
        launcher(*[object() for _ in range(7)], None, 1, 2, 3, 0, 4, 1e-5)
    assert not calls


def test_pure_v2_does_not_select_fused_translation_unit(jit):
    module, _, builds, _ = jit
    module._custom_ar_v2_module(4, module.SHOT_PUSH)
    assert not any("FUSED_RMSNORM" in flag for flag in builds[0][2])


def test_fused_translation_unit_has_four_typed_ffi_exports():
    source = V2_SOURCE.read_text()
    assert "#if !defined(SGL_CUSTOM_AR_FUSED_RMSNORM)" in source
    fused = source.split("#else  // SGL_CUSTOM_AR_FUSED_RMSNORM", 1)[1]
    assert fused.count("TVM_FFI_DLL_EXPORT_TYPED_FUNC(") == 4


@pytest.mark.parametrize("gemma,expected_bits", [(False, 0x3B80), (True, 0x3F81)])
def test_compiled_epilogue_preserves_gemma_rounding_boundary(tmp_path, gemma, expected_bits):
    """Compile the actual helper on the CPU with an IEEE BF16 scalar shim.

    At normalized=1.003 and raw weight=2^-8, prematurely rounding either the
    normalized value or weight+1 produces BF16 1.0, while Gemma requires
    1.0078125. This checks the actual helper branch, not a Python copy of it.
    """
    import re
    import shutil
    import subprocess

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the extracted-helper check")
    source = V2_SOURCE.read_text()
    helper = re.search(
        r"template <typename T>\n__device__ __forceinline__ T vllm_fused_apply_weight\(.*?\n\}",
        source,
        re.S,
    )
    assert helper is not None
    program = r'''
#include <cstdint>
#include <cstring>
#define __device__
#define __forceinline__ inline
struct BFloat16 {
  uint16_t bits;
  explicit BFloat16(float value) {
    uint32_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    raw += 0x7fff + ((raw >> 16) & 1);
    bits = static_cast<uint16_t>(raw >> 16);
  }
};
inline float upcast_s(BFloat16 value) {
  uint32_t raw = static_cast<uint32_t>(value.bits) << 16;
  float result;
  std::memcpy(&result, &raw, sizeof(result));
  return result;
}
template <typename T> inline T downcast_s(float value) { return T(value); }
'''
    program += helper.group()
    program += f'''
int main() {{
  const auto actual = vllm_fused_apply_weight<BFloat16>(1.003f, 0.00390625f);
  return actual.bits == {expected_bits} ? 0 : 1;
}}
'''
    path, executable = tmp_path / "epilogue.cpp", tmp_path / "epilogue"
    path.write_text(program)
    subprocess.run(
        [compiler, "-std=c++17", "-O2", f"-DVLLM_MUSA_FUSED_GEMMA={int(gemma)}", str(path), "-o", str(executable)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
