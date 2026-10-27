"""Source-level guards for the ported MUSA custom all-reduce v2 path.

The v2 kernel, its launcher and its runtime adapter are only exercised on real
MUSA hardware; CI has none. These guards keep the parts that must never drift
silently honest without a device:

* the fixed Signal ABI size (``2 * 512 * 128`` bytes) that both sides of the
  IPC handshake rely on,
* the shot enumeration values (they are baked into the compiled module name and
  into the kernel's compile-time macros),
* the exported TVM-FFI symbol name (renamed from ``sgl_`` to ``vllm_musa_``
  when the file was ported; a stray rename makes the launcher fail at runtime),
* the environment switch staying **off by default**, because turning it on
  changes decode numerics and latency,
* the fused all-reduce + RMSNorm entry points refusing to run instead of
  silently falling back to the v1 kernels.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / "vllm_musa/jit_kernel/csrc/distributed/custom_all_reduce_v2.mu"
JIT = ROOT / "vllm_musa/jit_kernel/csrc/allreduce.py"
ALGO = JIT
COMM = ROOT / "vllm_musa/distributed/device_communicators/musa_jit_custom_all_reduce.py"
V2_IMPL = COMM
OBJ = COMM
RUNTIME = COMM
FUSED = COMM
RESOLVER = ROOT / "vllm_musa/optimization_contract/resolver.py"

_SIGNAL_SLOTS = 512
_PULL_SEMAPHORE_BYTES = 128
_AR_SIGNAL_BYTES = 2 * _SIGNAL_SLOTS * _PULL_SEMAPHORE_BYTES


def _source(path: Path) -> str:
    assert path.is_file(), f"missing ported file: {path}"
    return path.read_text(encoding="utf-8")


def test_kernel_declares_fixed_signal_abi() -> None:
    """The v2 Signal struct is a fixed-size ABI; slots are addressed by index."""
    source = _source(KERNEL)
    assert "struct alignas(128) PullSemaphore" in source
    assert "struct alignas(128) Signal" in source
    # Slot count is a named constant so every JIT variant agrees on the ABI.
    assert "kSignalMaxBlocks = 512" in source
    assert "PullSemaphore pull[kSignalMaxBlocks]" in source
    assert "PullSemaphore publish[kSignalMaxBlocks]" in source
    assert "FlagType push_epoch[kSignalMaxBlocks]" in source
    # The kernel asserts its own size at compile time instead of trusting Python.
    assert "sizeof(Signal) == 2 * kSignalMaxBlocks * sizeof(PullSemaphore)" in source
    assert "static_assert(sizeof(PullSemaphore) == 128)" in source
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(" in source


def test_kernel_exports_only_the_renamed_symbol() -> None:
    """Ported file must not keep the upstream ``sgl_musa_custom_ar_`` prefix."""
    source = _source(KERNEL)
    assert "sgl_musa_custom_ar_" not in source
    assert (
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(\n"
        "    vllm_musa_custom_ar_launch_device_descriptor," in source
    )


def test_meta_size_matches_kernel_struct() -> None:
    """Python-side Signal size must equal the struct the kernel compiles."""
    source = _source(JIT)
    ns: dict[str, object] = {}
    for line in source.splitlines():
        if line.startswith("_V2_SIGNAL_BLOCKS") or line.startswith(
            "_V2_SEMAPHORE_BYTES"
        ):
            name, _, value = line.partition("=")
            ns[name.strip()] = int(value.strip())
    assert ns["_V2_SIGNAL_BLOCKS"] == _SIGNAL_SLOTS
    assert ns["_V2_SEMAPHORE_BYTES"] == _PULL_SEMAPHORE_BYTES
    tree = ast.parse(source)
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "allreduce_v2_meta_size"
    )
    returns = [node for node in ast.walk(func) if isinstance(node, ast.Return)]
    assert returns, "allreduce_v2_meta_size must return the Signal size"
    expr = ast.unparse(returns[0].value)
    assert "_V2_SIGNAL_BLOCKS" in expr and "_V2_SEMAPHORE_BYTES" in expr
    assert 2 * _SIGNAL_SLOTS * _PULL_SEMAPHORE_BYTES == _AR_SIGNAL_BYTES


def test_shot_values_are_stable() -> None:
    """Shot ids are compiled into the module name, so they must not drift."""
    jit_source = _source(JIT)
    expected = {
        "SHOT_PUSH": 0,
        "SHOT_ONE_STAGE": 1,
        "SHOT_TWO_STAGE": 2,
        "SHOT_PUSH_WIDE": 3,
        "SHOT_TWO_STAGE_512": 4,
    }
    for name, value in expected.items():
        assert re.search(
            rf"^{name} = {value}$", jit_source, re.MULTILINE
        ), f"{name} must stay {value}"
    algo_source = _source(ALGO)
    assert "class AllReduceAlgo(enum.Enum)" in algo_source
    # auto() keeps push/pull ordering tied to declaration order, not magic ints.
    assert algo_source.count("= enum.auto()") == 3
    assert "def is_push(self)" in algo_source
    assert "return 2 if self == AllReduceAlgo.TWO_SHOT_PULL else 1" in algo_source


def test_env_switch_is_off_by_default() -> None:
    """The switch is one explicit opt-in env var, read at construction only."""
    source = _source(COMM)
    assert 'VLLM_MUSA_CUSTOM_AR_V2"' in source
    on_line = next(
        line for line in source.splitlines() if line.startswith("_CUSTOM_AR_V2_ON")
    )
    off_line = next(
        line for line in source.splitlines() if line.startswith("_CUSTOM_AR_V2_OFF")
    )
    assert '"1"' in on_line
    assert '"0"' in off_line
    assert "def _custom_ar_v2_enabled()" in source
    helper = source.split("def _custom_ar_v2_enabled()", 1)[1].split("\ndef ", 1)[0]
    assert "os.environ.get(_CUSTOM_AR_V2_ENV" in helper
    # An unset or unrecognized value falls through to the contract.  The v2
    # feature is supported there but never preferred, so the default stays v1.
    assert "return _contract_prefers_v2()" in helper
    # Both implementations are chosen in the same constructor, exactly like
    # Upstream's dispatch_custom_allreduce() returns one class and lets the caller
    # instantiate it; nothing is selected later.
    ctor = source.split("if _custom_ar_v2_enabled():", 1)[1].split(
        "except Exception:", 1
    )[0]
    assert "_MusaCustomAllreduceV2Impl(" in ctor
    assert "_MusaJitCustomAllreduceImpl(" in ctor


def test_contract_advertises_v2_without_preferring_it() -> None:
    """Availability is not preference: TP models can opt in, defaults stay v1."""
    source = _source(RESOLVER)
    block = source.split(
        "if execution.tensor_parallel_size > 1 and not execution.is_pooling_model:",
        1,
    )[1].split("return contract", 1)[0]
    assert "supported_features=contract.supported_features" in block
    assert "preferred_features" not in block


def test_switch_never_mixes_implementations() -> None:
    """A v2-on run must not fall back to v1 kernels for larger payloads."""
    source = _source(COMM)
    assert "_v2_obj" not in source, "v1 impl must not hold a v2 object"
    assert "_init_custom_ar_v2" not in source
    impl = source.split("def _custom_all_reduce_impl(self", 1)[1].split(
        "\n    def ", 1
    )[0]
    assert "nbytes <=" not in impl, "no size gate in the hot path"
    assert "ONE_SHOT_PUSH" not in impl

    v2_impl = _source(V2_IMPL)
    assert "_v2_push_slot_bytes" not in v2_impl
    # Algorithm choice follows upstream's ModeConfig table, including the TP2
    # rule that the push threshold is 0.
    assert "mode_configs" in v2_impl
    assert "def _pick_algo(self, nbytes: int)" in v2_impl
    assert "ONE_SHOT_PULL" in v2_impl and "TWO_SHOT_PULL" in v2_impl


def test_runtime_adapter_maps_upstream_names() -> None:
    source = _source(RUNTIME)
    for name in (
        "cudaMalloc",
        "cudaMemset",
        "cudaMemcpy",
        "cudaIpcGetMemHandle",
        "cudaIpcOpenMemHandle",
        "cudaIpcCloseMemHandle",
        "cudaFree",
    ):
        assert f"def {name}(" in source, name
    assert "_MusaRTLibrary" in source, "adapter must reuse the v1 ctypes bindings"


def test_v2_object_declares_expected_modes() -> None:
    source = _source(OBJ)
    assert "mode_configs" in source
    assert "class MusaCustomAllReduceV2Obj" in source
    assert "def share_storage(self)" in source
    assert "def post_init(self, handles" in source


def test_fused_rmsnorm_refuses_instead_of_falling_back() -> None:
    source = _source(FUSED)
    assert source.count("NotImplementedError") >= 2
    assert "MusaCustomAllReduceV2RMSNorm" in source


def test_two_stage_keeps_the_tp4_fast_path_and_generic_reduce_scatter() -> None:
    """The 2stage kernel must keep both paths of the reference algorithm.

    The reference revision generalises nothing: a warp-aligned direct broadcast
    is only correct to prefer for the aligned TP4 512 KiB payload, while the
    generic path stays a reduce-scatter into local scratch followed by a
    broadcast. Promoting the fast path to the only path changes the algorithm.
    """
    source = _source(KERNEL)
    body = source.split("cross_device_reduce_2stage(", 1)[1].split(
        "template <typename T, int nranks, int vlen = 8>", 1
    )[0]
    assert "constexpr int kDirectBroadcastSize = (512 * 1024) / sizeof(P);" in body
    assert "if constexpr (nranks == 4) {" in body
    assert "if (size == kDirectBroadcastSize) {" in body
    assert "P* tmp = get_tmp_buf<P>(sg.signals[rank]);" in body
    assert "tmp[idx - start] = packed_reduce<P, nranks, A>(ptrs, idx);" in body
    assert "peer[rank * part + idx] = tmp[idx];" in body
    assert "if (data_ptr != nullptr) data = *data_ptr;" in body


def test_merged_modules_resolve_every_global() -> None:
    """Folding modules together must not orphan a name a body still reads.

    ``py_compile`` accepts a module that only fails at call time, so the merge
    needs a stronger check: every global a function body reads, and that is not
    assigned in that module, must exist at module level or in builtins.  The
    first merge attempt left ``Path`` behind in a removed file and the JIT build
    raised ``NameError`` only when the first kernel was compiled.
    """
    import builtins
    import symtable

    for path in (JIT, COMM):
        table = symtable.symtable(_source(path), str(path), "exec")
        declared = set(table.get_identifiers())
        unresolved: set[str] = set()

        def walk(node) -> None:
            for symbol in node.get_symbols():
                name = symbol.get_name()
                if (
                    symbol.is_global()
                    and not symbol.is_assigned()
                    and name not in declared
                    and not hasattr(builtins, name)
                ):
                    unresolved.add(name)
            for child in node.get_children():
                walk(child)

        walk(table)
        assert not unresolved, f"{path.name} reads undefined globals: {sorted(unresolved)}"
