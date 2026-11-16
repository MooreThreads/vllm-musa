# SPDX-License-Identifier: Apache-2.0
"""Rotary-embedding store-hint prewarm before graph capture.

No device, compiler, vLLM or tvm_ffi: both modules are loaded from source with
their imports stubbed, and ``load_musa_jit`` records what would be built.
"""

import functools
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ROPE_SOURCE = ROOT / "vllm_musa" / "jit_kernel" / "csrc" / "rope.py"
WARMUP_SOURCE = ROOT / "vllm_musa" / "model_executor" / "warmup" / "deep_gemm_warmup.py"
ENV = "VLLM_MUSA_ROPE_STORE_HINT"


def _stub(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _stub_package(monkeypatch, name: str, **attrs) -> types.ModuleType:
    """Stub ``name`` and every parent package, linked for a plain ``import``."""
    module = _stub(monkeypatch, name, **attrs)
    child = module
    while "." in child.__name__:
        parent_name, _, leaf = child.__name__.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is None or not getattr(parent, "_is_test_stub", False):
            parent = _stub(monkeypatch, parent_name, _is_test_stub=True)
        setattr(parent, leaf, child)
        child = parent
    return module


def _load(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cache_once(fn):
    results = {}

    @functools.wraps(fn)
    def wrapper(*args):
        if args not in results:
            results[args] = fn(*args)
        return results[args]

    return wrapper


@pytest.fixture
def rope(monkeypatch):
    loads: list[tuple[str, tuple[str, ...]]] = []
    launches: list[str] = []

    def load_musa_jit(name, sources, extra_musa_cflags=()):
        loads.append((name, tuple(extra_musa_cflags)))
        return types.SimpleNamespace(
            sgl_rotary_embedding=lambda *args: launches.append(name)
        )

    monkeypatch.delenv(ENV, raising=False)
    _stub(monkeypatch, "torch")
    _stub(
        monkeypatch,
        "vllm.utils.torch_utils",
        direct_register_custom_op=lambda **kwargs: None,
    )
    _stub(monkeypatch, "vllm_musa.jit_kernel.csrc.jit", load_musa_jit=load_musa_jit)
    _stub(monkeypatch, "vllm_musa.jit_kernel.utils", cache_once=_cache_once)
    module = _load(ROPE_SOURCE, "_rope_under_test")
    return types.SimpleNamespace(module=module, loads=loads, launches=launches)


def _names(*hints: str) -> list[str]:
    return [f"vllm_musa_rope_{hint}" for hint in hints]


def _launch(module, num_tokens: int, rot_dim: int) -> None:
    module._rotary_embedding_kernel(
        positions=None,
        query=None,
        key=None,
        cos_sin_cache=None,
        num_tokens=num_tokens,
        head_size=256,
        num_heads=8,
        num_kv_heads=2,
        rot_dim=rot_dim,
        seq_len=0,
        query_batch_stride=0,
        key_batch_stride=0,
        query_stride=2048,
        key_stride=512,
        query_head_stride=256,
        key_head_stride=256,
        query_dim_stride=1,
        key_dim_stride=1,
        is_neox=True,
        has_key=True,
    )


@pytest.mark.parametrize(
    ("rot_dim", "hints"),
    [
        (64, ("stwb", "stcg")),
        (128, ("stwb",)),
        (32, ("stwb",)),
        (None, ("stwb", "stcg")),
    ],
)
def test_auto_policy_prewarms_the_hints_it_can_pick(rope, rot_dim, hints) -> None:
    built = rope.module.prewarm_rope_modules(rot_dim)

    assert built == _names(*hints)
    assert [name for name, _ in rope.loads] == _names(*hints)


def test_prewarm_builds_each_hint_with_its_compile_flag(rope) -> None:
    rope.module.prewarm_rope_modules(64)

    assert rope.loads == [
        ("vllm_musa_rope_stwb", ("-DVLLM_MUSA_ROPE_STORE_HINT=1",)),
        ("vllm_musa_rope_stcg", ("-DVLLM_MUSA_ROPE_STORE_HINT=2",)),
    ]


def test_explicit_auto_is_the_auto_policy(rope, monkeypatch) -> None:
    monkeypatch.setenv(ENV, "AUTO")

    assert rope.module.prewarm_rope_modules(64) == _names("stwb", "stcg")


@pytest.mark.parametrize("rot_dim", [64, 128, None])
@pytest.mark.parametrize(
    ("value", "hint"),
    [
        ("stcs", "stcs"),
        ("STCG", "stcg"),
        ("default", "default"),
        ("stwb", "stwb"),
        ("unknown", "stwb"),
        ("", "stwb"),
    ],
)
def test_override_prewarms_only_the_forced_hint(
    rope, monkeypatch, rot_dim, value, hint
) -> None:
    monkeypatch.setenv(ENV, value)

    assert rope.module.prewarm_rope_modules(rot_dim) == _names(hint)
    assert [name for name, _ in rope.loads] == _names(hint)


@pytest.mark.parametrize("override", [None, "stcs", "unknown"])
def test_prewarm_covers_exactly_what_the_call_path_picks(
    rope, monkeypatch, override
) -> None:
    if override is not None:
        monkeypatch.setenv(ENV, override)
    picked_any: set[str] = set()
    for rot_dim in (16, 32, 64, 80, 128, 256):
        picked = {
            f"vllm_musa_rope_{rope.module._store_hint(num_tokens, rot_dim)}"
            for num_tokens in range(1, 8193)
        }
        picked_any |= picked

        assert picked == set(rope.module.prewarm_rope_modules(rot_dim))
    assert picked_any == set(rope.module.prewarm_rope_modules(None))


def test_call_path_reuses_the_prewarmed_modules(rope) -> None:
    rope.module.prewarm_rope_modules(64)

    for num_tokens in (16, 1024, 4096):
        _launch(rope.module, num_tokens, rot_dim=64)

    assert [name for name, _ in rope.loads] == _names("stwb", "stcg")
    assert rope.launches == _names("stwb", "stcg", "stwb")


PRESPLIT = "qwen3.qk_rope_kv_presplit"


class _FakeRope:
    """Stands in for MusaRotaryEmbedding, dispatched the way CustomOp does."""

    def __init__(self, cache_width: int | None, *, oot: bool = True) -> None:
        if cache_width is not None:
            self.cos_sin_cache = types.SimpleNamespace(shape=(40960, cache_width))
        self._forward_method = self.forward_oot if oot else self.forward_native

    def forward_oot(self, *args):
        raise AssertionError("warmup must not run the module")

    def forward_native(self, *args):
        raise AssertionError("warmup must not run the module")


class _FakeRopeSubclass(_FakeRope):
    """A subclass that inherits forward_oot."""


class _FakeVisionRope:
    """Stands in for MusaVisionApplyRotaryEmb."""

    def __init__(self, required_bf16_neox_shape: tuple[int, int] | None) -> None:
        self.required_bf16_neox_shape = required_bf16_neox_shape


class _FakeBackbone:
    """Stands in for a support_torch_compile module, which keeps its config."""

    def __init__(self, vllm_config: object) -> None:
        self.vllm_config = vllm_config


class _FakeModel:
    def __init__(self, *children: object) -> None:
        self.children = children

    def modules(self):
        return iter((self, *self.children))


def _config(*, preferred=()) -> types.SimpleNamespace:
    return types.SimpleNamespace(preferred_features=frozenset(preferred))


@pytest.fixture
def warmup(monkeypatch, rope):
    upstream: list[int] = []
    state = types.SimpleNamespace(current_config=None, use_native=False)

    def get_current_vllm_config_or_none():
        if isinstance(state.current_config, Exception):
            raise state.current_config
        return state.current_config

    def prefers_feature(vllm_config, feature) -> bool:
        return feature in vllm_config.preferred_features

    _stub(
        monkeypatch,
        "torch",
        Size=tuple,
        nn=types.SimpleNamespace(Module=object),
    )
    _stub_package(monkeypatch, "vllm.logger", init_logger=logging.getLogger)
    _stub_package(
        monkeypatch,
        "vllm.model_executor.warmup.deep_gemm_warmup",
        deep_gemm_warmup=lambda model, max_tokens: upstream.append(max_tokens),
    )
    _stub(
        monkeypatch,
        "vllm.config",
        get_current_vllm_config_or_none=get_current_vllm_config_or_none,
    )
    _stub(
        monkeypatch,
        "vllm_musa.model_executor.layers.rotary_embedding.base",
        MusaRotaryEmbedding=_FakeRope,
    )
    _stub(
        monkeypatch,
        "vllm_musa.optimization_contract.rotary",
        MusaVisionApplyRotaryEmb=_FakeVisionRope,
    )
    _stub(
        monkeypatch,
        "vllm_musa.optimization_contract.policy",
        prefers_feature=prefers_feature,
    )
    _stub(
        monkeypatch,
        "vllm_musa.optimization_contract.types",
        OptimizationFeature=types.SimpleNamespace(QWEN3_QK_ROPE_KV_PRESPLIT=PRESPLIT),
    )
    native_flag = types.SimpleNamespace(get=lambda: state.use_native)
    _stub(
        monkeypatch,
        "vllm_musa.utils.environ",
        envs=types.SimpleNamespace(VLLM_MUSA_CUSTOM_OP_USE_NATIVE=native_flag),
    )
    monkeypatch.setitem(sys.modules, "vllm_musa.jit_kernel.csrc.rope", rope.module)
    module = _load(WARMUP_SOURCE, "_deep_gemm_warmup_under_test")

    def built(model) -> list[str]:
        module._warmup_rope_store_hints(model)
        return [name for name, _ in rope.loads]

    return types.SimpleNamespace(
        module=module, state=state, upstream=upstream, rope=rope, built=built
    )


@pytest.mark.parametrize(
    "forward_method",
    [
        "forward_native",
        # CustomOp.maybe_compile wraps forward_native in a plain function.
        lambda *args: None,
    ],
)
def test_native_dispatch_builds_nothing(warmup, forward_method) -> None:
    rope = _FakeRope(64)
    rope._forward_method = (
        getattr(rope, forward_method)
        if isinstance(forward_method, str)
        else forward_method
    )
    model = _FakeModel(_FakeBackbone(_config()), rope)

    assert warmup.built(model) == []


@pytest.mark.parametrize(
    ("child", "hints"),
    [
        # head_dim 256 with partial_rotary_factor 0.25 rotates 64 lanes.
        (_FakeRope(64), ("stwb", "stcg")),
        (_FakeRopeSubclass(64), ("stwb", "stcg")),
        (_FakeRope(128), ("stwb",)),
        (_FakeRope(None), ("stwb", "stcg")),
    ],
)
def test_oot_dispatch_builds_every_auto_hint_for_its_width(
    warmup, child, hints
) -> None:
    assert warmup.built(_FakeModel(child)) == _names(*hints)


def test_layers_sharing_a_width_build_it_once(warmup) -> None:
    model = _FakeModel(_FakeRope(128), _FakeRope(128))

    assert warmup.built(model) == _names("stwb")


@pytest.mark.parametrize(
    "child", [_FakeRope(64), _FakeVisionRope((80, 40)), _FakeVisionRope(None)]
)
def test_native_route_builds_nothing(warmup, child) -> None:
    warmup.state.use_native = True

    assert warmup.built(_FakeModel(_FakeBackbone(_config()), child)) == []


def test_qwen3_presplit_builds_nothing(warmup) -> None:
    config = _config(preferred={PRESPLIT})
    model = _FakeModel(_FakeBackbone(config), _FakeRope(64), _FakeRope(128))

    assert warmup.built(model) == []


def test_presplit_reads_the_current_config_first(warmup) -> None:
    warmup.state.current_config = _config(preferred={PRESPLIT})
    model = _FakeModel(_FakeBackbone(_config()), _FakeRope(64))

    assert warmup.built(model) == []


def test_unreadable_contract_keeps_the_dispatch_decision(warmup) -> None:
    model = _FakeModel(_FakeBackbone(object()), _FakeRope(64))

    assert warmup.built(model) == _names("stwb", "stcg")


@pytest.mark.parametrize("current_config", [RuntimeError("config"), None])
@pytest.mark.parametrize(("oot", "hints"), [(False, ()), (True, ("stwb", "stcg"))])
def test_missing_config_keeps_the_dispatch_decision(
    warmup, current_config, oot, hints
) -> None:
    warmup.state.current_config = current_config
    model = _FakeModel(_FakeRope(64, oot=oot))

    assert warmup.built(model) == _names(*hints)


@pytest.mark.parametrize(
    ("child", "hints"),
    [
        (_FakeVisionRope((80, 40)), ("stwb",)),
        (_FakeVisionRope((128, 32)), ("stwb", "stcg")),
        (_FakeVisionRope(None), ("stwb", "stcg")),
    ],
)
def test_vision_adapter_builds_the_hints_for_its_width(warmup, child, hints) -> None:
    assert warmup.built(_FakeModel(child)) == _names(*hints)


def test_vision_adapter_ignores_the_text_rotary_decision(warmup) -> None:
    config = _config(preferred={PRESPLIT})
    model = _FakeModel(_FakeBackbone(config), _FakeVisionRope((80, 40)))

    assert warmup.module._rope_kernel_rotary_dims(model) == {80}


def test_model_without_the_kernel_is_not_prewarmed(warmup) -> None:
    assert warmup.built(_FakeModel(object(), object())) == []


def test_prewarm_failure_only_warns(warmup, monkeypatch, caplog) -> None:
    def boom(rot_dim=None) -> list[str]:
        raise RuntimeError("mcc failed")

    monkeypatch.setattr(warmup.rope.module, "prewarm_rope_modules", boom)

    with caplog.at_level(logging.WARNING):
        warmup.module._warmup_rope_store_hints(_FakeModel(_FakeRope(64)))

    assert any(
        record.levelno == logging.WARNING and "mcc failed" in record.getMessage()
        for record in caplog.records
    )


def test_unimportable_rope_kernel_only_warns(warmup, monkeypatch, caplog) -> None:
    monkeypatch.setitem(sys.modules, "vllm_musa.jit_kernel.csrc.rope", None)

    with caplog.at_level(logging.WARNING):
        warmup.module._warmup_rope_store_hints(_FakeModel(_FakeRope(64)))

    assert [record.levelno for record in caplog.records] == [logging.WARNING]


def test_deep_gemm_warmup_runs_the_rope_prewarm(warmup, monkeypatch) -> None:
    monkeypatch.setattr(warmup.module, "_warmup_bf16_grouped_moe", lambda *a: None)
    monkeypatch.setattr(
        warmup.module, "_warmup_paged_mqa_logits_metadata", lambda *a: None
    )

    warmup.module.deep_gemm_warmup(_FakeModel(_FakeRope(64)), 8192)

    assert warmup.upstream == [8192]
    assert [name for name, _ in warmup.rope.loads] == _names("stwb", "stcg")
