"""Arch resolution and build layout of the MUSA JIT loader (no compiler, no device)."""

import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tvm_ffi")
pytest.importorskip("vllm")

import torch  # noqa: E402
from torchada import arch as musa_arch  # noqa: E402

from vllm_musa.jit_kernel.csrc import jit  # noqa: E402

MP22 = musa_arch.MusaArch(2, 2)
MP31 = musa_arch.MusaArch(3, 1)
ROOT = Path(__file__).resolve().parents[2]
PROBE = "vllm_musa_arch_probe"
PROBE_SOURCES = ("norm/rmsnorm.mu",)


def _fake_musa(monkeypatch, *, initialized: bool, capabilities=()) -> list[str]:
    """Make torch.musa report an initialization state and per-device capabilities.

    Returns the list of device reads. A read while MUSA is not initialized is
    recorded and then fails, as it would initialize MUSA on real hardware.
    """
    reads: list[str] = []

    def _read(what):
        reads.append(what)
        if not initialized:
            raise AssertionError("device read while MUSA is not initialized")

    def device_count():
        _read("device_count")
        return len(capabilities)

    def get_device_capability(device=None):
        _read("get_device_capability")
        return capabilities[0 if device is None else int(device)]

    def current_device():
        _read("current_device")
        return 0

    musa = getattr(torch, "musa", None)
    if musa is None:
        musa = SimpleNamespace()
        monkeypatch.setattr(torch, "musa", musa, raising=False)
    monkeypatch.setattr(musa, "is_initialized", lambda: initialized, raising=False)
    monkeypatch.setattr(musa, "device_count", device_count, raising=False)
    monkeypatch.setattr(
        musa, "get_device_capability", get_device_capability, raising=False
    )
    monkeypatch.setattr(musa, "current_device", current_device, raising=False)
    return reads


@pytest.fixture
def jit_env(monkeypatch, tmp_path):
    """Run load_musa_jit against a tmp cache with ninja and the loader stubbed."""
    for name in jit._ARCH_ENV:
        monkeypatch.delenv(name, raising=False)
    cache = tmp_path / "cache"
    monkeypatch.setattr(jit, "_CACHE_DIR", cache)
    monkeypatch.setattr(jit, "_musa_home", lambda: tmp_path / "musa")
    monkeypatch.setattr(jit, "find_include_path", lambda: "/tvm/include")
    monkeypatch.setattr(jit, "find_dlpack_include_path", lambda: "/dlpack/include")
    monkeypatch.setattr(jit, "find_libtvm_ffi", lambda: "/tvm/lib/libtvm_ffi.so")

    commands = []
    loaded = []
    locks = []
    probes = []
    monkeypatch.setattr(
        jit,
        "subprocess",
        SimpleNamespace(run=lambda command, check: commands.append(command)),
    )
    monkeypatch.setattr(
        jit,
        "tvm_ffi",
        SimpleNamespace(load_module=lambda path: loaded.append(path) or path),
    )
    monkeypatch.setattr(
        jit, "FileLock", lambda path: locks.append(path) or contextlib.nullcontext()
    )

    def device_archs():
        probes.append(True)
        return ()

    # Every test either sets an arch variable or asserts on this stub, so the
    # device probe and its process-wide cache never take part.
    monkeypatch.setattr(musa_arch, "_device_archs", device_archs)
    reads = _fake_musa(monkeypatch, initialized=False)

    jit.load_musa_jit.cache_clear()
    yield SimpleNamespace(
        cache=cache,
        commands=commands,
        loaded=loaded,
        locks=locks,
        probes=probes,
        reads=reads,
        load=lambda: jit.load_musa_jit(PROBE, PROBE_SOURCES),
    )
    jit.load_musa_jit.cache_clear()


def _musa_cflags(build_dir: Path) -> list[str]:
    ninja = (build_dir / "build.ninja").read_text()
    prefix = "musa_cflags = "
    line = next(line for line in ninja.splitlines() if line.startswith(prefix))
    return line[len(prefix) :].split()


def _arch_flags(flags: list[str]) -> list[str]:
    return [flag for flag in flags if flag.startswith("--offload-arch")]


@pytest.mark.parametrize(
    "value",
    ["31", "3.1", "mp_31", "MP31", " 3.1 ", "31;3.1", "3.1 mp_31", "31;\tmp_31"],
)
def test_arch_list_forms_build_one_mp31_target(jit_env, monkeypatch, value):
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", value)

    jit_env.load()

    build_root = jit_env.cache / musa_arch.cache_tag((MP31,))
    flags = _musa_cflags(build_root / PROBE)
    assert _arch_flags(flags) == ["--offload-arch=mp_31"]
    assert flags[flags.index("-ffast-math") + 1] == "--offload-arch=mp_31"
    assert not any("__MUSA_ARCH" in flag for flag in flags)
    assert jit_env.probes == []
    assert jit_env.reads == []


def test_arch_list_with_two_archs_targets_both_in_one_tagged_root(jit_env, monkeypatch):
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "3.1;2.2")

    jit_env.load()

    build_root = jit_env.cache / musa_arch.cache_tag((MP22, MP31))
    assert _arch_flags(_musa_cflags(build_root / PROBE)) == [
        "--offload-arch=mp_22",
        "--offload-arch=mp_31",
    ]


def test_build_root_carries_the_arch_tag(jit_env, monkeypatch):
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "3.1")

    jit_env.load()

    tag = musa_arch.cache_tag((MP31,))
    build_root = jit_env.cache / tag
    build_dir = build_root / PROBE
    assert tag.startswith("mp31")
    assert sorted(path.name for path in jit_env.cache.iterdir()) == [tag]
    assert (build_dir / "build.ninja").is_file()
    assert (build_root / "tmp").is_dir()
    assert jit_env.locks == [str(build_root / "tmp" / f"{PROBE}.lock")]
    assert jit_env.loaded == [str(build_dir / f"{PROBE}.so")]
    assert len(jit_env.commands) == 1
    command = jit_env.commands[0]
    assert command[command.index("-C") + 1] == str(build_dir)
    assert command[command.index("-f") + 1] == str(build_dir / "build.ninja")
    ninja = (build_dir / "build.ninja").read_text()
    assert f"{build_dir.resolve().as_posix()}/{PROBE}.so" in ninja


def test_arch_list_310_raises_before_building(jit_env, monkeypatch):
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "310")

    with pytest.raises(ValueError) as excinfo:
        jit_env.load()

    assert "VLLM_MUSA_ARCH_LIST" in str(excinfo.value)
    assert "310" in str(excinfo.value)
    assert jit_env.commands == []
    assert jit_env.loaded == []


def test_vllm_arch_list_takes_precedence_over_mate(jit_env, monkeypatch):
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "3.1")
    # A later variable is not read, so a malformed value there is ignored.
    monkeypatch.setenv("MATE_MUSA_ARCH_LIST", "310")

    jit_env.load()

    build_root = jit_env.cache / musa_arch.cache_tag((MP31,))
    assert _arch_flags(_musa_cflags(build_root / PROBE)) == ["--offload-arch=mp_31"]


@pytest.mark.parametrize(("vllm_value", "arch"), [(None, MP31), ("", MP22)])
def test_mate_arch_list_decides_when_vllm_arch_list_is_unset_or_empty(
    jit_env, monkeypatch, vllm_value, arch
):
    if vllm_value is not None:
        monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", vllm_value)
    monkeypatch.setenv("MATE_MUSA_ARCH_LIST", f"{arch.major}.{arch.minor}")

    jit_env.load()

    build_root = jit_env.cache / musa_arch.cache_tag((arch,))
    assert _arch_flags(_musa_cflags(build_root / PROBE)) == [f"--offload-arch={arch}"]
    assert jit_env.probes == []


def test_unset_arch_lists_fall_back_to_mp31_without_devices(jit_env):
    jit_env.load()

    assert jit_env.probes == [True]
    build_root = jit_env.cache / musa_arch.cache_tag((MP31,))
    assert _arch_flags(_musa_cflags(build_root / PROBE)) == ["--offload-arch=mp_31"]


def test_device_check_rejects_targets_missing_a_visible_device(jit_env, monkeypatch):
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "3.1")
    _fake_musa(monkeypatch, initialized=True, capabilities=[(3, 1), (2, 2)])

    with pytest.raises(RuntimeError) as excinfo:
        jit_env.load()

    assert "VLLM_MUSA_ARCH_LIST" in str(excinfo.value)
    assert jit_env.commands == []
    assert jit_env.loaded == []


@pytest.mark.parametrize(
    ("value", "capabilities"),
    [("3.1", [(3, 1)]), ("3.1", [(3, 1), (3, 1)]), ("2.2;3.1", [(3, 1)])],
)
def test_device_check_accepts_targets_covering_every_device(
    jit_env, monkeypatch, value, capabilities
):
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", value)
    _fake_musa(monkeypatch, initialized=True, capabilities=capabilities)

    jit_env.load()

    assert len(jit_env.loaded) == 1


def test_device_check_is_skipped_while_musa_is_not_initialized(jit_env, monkeypatch):
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "2.2")

    jit_env.load()

    assert jit_env.reads == []
    assert len(jit_env.loaded) == 1


def test_importing_jit_resolves_no_arch(monkeypatch):
    monkeypatch.setenv("VLLM_MUSA_ARCH_LIST", "310")
    calls = []
    monkeypatch.setattr(
        musa_arch, "target_archs", lambda *args, **kwargs: calls.append(kwargs)
    )

    spec = importlib.util.spec_from_file_location(
        "_vllm_musa_jit_import_probe", Path(jit.__file__)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert calls == []
    assert module._ARCH_ENV == ("VLLM_MUSA_ARCH_LIST", "MATE_MUSA_ARCH_LIST")
    assert not hasattr(module, "_musa_arch_flags")
    assert not hasattr(module, "_normalize_musa_arch")


def test_rope_module_name_carries_only_the_store_hint(monkeypatch):
    from vllm_musa.jit_kernel.csrc import rope

    calls = []

    def fake_load(name, sources, **kwargs):
        calls.append((name, sources, kwargs))
        return "module"

    monkeypatch.setattr(rope, "load_musa_jit", fake_load)

    assert rope._rope_module.__wrapped__("stcg") == "module"
    assert calls == [
        (
            "vllm_musa_rope_stcg",
            ("rope/rotary_embedding.mu",),
            {"extra_musa_cflags": ("-DVLLM_MUSA_ROPE_STORE_HINT=2",)},
        )
    ]
    assert not hasattr(rope, "_musa_arch_tag")


def test_rope_call_path_reads_no_device_capability(monkeypatch):
    from vllm_musa.jit_kernel.csrc import rope

    launches = []
    hints = []

    def fake_rope_module(store_hint):
        hints.append(store_hint)
        return SimpleNamespace(sgl_rotary_embedding=lambda *args: launches.append(args))

    reads = _fake_musa(monkeypatch, initialized=True, capabilities=[(3, 1)])
    monkeypatch.setattr(rope, "_rope_module", fake_rope_module)
    monkeypatch.setenv("VLLM_MUSA_ROPE_STORE_HINT", "auto")

    rope._rotary_embedding_kernel(
        None, None, None, None, 1, 128, 4, 1, 128, 0, 0, 0, 0, 0, 0, 0, 1, 1, True, True
    )

    assert hints == ["stwb"]
    assert len(launches) == 1
    assert reads == []


def test_jit_modules_pass_no_arch_define(monkeypatch):
    pytest.importorskip("tilelang")
    from vllm_musa.jit_kernel.csrc import allreduce, norm

    calls = []

    def fake_load(name, sources, **kwargs):
        calls.append((name, kwargs.get("extra_musa_cflags", ())))
        return "module"

    monkeypatch.setattr(norm, "load_musa_jit", fake_load)
    monkeypatch.setattr(allreduce, "load_musa_jit", fake_load)

    norm._norm_module.__wrapped__()
    norm._qk_mrope_module.__wrapped__()
    allreduce._custom_ar_module.__wrapped__(8)

    assert [name for name, _ in calls][:2] == [
        "vllm_musa_norm",
        "vllm_musa_norm_qk_mrope",
    ]
    assert len(calls) == 3
    for _, flags in calls:
        assert not any("__MUSA_ARCH" in flag for flag in flags)
        assert not any(flag.startswith("--offload-arch") for flag in flags)


def test_no_source_defines_or_reads_musa_arch_list():
    suffixes = {".py", ".mu", ".muh", ".cu", ".cuh", ".h", ".hpp", ".cpp"}
    hits = [
        str(path.relative_to(ROOT))
        for top in ("vllm_musa", "csrc")
        for path in (ROOT / top).rglob("*")
        if path.suffix in suffixes
        and path.is_file()
        and "__MUSA_ARCH_LIST__" in path.read_text(errors="ignore")
    ]
    assert hits == []


def test_moe_glue_falls_back_when_the_arch_check_fails(monkeypatch):
    from vllm_musa.jit_kernel.csrc import moe

    def failing_module():
        raise RuntimeError("visible device arch is not among the JIT targets")

    monkeypatch.setattr(moe, "_moe_module", failing_module)

    assert moe._moe_available.__wrapped__() is False
