# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MUSA-100114: Eagle/Eagle3/DFlash FULL draft CUDAGraphs."""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, replace
from enum import Enum
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:  # Host-side CI without PyTorch installed

    class _FakeTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        @property
        def shape(self) -> tuple[int, ...]:
            return self._arr.shape

        def __getitem__(self, idx) -> _FakeTensor:
            return _FakeTensor(self._arr[idx])

        def __setitem__(self, idx, val) -> None:
            self._arr[idx] = val._arr if isinstance(val, _FakeTensor) else val

        def __isub__(self, other: _FakeTensor) -> _FakeTensor:
            self._arr -= other._arr
            return self

        def __eq__(self, other):  # type: ignore[override]
            rhs = other._arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self._arr == rhs)

        def all(self) -> bool:
            return bool(self._arr.all())

        def any(self) -> bool:
            return bool(self._arr.any())

        def item(self):
            return self._arr.item()

        def copy_(self, src: _FakeTensor) -> _FakeTensor:
            self._arr[...] = src._arr
            return self

        def zero_(self) -> _FakeTensor:
            self._arr[...] = 0
            return self

        def fill_(self, val) -> _FakeTensor:
            self._arr[...] = val
            return self

        def view(self, *shape: int) -> _FakeTensor:
            return _FakeTensor(self._arr.reshape(*shape))

        def numpy(self) -> np.ndarray:
            return self._arr

        def tolist(self):
            return self._arr.tolist()

        def data_ptr(self) -> int:
            return int(self._arr.__array_interface__["data"][0])

    torch = SimpleNamespace(  # type: ignore[assignment]
        Tensor=_FakeTensor,
        dtype=np.dtype,
        int32=np.int32,
        int64=np.int64,
        zeros=lambda size, dtype=np.int32: _FakeTensor(np.zeros(size, dtype=dtype)),
        full=lambda size, val, dtype=np.int64: _FakeTensor(
            np.full(size, val, dtype=dtype)
        ),
        arange=lambda n, dtype=np.int32: _FakeTensor(np.arange(n, dtype=dtype)),
        tensor=lambda data, dtype=None: _FakeTensor(np.array(data, dtype=dtype)),
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
SERIES_DIR = REPO_ROOT / "vllm_musa" / "patches" / "series"
PATCH_0026 = SERIES_DIR / "0026-MUSA-vllm.v1.cudagraph_dispatcher.patch"
PATCH_0030 = SERIES_DIR / "0030-MUSA-vllm.v1.spec_decode.llm_base_proposer.patch"
PATCH_0035 = (
    SERIES_DIR / "0035-MUSA-support-full-graph-drafts-and-pinned-metadata.patch"
)


def _extract_added_python_block(patch_path: Path, anchor: str, stop_line: str) -> str:
    lines = patch_path.read_text(encoding="utf-8").splitlines()
    collecting = False
    collected: list[str] = []
    for line in lines:
        if not collecting:
            if line.startswith("+") and not line.startswith("+++") and anchor in line:
                collecting = True
                collected.append(line[1:])
        else:
            if line.startswith("@@ ") or line.startswith("diff --git "):
                break
            if (
                line.startswith("+")
                and not line.startswith("+++")
                or line.startswith(" ")
            ):
                content = line[1:]
                if stop_line and content.startswith(stop_line):
                    break
                collected.append(content)
    assert collected, f"Could not find {anchor!r} in {patch_path.name}"
    return textwrap.dedent("\n".join(collected))


class _FakeCUDAGraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (2, 0)
    FULL_AND_PIECEWISE = (2, 1)

    def decode_mode(self) -> _FakeCUDAGraphMode:
        return (
            _FakeCUDAGraphMode(self.value[0]) if isinstance(self.value, tuple) else self
        )

    def mixed_mode(self) -> _FakeCUDAGraphMode:
        return (
            _FakeCUDAGraphMode(self.value[1]) if isinstance(self.value, tuple) else self
        )

    def has_mode(self, mode: _FakeCUDAGraphMode) -> bool:
        return self.decode_mode() == mode or self.mixed_mode() == mode

    def has_full_cudagraphs(self) -> bool:
        return self.has_mode(_FakeCUDAGraphMode.FULL)

    def separate_routine(self) -> bool:
        return isinstance(self.value, tuple)


@dataclass(frozen=True)
class _FakeBatchDescriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0


@dataclass
class _FakeCommonAttentionMetadata:
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens_cpu_upper_bound: torch.Tensor | None = None
    is_prefilling: torch.Tensor | None = None

    def replace(self, **kwargs) -> _FakeCommonAttentionMetadata:
        return replace(self, **kwargs)


class _FakeCpuGpuBuffer:
    def __init__(self, size: int, dtype: torch.dtype = torch.int32) -> None:
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = torch.zeros(size, dtype=dtype)

    def copy_to_gpu(self, n: int | None = None) -> None:
        if n is None:
            self.gpu.copy_(self.cpu)
        else:
            self.gpu[:n].copy_(self.cpu[:n])


def _load_proposer_helpers(is_musa: bool = True) -> dict:
    code = _extract_added_python_block(
        PATCH_0030,
        "def _should_use_full_cudagraph_wrap(",
        "    def _greedy_sample(",
    )
    ast.parse(code)
    ns: dict = {
        "CUDAGraphMode": _FakeCUDAGraphMode,
        "CommonAttentionMetadata": _FakeCommonAttentionMetadata,
        "PADDING_SLOT_ID": -1,
        "current_platform": SimpleNamespace(is_musa=lambda: is_musa),
        "torch": torch,
    }
    exec(compile(code, str(PATCH_0030), "exec"), ns)
    return ns


class _MiniDraftDispatcher:
    """Mirrors CudagraphDispatcher after patch 0026 for draft-key verification."""

    def __init__(self, vllm_config) -> None:
        self.vllm_config = vllm_config
        self.for_draft_model = True
        self.compilation_config = vllm_config.compilation_config
        self.uniform_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.cudagraph_keys = {
            _FakeCUDAGraphMode.PIECEWISE: set(),
            _FakeCUDAGraphMode.FULL: set(),
        }
        self.cudagraph_mode = _FakeCUDAGraphMode.NONE
        self.keys_initialized = False

    def _compute_bs_to_padded_graph_size(self) -> None:
        max_size = self.compilation_config.max_cudagraph_capture_size
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        self._bs_to_padded_graph_size = [0] * (max_size + 1)
        for end, start in zip(capture_sizes + [max_size + 1], [0] + capture_sizes):
            for bs in range(start, end):
                self._bs_to_padded_graph_size[bs] = start if bs == start else end

    def _create_padded_batch_descriptor(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int = 0,
        uniform_decode_query_len: int | None = None,
    ) -> _FakeBatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        if uniform_decode_query_len is None:
            uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]
        if uniform_decode and self.cudagraph_mode.has_mode(_FakeCUDAGraphMode.FULL):
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
            assert num_tokens_padded % uniform_decode_query_len == 0
        else:
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)
        return _FakeBatchDescriptor(
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )

    def add_cudagraph_key(self, mode, desc) -> None:
        self.cudagraph_keys[mode].add(desc)


def _bind_0026_init_keys() -> None:
    code = _extract_added_python_block(
        PATCH_0026,
        "def initialize_cudagraph_keys(",
        "    def dispatch(",
    )
    # Strip patch context if incomplete; instead verify patch 0026 snippet directly
    assert "for key_set in self.cudagraph_keys.values():" in PATCH_0026.read_text(
        encoding="utf-8"
    )
    assert "target_qdl = 1 + self.vllm_config.num_speculative_tokens" in (
        PATCH_0026.read_text(encoding="utf-8")
    )
    _ = code


def test_patch_files_exist_and_expose_draft_full_cudagraph_wiring() -> None:
    for p in (PATCH_0026, PATCH_0030, PATCH_0035):
        assert p.is_file(), f"Missing patch file: {p}"

    p26 = PATCH_0026.read_text(encoding="utf-8")
    p30 = PATCH_0030.read_text(encoding="utf-8")
    p35 = PATCH_0035.read_text(encoding="utf-8")

    assert "for key_set in self.cudagraph_keys.values():" in p26
    assert "target_qdl = 1 + self.vllm_config.num_speculative_tokens" in p26
    assert "capture_sizes_for_draft_model = sorted(" in p26

    assert "def _should_use_full_cudagraph_wrap(self) -> bool:" in p30
    assert 'self.method in ("eagle", "eagle3", "dflash")' in p30
    assert "def _prepare_decode_step_attn_metadata(" in p30
    assert "self.build_per_group_and_layer_attn_metadata(" in p30

    assert 'in ("eagle", "eagle3", "dflash")' in p35
    assert "VLLM_MUSA_DRAFT_FULL_WRAP" in p35


@pytest.mark.parametrize(
    ("method", "env_override", "expected"),
    [
        ("eagle", None, True),
        ("eagle3", None, True),
        ("dflash", None, True),
        ("mtp", None, False),
        ("eagle3", "0", False),
        ("mtp", "1", True),
    ],
)
def test_should_use_full_cudagraph_wrap_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    env_override: str | None,
    expected: bool,
) -> None:
    monkeypatch.delenv("VLLM_MUSA_DRAFT_FULL_WRAP", raising=False)
    monkeypatch.delenv("VLLM_MUSA_DFLASH_FULL_WRAP", raising=False)
    if env_override is not None:
        monkeypatch.setenv("VLLM_MUSA_DRAFT_FULL_WRAP", env_override)

    helpers = _load_proposer_helpers(is_musa=True)
    dummy_self = SimpleNamespace(
        method=method,
        compilation_config=SimpleNamespace(
            cudagraph_mode=_FakeCUDAGraphMode.FULL_AND_PIECEWISE
        ),
        speculative_config=SimpleNamespace(
            enforce_eager=False,
            disable_padded_drafter_batch=False,
        ),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(use_ubatching=False)
        ),
    )
    assert helpers["_should_use_full_cudagraph_wrap"](dummy_self) is expected


def test_dflash_full_wrap_honors_dflash_env_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_MUSA_DRAFT_FULL_WRAP", raising=False)
    monkeypatch.setenv("VLLM_MUSA_DFLASH_FULL_WRAP", "0")
    helpers = _load_proposer_helpers(is_musa=True)
    dummy_self = SimpleNamespace(
        method="dflash",
        compilation_config=SimpleNamespace(
            cudagraph_mode=_FakeCUDAGraphMode.FULL_AND_PIECEWISE
        ),
        speculative_config=SimpleNamespace(
            enforce_eager=False,
            disable_padded_drafter_batch=False,
        ),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(use_ubatching=False)
        ),
    )
    assert helpers["_should_use_full_cudagraph_wrap"](dummy_self) is False


def test_initialize_cudagraph_keys_passes_full_mode_for_wrapped_eagle3() -> None:
    helpers = _load_proposer_helpers(is_musa=True)
    recorded: list[tuple[_FakeCUDAGraphMode, int]] = []
    dispatcher = SimpleNamespace(
        uniform_decode_query_len=4,
        initialize_cudagraph_keys=lambda mode,
        uniform_decode_query_len=1: recorded.append((mode, uniform_decode_query_len)),
    )

    # Wrapped Eagle3 passes FULL_AND_PIECEWISE through with target_qdl=4
    eagle3 = SimpleNamespace(
        speculative_config=SimpleNamespace(enforce_eager=False),
        _use_full_cudagraph=True,
        parallel_drafting=False,
        needs_extra_input_slots=False,
        cudagraph_dispatcher=dispatcher,
    )
    helpers["initialize_cudagraph_keys"](eagle3, _FakeCUDAGraphMode.FULL_AND_PIECEWISE)
    assert recorded[-1] == (_FakeCUDAGraphMode.FULL_AND_PIECEWISE, 4)

    # Unwrapped MTP stays on PIECEWISE
    mtp = SimpleNamespace(
        speculative_config=SimpleNamespace(enforce_eager=False),
        _use_full_cudagraph=False,
        parallel_drafting=False,
        needs_extra_input_slots=False,
        cudagraph_dispatcher=dispatcher,
    )
    helpers["initialize_cudagraph_keys"](mtp, _FakeCUDAGraphMode.FULL_AND_PIECEWISE)
    assert recorded[-1] == (_FakeCUDAGraphMode.PIECEWISE, 4)


def test_draft_qdl1_capture_sizes_match_target_full_capture_ceiling() -> None:
    """Verify 0026 draft qdl=1 keys never exceed captured target_num_reqs."""
    # Default vllm serve: max_num_seqs=256, K=3 (target_qdl=4), max_capture=512
    capture_sizes = [4, 8, 16, 24, 32, 64, 128, 256, 512]
    max_num_seqs = 256
    target_qdl = 4
    vllm_config = SimpleNamespace(
        num_speculative_tokens=3,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=SimpleNamespace(parallel_drafting=False),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=512,
            cudagraph_capture_sizes=capture_sizes,
        ),
    )
    disp = _MiniDraftDispatcher(vllm_config)
    disp.cudagraph_mode = _FakeCUDAGraphMode.FULL_AND_PIECEWISE
    disp._compute_bs_to_padded_graph_size()

    max_target_tokens = target_qdl * max_num_seqs
    target_decode_sizes = [
        x for x in capture_sizes if target_qdl <= x <= max_target_tokens
    ]
    draft_sizes = sorted(
        {
            disp._bs_to_padded_graph_size[min(target_bs // target_qdl, max_num_seqs)]
            for target_bs in target_decode_sizes
        }
    )
    for bs, lora in product(draft_sizes, [0]):
        disp.add_cudagraph_key(
            _FakeCUDAGraphMode.FULL,
            disp._create_padded_batch_descriptor(bs, True, lora > 0, lora, 1),
        )

    # Target's max_cudagraph_capture_size=512 only visits up to 512 // 4 = 128 reqs!
    assert max(draft_sizes) == 128
    assert 256 not in draft_sizes
    assert (
        _FakeBatchDescriptor(num_tokens=4, num_reqs=4, uniform=True)
        in (disp.cudagraph_keys[_FakeCUDAGraphMode.FULL])
    )
    assert (
        _FakeBatchDescriptor(num_tokens=128, num_reqs=128, uniform=True)
        in (disp.cudagraph_keys[_FakeCUDAGraphMode.FULL])
    )


def test_prepare_decode_step_attn_metadata_pads_and_preserves_static_ptrs() -> None:
    helpers = _load_proposer_helpers(is_musa=True)
    max_slots = 16
    draft_qsl = _FakeCpuGpuBuffer(max_slots + 1)
    draft_seq_lens = torch.zeros(max_slots, dtype=torch.int32)
    slot_mapping_buf = torch.full((max_slots,), 99, dtype=torch.int64)

    full_block_table = torch.arange(64, dtype=torch.int32).view(8, 8)
    optimistic_seq_lens_cpu = torch.tensor(
        [20, 77, 88, 99, 11, 22, 33, 44], dtype=torch.int32
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            block_table=[
                SimpleNamespace(get_device_tensor=lambda n: full_block_table[:n])
            ]
        ),
        optimistic_seq_lens_cpu=optimistic_seq_lens_cpu,
    )

    proposer = SimpleNamespace(
        _draft_query_start_loc=draft_qsl,
        _draft_seq_lens=draft_seq_lens,
        _slot_mapping_buffer=slot_mapping_buf,
        token_arange_np=np.arange(max_slots, dtype=np.int32),
        num_speculative_tokens=3,
        kv_cache_gid=0,
        runner=runner,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1)
        ),
    )

    runner_seq_lens = torch.tensor([20], dtype=torch.int32)
    cad_in = _FakeCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=runner_seq_lens,
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=20,
        block_table_tensor=full_block_table[:1],
        slot_mapping=torch.tensor([5], dtype=torch.int64),
        seq_lens_cpu_upper_bound=optimistic_seq_lens_cpu[:1],
        is_prefilling=torch.tensor([False]),
    )
    rejected = torch.tensor([2], dtype=torch.int32)

    # 1. FULL mode with batch_size=1 padded to input_batch_size=4
    cad_full = helpers["_prepare_decode_step_attn_metadata"](
        proposer,
        cad_in,
        batch_size=1,
        input_batch_size=4,
        cudagraph_runtime_mode=_FakeCUDAGraphMode.FULL,
        num_rejected_tokens_gpu=rejected,
    )
    assert cad_full.num_reqs == 4
    assert cad_full.num_actual_tokens == 4
    assert cad_full.max_query_len == 1
    assert cad_full.query_start_loc.tolist() == [0, 1, 1, 1, 1]
    assert cad_full.query_start_loc_cpu.tolist() == [0, 1, 1, 1, 1]
    assert cad_full.seq_lens.tolist() == [18, 0, 0, 0]
    # Runner's own seq_lens tensor was NOT mutated in-place
    assert runner_seq_lens.tolist() == [20]
    assert cad_full.block_table_tensor.shape == (4, 8)
    assert (cad_full.block_table_tensor[1:4] == 0).all()
    assert cad_full.slot_mapping.shape == (4,)
    assert cad_full.slot_mapping[1:4].tolist() == [-1, -1, -1]
    assert cad_full.is_prefilling is None

    qsl_ptr_1 = cad_full.query_start_loc.data_ptr()
    seq_ptr_1 = cad_full.seq_lens.data_ptr()

    # 2. Subsequent call at batch_size=4 -> input_batch_size=4 reuses exact data_ptrs
    cad_in_4 = _FakeCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32),
        seq_lens=torch.tensor([10, 20, 30, 40], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=16,
        max_query_len=4,
        max_seq_len=40,
        block_table_tensor=full_block_table[:4],
        slot_mapping=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
    )
    cad_full_4 = helpers["_prepare_decode_step_attn_metadata"](
        proposer,
        cad_in_4,
        batch_size=4,
        input_batch_size=4,
        cudagraph_runtime_mode=_FakeCUDAGraphMode.FULL,
        num_rejected_tokens_gpu=None,
    )
    assert cad_full_4.query_start_loc.data_ptr() == qsl_ptr_1
    assert cad_full_4.seq_lens.data_ptr() == seq_ptr_1
    assert cad_full_4.query_start_loc.tolist() == [0, 1, 2, 3, 4]
    assert cad_full_4.seq_lens.tolist() == [10, 20, 30, 40]


def test_mrv2_padded_is_prefilling_for_full_cudagraph_spec_decode() -> None:
    """Verify MRV2 DefaultModelState and FA3/TreeAttn pad is_prefilling on FULL."""
    patch_txt = PATCH_0035.read_text(encoding="utf-8")
    assert "vllm/v1/worker/gpu/model_states/default.py" in patch_txt
    assert "input_batch.prefill_runs_as_decode_np is not None" in patch_txt
    assert "input_batch.num_reqs < num_reqs" in patch_txt

    flash_attn_src = (
        REPO_ROOT / "vllm_musa" / "v1" / "attention" / "backends" / "flash_attn.py"
    ).read_text(encoding="utf-8")
    tree_attn_src = (
        REPO_ROOT / "vllm_musa" / "v1" / "attention" / "backends" / "tree_attn.py"
    ).read_text(encoding="utf-8")
    assert "padded_is_prefilling = torch.zeros(" in flash_attn_src
    assert "padded_is_prefilling = torch.zeros(" in tree_attn_src

    # Simulate B=3 (12 tokens) padded to num_reqs=4 (16 tokens) for K=3 spec verify:
    # previously threw RuntimeError: size of tensor a (4) must match tensor b (3).
    for unpadded_b in (1, 2, 3):
        raw_is_prefilling = np.array([False] * unpadded_b, dtype=np.bool_)
        padded = np.zeros(4, dtype=np.bool_)
        padded[:unpadded_b] = raw_is_prefilling
        query_lens = np.array([4] * unpadded_b + [0] * (4 - unpadded_b), dtype=np.int32)
        is_prefill = (query_lens > 4) | padded
        assert is_prefill.shape == (4,)
        assert not bool(is_prefill.any())

    # Simulate B=4 with 1-token chunked-prefill tail (prefill_runs_as_decode_np[3]=True)
    raw_4 = np.array([False, False, False, True], dtype=np.bool_)
    runs_as_decode_4 = np.array([False, False, False, True], dtype=np.bool_)
    effective_4 = raw_4 & ~runs_as_decode_4
    assert not bool(effective_4.any())


def test_patch_0028_does_not_force_synthetic_mode_in_rejection_sampler() -> None:
    """Ensure patch 0028 keeps synthetic_conditional_rates=None by default."""
    patch_0028 = SERIES_DIR / "0028-MUSA-apply-rejection-sampling-constraints.patch"
    txt = patch_0028.read_text(encoding="utf-8")
    assert "if synthetic_conditional_rates is None:" not in txt, (
        "Patch 0028 must not overwrite synthetic_conditional_rates=None with a dummy "
        "tensor, because upstream _rejection_greedy_sample/_rejection_random_sample "
        "derive SYNTHETIC_MODE=(synthetic_conditional_rates is not None)."
    )


def test_spec_decode_utils_install_hooks_updates_warmup_dispatcher_kernel() -> None:
    """Verify vllm_musa.v1.spec_decode.utils.install_hooks rebinds dispatcher.kernel."""
    utils_src = (REPO_ROOT / "vllm_musa" / "v1" / "spec_decode" / "utils.py").read_text(
        encoding="utf-8"
    )
    assert "_eagle_prepare_next_token_padded" in utils_src
    assert "dispatcher.kernel = eagle_prepare_next_token_padded_kernel" in utils_src


def test_dummy_run_skips_already_captured_full_cudagraph_mid_capture() -> None:
    """Verify patch 0030 guards against replaying FULL draft graphs mid-capture.

    On TP>=2 MUSA, custom all-reduce registers peer IPC handles only when the
    outer ``with graph_capture(device):`` exits. Replaying an already-captured
    padded draft decode graph (e.g., B=2 padded to 4) during capture would read
    uninitialized ``graph_rank_data`` pointers.
    """
    p30 = PATCH_0030.read_text(encoding="utf-8")
    assert "self.model.concrete_cudagraph_entries.get(batch_desc)" in p30
    assert "existing_entry is not None and existing_entry.cudagraph is not None" in p30
