# SPDX-License-Identifier: Apache-2.0
"""Exercise MRv2 PPHandler packed Gloo CPU ring, GIL-free event sync, and shutdown."""

import ast
import ctypes
import queue
import threading
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
PP_UTILS = ROOT / "third_party/vllm/vllm/v1/worker/gpu/pp_utils.py"
MODEL_RUNNER = ROOT / "third_party/vllm/vllm/v1/worker/gpu/model_runner.py"


class _Arr(list):
    def _bin(self, o, fn):
        rhs = o if isinstance(o, list) else [o] * len(self)
        return _Arr(fn(a, b) for a, b in zip(self, rhs))

    __add__ = lambda s, o: s._bin(o, lambda a, b: a + b)  # noqa: E731
    __ge__ = lambda s, o: s._bin(o, lambda a, b: a >= b)  # noqa: E731
    __ne__ = lambda s, o: s._bin(o, lambda a, b: a != b)  # noqa: E731
    __and__ = lambda s, o: s._bin(o, lambda a, b: a and b)  # noqa: E731
    __or__ = lambda s, o: s._bin(o, lambda a, b: a or b)  # noqa: E731
    __invert__ = lambda s: _Arr(not v for v in s)  # noqa: E731
    any = lambda s: any(s)  # noqa: E731
    all = lambda s: all(s)  # noqa: E731

    def __getitem__(self, idx):
        if isinstance(idx, _Arr):
            if idx and isinstance(idx[0], bool):
                return _Arr(v for v, k in zip(self, idx) if k)
            return _Arr(self[i] for i in idx)
        return super().__getitem__(idx)


class _Buf:
    def __init__(
        self, flat, shape, dtype="int64", device="cpu", pin_memory=False, off=0
    ):
        self._s, self._o, self.shape = flat, off, shape
        self.dtype, self.device, self.pin_memory = dtype, device, pin_memory

    def numel(self):
        return self.shape[0] * (self.shape[1] if len(self.shape) == 2 else 1)

    def _vals(self):
        return self._s[self._o : self._o + self.numel()]

    @property
    def data(self):
        v = self._vals()
        if len(self.shape) == 2:
            c = self.shape[1]
            return [v[r * c : (r + 1) * c] for r in range(self.shape[0])]
        return v

    def contiguous(self):
        return self

    def record_stream(self, _):
        pass

    def reshape(self, _):
        return _Buf(
            self._s, (self.numel(),), self.dtype, self.device, self.pin_memory, self._o
        )

    def copy_(self, other, non_blocking=False):
        v = other._vals()
        self._s[self._o : self._o + len(v)] = v
        return self

    def view(self, *args):
        v = self._vals()
        if args == ("int32",):
            return _Buf(
                [w for x in v for w in (x & 0xFFFFFFFF, x >> 32)],
                (len(v) * 2,),
                "int32",
                self.device,
            )
        if args == ("int64",):
            return _Buf(
                [(v[i + 1] << 32) | (v[i] & 0xFFFFFFFF) for i in range(0, len(v), 2)],
                (len(v) // 2,),
                "int64",
                self.device,
            )
        return _Buf(
            self._s, tuple(args), self.dtype, self.device, self.pin_memory, self._o
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            a, b, _ = idx.indices(self.shape[0])
            step = self.shape[1] if len(self.shape) == 2 else 1
            sh = (b - a, self.shape[1]) if len(self.shape) == 2 else (b - a,)
            return _Buf(
                self._s,
                sh,
                self.dtype,
                self.device,
                self.pin_memory,
                self._o + a * step,
            )
        rows = (
            [r for r, k in zip(self.data, idx.data) if k]
            if idx.dtype == "bool"
            else [self.data[i] for i in idx.data]
        )
        return _Buf(
            [x for r in rows for x in r],
            (len(rows), len(rows[0])),
            self.dtype,
            self.device,
        )

    def __setitem__(self, idx, val):
        c = self.shape[1]
        for p, dst in enumerate(idx.data):
            self._s[self._o + dst * c : self._o + (dst + 1) * c] = val.data[p]


def _make_tensor(rows, dtype="int64", device="musa:0"):
    if rows and isinstance(rows[0], list):
        return _Buf(
            [x for r in rows for x in r], (len(rows), len(rows[0])), dtype, device
        )
    return _Buf(list(rows), (len(rows),), dtype, device)


def _build_env(
    *, is_musa=True, world_size=2, rank=0, group_ranks=None, sync_fn=None, ev_fn=None
):
    group_ranks = group_ranks or [list(range(world_size))]
    my = next(g for g in group_ranks if rank in g)
    pp_group = NS(
        world_size=len(my),
        rank=rank,
        last_rank=my[-1],
        is_last_rank=(rank == my[-1]),
        group_ranks=group_ranks,
        make_sibling_device_group=MagicMock(return_value="dev_pg"),
    )
    wire, groups, waited = queue.SimpleQueue(), [], []

    def new_group(ranks, backend=None, group_desc=None, timeout=None):
        g = ("gloo", tuple(ranks), backend, group_desc, timeout)
        groups.append(g)
        return g

    def bcast(t, src=None, group=None, async_op=False):
        if async_op:
            return NS(
                wait=lambda: (t.copy_(wire.get(timeout=5.0)), waited.append(t.shape))
            )
        if pp_group.is_last_rank:
            wire.put(_Buf(list(t._vals()), t.shape, t.dtype, t.device))

    def empty(*sh, dtype="int64", device="cpu", pin_memory=False):
        sh = sh[0] if len(sh) == 1 and isinstance(sh[0], tuple) else sh
        n = sh[0] * (sh[1] if len(sh) == 2 else 1)
        return _Buf([0] * n, sh, dtype, device, pin_memory)

    ev_fn = ev_fn or (lambda: NS(musa_event=0xAA00, synchronize=MagicMock()))
    torch_mod = NS(
        int64="int64",
        int32="int32",
        empty=empty,
        cat=lambda ts: _Buf(
            [x for t in ts for x in t._vals()],
            (sum(t.numel() for t in ts),),
            ts[0].dtype,
            ts[0].device,
        ),
        stack=lambda ts, dim=0: _make_tensor(
            [t.data for t in ts], ts[0].dtype, ts[0].device
        ),
        as_tensor=lambda a, device=None: _make_tensor(list(a), "bool", device),
        nn=NS(
            functional=NS(
                pad=lambda t, p: _make_tensor(
                    [r + [0] * p[1] for r in t.data], t.dtype, t.device
                )
            )
        ),
        cuda=NS(
            current_stream=lambda d=None: NS(
                wait_event=MagicMock(), synchronize=MagicMock()
            ),
            Stream=lambda d=None: NS(
                wait_stream=MagicMock(), record_event=MagicMock(side_effect=ev_fn)
            ),
            stream=lambda _: nullcontext(),
            set_device=MagicMock(),
        ),
        distributed=NS(new_group=new_group, broadcast=MagicMock(side_effect=bcast)),
    )
    np_mod = NS(
        array=_Arr,
        zeros=lambda n, dtype: _Arr([0] * n),
        int32="int32",
        where=lambda m, x, y: _Arr(x if f else v for f, v in zip(m, y)),
    )
    tree = ast.parse(PP_UTILS.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    ns = dict(
        ctypes=ctypes,
        dataclass=dataclass,
        deque=deque,
        queue=queue,
        threading=threading,
        np=np_mod,
        envs=NS(VLLM_XPU_PP_MICROBATCH=True),
        logger=MagicMock(),
        torch=torch_mod,
        get_pp_group=lambda: pp_group,
        get_cpu_distributed_timeout_or_none=lambda: 60,
        current_platform=NS(
            is_musa=lambda: is_musa,
            is_xpu=lambda: False,
            is_pin_memory_available=lambda: True,
        ),
        async_tensor_h2d=lambda d, device=None: _Buf(
            list(d._vals()), d.shape, d.dtype, device
        )
        if isinstance(d, _Buf)
        else _make_tensor(list(d), "int32", device),
    )
    exec("from __future__ import annotations\n" + ast.unparse(tree), ns)
    ns["_load_musart_event_sync"] = lambda: sync_fn
    return ns["PPHandler"], np_mod, pp_group, torch_mod, wire, groups, waited


def _batch(np_mod, idxs=(0, 1)):
    n = len(idxs)
    return NS(
        num_reqs=n,
        idx_mapping=_make_tensor(list(idxs), "int32"),
        idx_mapping_np=np_mod.array(idxs),
        num_computed_tokens_np=np_mod.array([1] * n),
        prefill_len_np=np_mod.array([1] * n),
        num_scheduled_tokens=1,
        max_seq_len_np=None,
        is_prefilling_np=np_mod.array([False] * n),
    )


def test_musa_sibling_gloo_group_vs_non_musa_device_group():
    cls_m, _, pp_m, _, _, groups_m, _ = _build_env(
        is_musa=True, rank=2, group_ranks=[[0, 2, 4], [1, 3, 5]]
    )
    h_m = cls_m(4, 0, "musa:0")
    try:
        pp_m.make_sibling_device_group.assert_not_called()
        assert h_m.broadcast_group == ("gloo", (0, 2, 4), "gloo", "pp_broadcast", 60)
        assert len(groups_m) == 2 and h_m._cpu_ring[0].pin_memory
    finally:
        h_m.shutdown()

    cls_c, _, pp_c, _, _, groups_c, _ = _build_env(is_musa=False)
    h_c = cls_c(4, 0, "cuda:0")
    pp_c.make_sibling_device_group.assert_called_once_with(group_desc="pp_broadcast")
    assert groups_c == [] and h_c._cpu_ring == []


def test_pp3_packed_gloo_broadcast_spec_drafts_and_gil_free_sync():
    ptrs, py_sync = [], MagicMock()
    s_cls, np_s, _, torch_s, wire, _, _ = _build_env(
        world_size=3,
        rank=2,
        sync_fn=lambda p: (ptrs.append(p.value) or 0),
        ev_fn=lambda: NS(musa_event=0x7788, synchronize=py_sync),
    )
    r_cls, np_r, _, torch_r, _, _, waited = _build_env(world_size=3, rank=1)
    torch_r.distributed.broadcast.side_effect = lambda t, **kw: NS(
        wait=lambda: (t.copy_(wire.get(timeout=5.0)), waited.append(t.shape))
    )
    sender, receiver = s_cls(4, 2, "musa:2"), r_cls(4, 2, "musa:1")
    try:
        assert receiver.get_prev_sampled_outputs() is None
        assert receiver.receive(_batch(np_r))
        sender.broadcast(
            _make_tensor([[11, 12], [21, 22]]),
            _make_tensor([2, 1], "int32"),
            _make_tensor([0, 1], "int32"),
            _batch(np_s),
        )
        sender.broadcast_drafts(
            _make_tensor([[101, 102], [201, 202], [9, 9], [8, 8]]), _batch(np_s)
        )
        assert receiver.get_prev_sampled_outputs() is None
        assert receiver.get_prev_sampled_outputs() is None

        receiver.on_req_idx_freed(0)
        draft_tbl = _make_tensor([[0, 0]] * 4, device="musa:1")
        out = receiver.get_prev_sampled_outputs(draft_tokens_to_update=draft_tbl)
        assert waited == [(16,), (2, 2)]
        assert out["sampled_tokens"].data == [[11, 12, 0], [21, 22, 0]]
        assert out["num_sampled"].data == [2, 1] and out["num_rejected"].data == [0, 1]
        assert out["idx_mapping"].data == [-1, 1]
        assert draft_tbl.data[:2] == [[0, 0], [201, 202]]
        assert ptrs == [0x7788, 0x7788]
        py_sync.assert_not_called()
        torch_s.cuda.set_device.assert_called_once_with("musa:2")
    finally:
        sender.shutdown()
        receiver.shutdown()


def test_event_sync_delegation_errors_and_shutdown_drain():
    calls = []
    s_cls, np_s, _, _, wire, _, _ = _build_env(world_size=2, rank=1)
    r_cls, np_r, _, torch_r, _, _, waited = _build_env(
        world_size=2,
        rank=0,
        sync_fn=lambda p: (calls.append(p.value) or 17),
    )
    torch_r.distributed.broadcast.side_effect = lambda t, **kw: NS(
        wait=lambda: (t.copy_(wire.get(timeout=5.0)), waited.append(t.shape))
    )
    sender, receiver = s_cls(4, 0, "musa:1"), r_cls(4, 0, "musa:0")
    try:
        gil_ok = MagicMock()
        gil_ok._torchada_gil_free = True
        receiver._wait_event_gil_free(NS(musa_event=0x11, synchronize=gil_ok))
        gil_ok.assert_called_once_with()
        assert calls == []

        with pytest.raises(
            RuntimeError, match="musaEventSynchronize failed with error 17"
        ):
            receiver._wait_event_gil_free(NS(musa_event=0x22, synchronize=MagicMock()))
        assert calls == [0x22]

        for tok in (7, 8):
            assert receiver.get_prev_sampled_outputs() is None
            receiver.receive(_batch(np_r, (0,)))
            sender.broadcast(
                _make_tensor([[tok]]),
                _make_tensor([1], "int32"),
                _make_tensor([0], "int32"),
                _batch(np_s, (0,)),
            )
            if tok == 7:
                receiver.on_req_idx_freed(0)
        assert receiver.get_prev_sampled_outputs() is None
        assert waited == [(4,)]
        sender.shutdown()
        receiver.shutdown()
        assert waited == [(4,), (4,)]
    finally:
        sender.shutdown()
        receiver.shutdown()


def test_model_runner_shutdown_calls_pp_handler_shutdown():
    cls = next(
        n
        for n in ast.parse(MODEL_RUNNER.read_text()).body
        if getattr(n, "name", None) == "GPUModelRunner"
    )
    fn = next(n for n in cls.body if getattr(n, "name", None) == "shutdown")
    events, scope = (
        [],
        dict(
            torch=NS(
                accelerator=NS(
                    synchronize=lambda: events.append("sync"), empty_cache=lambda: None
                )
            ),
            set_offloader=lambda _: None,
            free_before_shutdown=lambda _: None,
            clear_layer_kv_caches=lambda _: None,
            gc=NS(collect=lambda: None),
            logger=MagicMock(),
        ),
    )
    exec("from __future__ import annotations\n" + ast.unparse(fn), scope)
    scope["shutdown"](
        NS(
            pp_handler=NS(shutdown=lambda: events.append("pp")),
            aux_output_connector=None,
            vllm_config=object(),
        )
    )
    assert events == ["pp", "sync"]
