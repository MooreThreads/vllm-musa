"""Pytest-only PP transport contracts execute candidate AST and upstream helpers."""

import ast
import subprocess
import sys
import types
from itertools import islice
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/vllm/vllm/model_executor/models/deepseek_v2.py"
UTILS = ROOT / "third_party/vllm/vllm/model_executor/models/utils.py"
PIN = next(
    line.split("=", 1)[1]
    for line in (ROOT / "third_party/PINS").read_text().splitlines()
    if line.startswith("VLLM_COMMIT=")
)


def extract(path, names, cls=None):
    tree = ast.parse(path.read_text())
    body = (
        tree.body
        if cls is None
        else next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    )
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    return ast.parse(
        "from __future__ import annotations\n"
        + "\n".join(ast.unparse(n) for n in nodes)
    )


class RowBuffer:
    """Token-row transport spy: slices alias storage; only copy_ mutates it."""

    def __init__(self, shape, dtype="float32", *, storage=None, start=0, events=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.storage = (
            storage
            if storage is not None
            else [[0] * shape[1] for _ in range(shape[0])]
        )
        self.start = start
        self.events = events if events is not None else []

    def __getitem__(self, rows):
        assert isinstance(rows, slice) and rows.step is None
        start, stop, _ = rows.indices(self.shape[0])
        return RowBuffer(
            (stop - start, self.shape[1]),
            self.dtype,
            storage=self.storage,
            start=self.start + start,
            events=self.events,
        )

    def copy_(self, source):
        assert self.shape == source.shape
        self.events.append(("copy", self.shape[0]))
        values = [
            row[:]
            for row in source.storage[source.start : source.start + source.shape[0]]
        ]
        self.storage[self.start : self.start + self.shape[0]] = values
        return self

    def new_empty(self, *shape):
        return RowBuffer(shape, self.dtype)


def zeros(shape, *, dtype, device):
    assert device == "cpu"
    return RowBuffer(shape, dtype)


@pytest.fixture
def env(monkeypatch):
    if not SOURCE.is_file() or not UTILS.is_file():
        pytest.skip("requires applied third_party/vllm tree")
    group = NS(world_size=2, is_first_rank=True, is_last_rank=False)
    state = types.ModuleType("vllm.distributed.parallel_state")
    state.get_pp_group = lambda: group
    distutils = types.ModuleType("vllm.distributed.utils")
    # Mock partition service only; sharing algorithms are actual upstream code.
    distutils.get_pp_indices = lambda total, rank, size: (group.split, total)
    monkeypatch.setitem(sys.modules, state.__name__, state)
    monkeypatch.setitem(sys.modules, distutils.__name__, distutils)
    ns = dict(
        torch=NS(zeros=zeros),
        IntermediateTensors=dict,
        get_pp_group=lambda: group,
        islice=islice,
    )
    exec(
        compile(
            extract(
                UTILS,
                [
                    "compute_skip_topk_layers",
                    "pp_splits_layer_sharing",
                    "pp_topk_buffers",
                    "copy_pp_buffers",
                    "make_empty_intermediate_tensors_factory",
                ],
            ),
            str(UTILS),
            "exec",
        ),
        ns,
    )
    exec(
        compile(extract(SOURCE, ["forward"], "DeepseekV2Model"), str(SOURCE), "exec"),
        ns,
    )
    return group, ns


def config(dsa=True):
    c = NS(
        num_hidden_layers=78,
        hidden_size=4,
        vocab_size=10,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )
    if dsa:
        c.index_topk = 3
    return c


def candidate_buffers(ns, c, buffer):
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2Model"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    obj = NS(hidden_size=c.hidden_size)
    scope = dict(ns, self=obj, config=c, topk_indices_buffer=buffer)
    for attr in ("_pp_buffers", "make_empty_intermediate_tensors"):
        node = next(
            n
            for n in init.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == attr for t in n.targets)
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"),
            scope,
        )
    return obj


@pytest.mark.parametrize("split,transport", [(38, False), (39, True)])
def test_shared_schema_and_boundary(env, split, transport):
    group, ns = env
    group.split = split
    models = [
        candidate_buffers(ns, config(), RowBuffer((12, 3), "int32")) for _ in range(2)
    ]
    assert [set(m._pp_buffers) for m in models] == [
        {"topk_indices"} if transport else set()
    ] * 2
    for m in models:
        empty = m.make_empty_intermediate_tensors(5, "float32", "cpu")
        assert set(empty) == {"hidden_states", "residual"} | set(m._pp_buffers)
        if transport:
            assert empty["topk_indices"].shape == (5, 3)
            assert empty["topk_indices"].dtype == "int32"


def test_non_dsa_empty(env):
    group, ns = env
    group.split = 39
    m = candidate_buffers(ns, config(False), None)
    assert m._pp_buffers == {}
    assert set(m.make_empty_intermediate_tensors(2, "float32", "cpu")) == {
        "hidden_states",
        "residual",
    }


def test_poisoned_receiver_multiframe_actual_forward(env):
    group, ns = env
    group.split = 39
    tx = candidate_buffers(ns, config(), RowBuffer((12, 3), "int32"))
    rx = candidate_buffers(ns, config(), RowBuffer((12, 3), "int32"))
    for m in (tx, rx):
        m.config = config()
        m.aux_hidden_state_layers = ()
        m.start_layer, m.end_layer = 0, 1
    tx.embed_input_ids = lambda ids: RowBuffer((ids.shape[0], 4))

    def producer(pos, hidden, residual, scaling):
        rows = pos.shape[0]
        tx._pp_buffers["topk_indices"].storage[:rows] = [
            [rows * 100 + row * 3 + col for col in range(3)] for row in range(rows)
        ]
        return hidden, RowBuffer(hidden.shape)

    class Layer:
        use_sequence_parallel_moe = False

        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    tx.layers = [Layer(producer)]
    for rows in (1, 3, 5, 7, 8, 1, 7, 3):
        group.is_first_rank, group.is_last_rank = True, False
        positions = NS(shape=(rows,))
        frame = ns["forward"](tx, positions, positions, None)
        assert frame["topk_indices"].shape == (rows, 3)
        assert frame["topk_indices"].storage is tx._pp_buffers["topk_indices"].storage
        expected = [row[:] for row in frame["topk_indices"].storage[:rows]]
        receiver = rx._pp_buffers["topk_indices"]
        receiver.storage[:] = [[-999] * 3 for _ in range(12)]
        receiver.events.clear()
        consumer_calls = []

        def consumer(pos, hidden, residual, scaling):
            consumer_calls.append(rows)
            assert rx._pp_buffers["topk_indices"] is receiver
            assert receiver.events == [("copy", rows)]
            assert receiver.storage[:rows] == expected
            assert receiver.storage[rows:] == [[-999] * 3 for _ in range(12 - rows)]
            receiver.events.append(("consumer", rows))
            return hidden, residual

        rx.layers = [Layer(consumer)]
        rx.norm = lambda h, r: (h, r)
        group.is_first_rank, group.is_last_rank = False, True
        assert ns["forward"](rx, None, positions, frame).shape == (rows, 4)
        assert consumer_calls == [rows]


def test_attention_actual_skip_assignment_and_preservation(env):
    _, ns = env
    tree = ast.parse(SOURCE.read_text())
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_skip_topk" for t in n.targets)
        and isinstance(n.value, ast.Compare)
    )
    legacy_source = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT / "third_party/vllm"),
            "show",
            f"{PIN}:vllm/model_executor/models/deepseek_v2.py",
        ],
        text=True,
    )
    baseline_tree = ast.parse(legacy_source)
    legacy = next(
        n
        for n in ast.walk(baseline_tree)
        if isinstance(n, ast.If)
        and any(
            isinstance(x, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_index_topk_freq"
                for t in x.targets
            )
            for x in n.body
        )
    )
    for layer in range(80):
        scope = dict(
            ns,
            self=NS(is_v32=True),
            config=config(),
            prefix=str(layer),
            extract_layer_index=int,
            _skip_topk=False,
        )
        exec(
            compile(
                ast.Module(body=[legacy], type_ignores=[]),
                f"{PIN}:deepseek_v2.py",
                "exec",
            ),
            scope,
        )
        actual = layer in ns["compute_skip_topk_layers"](config())
        assert (not scope["_skip_topk"] or scope["is_mtp_layer"]) == (
            not actual or layer >= 78
        )
    for layer in (0, 38, 39, 77, 78):
        scope = dict(ns, config=config(), layer_id=layer)
        exec(
            compile(
                ast.Module(body=[assignment], type_ignores=[]), str(SOURCE), "exec"
            ),
            scope,
        )
        assert scope["_skip_topk"] == (
            layer in ns["compute_skip_topk_layers"](config())
        )
    assert "if self.is_v32 and (not _skip_topk or is_mtp_layer):" in SOURCE.read_text()
