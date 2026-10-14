"""CPU state-machine checks of actual BlockTable methods, not GPU acceptance."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(os.environ.get(
    "MUSA_BLOCK_TABLE_SOURCE",
    str(Path(__file__).resolve().parents[1] / "third_party/vllm/vllm/v1/worker/block_table.py"),
))
if not SOURCE.is_file():
    pytest.skip(
        "vllm/v1/worker/block_table.py not found at "
        f"{SOURCE}; set MUSA_BLOCK_TABLE_SOURCE to point at it",
        allow_module_level=True,
    )

METHODS = {"append_row", "add_row", "clear_row", "move_row", "swap_row",
           "commit_block_table", "clear"}


class FakeTensor:
    def __init__(self, array):
        self.array = array

    def fill_(self, value):
        self.array.fill(value)


def make_table(mamba=True):
    source = ast.parse(SOURCE.read_text())
    klass = next(x for x in source.body if isinstance(x, ast.ClassDef) and x.name == "BlockTable")
    klass.body = [m for m in klass.body if isinstance(m, ast.FunctionDef) and m.name in METHODS]
    namespace = {"np": np, "SlotMappingMode": SimpleNamespace(NONE=0)}
    exec(compile("from __future__ import annotations\n" + ast.unparse(klass), str(SOURCE), "exec"), namespace)
    table = namespace["BlockTable"]()
    table.max_num_reqs = 8
    table.num_blocks_per_row = np.zeros(8, dtype=np.int32)
    table.use_hybrid_blocks = False
    table.slot_mapping_mode = 0 if mamba else 1
    table._mamba_table_dirty = mamba
    table._mamba_uploaded_rows = 0
    cpu = np.zeros((8, 32), dtype=np.int32)
    gpu = np.full_like(cpu, -1)
    uploads = []

    def upload(n):
        uploads.append(n)
        gpu[:n] = cpu[:n]

    table.block_table = SimpleNamespace(np=cpu, cpu=FakeTensor(cpu), gpu=FakeTensor(gpu), copy_to_gpu=upload)
    return table, cpu, gpu, uploads


def test_clean_mamba_skips_but_attention_keeps_uploading():
    for mamba, count in [(True, 1), (False, 3)]:
        t, cpu, gpu, uploads = make_table(mamba)
        t.add_row([7, 8], 0)
        for _ in range(3):
            t.commit_block_table(1)
        assert len(uploads) == count
        np.testing.assert_array_equal(gpu[:1], cpu[:1])


def test_partial_commit_must_not_lose_dirty_tail_on_batch_growth():
    t, cpu, gpu, uploads = make_table()
    t.add_row([101], 6)
    t.commit_block_table(2)
    t.commit_block_table(8)
    np.testing.assert_array_equal(gpu, cpu)
    assert uploads == [2, 8]
    t.add_row([202], 7)
    t.commit_block_table(1)
    t.commit_block_table(8)
    np.testing.assert_array_equal(gpu, cpu)
    assert uploads == [2, 8, 1, 8]


@pytest.mark.parametrize("operation", ["add", "append", "clear_row", "move", "swap", "clear"])
def test_mutations_reach_device(operation):
    t, cpu, gpu, _ = make_table()
    t.add_row([11, 12], 0)
    t.add_row([21, 22], 1)
    t.commit_block_table(8)
    {"add": lambda: t.add_row([31], 0),
     "append": lambda: t.append_row([32], 0),
     "clear_row": lambda: t.clear_row(0),
     "move": lambda: t.move_row(0, 4),
     "swap": lambda: t.swap_row(0, 1),
     "clear": t.clear}[operation]()
    t.commit_block_table(8)
    np.testing.assert_array_equal(gpu, cpu)


def test_randomized_mutations_and_partial_commits():
    rng = np.random.default_rng(100020)
    t, cpu, gpu, _ = make_table()
    for _ in range(2000):
        row, other = rng.choice(8, 2, replace=False)
        op = rng.integers(6)
        if op == 0:
            t.clear_row(row)
            t.add_row(rng.integers(1, 999, size=rng.integers(0, 12)).tolist(), row)
        elif op == 1 and t.num_blocks_per_row[row] < 30:
            t.append_row([int(rng.integers(1, 999))], row)
        elif op == 2:
            t.move_row(row, other)
        elif op == 3:
            t.swap_row(row, other)
        elif op == 4:
            t.clear_row(row)
        n = int(rng.integers(0, 9))
        t.commit_block_table(n)
        np.testing.assert_array_equal(gpu[:n], cpu[:n])
    t.commit_block_table(8)
    np.testing.assert_array_equal(gpu, cpu)
