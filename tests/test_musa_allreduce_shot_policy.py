import ast
from pathlib import Path

import pytest


@pytest.fixture
def preferred_shot():
    path = (
        Path(__file__).resolve().parents[1] / "vllm_musa/jit_kernel/csrc/allreduce.py"
    )
    tree = ast.parse(path.read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "preferred_shot"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["preferred_shot"]


@pytest.mark.parametrize("nbytes", [16, 10240, 40960, 65536])
def test_tp8_small_messages_keep_two_shot(preferred_shot, nbytes):
    assert preferred_shot(8, nbytes) == 2


@pytest.mark.parametrize("nbytes", [0, 65552, 1024 * 1024, 16 * 1024 * 1024])
def test_tp8_larger_messages_keep_two_shot(preferred_shot, nbytes):
    assert preferred_shot(8, nbytes) == 2


@pytest.mark.parametrize("world_size", [4, 6])
@pytest.mark.parametrize("nbytes", [16, 65536, 16 * 1024 * 1024])
def test_other_multi_rank_policies_are_unchanged(preferred_shot, world_size, nbytes):
    assert preferred_shot(world_size, nbytes) == 2


@pytest.mark.parametrize(
    "nbytes,expected",
    [(10240, 2), (65536, 2), (16 * 1024 * 1024 - 16, 2), (16 * 1024 * 1024, 1)],
)
def test_two_rank_threshold_is_unchanged(preferred_shot, nbytes, expected):
    assert preferred_shot(2, nbytes) == expected
