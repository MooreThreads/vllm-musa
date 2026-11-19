# SPDX-License-Identifier: Apache-2.0
"""CPU admission/state tests against the native NGram processor.

These execute the actual admission methods extracted from the module. They do
not substitute for importing the real V2 loader or running the Triton kernel.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "vllm_musa/v1/worker/gpu/sample/no_repeat_ngram.py"


@pytest.fixture(scope="module")
def source():
    return MODULE.read_text()


@pytest.fixture(scope='module')
def admission_class(source):
    tree = ast.parse(source)
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NoRepeatNGramState')
    # The real methods execute with mutable CPU slots. GPU buffer construction,
    # NumPy dtype behavior, and kernel launch are outside this CPU-only evidence.
    cls = ast.ClassDef(
        name=original.name,
        bases=[],
        keywords=[],
        body=[n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in ('validate_params', 'add_request')],
        decorator_list=[],
    )
    program = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {'SamplingParams': SimpleNamespace, 'Iterable': Iterable, 'MAX_WHITELIST_TOKEN_IDS': 1024}
    exec(compile(program, str(MODULE), 'exec'), namespace)
    return namespace['NoRepeatNGramState']


@pytest.mark.parametrize('extra_args', [
    None, {}, {'ngram_size': None}, {'ngram_size': 1},
    {'ngram_size': 3, 'window_size': 90, 'whitelist_token_ids': [3, 3, 7]},
    {'no_repeat_ngram_size': 1},
])
def test_frontend_accepts_declared_parameters(admission_class, extra_args):
    admission_class.validate_params(SimpleNamespace(extra_args=extra_args))


@pytest.mark.parametrize('extra_args', [
    {'ngram_size': 0}, {'ngram_size': -1}, {'ngram_size': True},
    {'ngram_size': 2.0}, {'ngram_size': '2'},
    {'ngram_size': 2, 'no_repeat_ngram_size': 2},
    {'ngram_size': 2, 'window_size': 0},
    {'ngram_size': 2, 'window_size': True},
    {'ngram_size': 2, 'whitelist_token_ids': 'abc'},
    {'ngram_size': 2, 'whitelist_token_ids': [False]},
    {'ngram_size': 2, 'whitelist_token_ids': list(range(1025))},
])
def test_frontend_rejects_declared_invalid_parameters(admission_class, extra_args):
    with pytest.raises(ValueError):
        admission_class.validate_params(SimpleNamespace(extra_args=extra_args))


@pytest.fixture
def instance(admission_class):
    state = admission_class()
    state.max_model_len = 128
    state.req_states = SimpleNamespace(vocab_size=64)
    state.ngram_sizes = SimpleNamespace(np=[0, 0])
    state.window_sizes = SimpleNamespace(np=[0, 0])
    state.whitelist_lens = SimpleNamespace(np=[0, 0])
    state.staged_whitelist = []
    state.whitelist_token_ids = SimpleNamespace(
        stage_write=lambda *args: state.staged_whitelist.append(args)
    )
    return state


def test_slot_reuse_clears_previous_configuration(instance):
    assert instance.add_request(1, SimpleNamespace(extra_args={'ngram_size': 3, 'whitelist_token_ids': [5, 5, 2]}))
    assert instance.staged_whitelist == [(1, 0, [2, 5])]
    assert instance.whitelist_lens.np[1] == 2
    assert not instance.add_request(1, SimpleNamespace(extra_args=None))
    assert instance.ngram_sizes.np[1] == instance.window_sizes.np[1] == instance.whitelist_lens.np[1] == 0


def test_legacy_and_canonical_unigram_have_deliberately_different_meaning(instance):
    assert not instance.add_request(0, SimpleNamespace(extra_args={'ngram_size': 1}))
    assert instance.add_request(0, SimpleNamespace(extra_args={'no_repeat_ngram_size': 1}))
    assert instance.ngram_sizes.np[0] == 1


@pytest.mark.parametrize('extra_args', [
    {'ngram_size': 129},
    {'ngram_size': 2, 'whitelist_token_ids': [-1]},
    {'ngram_size': 2, 'whitelist_token_ids': [64]},
])
def test_worker_checks_model_dependent_boundaries(instance, extra_args):
    with pytest.raises(ValueError):
        instance.add_request(0, SimpleNamespace(extra_args=extra_args))
