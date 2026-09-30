"""MRV2 decode-graph eligibility for a padded final prompt token."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

model_runner = pytest.importorskip("vllm.v1.worker.gpu.model_runner")
GPUModelRunner = model_runner.GPUModelRunner

DECODE_QUERY_LEN = 5


def _runner(req_states: dict[str, tuple[int, int]], enabled: bool) -> Any:
    """Stub from {req_id: (num_computed_tokens, prefill_len)}."""
    prefill_lens = np.array([s[1] for s in req_states.values()], dtype=np.int32)
    num_computed = np.array([s[0] for s in req_states.values()], dtype=np.int32)
    runner: Any = GPUModelRunner.__new__(GPUModelRunner)
    runner.decode_query_len = DECODE_QUERY_LEN
    runner.adaptive_verification = None
    runner._final_prompt_token_decode_graph = enabled
    runner.req_states = SimpleNamespace(
        req_id_to_index={req_id: i for i, req_id in enumerate(req_states)},
        num_computed_prefill_tokens=np.minimum(num_computed, prefill_lens),
        prefill_len=SimpleNamespace(np=prefill_lens),
    )
    return runner


def _gather(runner: Any, drafts: dict[str, list[int]], query_len: int = 5):
    num_scheduled = {req_id: query_len for req_id in runner.req_states.req_id_to_index}
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens=drafts,
    )
    return runner.gather_batch_req_state(scheduler_output, False)


def _batch(new_req_computed: int, prefill_len: int = 32000):
    states = {f"d{i}": (prefill_len + 40, prefill_len) for i in range(4)}
    states["new"] = (new_req_computed, prefill_len)
    drafts = {req_id: [7, 8, 9, 10] for req_id in states}
    drafts["new"] = [-1, -1, -1, -1]
    return states, drafts


def test_padded_final_prompt_token_replays_decode_graph() -> None:
    states, drafts = _batch(new_req_computed=31999)
    state, uniform = _gather(_runner(states, enabled=True), drafts)

    assert uniform == DECODE_QUERY_LEN
    # The prompt token must still be written by prepare_prefill_inputs.
    assert state.has_prefill
    assert state.is_prefilling_np.tolist() == [False] * 4 + [True]


def test_padded_final_prompt_token_stays_eager_without_contract() -> None:
    states, drafts = _batch(new_req_computed=31999)
    _, uniform = _gather(_runner(states, enabled=False), drafts)

    assert uniform is None


def test_prompt_chunk_of_verify_length_stays_eager() -> None:
    # Five real prompt tokens remain: a genuine chunk, not a padded final token.
    states, drafts = _batch(new_req_computed=31995)
    _, uniform = _gather(_runner(states, enabled=True), drafts)

    assert uniform is None


def test_final_prompt_token_without_draft_slots_stays_eager() -> None:
    states, drafts = _batch(new_req_computed=31999)
    del drafts["new"]
    _, uniform = _gather(_runner(states, enabled=True), drafts)

    assert uniform is None
