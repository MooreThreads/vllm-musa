# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: I001

from types import SimpleNamespace

# torchada must patch CUDA-facing symbols before torch is imported.
# isort: off
import numpy as np
import torchada  # noqa: F401
import torch

# isort: on

from tests.qwen_runtime_plan_test_utils import qwen_sampler
from vllm_musa.v1.worker import qwen_fused_next_input_ids as fused_inputs


def make_runner(capacity: int = 8, *, is_qwen_family: bool = True):
    return SimpleNamespace(
        input_buffers=SimpleNamespace(
            input_ids=torch.zeros(capacity, dtype=torch.int32)
        ),
        model_state=SimpleNamespace(num_new_sampled_tokens_per_step=1),
        sampler=qwen_sampler(enabled=is_qwen_family),
        use_pp=False,
    )


def call_selector(runner, batch_size: int = 4, **overrides):
    arguments = {
        "req_ids": [f"req-{index}" for index in range(batch_size)],
        "num_scheduled_tokens": np.ones(batch_size, dtype=np.int32),
        "is_prefilling_np": np.zeros(batch_size, dtype=np.bool_),
        "total_num_draft_tokens": 0,
        "total_num_logits": batch_size,
        "num_tokens": batch_size,
        "num_tokens_after_padding": batch_size,
        "num_reqs_after_padding": batch_size,
    }
    arguments.update(overrides)
    return fused_inputs.select_qwen_fused_decode_inputs(runner, **arguments)


def install_test_gates(monkeypatch) -> None:
    monkeypatch.setattr(fused_inputs.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(fused_inputs, "_is_musa_tensor", lambda tensor: True)


def test_qwen_fused_next_input_ids_primes_then_reuses(monkeypatch) -> None:
    install_test_gates(monkeypatch)
    runner = make_runner()
    log_calls = []
    monkeypatch.setattr(
        fused_inputs.logger,
        "info_once",
        lambda *args, **kwargs: log_calls.append((args, kwargs)),
    )

    assert call_selector(runner) is None
    next_input_ids = fused_inputs.select_qwen_next_input_ids_buffer(runner)
    assert next_input_ids is runner.input_buffers.input_ids

    selected = call_selector(runner)
    assert selected is not None
    input_ids, logits_indices = selected
    assert input_ids.data_ptr() == runner.input_buffers.input_ids.data_ptr()
    assert input_ids.shape == (4,)
    assert logits_indices.tolist() == [0, 1, 2, 3]

    fused_inputs.select_qwen_next_input_ids_buffer(runner)
    selected_again = call_selector(runner)
    assert selected_again is not None
    assert selected_again[1].data_ptr() == logits_indices.data_ptr()
    assert len(log_calls) == 1


def test_qwen_fused_next_input_ids_request_change_falls_back(monkeypatch) -> None:
    install_test_gates(monkeypatch)
    runner = make_runner()
    assert call_selector(runner) is None
    fused_inputs.select_qwen_next_input_ids_buffer(runner)

    assert call_selector(runner, req_ids=["req-0", "req-1", "req-2", "new"]) is None


def test_qwen_fused_next_input_ids_preserves_batch_order(monkeypatch) -> None:
    install_test_gates(monkeypatch)
    runner = make_runner()
    req_ids = ["req-3", "req-0", "req-2", "req-1"]
    assert call_selector(runner, req_ids=req_ids) is None
    next_input_ids = fused_inputs.select_qwen_next_input_ids_buffer(runner)
    assert next_input_ids is not None
    next_input_ids[:4] = torch.tensor([13, 10, 12, 11], dtype=torch.int32)

    selected = call_selector(runner, req_ids=req_ids)
    assert selected is not None
    assert selected[0].tolist() == [13, 10, 12, 11]
    assert selected[1].tolist() == [0, 1, 2, 3]


def test_qwen_fused_next_input_ids_preserves_graph_padding(monkeypatch) -> None:
    install_test_gates(monkeypatch)
    runner = make_runner()
    padding = {
        "num_tokens_after_padding": 8,
        "num_reqs_after_padding": 8,
    }
    assert call_selector(runner, **padding) is None
    fused_inputs.select_qwen_next_input_ids_buffer(runner)

    selected = call_selector(runner, **padding)
    assert selected is not None
    assert selected[0].shape == (8,)
    assert selected[1].shape == (4,)


def test_qwen_fused_next_input_ids_fail_closed(monkeypatch) -> None:
    install_test_gates(monkeypatch)
    batch_size = 4
    cases = [
        {"num_scheduled_tokens": np.array([1, 1, 2, 1], dtype=np.int32)},
        {"is_prefilling_np": np.array([False, True, False, False])},
        {"total_num_draft_tokens": 1},
        {"total_num_logits": 5},
        {"num_tokens": 5},
        {"num_tokens_after_padding": 3},
        {"num_reqs_after_padding": 3},
        {"num_tokens_after_padding": 8, "num_reqs_after_padding": 4},
    ]
    for overrides in cases:
        runner = make_runner()
        runner._musa_qwen_primed_input_req_ids = tuple(
            f"req-{index}" for index in range(batch_size)
        )
        assert call_selector(runner, **overrides) is None
        assert fused_inputs.select_qwen_next_input_ids_buffer(runner) is None

    runner = make_runner()
    runner.model_state.num_new_sampled_tokens_per_step = 0
    runner._musa_qwen_primed_input_req_ids = tuple(
        f"req-{index}" for index in range(batch_size)
    )
    assert call_selector(runner) is None


def test_qwen_fused_next_input_ids_rejects_non_qwen_trait(monkeypatch) -> None:
    install_test_gates(monkeypatch)
    runner = make_runner(is_qwen_family=False)
    assert call_selector(runner) is None
