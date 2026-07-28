from types import SimpleNamespace

import numpy as np
import torch

from vllm_musa.v1.worker import qwen_identity_logits_view as identity_view


def make_runner(architecture: str = "Qwen3ForCausalLM", sampled_tokens: int = 1):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(architectures=[architecture])
        ),
        model_state=SimpleNamespace(num_new_sampled_tokens_per_step=sampled_tokens),
    )


def make_batch(batch_size: int = 4):
    return SimpleNamespace(
        num_reqs=batch_size,
        num_scheduled_tokens=np.ones(batch_size, dtype=np.int32),
        num_draft_tokens=0,
        logits_indices=torch.arange(batch_size, dtype=torch.int64),
    )


def test_qwen_identity_logits_view_accepts_uniform_decode(monkeypatch) -> None:
    monkeypatch.setattr(identity_view, "_MUSA_QWEN_IDENTITY_LOGITS_VIEW_ENABLED", True)
    monkeypatch.setattr(identity_view.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(identity_view, "_is_musa_tensor", lambda _tensor: True)
    hidden_states = torch.randn((4, 32), dtype=torch.bfloat16)

    selected = identity_view.select_qwen_identity_logits_view(
        make_runner(), hidden_states, make_batch()
    )

    assert selected is not None
    assert selected.data_ptr() == hidden_states.data_ptr()
    assert torch.equal(selected, hidden_states)


def test_qwen_identity_logits_view_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(identity_view, "_MUSA_QWEN_IDENTITY_LOGITS_VIEW_ENABLED", True)
    monkeypatch.setattr(identity_view.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(identity_view, "_is_musa_tensor", lambda _tensor: True)
    hidden_states = torch.randn((4, 32), dtype=torch.bfloat16)

    cases = []
    batch = make_batch()
    batch.num_scheduled_tokens[0] = 2
    cases.append((make_runner(), hidden_states, batch))
    batch = make_batch()
    batch.num_draft_tokens = 1
    cases.append((make_runner(), hidden_states, batch))
    batch = make_batch()
    batch.logits_indices = batch.logits_indices[:3]
    cases.append((make_runner(), hidden_states, batch))
    cases.append((make_runner(sampled_tokens=2), hidden_states, make_batch()))
    cases.append((make_runner("LlamaForCausalLM"), hidden_states, make_batch()))
    cases.append((make_runner(), hidden_states.float(), make_batch()))

    for runner, candidate_hidden_states, batch in cases:
        assert (
            identity_view.select_qwen_identity_logits_view(
                runner, candidate_hidden_states, batch
            )
            is None
        )


def test_qwen_identity_logits_view_flag_off(monkeypatch) -> None:
    monkeypatch.setattr(identity_view, "_MUSA_QWEN_IDENTITY_LOGITS_VIEW_ENABLED", False)
    hidden_states = torch.randn((4, 32), dtype=torch.bfloat16)
    assert (
        identity_view.select_qwen_identity_logits_view(
            make_runner(), hidden_states, make_batch()
        )
        is None
    )
