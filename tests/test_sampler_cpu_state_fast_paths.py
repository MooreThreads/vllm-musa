# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001
"""CPU-state contracts for MUSA V2 sampling fast paths."""

import inspect
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

# isort: off
import torchada  # noqa: F401
import torch

# isort: on

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_musa.v1.sample import topk_topp_sampler as sampler

requires_musa = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires a MUSA device",
)


def test_legacy_seeded_multinomial_is_limited_to_qwen_text_vocabs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sampler, "musa_seeded_multinomial_enabled", lambda: True)
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    generators = {0: object()}

    assert sampler.can_use_musa_seeded_multinomial(
        torch.empty((1, 151936)), generators, "raw_logprobs"
    )
    assert sampler.can_use_musa_seeded_multinomial(
        torch.empty((1, 152064)), generators, "raw_logprobs"
    )
    assert sampler.can_use_musa_seeded_multinomial(
        torch.empty((1, 248320)), generators, "raw_logprobs"
    )
    assert not sampler.can_use_musa_seeded_multinomial(
        torch.empty((1, 102400)), generators, "raw_logprobs"
    )
    assert not sampler.can_use_musa_seeded_multinomial(
        torch.empty((1, 8448)), generators, "raw_logprobs"
    )
    assert not sampler.can_use_musa_seeded_multinomial(
        torch.empty((1, 151936)), {}, "raw_logprobs"
    )
    assert not sampler.can_use_musa_seeded_multinomial(
        torch.empty((1, 151936)), generators, "processed_logprobs"
    )


def test_uniform_sampler_metadata_patch_is_active_and_qwen_gated() -> None:
    metadata_fields = {field.name for field in fields(SamplingMetadata)}
    assert "uniform_top_k" in metadata_fields
    assert "uniform_temperature" in metadata_fields
    source = inspect.getsource(InputBatch._make_sampling_metadata)
    assert "uniform_top_k" in source
    assert "uniform_temperature" in source
    assert "self.vocab_size in (151936, 248320)" in source
    assert "candidate_top_k == 50" in source
    assert "self.all_random" in source
    assert "temperature_cpu == np.float32(1.0)" in source


def test_legacy_qwen_unit_temperature_skip_is_exact_and_vocab_gated() -> None:
    enabled_sampler = SimpleNamespace(_musa_qwen_skip_unit_temperature=True)
    metadata = SimpleNamespace(all_random=True, uniform_temperature=1.0)
    assert sampler._can_skip_legacy_qwen_unit_temperature(
        enabled_sampler, torch.empty((4, 248320)), metadata
    )
    assert sampler._can_skip_legacy_qwen_unit_temperature(
        enabled_sampler, torch.empty((1, 151936)), metadata
    )

    metadata.uniform_temperature = None
    assert not sampler._can_skip_legacy_qwen_unit_temperature(
        enabled_sampler, torch.empty((4, 248320)), metadata
    )
    metadata.uniform_temperature = 0.7
    assert not sampler._can_skip_legacy_qwen_unit_temperature(
        enabled_sampler, torch.empty((4, 248320)), metadata
    )
    metadata.uniform_temperature = 1.0
    assert not sampler._can_skip_legacy_qwen_unit_temperature(
        enabled_sampler, torch.empty((4, 32000)), metadata
    )
    metadata.all_random = False
    assert not sampler._can_skip_legacy_qwen_unit_temperature(
        enabled_sampler, torch.empty((4, 248320)), metadata
    )
    metadata.all_random = True
    assert not sampler._can_skip_legacy_qwen_unit_temperature(
        SimpleNamespace(_musa_qwen_skip_unit_temperature=False),
        torch.empty((4, 248320)),
        metadata,
    )


def test_legacy_sample_skips_only_cpu_proven_qwen_unit_temperature(
    monkeypatch,
) -> None:
    temperature_calls = []

    def apply_temperature(logits, temperature, all_random):
        temperature_calls.append((temperature, all_random))
        return logits

    fake_sampler = SimpleNamespace(
        logprobs_mode="raw_logprobs",
        apply_temperature=apply_temperature,
        topk_topp_sampler=SimpleNamespace(
            forward=SimpleNamespace(__name__="forward_native")
        ),
        use_fp64_gumbel=False,
        _musa_qwen_skip_unit_temperature=True,
    )
    metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.ones(4),
        top_k=None,
        top_p=None,
        generators={},
        logitsprocs=SimpleNamespace(argmax_invariant=[]),
        uniform_temperature=1.0,
    )
    monkeypatch.setattr(sampler, "can_use_qwen_legacy_gumbel", lambda *_args: False)
    monkeypatch.setattr(
        sampler,
        "_call_topk_topp_sampler",
        lambda *_args, **_kwargs: (torch.zeros(4, dtype=torch.int64), None),
    )

    sampler._sample(fake_sampler, torch.empty((4, 248320)), metadata)
    assert temperature_calls == []

    metadata.uniform_temperature = None
    sampler._sample(fake_sampler, torch.empty((4, 248320)), metadata)
    assert len(temperature_calls) == 1

    metadata.uniform_temperature = 1.0
    sampler._sample(fake_sampler, torch.empty((4, 32000)), metadata)
    assert len(temperature_calls) == 2

    fake_sampler._musa_qwen_skip_unit_temperature = False
    sampler._sample(fake_sampler, torch.empty((4, 248320)), metadata)
    assert len(temperature_calls) == 3


def _state_array(
    values: list[int] | list[float], dtype: torch.dtype
) -> SimpleNamespace:
    numpy_dtype = np.int32 if dtype == torch.int32 else np.float32
    return SimpleNamespace(
        np=np.asarray(values, dtype=numpy_dtype),
        gpu=torch.tensor(values, dtype=dtype),
    )


def _gumbel_gate_sampler(
    rows: int,
    *,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    min_p: float = 0.05,
    all_seeded: bool = True,
    num_speculative_tokens: int = 1,
    use_fp64_gumbel: bool = False,
    logprobs_mode: str = "raw_logprobs",
    is_qwen_family: bool = True,
) -> SimpleNamespace:
    states = SimpleNamespace(
        temperature=_state_array([temperature] * rows, torch.float32),
        top_k=_state_array([top_k] * rows, torch.int32),
        top_p=_state_array([top_p] * rows, torch.float32),
        min_p=_state_array([min_p] * rows, torch.float32),
        has_user_seed=np.full(rows, all_seeded, dtype=np.bool_),
    )
    return SimpleNamespace(
        sampling_states=states,
        num_speculative_tokens=num_speculative_tokens,
        use_fp64_gumbel=use_fp64_gumbel,
        logprobs_mode=logprobs_mode,
        _musa_qwen_family=is_qwen_family,
    )


def _unfiltered_gumbel_gate_sampler(
    rows: int, vocab_size: int = 151936, *, is_qwen_family: bool = True
):
    states = SimpleNamespace(
        temperature=_state_array([1.0] * rows, torch.float32),
        top_k=_state_array([vocab_size] * rows, torch.int32),
        top_p=_state_array([1.0] * rows, torch.float32),
        min_p=_state_array([0.0] * rows, torch.float32),
        seeds=SimpleNamespace(gpu=object()),
        has_user_seed=np.zeros(rows, dtype=np.bool_),
    )
    return SimpleNamespace(
        sampling_states=states,
        num_speculative_tokens=1,
        use_fp64_gumbel=False,
        logprobs_mode="raw_logprobs",
        _musa_qwen_family=is_qwen_family,
        logit_bias_state=SimpleNamespace(use_logit_bias=np.zeros(rows, dtype=np.bool_)),
        penalties_state=SimpleNamespace(use_penalty=np.zeros(rows, dtype=np.bool_)),
        bad_words_state=SimpleNamespace(
            num_bad_words=SimpleNamespace(np=np.zeros(rows, dtype=np.int32))
        ),
    )


class _FakeGenerator:
    def __init__(
        self,
        seed: int,
        offset: int = 0,
        device: torch.device | None = None,
        fail_offset: int | None = None,
    ) -> None:
        self.seed = seed
        self.offset = offset
        self.device = device or torch.device("cpu")
        self.fail_offset = fail_offset

    def initial_seed(self) -> int:
        return self.seed

    def get_offset(self) -> int:
        return self.offset

    def set_offset(self, offset: int) -> None:
        if offset == self.fail_offset:
            raise RuntimeError("injected set_offset failure")
        self.offset = offset


def _legacy_gumbel_metadata(rows: int, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        max_num_logprobs=kwargs.get("max_num_logprobs"),
        logprob_token_ids=kwargs.get("logprob_token_ids"),
        all_random=kwargs.get("all_random", True),
        top_k=kwargs.get("top_k", torch.full((rows,), 50, dtype=torch.int32)),
        top_p=kwargs.get("top_p"),
        spec_token_ids=kwargs.get("spec_token_ids", [[] for _ in range(rows)]),
        generators=kwargs.get(
            "generators", {row: _FakeGenerator(60043 + row) for row in range(rows)}
        ),
    )


class MinPLogitsProcessor:
    def __init__(self) -> None:
        self.min_p_count = 0


class LogitBiasLogitsProcessor:
    def __init__(self) -> None:
        self.biases = {}


class MinTokensLogitsProcessor:
    def __init__(self) -> None:
        self.min_toks = {}


def _legacy_unfiltered_metadata(rows: int) -> SimpleNamespace:
    return SimpleNamespace(
        all_random=True,
        all_greedy=False,
        uniform_temperature=1.0,
        top_k=None,
        top_p=None,
        generators={},
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        spec_token_ids=[[] for _ in range(rows)],
        thinking_budget_state_holder=None,
        logitsprocs=SimpleNamespace(
            argmax_invariant=[MinPLogitsProcessor()],
            non_argmax_invariant=[
                LogitBiasLogitsProcessor(),
                MinTokensLogitsProcessor(),
            ],
        ),
    )


@pytest.mark.parametrize("vocab_size", [151936, 248320])
def test_qwen_legacy_unfiltered_gumbel_gate_accepts_exact_contract(
    monkeypatch, vocab_size: int
) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)

    assert sampler.can_use_qwen_legacy_unfiltered_gumbel(
        torch.empty((2, vocab_size), dtype=torch.bfloat16),
        _legacy_unfiltered_metadata(2),
        "raw_logprobs",
        False,
        False,
        True,
    )


def test_qwen_legacy_unfiltered_gumbel_gate_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    logits = torch.empty((2, 248320), dtype=torch.bfloat16)

    rejected = []
    value = _legacy_unfiltered_metadata(2)
    value.uniform_temperature = 0.8
    rejected.append(value)
    value = _legacy_unfiltered_metadata(2)
    value.top_k = torch.full((2,), 50)
    rejected.append(value)
    value = _legacy_unfiltered_metadata(2)
    value.generators = {0: _FakeGenerator(1)}
    rejected.append(value)
    value = _legacy_unfiltered_metadata(2)
    value.no_penalties = False
    rejected.append(value)
    value = _legacy_unfiltered_metadata(2)
    value.logitsprocs.non_argmax_invariant[0].biases[0] = {1: 1.0}
    rejected.append(value)
    value = _legacy_unfiltered_metadata(2)
    value.logitsprocs.argmax_invariant[0].min_p_count = 1
    rejected.append(value)
    value = _legacy_unfiltered_metadata(2)
    value.logitsprocs.argmax_invariant.append(SimpleNamespace())
    rejected.append(value)
    value = _legacy_unfiltered_metadata(2)
    value.spec_token_ids[0] = [1]
    rejected.append(value)

    for metadata in rejected:
        assert not sampler.can_use_qwen_legacy_unfiltered_gumbel(
            logits, metadata, "raw_logprobs", False, False, True
        )
    assert not sampler.can_use_qwen_legacy_unfiltered_gumbel(
        logits, _legacy_unfiltered_metadata(2), "raw_logprobs", True, False, True
    )
    assert not sampler.can_use_qwen_legacy_unfiltered_gumbel(
        logits, _legacy_unfiltered_metadata(2), "raw_logprobs", False, True, True
    )
    assert not sampler.can_use_qwen_legacy_unfiltered_gumbel(
        logits,
        _legacy_unfiltered_metadata(2),
        "processed_logits",
        False,
        False,
        True,
    )
    assert not sampler.can_use_qwen_legacy_unfiltered_gumbel(
        logits.float(),
        _legacy_unfiltered_metadata(2),
        "raw_logprobs",
        False,
        False,
        True,
    )


def test_qwen_legacy_unfiltered_gumbel_advances_private_stream(monkeypatch) -> None:
    rows = 4
    logits = torch.empty((rows, 151936), dtype=torch.bfloat16)
    generator = _FakeGenerator(60043, offset=8, device=logits.device)
    fake_sampler = SimpleNamespace(_musa_qwen_unfiltered_generator=generator)
    calls = []

    def fake_gumbel(*args, **kwargs):
        calls.append(
            (
                tuple(
                    value.clone() if isinstance(value, torch.Tensor) else value
                    for value in args
                ),
                kwargs,
            )
        )
        return torch.arange(rows, dtype=torch.int64)

    monkeypatch.setattr(sampler.vllm_worker_sampler, "gumbel_sample", fake_gumbel)
    first = sampler.sample_qwen_legacy_unfiltered_gumbel(fake_sampler, logits)
    second = sampler.sample_qwen_legacy_unfiltered_gumbel(fake_sampler, logits)

    assert first.tolist() == second.tolist() == [0, 1, 2, 3]
    assert calls[0][0][1].tolist() == [0, 0, 0, 0]
    assert calls[0][0][2].tolist() == [1.0]
    assert calls[0][0][3].tolist() == [60043]
    assert calls[0][0][4].tolist() == [2, 3, 4, 5]
    assert calls[1][0][4].tolist() == [6, 7, 8, 9]
    assert (
        calls[0][1]
        == calls[1][1]
        == {
            "apply_temperature": False,
            "use_fp64": False,
        }
    )
    assert generator.get_offset() == 40


def test_qwen_legacy_unfiltered_gumbel_rejects_non_qwen_trait(monkeypatch) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    assert not sampler.can_use_qwen_legacy_unfiltered_gumbel(
        torch.empty((2, 248320), dtype=torch.bfloat16),
        _legacy_unfiltered_metadata(2),
        "raw_logprobs",
        False,
        False,
        False,
    )


def _can_use_qwen_legacy_gumbel(*args, **kwargs) -> bool:
    return sampler.can_use_qwen_legacy_gumbel(*args, **kwargs, is_qwen_family=True)


def test_qwen_legacy_gumbel_gate_and_generator_handoff(monkeypatch) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 4
    logits = torch.randn((rows, 248320))
    metadata = _legacy_gumbel_metadata(rows)

    assert not sampler.can_use_qwen_legacy_gumbel(
        logits,
        metadata,
        "raw_logprobs",
        None,
        False,
        is_qwen_family=False,
    )
    assert _can_use_qwen_legacy_gumbel(logits, metadata, "raw_logprobs", None, False)
    metadata.generators = dict(reversed(tuple(metadata.generators.items())))
    assert _can_use_qwen_legacy_gumbel(logits, metadata, "raw_logprobs", None, False)
    state = sampler.get_qwen_legacy_generator_state(metadata.generators, rows)
    assert state == ([60043, 60044, 60045, 60046], [0, 0, 0, 0])

    captured = {}

    def fake_gumbel_sample(
        processed_logits,
        mapping,
        temperature,
        seeds,
        positions,
        **kwargs,
    ):
        captured.update(
            processed_logits=processed_logits,
            mapping=mapping,
            temperature=temperature,
            seeds=seeds,
            positions=positions,
            kwargs=kwargs,
        )
        return torch.arange(rows, dtype=torch.int64)

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", lambda tensor, *_args: tensor)
    monkeypatch.setattr(
        sampler.vllm_worker_sampler, "gumbel_sample", fake_gumbel_sample
    )
    sampled = sampler.sample_qwen_legacy_gumbel(
        logits, metadata.generators, metadata.top_k, state
    )

    assert sampled.tolist() == [0, 1, 2, 3]
    assert captured["mapping"].tolist() == [0, 1, 2, 3]
    assert captured["temperature"].tolist() == [1.0] * rows
    assert captured["seeds"].tolist() == [60043, 60044, 60045, 60046]
    assert captured["positions"].tolist() == [0] * rows
    assert captured["kwargs"] == {"apply_temperature": False, "use_fp64": False}
    assert [generator.get_offset() for generator in metadata.generators.values()] == [
        4
    ] * rows


def test_qwen_legacy_gumbel_accepts_unfiltered_seeded_rows(monkeypatch) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 16
    logits = torch.randn((rows, 248320))
    metadata = _legacy_gumbel_metadata(rows, top_k=None)
    for generator in metadata.generators.values():
        generator.set_offset(64)

    assert _can_use_qwen_legacy_gumbel(
        logits,
        metadata,
        "raw_logprobs",
        None,
        False,
    )

    filter_args = []

    def fake_apply_top_k_top_p(logits_arg, top_k_arg, top_p_arg):
        filter_args.append((top_k_arg, top_p_arg))
        return logits_arg

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    monkeypatch.setattr(
        sampler.vllm_worker_sampler,
        "gumbel_sample",
        lambda *args, **kwargs: torch.arange(rows, dtype=torch.int64),
    )

    sampled = sampler.sample_qwen_legacy_gumbel_partitioned(
        logits,
        metadata.generators,
        None,
    )

    assert sampled is not None
    assert sampled.tolist() == list(range(rows))
    assert filter_args == [(None, None)]
    assert [generator.get_offset() for generator in metadata.generators.values()] == [
        68
    ] * rows


def test_qwen_legacy_gumbel_rejects_small_unfiltered_batches(monkeypatch) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 15
    logits = torch.randn((rows, 248320))
    metadata = _legacy_gumbel_metadata(rows, top_k=None)

    assert not _can_use_qwen_legacy_gumbel(
        logits,
        metadata,
        "raw_logprobs",
        None,
        False,
    )


def test_qwen_legacy_gumbel_waits_for_unfiltered_seed_offsets(monkeypatch) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 16
    logits = torch.randn((rows, 248320))
    metadata = _legacy_gumbel_metadata(rows, top_k=None)
    for generator in metadata.generators.values():
        generator.set_offset(60)

    assert not _can_use_qwen_legacy_gumbel(
        logits,
        metadata,
        "raw_logprobs",
        None,
        False,
    )


def test_qwen_legacy_gumbel_gate_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    logits = torch.randn((4, 248320))

    for rows in (1, 2, 3):
        assert not _can_use_qwen_legacy_gumbel(
            torch.randn((rows, 248320)),
            _legacy_gumbel_metadata(rows),
            "raw_logprobs",
            None,
            False,
        )
    assert not _can_use_qwen_legacy_gumbel(
        torch.randn((4, 151936)),
        _legacy_gumbel_metadata(4),
        "raw_logprobs",
        None,
        False,
    )
    assert not _can_use_qwen_legacy_gumbel(
        logits,
        _legacy_gumbel_metadata(4, max_num_logprobs=0),
        "raw_logprobs",
        None,
        False,
    )
    assert _can_use_qwen_legacy_gumbel(
        logits,
        _legacy_gumbel_metadata(4, generators={0: _FakeGenerator(1)}),
        "raw_logprobs",
        None,
        False,
    )
    assert not _can_use_qwen_legacy_gumbel(
        logits,
        _legacy_gumbel_metadata(4, generators={}),
        "raw_logprobs",
        None,
        False,
    )
    assert not _can_use_qwen_legacy_gumbel(
        torch.randn((16, 248320)),
        _legacy_gumbel_metadata(16, top_k=None, generators={}),
        "raw_logprobs",
        None,
        False,
    )
    assert not _can_use_qwen_legacy_gumbel(
        logits,
        _legacy_gumbel_metadata(4, generators={4: _FakeGenerator(1)}),
        "raw_logprobs",
        None,
        False,
    )
    assert not _can_use_qwen_legacy_gumbel(
        logits,
        _legacy_gumbel_metadata(4, spec_token_ids=[[1], [], [], []]),
        "raw_logprobs",
        None,
        False,
    )
    assert not _can_use_qwen_legacy_gumbel(
        logits, _legacy_gumbel_metadata(4), "raw_logprobs", None, True
    )
    invalid = _legacy_gumbel_metadata(4)
    invalid.generators[0].offset = 2
    assert sampler.get_qwen_legacy_generator_state(invalid.generators, 4) is None


def test_qwen_legacy_gumbel_rolls_back_partial_generator_advance(
    monkeypatch,
) -> None:
    rows = 4
    logits = torch.randn((rows, 248320))
    generators = {
        row: _FakeGenerator(
            60043 + row,
            device=logits.device,
            fail_offset=4 if row == 1 else None,
        )
        for row in range(rows)
    }
    state = sampler.get_qwen_legacy_generator_state(generators, rows)
    monkeypatch.setattr(sampler, "_apply_top_k_top_p", lambda tensor, *_args: tensor)
    monkeypatch.setattr(
        sampler.vllm_worker_sampler,
        "gumbel_sample",
        lambda *_args, **_kwargs: torch.arange(rows, dtype=torch.int64),
    )

    with pytest.raises(RuntimeError, match="Failed to advance legacy MUSA generators"):
        sampler.sample_qwen_legacy_gumbel(
            logits,
            generators,
            torch.full((rows,), 50, dtype=torch.int32),
            state,
        )

    assert [generator.get_offset() for generator in generators.values()] == [0] * rows


def test_qwen_legacy_gumbel_partitions_seeded_and_unseeded_rows(
    monkeypatch,
) -> None:
    rows = 4
    logits = torch.randn((rows, 248320))
    generators = {0: _FakeGenerator(60043), 2: _FakeGenerator(60045)}
    captured = {}

    def fake_gumbel_sample(
        processed_logits,
        mapping,
        temperature,
        seeds,
        positions,
        **kwargs,
    ):
        captured.update(
            processed_logits=processed_logits.clone(),
            mapping=mapping.clone(),
            temperature=temperature.clone(),
            seeds=seeds.clone(),
            positions=positions.clone(),
            kwargs=kwargs,
        )
        return torch.tensor([10, 20], dtype=torch.int64)

    def fake_multinomial(probs, subset_generators):
        captured.update(
            unseeded_probs=probs.clone(),
            unseeded_generators=subset_generators,
        )
        return torch.tensor([11, 13], dtype=torch.int64)

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", lambda tensor, *_args: tensor)
    monkeypatch.setattr(
        sampler.vllm_worker_sampler, "gumbel_sample", fake_gumbel_sample
    )
    monkeypatch.setattr(sampler, "sample_probs_seeded_multinomial", fake_multinomial)

    sampled = sampler.sample_qwen_legacy_gumbel_partitioned(
        logits,
        generators,
        torch.full((rows,), 50, dtype=torch.int32),
    )

    assert sampled.tolist() == [10, 11, 20, 13]
    assert torch.equal(captured["processed_logits"], logits[[0, 2]])
    assert captured["mapping"].tolist() == [0, 1]
    assert captured["seeds"].tolist() == [60043, 60045]
    assert captured["positions"].tolist() == [0, 0]
    assert captured["kwargs"] == {"apply_temperature": False, "use_fp64": False}
    assert torch.allclose(
        captured["unseeded_probs"],
        logits[[1, 3]].softmax(dim=-1, dtype=torch.float32),
    )
    assert captured["unseeded_generators"] == {}
    assert [generator.get_offset() for generator in generators.values()] == [4, 4]


def test_qwen_v2_gumbel_gate_accepts_only_exact_contract(monkeypatch) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 4
    logits = torch.randn((rows, 151936))
    mapping = torch.arange(rows, dtype=torch.int64)
    mapping_np = np.arange(rows, dtype=np.int64)
    pos = mapping.clone()

    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows, is_qwen_family=False),
        logits,
        mapping,
        mapping_np,
        pos,
        False,
    )
    assert sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows), logits, mapping, mapping_np, pos, False
    )
    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows, top_p=0.9),
        logits,
        mapping,
        mapping_np,
        pos,
        False,
    )
    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows, all_seeded=False),
        logits,
        mapping,
        mapping_np,
        pos,
        False,
    )
    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows, num_speculative_tokens=2),
        logits,
        mapping,
        mapping_np,
        pos,
        False,
    )
    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows), logits, mapping, mapping_np, pos, True
    )


def test_qwen_v2_gumbel_gate_rejects_non_qwen_vocab(monkeypatch) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 4
    mapping = torch.arange(rows, dtype=torch.int64)

    assert not sampler.can_use_qwen_v2_gumbel(
        _gumbel_gate_sampler(rows),
        torch.randn((rows, 131072)),
        mapping,
        np.arange(rows, dtype=np.int64),
        mapping,
        False,
    )


@pytest.mark.parametrize("vocab_size", [151936, 248320])
def test_qwen_v2_unfiltered_gumbel_gate_accepts_exact_contract(
    monkeypatch, vocab_size: int
) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 2
    mapping = torch.arange(rows, dtype=torch.int64)

    assert sampler.can_use_qwen_v2_unfiltered_gumbel(
        _unfiltered_gumbel_gate_sampler(rows, vocab_size),
        torch.empty((rows, vocab_size), dtype=torch.bfloat16),
        mapping,
        np.arange(rows, dtype=np.int64),
        mapping,
        False,
    )


def test_qwen_v2_unfiltered_gumbel_gate_rejects_noncontract(monkeypatch) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 2
    mapping = torch.arange(rows, dtype=torch.int64)
    mapping_np = np.arange(rows, dtype=np.int64)
    logits = torch.empty((rows, 151936), dtype=torch.bfloat16)

    rejected = []

    value = _unfiltered_gumbel_gate_sampler(rows)
    value.sampling_states.temperature.np[0] = 0.8
    rejected.append((value, False))

    value = _unfiltered_gumbel_gate_sampler(rows)
    value.sampling_states.top_k.np[0] = 50
    rejected.append((value, False))

    value = _unfiltered_gumbel_gate_sampler(rows)
    value.sampling_states.top_p.np[0] = 0.9
    rejected.append((value, False))

    value = _unfiltered_gumbel_gate_sampler(rows)
    value.sampling_states.min_p.np[0] = 0.05
    rejected.append((value, False))

    value = _unfiltered_gumbel_gate_sampler(rows)
    value.logit_bias_state.use_logit_bias[0] = True
    rejected.append((value, False))

    value = _unfiltered_gumbel_gate_sampler(rows)
    value.penalties_state.use_penalty[0] = True
    rejected.append((value, False))

    value = _unfiltered_gumbel_gate_sampler(rows)
    value.bad_words_state.num_bad_words.np[0] = 1
    rejected.append((value, False))

    value = _unfiltered_gumbel_gate_sampler(rows)
    value.num_speculative_tokens = 2
    rejected.append((value, False))

    value = _unfiltered_gumbel_gate_sampler(rows)
    value.sampling_states.has_user_seed[0] = True
    rejected.append((value, False))

    value = _unfiltered_gumbel_gate_sampler(rows)
    del value.sampling_states.has_user_seed
    rejected.append((value, False))

    value = _unfiltered_gumbel_gate_sampler(rows)
    rejected.append((value, True))

    for candidate, return_logprobs in rejected:
        assert not sampler.can_use_qwen_v2_unfiltered_gumbel(
            candidate,
            logits,
            mapping,
            mapping_np,
            mapping,
            return_logprobs,
        )


def test_qwen_v2_unfiltered_gumbel_gate_rejects_non_qwen_trait(monkeypatch) -> None:
    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _tensor: True)
    rows = 1
    mapping = torch.zeros(rows, dtype=torch.int64)

    assert not sampler.can_use_qwen_v2_unfiltered_gumbel(
        _unfiltered_gumbel_gate_sampler(rows, is_qwen_family=False),
        torch.empty((rows, 151936), dtype=torch.bfloat16),
        mapping,
        np.zeros(rows, dtype=np.int64),
        mapping,
        False,
    )


def test_qwen_v2_unfiltered_gumbel_uses_existing_seed_state(monkeypatch) -> None:
    sentinel = torch.tensor([7, 11])
    temperature = object()
    seeds = object()
    worker = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=temperature),
            seeds=SimpleNamespace(gpu=seeds),
        )
    )
    logits = torch.empty((2, 151936), dtype=torch.bfloat16)
    idx_mapping = torch.arange(2)
    positions = torch.arange(2)
    captured = {}

    def fake_gumbel(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(sampler.vllm_worker_sampler, "gumbel_sample", fake_gumbel)
    sampled, processed_logits = sampler.sample_worker_logits_qwen_v2_unfiltered_gumbel(
        worker, logits, idx_mapping, positions
    )

    assert sampled is sentinel
    assert processed_logits is logits
    assert captured["args"] == (
        logits,
        idx_mapping,
        temperature,
        seeds,
        positions,
    )
    assert captured["kwargs"] == {
        "apply_temperature": False,
        "use_fp64": False,
    }


def test_uniform_active_min_p_uses_cpu_state() -> None:
    assert sampler._uniform_active_min_p(
        np.asarray([0.05], dtype=np.float32)
    ) == pytest.approx(0.05)
    assert sampler._uniform_active_min_p(
        np.asarray([0.05, 0.05], dtype=np.float32)
    ) == pytest.approx(0.05)
    assert sampler._uniform_active_min_p(np.asarray([], dtype=np.float32)) is None
    assert (
        sampler._uniform_active_min_p(np.asarray([0.0, 0.0], dtype=np.float32)) is None
    )
    assert (
        sampler._uniform_active_min_p(np.asarray([0.05, 0.1], dtype=np.float32)) is None
    )


def test_legacy_uniform_min_p_uses_cpu_hint_only_for_qwen_vocab() -> None:
    min_p = torch.full((4,), 0.05, dtype=torch.float32)
    processor = SimpleNamespace(
        min_p=min_p,
        min_p_cpu=np.full((4,), 0.05, dtype=np.float32),
    )

    qwen_logits = torch.empty((4, 248320))
    assert sampler._legacy_min_p_with_cpu_hint(processor, qwen_logits) == pytest.approx(
        0.05
    )

    non_qwen_logits = torch.empty((4, 32000))
    assert sampler._legacy_min_p_with_cpu_hint(processor, non_qwen_logits) is min_p

    processor.min_p_cpu[1] = 0.04
    assert sampler._legacy_min_p_with_cpu_hint(processor, qwen_logits) is min_p


def test_legacy_uniform_top_k_uses_cpu_hint_only_for_musa_native_sampler(
    monkeypatch,
) -> None:
    top_k = torch.full((4,), 50, dtype=torch.int32)
    metadata = SimpleNamespace(uniform_top_k=50, generators={})
    musa_sampler = SimpleNamespace(forward=SimpleNamespace(__name__="forward_musa"))
    logits = torch.empty((4, 248320))

    def can_use_native(_logits, generators, logprobs_mode):
        return not generators and logprobs_mode == "raw_logprobs"

    monkeypatch.setattr(sampler, "can_use_musa_sampler", can_use_native)

    assert (
        sampler._legacy_top_k_with_cpu_hint(
            metadata, top_k, musa_sampler, logits, "raw_logprobs"
        )
        == 50
    )

    metadata.uniform_top_k = 49
    assert (
        sampler._legacy_top_k_with_cpu_hint(
            metadata, top_k, musa_sampler, logits, "raw_logprobs"
        )
        is top_k
    )

    metadata.uniform_top_k = 50
    fallback_sampler = SimpleNamespace(
        forward=SimpleNamespace(__name__="forward_native")
    )
    assert (
        sampler._legacy_top_k_with_cpu_hint(
            metadata, top_k, fallback_sampler, logits, "raw_logprobs"
        )
        is top_k
    )

    metadata.uniform_top_k = None
    heterogeneous_top_k = torch.tensor([50, 49, 50, 49], dtype=torch.int32)
    assert (
        sampler._legacy_top_k_with_cpu_hint(
            metadata,
            heterogeneous_top_k,
            musa_sampler,
            logits,
            "raw_logprobs",
        )
        is heterogeneous_top_k
    )

    metadata.uniform_top_k = 50
    metadata.generators = {0: object()}
    assert (
        sampler._legacy_top_k_with_cpu_hint(
            metadata, top_k, musa_sampler, logits, "raw_logprobs"
        )
        is top_k
    )

    metadata.generators = {}
    assert (
        sampler._legacy_top_k_with_cpu_hint(
            metadata, top_k, musa_sampler, logits, "processed_logprobs"
        )
        is top_k
    )

    non_qwen_logits = torch.empty((4, 32000))
    assert (
        sampler._legacy_top_k_with_cpu_hint(
            metadata, top_k, musa_sampler, non_qwen_logits, "raw_logprobs"
        )
        is top_k
    )


def test_legacy_gumbel_uniform_top_k_uses_cpu_hint_with_generators() -> None:
    top_k = torch.full((4,), 50, dtype=torch.int32)
    metadata = SimpleNamespace(uniform_top_k=50)

    assert (
        sampler._legacy_gumbel_top_k_with_cpu_hint(
            metadata, torch.empty((4, 248320)), top_k
        )
        == 50
    )

    metadata.uniform_top_k = None
    assert (
        sampler._legacy_gumbel_top_k_with_cpu_hint(
            metadata, torch.empty((4, 248320)), top_k
        )
        is top_k
    )

    metadata.uniform_top_k = None
    heterogeneous_top_k = torch.tensor([50, 49, 50, 49], dtype=torch.int32)
    assert (
        sampler._legacy_gumbel_top_k_with_cpu_hint(
            metadata, torch.empty((4, 248320)), heterogeneous_top_k
        )
        is heterogeneous_top_k
    )
    metadata.uniform_top_k = 50
    assert (
        sampler._legacy_gumbel_top_k_with_cpu_hint(
            metadata, torch.empty((4, 131072)), top_k
        )
        is top_k
    )


def test_uniform_top_k_threshold_preserves_ties() -> None:
    logits = torch.arange(64, dtype=torch.float32).repeat(2, 1)
    logits[:, 12:17] = 14.0
    expected = sampler.vllm_topk_topp_sampler.apply_top_k_only(
        logits.clone(), torch.full((2,), 50, dtype=torch.int32)
    )

    actual = sampler._apply_top_k_top_p(logits.clone(), 50, None)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.isfinite(), expected.isfinite())
    assert actual.isfinite().sum(dim=1).tolist() == [52, 52]


def test_tensor_top_k_top_p_prefilter_matches_upstream() -> None:
    torch.manual_seed(60046)
    logits = torch.randn((4, 256), dtype=torch.float32)
    top_k = torch.tensor([20, 50, 7, 49], dtype=torch.int32)
    top_p = torch.tensor([0.75, 0.8, 0.9, 0.95], dtype=torch.float32)
    expected = sampler.vllm_topk_topp_sampler.apply_top_k_top_p_pytorch(
        logits.clone(), top_k, top_p
    )

    actual_input = logits.clone()
    actual = sampler._apply_top_k_top_p_musa_topk_prefilter(actual_input, top_k, top_p)

    assert actual.data_ptr() == actual_input.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.isfinite(), expected.isfinite())


def test_seeded_filters_reuse_processed_logits_only_when_allowed() -> None:
    sampling_states = SimpleNamespace(
        vocab_size=64,
        top_k=_state_array([50, 50], torch.int32),
        top_p=_state_array([1.0, 1.0], torch.float32),
        min_p=_state_array([0.0, 0.0], torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(sampling_states=sampling_states)
    mapping = torch.tensor([0, 1], dtype=torch.int64)
    mapping_np = np.asarray([0, 1], dtype=np.int64)
    processed = torch.randn((2, 64), dtype=torch.float32)
    original = processed.clone()

    reused = sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        processed,
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )
    assert reused.data_ptr() == processed.data_ptr()
    assert not torch.equal(processed, original)

    preserved_input = original.clone()
    copied = sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        preserved_input,
        mapping,
        mapping_np,
        preserve_processed_logits=True,
    )
    assert copied.data_ptr() != preserved_input.data_ptr()
    torch.testing.assert_close(preserved_input, original, rtol=0, atol=0)
    torch.testing.assert_close(copied, reused, rtol=0, atol=0)


def test_seeded_top_p_keeps_tensor_top_k_fallback(monkeypatch) -> None:
    captured = {}

    def fake_apply_top_k_top_p(logits, top_k, top_p):
        captured.update(top_k=top_k, top_p=top_p)
        return logits

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    sampling_states = SimpleNamespace(
        vocab_size=64,
        top_k=_state_array([50, 50], torch.int32),
        top_p=_state_array([0.9, 0.9], torch.float32),
        min_p=_state_array([0.0, 0.0], torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(sampling_states=sampling_states)
    mapping = torch.tensor([0, 1], dtype=torch.int64)
    mapping_np = np.asarray([0, 1], dtype=np.int64)

    sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        torch.randn((2, 64)),
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )

    assert isinstance(captured["top_k"], torch.Tensor)
    assert isinstance(captured["top_p"], torch.Tensor)


def test_seeded_large_uniform_top_k_top_p_uses_cpu_scalar(monkeypatch) -> None:
    captured = {}

    def fake_apply_top_k_top_p(logits, top_k, top_p):
        captured.update(top_k=top_k, top_p=top_p)
        return logits

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    rows = 16
    vocab_size = 248320
    sampling_states = SimpleNamespace(
        vocab_size=vocab_size,
        top_k=_state_array([50] * rows, torch.int32),
        top_p=_state_array([0.9] * rows, torch.float32),
        min_p=_state_array([0.0] * rows, torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(
        sampling_states=sampling_states, _musa_qwen_family=True
    )
    mapping = torch.arange(rows, dtype=torch.int64)
    mapping_np = np.arange(rows, dtype=np.int64)

    sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        torch.randn((rows, vocab_size)),
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )

    assert captured["top_k"] == 50
    assert isinstance(captured["top_p"], torch.Tensor)


def test_seeded_small_batch_qwen_vocab_uses_cpu_scalar(monkeypatch) -> None:
    captured = {}

    def fake_apply_top_k_top_p(logits, top_k, top_p):
        captured.update(top_k=top_k, top_p=top_p)
        return logits

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    rows = 4
    vocab_size = 151936
    sampling_states = SimpleNamespace(
        vocab_size=vocab_size,
        top_k=_state_array([50] * rows, torch.int32),
        top_p=_state_array([1.0] * rows, torch.float32),
        min_p=_state_array([0.0] * rows, torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(
        sampling_states=sampling_states, _musa_qwen_family=True
    )
    mapping = torch.arange(rows, dtype=torch.int64)
    mapping_np = np.arange(rows, dtype=np.int64)

    sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        torch.randn((rows, vocab_size)),
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )

    assert captured == {"top_k": 50, "top_p": None}


def test_qwen_vocab_without_model_trait_keeps_tensor_top_k(monkeypatch) -> None:
    captured = {}

    def fake_apply_top_k_top_p(logits, top_k, top_p):
        captured.update(top_k=top_k, top_p=top_p)
        return logits

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    rows = 4
    vocab_size = 151936
    sampling_states = SimpleNamespace(
        vocab_size=vocab_size,
        top_k=_state_array([50] * rows, torch.int32),
        top_p=_state_array([1.0] * rows, torch.float32),
        min_p=_state_array([0.0] * rows, torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(sampling_states=sampling_states)
    mapping = torch.arange(rows, dtype=torch.int64)
    mapping_np = np.arange(rows, dtype=np.int64)

    sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        torch.randn((rows, vocab_size)),
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )

    assert isinstance(captured["top_k"], torch.Tensor)
    assert captured["top_p"] is None


def test_non_qwen_vocab_keeps_tensor_top_k_path(monkeypatch) -> None:
    captured = {}

    def fake_apply_top_k_top_p(logits, top_k, top_p):
        captured.update(top_k=top_k, top_p=top_p)
        return logits

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    rows = 4
    vocab_size = 131072
    sampling_states = SimpleNamespace(
        vocab_size=vocab_size,
        top_k=_state_array([50] * rows, torch.int32),
        top_p=_state_array([1.0] * rows, torch.float32),
        min_p=_state_array([0.0] * rows, torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(
        sampling_states=sampling_states, _musa_qwen_family=True
    )
    mapping = torch.arange(rows, dtype=torch.int64)
    mapping_np = np.arange(rows, dtype=np.int64)

    sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        torch.randn((rows, vocab_size)),
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )

    assert isinstance(captured["top_k"], torch.Tensor)


def test_seeded_bs1_active_top_p_keeps_tensor_path(monkeypatch) -> None:
    captured = {}

    def fake_apply_top_k_top_p(logits, top_k, top_p):
        captured.update(top_k=top_k, top_p=top_p)
        return logits

    monkeypatch.setattr(sampler, "_apply_top_k_top_p", fake_apply_top_k_top_p)
    vocab_size = 151936
    sampling_states = SimpleNamespace(
        vocab_size=vocab_size,
        top_k=_state_array([50], torch.int32),
        top_p=_state_array([0.9], torch.float32),
        min_p=_state_array([0.0], torch.float32),
        apply_min_p=lambda *_args: None,
    )
    fake_sampler = SimpleNamespace(sampling_states=sampling_states)
    mapping = torch.zeros(1, dtype=torch.int64)
    mapping_np = np.zeros(1, dtype=np.int64)

    sampler._apply_worker_sampling_filters_for_seeded_multinomial(
        fake_sampler,
        torch.randn((1, vocab_size)),
        mapping,
        mapping_np,
        preserve_processed_logits=False,
    )

    assert isinstance(captured["top_k"], torch.Tensor)
    assert isinstance(captured["top_p"], torch.Tensor)


def test_uniform_top_k_top_p_prefilter_matches_upstream() -> None:
    torch.manual_seed(60043)
    logits = torch.randn((4, 256), dtype=torch.float32)
    top_p = torch.tensor([0.75, 0.8, 0.9, 0.95], dtype=torch.float32)
    expected = sampler.vllm_topk_topp_sampler.apply_top_k_top_p_pytorch(
        logits.clone(), torch.full((4,), 50, dtype=torch.int32), top_p
    )

    actual_input = logits.clone()
    actual = sampler._apply_top_k_top_p_musa_uniform_k_prefilter(
        actual_input, 50, top_p
    )

    assert actual.data_ptr() == actual_input.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.isfinite(), expected.isfinite())


def test_uniform_top_k_top_p_prefilter_falls_back_for_boundary_ties() -> None:
    logits = torch.arange(64, dtype=torch.float32).repeat(2, 1)
    logits[:, 12:17] = 14.0
    top_p = torch.tensor([0.9, 0.95], dtype=torch.float32)
    expected = sampler.vllm_topk_topp_sampler.apply_top_k_top_p_pytorch(
        logits.clone(), torch.full((2,), 50, dtype=torch.int32), top_p
    )

    actual = sampler._apply_top_k_top_p_musa_uniform_k_prefilter(
        logits.clone(), 50, top_p
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.isfinite(), expected.isfinite())


def test_unseeded_uniform_min_p_reaches_sampler_as_scalar(monkeypatch) -> None:
    captured = {}

    def fake_sample_from_logits(logits, top_k, top_p, min_p):
        captured.update(top_k=top_k, top_p=top_p, min_p=min_p)
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(sampler, "sample_from_logits", fake_sample_from_logits)
    sampling_states = SimpleNamespace(
        vocab_size=128,
        top_k=_state_array([128, 128], torch.int32),
        top_p=_state_array([1.0, 1.0], torch.float32),
        min_p=_state_array([0.05, 0.05], torch.float32),
    )
    mapping = torch.tensor([0, 1], dtype=torch.int64)
    mapping_np = np.asarray([0, 1], dtype=np.int64)

    sampler.sample_worker_logits(
        torch.randn((2, 128)), sampling_states, mapping, mapping_np, False
    )

    assert captured == {"top_k": None, "top_p": None, "min_p": pytest.approx(0.05)}


def test_unseeded_mixed_min_p_keeps_tensor_fallback(monkeypatch) -> None:
    captured = {}

    def fake_sample_from_logits(logits, top_k, top_p, min_p):
        captured["min_p"] = min_p
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(sampler, "sample_from_logits", fake_sample_from_logits)
    sampling_states = SimpleNamespace(
        vocab_size=128,
        top_k=_state_array([128, 128], torch.int32),
        top_p=_state_array([1.0, 1.0], torch.float32),
        min_p=_state_array([0.05, 0.1], torch.float32),
    )
    mapping = torch.tensor([0, 1], dtype=torch.int64)
    mapping_np = np.asarray([0, 1], dtype=np.int64)

    sampler.sample_worker_logits(
        torch.randn((2, 128)), sampling_states, mapping, mapping_np, False
    )

    assert isinstance(captured["min_p"], torch.Tensor)
    torch.testing.assert_close(
        captured["min_p"], torch.tensor([0.05, 0.1]), rtol=0, atol=0
    )


@requires_musa
@pytest.mark.parametrize("vocab_size", [151936, 248320])
@pytest.mark.parametrize("rows", [1, 4, 16, 64])
def test_uniform_min_p_scalar_preserves_tokens_and_generator_state(
    vocab_size: int, rows: int
) -> None:
    torch.manual_seed(60043 + rows)
    probs = torch.softmax(
        torch.randn((rows, vocab_size), device="musa", dtype=torch.float32), dim=-1
    )
    probs = sampler._ops.top_k_renorm_probs(probs, 50)
    scalar_generator = torch.Generator(device="musa").manual_seed(5678)
    tensor_generator = torch.Generator(device="musa").manual_seed(5678)

    scalar_tokens = sampler._ops.min_p_sampling_from_probs(
        probs, 0.05, generator=scalar_generator
    )
    tensor_tokens = sampler._ops.min_p_sampling_from_probs(
        probs,
        torch.full((rows,), 0.05, device="musa", dtype=torch.float32),
        generator=tensor_generator,
    )
    torch.musa.synchronize()

    assert torch.equal(scalar_tokens, tensor_tokens)
    assert torch.equal(scalar_generator.get_state(), tensor_generator.get_state())
