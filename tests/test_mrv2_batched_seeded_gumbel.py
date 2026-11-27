# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit and source tests for MRV2 batched seeded Gumbel and Qwen2.5 152064 hints."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLER_PATH = REPO_ROOT / "vllm_musa" / "v1" / "sample" / "topk_topp_sampler.py"
PATCH_0082_PATH = (
    REPO_ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0082-MUSA-perf-musa-unify-Qwen-runtime-fast-paths.patch"
)


def test_patch_0082_includes_qwen25_152064_vocab_in_cpu_hints() -> None:
    patch_text = PATCH_0082_PATH.read_text(encoding="utf-8")
    assert patch_text.count("self.vocab_size in (151936, 152064, 248320)") == 2
    assert "self.vocab_size in (151936, 248320)" not in patch_text


def test_worker_sample_routes_seeded_batches_to_batched_gumbel_before_multinomial() -> (
    None
):
    source = SAMPLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "can_use_worker_seeded_gumbel" in fn_names
    assert "sample_worker_logits_seeded_gumbel" in fn_names

    worker_start = source.index("def _worker_sample(")
    worker_body = source[worker_start:]
    gumbel_pos = worker_body.index("can_use_worker_seeded_gumbel(")
    multinomial_pos = worker_body.index("can_use_worker_seeded_multinomial(")
    assert gumbel_pos < multinomial_pos


def test_add_request_evicts_cached_generator_without_eager_musa_alloc() -> None:
    source = SAMPLER_PATH.read_text(encoding="utf-8")
    start = source.index("def _sampling_states_add_request(")
    end = source.index("\ndef ", start + 1)
    body = source[start:end]
    assert "self.musa_generators.pop(req_idx, None)" in body
    assert 'torch.Generator(device="musa")' not in body


def test_mrv2_seeded_gumbel_runtime_gates_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm_musa.v1.sample import topk_topp_sampler as sampler

    monkeypatch.setattr(sampler.current_platform, "is_musa", lambda: True)
    monkeypatch.setattr(sampler, "is_musa_tensor", lambda _t: True)

    rows = 4
    # Test non-Qwen vocabs (Llama-3 / DeepSeek) and Qwen2.5 152064.
    for vocab_size in (102400, 128256, 152064, 248320):
        logits = torch.randn((rows, vocab_size), dtype=torch.bfloat16)
        mapping = torch.arange(rows, dtype=torch.int64)
        mapping_np = np.arange(rows, dtype=np.int64)
        pos = torch.tensor([5, 12, 0, 33], dtype=torch.int64)

        seeds_gpu = torch.tensor([101, 202, 303, 404], dtype=torch.int64)
        temp_gpu = torch.tensor([0.7, 1.0, 0.8, 0.9], dtype=torch.float32)
        states = SimpleNamespace(
            vocab_size=vocab_size,
            temperature=SimpleNamespace(
                np=np.asarray([0.7, 1.0, 0.8, 0.9], dtype=np.float32),
                gpu=temp_gpu,
            ),
            top_k=SimpleNamespace(
                np=np.asarray([50, vocab_size, 20, 50], dtype=np.int32),
                gpu=torch.tensor([50, vocab_size, 20, 50], dtype=torch.int32),
            ),
            top_p=SimpleNamespace(
                np=np.asarray([0.95, 1.0, 0.9, 1.0], dtype=np.float32),
                gpu=torch.tensor([0.95, 1.0, 0.9, 1.0], dtype=torch.float32),
            ),
            min_p=SimpleNamespace(
                np=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                gpu=torch.zeros(rows, dtype=torch.float32),
            ),
            seeds=SimpleNamespace(
                np=np.asarray([101, 202, 303, 404], dtype=np.int64),
                gpu=seeds_gpu,
            ),
            has_user_seed=np.asarray([True, False, True, False], dtype=np.bool_),
            musa_generators={},
            apply_temperature=lambda *_args: None,
            apply_min_p=lambda *_args: None,
        )
        fake_worker = SimpleNamespace(
            sampling_states=states,
            logprobs_mode="raw_logprobs",
            use_fp64_gumbel=False,
            needs_logits_processing=np.asarray([True, False, True, True], dtype=bool),
            logits_processors=[],
            thinking_budget_state=SimpleNamespace(apply=lambda *_args: None),
        )

        assert sampler.can_use_worker_seeded_gumbel(
            fake_worker, logits, mapping, mapping_np, pos
        )

        # Reject processed logprobs/logits and use_fp64_gumbel.
        fake_worker.logprobs_mode = "processed_logprobs"
        assert not sampler.can_use_worker_seeded_gumbel(
            fake_worker, logits, mapping, mapping_np, pos
        )
        fake_worker.logprobs_mode = "processed_logits"
        assert not sampler.can_use_worker_seeded_gumbel(
            fake_worker, logits, mapping, mapping_np, pos
        )
        fake_worker.logprobs_mode = "raw_logprobs"
        fake_worker.use_fp64_gumbel = True
        assert not sampler.can_use_worker_seeded_gumbel(
            fake_worker, logits, mapping, mapping_np, pos
        )
        fake_worker.use_fp64_gumbel = False

        captured: dict = {}

        def fake_gumbel_sample(
            sampled_logits,
            exp_map,
            temp,
            seeds,
            positions,
            _captured=captured,
            **kwargs,
        ):
            _captured.update(
                dtype=sampled_logits.dtype,
                shape=tuple(sampled_logits.shape),
                exp_map=exp_map,
                temp=temp,
                seeds=seeds,
                positions=positions,
                kwargs=kwargs,
            )
            return torch.arange(rows, dtype=torch.int64)

        monkeypatch.setattr(
            sampler.vllm_worker_sampler, "gumbel_sample", fake_gumbel_sample
        )
        monkeypatch.setattr(
            sampler,
            "sample_worker_logits_seeded_multinomial",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("must not fall back to O(B) multinomial")
            ),
        )

        out_tokens, out_logits = sampler._worker_sample(
            fake_worker,
            logits,
            mapping,
            mapping,
            mapping_np,
            pos,
            torch.zeros(rows, dtype=torch.int32),
            torch.zeros(rows, dtype=torch.int32),
            np.ones(rows, dtype=np.int32),
            return_logprobs=True,
        )
        assert out_tokens.tolist() == [0, 1, 2, 3]
        assert out_logits.dtype == torch.float32
        # Original bf16 logits preserved untouched for raw_logprobs!
        assert logits.dtype == torch.bfloat16
        assert captured["seeds"] is seeds_gpu
        assert captured["positions"] is pos
        assert captured["kwargs"] == {
            "apply_temperature": False,
            "is_drafting": False,
            "use_fp64": False,
        }

        # Zero-copy unfiltered seeded path when needs_logits_processing is all False
        fake_worker.needs_logits_processing[:] = False
        captured.clear()
        out_tokens_zc, out_logits_zc = sampler._worker_sample(
            fake_worker,
            logits,
            mapping,
            mapping,
            mapping_np,
            pos,
            torch.zeros(rows, dtype=torch.int32),
            torch.zeros(rows, dtype=torch.int32),
            np.ones(rows, dtype=np.int32),
            return_logprobs=False,
        )
        assert out_tokens_zc.tolist() == [0, 1, 2, 3]
        assert out_logits_zc is logits
        assert captured["dtype"] == torch.bfloat16
