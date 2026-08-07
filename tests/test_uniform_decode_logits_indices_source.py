# SPDX-License-Identifier: Apache-2.0
"""Source contract for uniform single-token decode logits indices."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0090-perf-musa-unify-Qwen-runtime-fast-paths.patch"
)
CONTRACT = ROOT / "vllm_musa" / "optimization_contract" / "qwen.py"


def test_uniform_decode_patch_reuses_uploaded_request_indices() -> None:
    source = PATCH.read_text()
    contract = CONTRACT.read_text()

    assert "current_platform.is_musa()" in source
    assert "not execution.has_speculative_config" in contract
    assert "not execution.is_pooling_model" in contract
    for architecture in (
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        assert architecture in contract
    assert "VLLM_MUSA_QWEN_UNIFORM_DECODE_LOGITS_INDICES" not in source
    assert "max_num_scheduled_tokens == 1" in source
    assert "num_tokens_unpadded == num_reqs" in source
    assert "torch.arange(self.max_num_reqs, dtype=torch.int32" in source
    assert source.count("self._musa_qwen_uniform_decode_logits_indices[") == 1
    assert "use_cached_decode_logits_indices: bool = False" in source
    assert "+                logits_indices = query_start_loc[1:] - 1" in source
    assert "query_start_loc path" in source


def test_uniform_decode_gate_is_exact_for_positive_query_lengths() -> None:
    cases = (
        [1],
        [1] * 4,
        [1] * 16,
        [1] * 64,
        [2],
        [1, 2],
        [4, 1, 1],
        [0, 2],
    )

    for query_lens in cases:
        total = sum(query_lens)
        num_reqs = len(query_lens)
        use_fast_path = max(query_lens) == 1 and total == num_reqs

        query_start_loc = [0]
        for query_len in query_lens:
            assert query_len >= 0
            query_start_loc.append(query_start_loc[-1] + query_len)
        baseline = [end - 1 for end in query_start_loc[1:]]
        request_indices = [
            req_idx
            for req_idx, query_len in enumerate(query_lens)
            for _ in range(query_len)
        ]

        if use_fast_path:
            assert request_indices == baseline
        else:
            assert request_indices != baseline


def test_uniform_decode_hidden_view_patch_preserves_fallbacks() -> None:
    source = PATCH.read_text()

    assert source.count("+                if use_cached_decode_logits_indices:") == 2
    assert source.count("sample_hidden_states = hidden_states\n") == 2
    assert source.count("sample_hidden_states = hidden_states[:num_reqs]") == 2
    assert (
        source.count(
            "+                    sample_hidden_states = hidden_states[logits_indices]"
        )
        == 2
    )
    assert source.count("hidden_states.shape[0] != num_reqs") == 2


def test_identity_hidden_selection_preserves_padding_and_fallback_semantics() -> None:
    def select_hidden_states(
        hidden_states: list[str],
        num_reqs: int,
        logits_indices: list[int],
        use_cached_decode_logits_indices: bool,
    ) -> list[str]:
        if use_cached_decode_logits_indices:
            if len(hidden_states) == num_reqs:
                return hidden_states
            return hidden_states[:num_reqs]
        return [hidden_states[index] for index in logits_indices]

    for num_reqs, padded_reqs in ((1, 1), (4, 4), (16, 16), (63, 64), (64, 64)):
        hidden_states = [f"row-{index}" for index in range(padded_reqs)]
        selected = select_hidden_states(
            hidden_states,
            num_reqs,
            list(range(num_reqs)),
            use_cached_decode_logits_indices=True,
        )
        assert selected == hidden_states[:num_reqs]
        if num_reqs == padded_reqs:
            assert selected is hidden_states

    hidden_states = ["row-0", "row-1", "row-2", "row-3"]
    assert select_hidden_states(
        hidden_states,
        num_reqs=2,
        logits_indices=[3, 1],
        use_cached_decode_logits_indices=False,
    ) == ["row-3", "row-1"]
