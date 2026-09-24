from types import SimpleNamespace

import pytest

from vllm_musa.optimization_contract import (
    OptimizationFeature,
    resolve_glm_contract,
    resolve_optimization_contract,
)


def _config(
    *,
    architecture="GlmMoeDsaForCausalLM",
    model_type="glm_moe_dsa",
    dtype="bfloat16",
    quantization="fp8",
    use_mla=True,
    index_topk=2048,
):
    text = SimpleNamespace(
        architectures=[architecture],
        model_type=model_type,
        hidden_size=4096,
        intermediate_size=13696,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=128,
        index_topk=index_topk,
        quantization_config={"quant_method": quantization},
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=[architecture],
            model_type=model_type,
            hf_text_config=text,
            dtype=dtype,
            quantization=quantization,
            use_mla=use_mla,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=2,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        scheduler_config=SimpleNamespace(max_num_seqs=1, async_scheduling=True),
        speculative_config=None,
        quant_config=None,
    )


def test_glm_contract_prefers_both_glm_features() -> None:
    contract = resolve_optimization_contract(_config())
    assert contract.prefers(OptimizationFeature.GLM5_EAGER_MCCL_INIT)
    assert contract.prefers(OptimizationFeature.GLM5_SPARSE_MLA_MATE_PREFILL)


@pytest.mark.parametrize(
    "override",
    [
        {"dtype": "float16"},
        {"quantization": "compressed-tensors"},
        {"use_mla": False},
        {"index_topk": 1024},
    ],
)
def test_glm_mate_sparse_prefill_requires_full_signature(override) -> None:
    contract = resolve_optimization_contract(_config(**override))
    assert contract.prefers(OptimizationFeature.GLM5_EAGER_MCCL_INIT)
    assert not contract.prefers(OptimizationFeature.GLM5_SPARSE_MLA_MATE_PREFILL)


def test_non_glm_contract_fails_closed() -> None:
    contract = resolve_optimization_contract(
        _config(architecture="Qwen3ForCausalLM", model_type="qwen3")
    )
    assert not contract.prefers(OptimizationFeature.GLM5_EAGER_MCCL_INIT)
    assert not contract.prefers(OptimizationFeature.GLM5_SPARSE_MLA_MATE_PREFILL)
    assert resolve_glm_contract(contract.model, contract.execution) is None
