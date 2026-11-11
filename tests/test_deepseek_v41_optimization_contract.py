from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_musa.optimization_contract import (
    ModelFamily,
    ModelRole,
    OptimizationFeature,
    resolve_optimization_contract,
)

from .test_deepseek_v4_optimization_contract import _flash_base_config

FOLD = OptimizationFeature.DEEPSEEK_V41_SHARED_EXPERT_FOLD
W4AFP8_OVERLAP = OptimizationFeature.DEEPSEEK_V41_W4AFP8_SHARED_EXPERT_OVERLAP


def _v41_flash_config(
    *,
    tp: int = 8,
    speculative_method: str | None = None,
    attention_backend: str = "FLASHMLA",
    **text_overrides,
):
    """The DeepSeek-V4.1-Flash FP8 serving config as the resolver sees it."""
    text_config = SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        hidden_size=5120,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        vocab_size=129280,
        n_routed_experts=384,
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=2304,
        expert_dtype="fp8",
        hidden_act="silu",
        swiglu_limit=10.0,
        index_topk=512,
        quantization_config={"quant_method": "fp8", "weight_block_size": [32, 32]},
    )
    for key, value in text_overrides.items():
        setattr(text_config, key, value)
    model_config = SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        hf_text_config=text_config,
        hf_config=text_config,
        dtype="bfloat16",
        quantization="deepseek_v4_fp8",
        use_mla=True,
        is_hybrid=False,
        is_moe=True,
        enforce_eager=False,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8", block_size=64),
        scheduler_config=SimpleNamespace(
            max_num_seqs=64, max_num_batched_tokens=8192, async_scheduling=True
        ),
        attention_config=SimpleNamespace(backend=attention_backend),
        compilation_config=SimpleNamespace(
            mode="NONE", cudagraph_mode="FULL_DECODE_ONLY"
        ),
        speculative_config=(
            SimpleNamespace(method=speculative_method, num_speculative_tokens=4)
            if speculative_method is not None
            else None
        ),
        quant_config=SimpleNamespace(weight_block_size=[32, 32]),
    )


def _v41_w4afp8_config(*, expert_dtype="fp4", **overrides):
    config = _v41_flash_config(expert_dtype=expert_dtype, **overrides)
    text_config = config.model_config.hf_text_config
    text_config.quantization_config = {
        "quant_method": "w4afp8",
        "expert_dtype": expert_dtype,
        "weight_block_size": [32, 32],
        "moe_group_size": 128,
    }
    config.model_config.quantization = "w4afp8"
    config.parallel_config.enable_expert_parallel = True
    return config


def test_v41_flash_tp8_prefers_the_shared_expert_fold() -> None:
    contract = resolve_optimization_contract(_v41_flash_config())

    assert contract.model.family is ModelFamily.DEEPSEEK_V41
    assert contract.model.role is ModelRole.TEXT
    assert contract.profile == "deepseek_v4.1.tp8_flash"
    assert contract.prefers(FOLD)
    assert contract.preferred_features == {FOLD}


@pytest.mark.parametrize("expert_dtype", ["int4", "fp4"])
def test_v41_w4afp8_ep_prefers_external_shared_expert_overlap(expert_dtype) -> None:
    contract = resolve_optimization_contract(
        _v41_w4afp8_config(expert_dtype=expert_dtype)
    )

    assert contract.model.family is ModelFamily.DEEPSEEK_V41
    assert contract.supported_features == {W4AFP8_OVERLAP}
    assert contract.prefers(W4AFP8_OVERLAP)
    assert not contract.prefers(FOLD)


@pytest.mark.parametrize("expert_dtype", ["int4", "fp4"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 4),
        ("enable_expert_parallel", False),
        ("data_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("enable_eplb", True),
        ("enable_dbo", True),
    ],
)
def test_v41_w4afp8_overlap_fails_closed_on_unvalidated_execution(
    expert_dtype, field, value
) -> None:
    config = _v41_w4afp8_config(expert_dtype=expert_dtype)
    setattr(config.parallel_config, field, value)

    contract = resolve_optimization_contract(config)

    assert contract.supports(W4AFP8_OVERLAP)
    assert not contract.prefers(W4AFP8_OVERLAP)


@pytest.mark.parametrize("expert_dtype", ["fp8", "int8", None])
def test_v41_w4afp8_rejects_other_expert_dtypes(expert_dtype) -> None:
    contract = resolve_optimization_contract(
        _v41_w4afp8_config(expert_dtype=expert_dtype)
    )

    assert not contract.supports(W4AFP8_OVERLAP)
    assert not contract.supports(FOLD)


@pytest.mark.parametrize("quantization", ["fp8", "deepseek_v4_fp8", "w4afp8"])
@pytest.mark.parametrize("expert_dtype", ["fp8", "int4", "fp4"])
def test_v41_quantization_and_expert_dtype_must_match(
    quantization, expert_dtype
) -> None:
    config = _v41_w4afp8_config(expert_dtype=expert_dtype)
    config.model_config.quantization = quantization
    config.model_config.hf_text_config.quantization_config["quant_method"] = (
        quantization
    )
    contract = resolve_optimization_contract(config)

    assert contract.supports(FOLD) == (
        quantization in ("fp8", "deepseek_v4_fp8") and expert_dtype == "fp8"
    )
    assert contract.supports(W4AFP8_OVERLAP) == (
        quantization == "w4afp8" and expert_dtype in ("int4", "fp4")
    )


def test_v41_dspark_target_and_draft_prefer_the_fold() -> None:
    target = _v41_flash_config(speculative_method="dspark")
    contract = resolve_optimization_contract(target)
    assert contract.profile == "deepseek_v4.1.tp8_flash_dspark"
    assert contract.prefers(FOLD)

    # The DSpark draft keeps the target model config and swaps the attention backend.
    draft = SimpleNamespace(
        **{
            **vars(target),
            "attention_config": SimpleNamespace(backend="FLASHMLA_SPARSE_DSV41"),
        }
    )
    assert resolve_optimization_contract(draft).prefers(FOLD)


@pytest.mark.parametrize(
    "config",
    [
        _v41_flash_config(tp=4),
        _v41_flash_config(speculative_method="mtp"),
        _v41_flash_config(speculative_method="eagle"),
    ],
)
def test_v41_fold_is_preferred_only_on_the_validated_execution(config) -> None:
    contract = resolve_optimization_contract(config)

    assert contract.model.family is ModelFamily.DEEPSEEK_V41
    assert contract.supports(FOLD)
    assert not contract.prefers(FOLD)
    assert contract.profile == "deepseek_v4.1.unvalidated"


def test_v41_fold_skips_parallel_layouts_and_runtimes_it_was_not_measured_on() -> None:
    for name, value in (
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 2),
        ("decode_context_parallel_size", 2),
    ):
        config = _v41_flash_config()
        setattr(config.parallel_config, name, value)
        assert not resolve_optimization_contract(config).prefers(FOLD), name

    no_quant = _v41_flash_config()
    no_quant.quant_config = None
    assert not resolve_optimization_contract(no_quant).prefers(FOLD)
    assert not resolve_optimization_contract(
        _v41_flash_config(), is_pooling_model=True
    ).prefers(FOLD)


def test_v41_fold_is_off_under_batch_invariance(monkeypatch) -> None:
    fake_vllm = ModuleType("vllm")
    fake_envs = ModuleType("vllm.envs")
    fake_envs.VLLM_BATCH_INVARIANT = True
    fake_vllm.envs = fake_envs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", fake_envs)

    contract = resolve_optimization_contract(_v41_flash_config())

    assert contract.supports(FOLD)
    assert not contract.prefers(FOLD)


@pytest.mark.parametrize(
    "overrides",
    [
        {"hidden_size": 4096},
        {"n_routed_experts": 256},
        {"num_experts_per_tok": 8},
        {"moe_intermediate_size": 2048},
        {"expert_dtype": "fp4"},
        {"swiglu_limit": None},
        {
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
        },
    ],
)
def test_v41_fold_fails_closed_on_other_model_traits(overrides) -> None:
    config = _v41_flash_config(**overrides)
    if "quantization_config" in overrides:
        config.quant_config = SimpleNamespace(weight_block_size=[128, 128])
    contract = resolve_optimization_contract(config)

    assert contract.model.family is ModelFamily.DEEPSEEK_V41
    assert not contract.supports(FOLD)
    assert contract.profile == "deepseek_v4.1.unvalidated"


def test_v41_identity_must_be_complete() -> None:
    config = _v41_flash_config()
    config.model_config.architectures = ["DSparkV41DraftModel"]
    contract = resolve_optimization_contract(config)

    assert contract.model.family is ModelFamily.DEEPSEEK_V41
    assert not contract.supports(FOLD)


def test_deepseek_v4_flash_never_takes_the_fold() -> None:
    for config in (
        _flash_base_config(),
        _flash_base_config(speculative=True, speculative_method="dspark"),
        _flash_base_config(max_num_seqs=64),
    ):
        contract = resolve_optimization_contract(config)
        assert contract.model.family is ModelFamily.DEEPSEEK_V4
        assert not contract.supports(FOLD)


def test_v41_stays_outside_the_deepseek_v4_family() -> None:
    contract = resolve_optimization_contract(_v41_flash_config())

    assert contract.model.family is not ModelFamily.DEEPSEEK_V4
    assert contract.supported_features == {FOLD}
