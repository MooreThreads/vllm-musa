"""A heterogeneous per-layer config must not abort the contract resolver.

transformers raises AmbiguousGlobalPerLayerAttributeError, a RuntimeError, for a
per-layer attribute that varies across layers; getattr's default does not cover
it and engine startup would abort before the contract fails closed.
"""

from types import SimpleNamespace

import pytest
from transformers.integrations.heterogeneity import (
    AmbiguousGlobalPerLayerAttributeError,
)

from vllm_musa.optimization_contract import (
    ModelFamily,
    OptimizationFeature,
    resolve_optimization_contract,
)

# The attributes a heterogeneous config may serve directly; every other read is
# per-layer geometry and must go through config_attr.
HOMOGENEOUS = frozenset(
    {
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "vocab_size",
        "layer_types",
        "error",
        "__dict__",
        "__class__",
    }
)


class _Subclassed(AmbiguousGlobalPerLayerAttributeError):
    pass


class _TextConfig(SimpleNamespace):
    error: type = AmbiguousGlobalPerLayerAttributeError

    def __getattribute__(self, name: str):
        if name in HOMOGENEOUS:
            return super().__getattribute__(name)
        raise self.error(f"{name!r} is a per-layer attribute")


def _config(error: type = AmbiguousGlobalPerLayerAttributeError) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["DiffusionGemmaForBlockDiffusion"],
            model_type="diffusion_gemma",
            hf_text_config=_TextConfig(
                error=error,
                architectures=["DiffusionGemmaForBlockDiffusion"],
                model_type="diffusion_gemma",
                hidden_size=2816,
                intermediate_size=2112,
                num_hidden_layers=26,
                vocab_size=262144,
                layer_types=["sliding_attention", "full_attention"],
            ),
            dtype="bfloat16",
            quantization=None,
            use_mla=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        scheduler_config=SimpleNamespace(max_num_seqs=1, async_scheduling=True),
        speculative_config=None,
        quant_config=None,
    )


@pytest.mark.parametrize("error", [AmbiguousGlobalPerLayerAttributeError, _Subclassed])
def test_per_layer_ambiguity_does_not_abort_the_resolver(error: type) -> None:
    contract = resolve_optimization_contract(_config(error))
    model = contract.model
    # The ambiguous geometry is missing, not guessed.
    assert model.num_key_value_heads is None
    assert model.num_attention_heads is None
    assert model.head_dim is None
    assert model.hidden_size == 2816
    assert model.num_hidden_layers == 26
    assert model.family is ModelFamily.UNKNOWN
    assert not contract.prefers(OptimizationFeature.GLM5_EAGER_MCCL_INIT)


def test_other_runtime_errors_propagate() -> None:
    class _Broken(SimpleNamespace):
        def __getattribute__(self, name: str):
            if name == "hidden_size":
                raise RuntimeError("unrelated")
            return super().__getattribute__(name)

    config = _config()
    config.model_config.hf_text_config = _Broken(hidden_size=2816)
    with pytest.raises(RuntimeError, match="unrelated"):
        resolve_optimization_contract(config)
