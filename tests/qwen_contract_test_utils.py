from types import SimpleNamespace

from vllm_musa.optimization_contract import resolve_optimization_contract


def qwen_sampler(*, enabled: bool = True, legacy: bool = True):
    architecture = "Qwen3ForCausalLM" if enabled else "LlamaForCausalLM"
    model_type = "qwen3" if enabled else "llama"
    contract = resolve_optimization_contract(
        SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=[architecture],
                hf_text_config=SimpleNamespace(
                    model_type=model_type,
                    vocab_size=151936,
                ),
                dtype="bfloat16",
                quantization=None,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
            speculative_config=None,
            quant_config=None,
        )
    )
    if not legacy:
        contract = resolve_optimization_contract(
            SimpleNamespace(
                model_config=SimpleNamespace(
                    architectures=["Qwen3_5ForCausalLM"],
                    hf_text_config=SimpleNamespace(
                        model_type="qwen3_5_text",
                        vocab_size=248320,
                    ),
                    dtype="bfloat16",
                    quantization=None,
                )
            )
        )
    return SimpleNamespace(_musa_optimization_contract=contract)


def qwen_hybrid_contract():
    return resolve_optimization_contract(
        SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=["Qwen3_5ForConditionalGeneration"],
                hf_text_config=SimpleNamespace(
                    model_type="qwen3_5_text",
                    vocab_size=248320,
                ),
                dtype="bfloat16",
                quantization=None,
                is_hybrid=True,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                decode_context_parallel_size=1,
            ),
        )
    )
