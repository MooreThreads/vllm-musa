# SPDX-License-Identifier: Apache-2.0
"""MUSA cat-6 object patch for the vLLM compilation backend.

Accept torch.compile backend keyword options on this vLLM snapshot and apply
the exact Qwen-family rewrites before vLLM splits the raw FX graph.
"""

from functools import wraps
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

PATCHES: list = []


def _try_qwen2_rope_kv_presplit(backend, graph) -> int:
    vllm_config = getattr(backend, "vllm_config", None)
    compilation_config = getattr(backend, "compilation_config", None)
    if vllm_config is None or compilation_config is None:
        return 0

    from vllm_musa.compilation import qwen2_rope_kv_presplit as presplit
    from vllm_musa.optimization_contract import policy
    from vllm_musa.optimization_contract.types import OptimizationFeature

    if not policy.prefers_feature(
        vllm_config, OptimizationFeature.QWEN2_ROPE_KV_PRESPLIT
    ):
        return 0

    if not presplit.qwen2_rope_kv_backend_supported(vllm_config):
        logger.warning_once(
            "MUSA Qwen2 RoPE+KV fusion requires all 24 attention layers to "
            "use the MUSA FlashAttention3 fused-cache implementation; "
            "keeping the baseline split graph."
        )
        return 0

    splitting_ops = compilation_config.splitting_ops
    if (
        compilation_config.use_inductor_graph_partition
        or splitting_ops is None
        or presplit.KV_UPDATE_SPLITTING_OP not in splitting_ops
    ):
        logger.warning_once(
            "MUSA Qwen2 RoPE+KV fusion requires the baseline Dynamo KV-cache "
            "split; keeping the unfused graph."
        )
        return 0

    candidates = presplit.plan_qwen2_rope_kv_presplit(graph)
    if candidates is None:
        logger.warning_once(
            "MUSA Qwen2 RoPE+KV fusion did not find the exact 24-layer raw FX "
            "graph; keeping the baseline split graph."
        )
        return 0

    matched = presplit.apply_qwen2_rope_kv_presplit(graph, candidates)
    if presplit.FUSED_SPLITTING_OP not in splitting_ops:
        compilation_config.splitting_ops = [
            *splitting_ops,
            presplit.FUSED_SPLITTING_OP,
        ]

    compilation_config.traced_files.update(
        {
            str(Path(__file__).resolve()),
            str(Path(presplit.__file__).resolve()),
        }
    )
    logger.info_once(
        "Applied the MUSA Qwen2 RoPE+KV pre-split fusion to %d layers.", matched
    )
    return matched


def _try_qwen3_qk_rope_kv_presplit(backend, graph) -> int:
    vllm_config = getattr(backend, "vllm_config", None)
    compilation_config = getattr(backend, "compilation_config", None)
    if vllm_config is None or compilation_config is None:
        return 0

    from vllm_musa.compilation import qwen3_qk_rope_kv_presplit as presplit
    from vllm_musa.optimization_contract import policy
    from vllm_musa.optimization_contract.types import OptimizationFeature

    if not policy.prefers_feature(
        vllm_config, OptimizationFeature.QWEN3_QK_ROPE_KV_PRESPLIT
    ):
        return 0
    model_config = getattr(vllm_config, "model_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is None:
        hf_config = getattr(model_config, "hf_config", None)
        hf_text_config = getattr(hf_config, "text_config", hf_config)
    expected_sites = getattr(hf_text_config, "num_hidden_layers", None)
    if not isinstance(expected_sites, int) or expected_sites <= 0:
        logger.warning_once(
            "MUSA Qwen3 QK-RoPE-KV fusion skipped: missing layer count."
        )
        return 0
    if not presplit.qwen3_qk_rope_kv_backend_supported(vllm_config, expected_sites):
        logger.warning_once(
            "MUSA Qwen3 QK-RoPE-KV fusion requires every attention layer "
            "to use the exact FlashAttention3 BF16/NHD implementation; "
            "keeping the baseline split graph."
        )
        return 0

    splitting_ops = compilation_config.splitting_ops
    if (
        compilation_config.use_inductor_graph_partition
        or splitting_ops is None
        or presplit.KV_UPDATE_SPLITTING_OP not in splitting_ops
    ):
        logger.warning_once(
            "MUSA Qwen3 QK-RoPE-KV fusion requires the baseline Dynamo "
            "KV-cache split; keeping the unfused graph."
        )
        return 0

    candidates = presplit.plan_qwen3_qk_rope_kv_presplit(graph, expected_sites)
    if candidates is None:
        logger.warning_once(
            "MUSA Qwen3 QK-RoPE-KV fusion did not find the exact all-layer "
            "raw FX graph; keeping the baseline split graph."
        )
        return 0
    matched = presplit.apply_qwen3_qk_rope_kv_presplit(graph, candidates)
    if presplit.FUSED_SPLITTING_OP not in splitting_ops:
        compilation_config.splitting_ops = [
            *splitting_ops,
            presplit.FUSED_SPLITTING_OP,
        ]
    compilation_config.traced_files.update(
        {
            str(Path(__file__).resolve()),
            str(Path(presplit.__file__).resolve()),
        }
    )
    logger.info_once(
        "Applied the MUSA Qwen3 QK-RoPE-KV pre-split fusion to %d layers.",
        matched,
    )
    return matched


def apply() -> None:
    try:
        from vllm.compilation.backends import VllmBackend
    except Exception as e:
        logger.debug("Skipping VllmBackend options patch: %s", e)
        return

    original_call = VllmBackend.__call__
    if getattr(original_call, "_musa_accepts_backend_options", False):
        return

    @wraps(original_call)
    def call_with_ignored_options(self, graph, example_inputs, **kwargs):
        _try_qwen2_rope_kv_presplit(self, graph)
        _try_qwen3_qk_rope_kv_presplit(self, graph)
        return original_call(self, graph, example_inputs)

    call_with_ignored_options._musa_accepts_backend_options = True
    VllmBackend.__call__ = call_with_ignored_options
