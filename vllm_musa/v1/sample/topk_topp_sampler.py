# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Any

import numpy as np
import torch
import vllm.v1.sample.ops.topk_topp_sampler as vllm_topk_topp_sampler
import vllm.v1.sample.sampler as vllm_sample_sampler
import vllm.v1.worker.gpu.sample.sampler as vllm_worker_sampler
import vllm.v1.worker.gpu.sample.states as vllm_worker_states
from vllm.config.model import LogprobsMode
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.platforms import current_platform

from vllm_musa import _custom_ops as _ops
from vllm_musa.optimization_contract import (
    OptimizationFeature,
    prefers_optimization,
    resolve_optimization_contract,
)
from vllm_musa.utils.environ import envs

logger = logging.getLogger(__name__)

_SAMPLING_EPS = 1e-5
_MUSA_QWEN_SAMPLER_VOCAB_SIZES = frozenset((151936, 152064, 248320))
# Keep chunked-prefill and first-token response draining on the original path.
_QWEN_LEGACY_UNFILTERED_GUMBEL_MIN_ROWS = 16
_QWEN_LEGACY_UNFILTERED_GUMBEL_MIN_OFFSET = 64
_MUSA_QWEN_SHARDED_MIN_BATCH = 32


def _is_qwen_sampler_vocab(logits: torch.Tensor) -> bool:
    """Limit scalar sampler specializations to validated Qwen vocabularies."""
    return logits.ndim == 2 and logits.shape[1] in _MUSA_QWEN_SAMPLER_VOCAB_SIZES


def _is_qwen_sharded_logits(sampler: Any, logits: torch.Tensor) -> bool:
    if not getattr(sampler, "_musa_qwen_sharded_logits", False):
        return False
    global_vocab = getattr(sampler, "_musa_qwen_global_vocab_size", 0)
    tp_size = getattr(sampler, "_musa_qwen_tp_size", 0)
    return (
        logits.ndim == 2
        and global_vocab in _MUSA_QWEN_SAMPLER_VOCAB_SIZES
        and tp_size == get_tensor_model_parallel_world_size()
        and tp_size == 4
        and logits.shape[0] >= _MUSA_QWEN_SHARDED_MIN_BATCH
        and logits.shape[0] <= 64
        and logits.shape[1] * tp_size == global_vocab
        and logits.dtype == torch.bfloat16
        and logits.is_contiguous()
        and logits.stride(-1) == 1
    )


def _is_uniform_top_k_50(top_k: np.ndarray) -> bool:
    """Whether existing CPU sampling state selects the k=50 specialization."""
    return top_k.size > 0 and bool(np.all(top_k == 50))


def _uniform_active_min_p(min_p: np.ndarray) -> float | None:
    """Return a uniform active min-p value from the existing CPU state."""
    if min_p.size == 0 or not bool(np.all(min_p == min_p[0])):
        return None
    value = float(min_p[0])
    return value if value != 0.0 else None


def _can_skip_legacy_qwen_unit_temperature(
    sampler: Any, logits: torch.Tensor, sampling_metadata: Any
) -> bool:
    """Use the scheduler's exact CPU hint to avoid a device divide by one."""
    return (
        prefers_optimization(sampler, OptimizationFeature.QWEN_LEGACY_SAMPLING)
        and getattr(sampling_metadata, "all_random", False)
        and _is_qwen_sampler_vocab(logits)
        and getattr(sampling_metadata, "uniform_temperature", None) == np.float32(1.0)
    )


def musa_seeded_multinomial_enabled() -> bool:
    return envs.VLLM_MUSA_SEEDED_MULTINOMIAL.get()


def is_musa_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "musa"


def can_use_musa_sampler(
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    logprobs_mode: LogprobsMode,
) -> bool:
    if not current_platform.is_musa() or not is_musa_tensor(logits):
        return False
    if generators:
        return False
    return logprobs_mode not in ("processed_logits", "processed_logprobs")


def can_use_musa_seeded_multinomial(
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    logprobs_mode: LogprobsMode,
) -> bool:
    """Gate the legacy per-request multinomial path to validated vocabularies."""
    return (
        bool(generators)
        and musa_seeded_multinomial_enabled()
        and current_platform.is_musa()
        and is_musa_tensor(logits)
        # Keep non-Qwen and small codec vocabularies on the upstream sampler.
        and _is_qwen_sampler_vocab(logits)
        and logprobs_mode not in ("processed_logits", "processed_logprobs")
    )


def _squeeze_filter_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim > 1 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    return value.contiguous()


def sample_from_probs(
    probs: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
    min_p: torch.Tensor | float | None = None,
) -> torch.Tensor:
    top_k = _squeeze_filter_tensor(top_k) if isinstance(top_k, torch.Tensor) else top_k
    top_p = _squeeze_filter_tensor(top_p) if isinstance(top_p, torch.Tensor) else top_p
    min_p = _squeeze_filter_tensor(min_p) if isinstance(min_p, torch.Tensor) else min_p

    # Sampling metadata already uses valid sentinel values for inactive
    # filters (vocab_size, 1.0, and 0.0). Keep those tensors on device and let
    # the native sampler consume them instead of calling .item(), which
    # synchronizes the host with the preceding decode graph.
    use_top_k = top_k is not None
    use_top_p = top_p is not None
    use_min_p = min_p is not None

    if use_min_p:
        if use_top_k:
            probs = _ops.top_k_renorm_probs(probs, top_k)
        if use_top_p:
            probs = _ops.top_p_renorm_probs(probs, top_p)
        return _ops.min_p_sampling_from_probs(probs, min_p).long().view(-1)

    if use_top_k and use_top_p:
        return (
            _ops.top_k_top_p_sampling_from_probs(
                probs, top_k, top_p, filter_apply_order="joint"
            )
            .long()
            .view(-1)
        )

    if use_top_k:
        return (
            _ops.top_k_top_p_sampling_from_probs(
                probs, top_k, 1.0, filter_apply_order="joint"
            )
            .long()
            .view(-1)
        )

    if use_top_p:
        return _ops.top_p_sampling_from_probs(probs, top_p).long().view(-1)

    return _ops.top_p_sampling_from_probs(probs, 1.0).long().view(-1)


def sample_from_logits(
    logits: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
    min_p: torch.Tensor | float | None = None,
) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
    return sample_from_probs(probs, top_k, top_p, min_p)


def sample_probs_seeded_multinomial(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    samples = []
    for row_idx in range(probs.shape[0]):
        generator = generators.get(row_idx)
        if generator is None:
            sample = torch.multinomial(probs[row_idx], num_samples=1, replacement=True)
        else:
            sample = torch.multinomial(
                probs[row_idx],
                num_samples=1,
                replacement=True,
                generator=generator,
            )
        samples.append(sample)
    return torch.cat(samples, dim=0).to(dtype=torch.long).view(-1)


def _apply_top_k_top_p_musa_topk_prefilter(
    logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor
) -> torch.Tensor:
    vocab_size = logits.shape[1]
    k_long = k.to(torch.long)
    if bool((k_long >= vocab_size).any().item()):
        return vllm_topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)

    max_top_k = int(k_long.max().item())
    if max_top_k <= 0 or max_top_k > 1024:
        return vllm_topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)

    values, indices = logits.topk(max_top_k, dim=-1, largest=True, sorted=True)
    gather_idx = (k_long.clamp_min(1) - 1).unsqueeze(1)
    threshold = values.gather(1, gather_idx)

    num_ge = (logits >= threshold).sum(dim=-1)
    if bool((num_ge != k_long).any().item()):
        return vllm_topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)

    valid = torch.arange(max_top_k, device=logits.device).unsqueeze(0)
    values = values.masked_fill(~(valid < k_long.unsqueeze(1)), -float("inf"))
    logits_sort = values.flip(dims=(-1,))
    logits_idx = indices.flip(dims=(-1,))

    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # ``values`` and ``indices`` do not alias ``logits``. Reuse the input
    # buffer instead of allocating a full-vocabulary output and copying it
    # back after the scatter.
    logits.fill_(-float("inf"))
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def _apply_top_k_top_p_musa_uniform_k_prefilter(
    logits: torch.Tensor, k: int, p: torch.Tensor
) -> torch.Tensor:
    """Apply uniform top-k/top-p without reading sampling state from MUSA.

    The CPU sampling state has already proved that every active row uses the
    same ``k``. The only host read is the post-topk tie check: it preserves the
    upstream rule that all values tied at the kth boundary remain eligible.
    """
    values, indices = logits.topk(k, dim=-1, largest=True, sorted=True)
    threshold = values[:, -1:]
    num_ge = (logits >= threshold).sum(dim=-1)
    if bool((num_ge != k).any().item()):
        k_tensor = torch.full(
            (logits.shape[0],), k, dtype=torch.int32, device=logits.device
        )
        return vllm_topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k_tensor, p)

    logits_sort = values.flip(dims=(-1,))
    logits_idx = indices.flip(dims=(-1,))
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    logits.fill_(-float("inf"))
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def _apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | int | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    # Uniform k=50 is already known from CPU sampling state. Keep the exact
    # upstream threshold/tie semantics without materializing a device k tensor
    # or synchronizing it back to the CPU before submitting topk.
    if isinstance(k, int):
        if k <= 0 or k >= logits.shape[1]:
            return logits
        if p is None:
            threshold = logits.topk(k, dim=1).values[:, -1:]
            return logits.masked_fill_(logits < threshold, -float("inf"))
        return _apply_top_k_top_p_musa_uniform_k_prefilter(logits, k, p)

    if current_platform.is_musa():
        if isinstance(k, torch.Tensor) and logits.shape[0] >= 16:
            if p is None and logits.shape[1] >= 65536:
                max_top_k = int(k.to(torch.long).max().item())
                if 0 < max_top_k <= 1024:
                    return vllm_topk_topp_sampler.apply_top_k_only(logits, k)
            elif logits.shape[1] >= 65536:
                return _apply_top_k_top_p_musa_topk_prefilter(logits, k, p)

    if (
        vllm_topk_topp_sampler.HAS_TRITON
        and logits.shape[0] >= 8
        and not current_platform.is_musa()
    ):
        return vllm_topk_topp_sampler.apply_top_k_top_p_triton(logits, k, p)

    return vllm_topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


def forward_musa(
    self: Any,
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    k: torch.Tensor | int | None,
    p: torch.Tensor | float | None,
    min_p: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if can_use_musa_seeded_multinomial(logits, generators, self.logprobs_mode):
        if min_p is not None:
            return self.forward_native(logits, generators, k, p)
        logits = _apply_top_k_top_p(logits, k, p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        vllm_topk_topp_sampler.logger.info_once(
            "Using MUSA seeded multinomial sampling for per-request generators.",
            scope="global",
        )
        return sample_probs_seeded_multinomial(probs, generators), None

    if not can_use_musa_sampler(logits, generators, self.logprobs_mode):
        if generators:
            logger.debug(
                "MUSA native sampling ops do not support per-request generators; "
                "falling back to PyTorch-native sampling."
            )
        return self.forward_native(logits, generators, k, p)

    return sample_from_logits(logits, k, p, min_p), None


def _topk_topp_sampler_init(
    self: Any,
    logprobs_mode: LogprobsMode = "raw_logprobs",
    use_fp64_gumbel: bool = False,
):
    original_init = vllm_topk_topp_sampler.TopKTopPSampler._musa_original_init
    original_init(self, logprobs_mode, use_fp64_gumbel)
    self._musa_optimization_contract = resolve_optimization_contract()
    if (
        logprobs_mode not in ("processed_logits", "processed_logprobs")
        and current_platform.is_musa()
    ):
        vllm_topk_topp_sampler.logger.info_once(
            "Using MUSA native ops for top-p/top-k/min-p sampling.",
            scope="global",
        )
        self.forward = self.forward_musa


def is_min_p_logits_processor(processor: Any) -> bool:
    return processor.__class__.__name__ == "MinPLogitsProcessor"


def should_defer_min_p_processor(
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    logprobs_mode: LogprobsMode,
    processor: Any,
) -> bool:
    if not is_min_p_logits_processor(processor):
        return False
    if not getattr(processor, "min_p_count", 0):
        return False
    return can_use_musa_sampler(logits, generators, logprobs_mode)


def get_processor_min_p(processor: Any) -> torch.Tensor:
    min_p = getattr(processor, "min_p")
    return _squeeze_filter_tensor(min_p)


def _legacy_min_p_with_cpu_hint(
    processor: Any,
    logits: torch.Tensor,
) -> torch.Tensor | float:
    min_p = get_processor_min_p(processor)
    min_p_cpu = getattr(processor, "min_p_cpu", None)
    if min_p_cpu is None or not _is_qwen_sampler_vocab(logits):
        return min_p
    uniform_min_p = _uniform_active_min_p(min_p_cpu[: logits.shape[0]])
    return uniform_min_p if uniform_min_p is not None else min_p


def _legacy_top_k_with_cpu_hint(
    sampling_metadata: Any,
    top_k: torch.Tensor | None,
    topk_topp_sampler: Any,
    logits: torch.Tensor,
    logprobs_mode: LogprobsMode,
) -> int | torch.Tensor | None:
    if (
        top_k is not None
        and getattr(topk_topp_sampler.forward, "__name__", "") == "forward_musa"
        and can_use_musa_sampler(
            logits,
            sampling_metadata.generators,
            logprobs_mode,
        )
        and _is_qwen_sampler_vocab(logits)
        and getattr(sampling_metadata, "uniform_top_k", None) == 50
    ):
        return 50
    return top_k


def _legacy_gumbel_top_k_with_cpu_hint(
    sampling_metadata: Any,
    logits: torch.Tensor,
    top_k: torch.Tensor | None,
) -> int | torch.Tensor | None:
    """Use the CPU-proven uniform k in the legacy seeded path.

    Legacy Gumbel still needs the top-k support mask, but it does not need the
    per-row device tensor when the scheduler already proved that every Qwen
    row uses k=50.  Passing the scalar keeps the fixed-k path from reading a
    device tensor back to the host for dispatch checks.  Heterogeneous and
    non-Qwen requests retain the original tensor path.
    """
    if getattr(
        sampling_metadata, "uniform_top_k", None
    ) == 50 and _is_qwen_sampler_vocab(logits):
        vllm_topk_topp_sampler.logger.info_once(
            "Using the CPU uniform top-k hint for gated MUSA legacy Gumbel.",
            scope="global",
        )
        return 50
    return top_k


def _call_topk_topp_sampler(
    sampler: Any,
    logprobs_mode: LogprobsMode,
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
    min_p: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    original_logprobs_mode = sampler.logprobs_mode
    sampler.logprobs_mode = logprobs_mode
    try:
        if min_p is None:
            return sampler(logits, generators, top_k, top_p)
        return sampler(logits, generators, top_k, top_p, min_p=min_p)
    finally:
        sampler.logprobs_mode = original_logprobs_mode


def _legacy_logits_processors_are_inactive(sampling_metadata: Any) -> bool:
    logitsprocs = getattr(sampling_metadata, "logitsprocs", None)
    if logitsprocs is None:
        return False
    for group_name in ("non_argmax_invariant", "argmax_invariant"):
        processors = getattr(logitsprocs, group_name, None)
        if processors is None:
            return False
        for processor in processors:
            name = processor.__class__.__name__
            if name == "MinPLogitsProcessor":
                if getattr(processor, "min_p_count", None) != 0:
                    return False
            elif name == "LogitBiasLogitsProcessor":
                biases = getattr(processor, "biases", None)
                if biases is None or biases:
                    return False
            elif name == "MinTokensLogitsProcessor":
                min_toks = getattr(processor, "min_toks", None)
                if min_toks is None or min_toks:
                    return False
            else:
                return False
    return True


def can_use_qwen_legacy_unfiltered_metadata(
    sampling_metadata: Any,
    logprobs_mode: LogprobsMode,
    predict_bonus_token: bool,
    use_fp64_gumbel: bool,
    is_qwen_family: bool = False,
) -> bool:
    """Gate raw-logits Gumbel using metadata available before lm-head."""
    if not is_qwen_family or not current_platform.is_musa():
        return False
    if predict_bonus_token or use_fp64_gumbel or logprobs_mode != "raw_logprobs":
        return False
    if (
        not getattr(sampling_metadata, "all_random", False)
        or getattr(sampling_metadata, "all_greedy", True)
        or getattr(sampling_metadata, "uniform_temperature", None) != np.float32(1.0)
    ):
        return False
    if (
        getattr(sampling_metadata, "top_k", object()) is not None
        or getattr(sampling_metadata, "top_p", object()) is not None
        or getattr(sampling_metadata, "generators", None) != {}
    ):
        return False
    if (
        getattr(sampling_metadata, "max_num_logprobs", object()) is not None
        or getattr(sampling_metadata, "logprob_token_ids", None)
        or getattr(sampling_metadata, "no_penalties", False) is not True
        or getattr(sampling_metadata, "allowed_token_ids_mask", object()) is not None
        or getattr(sampling_metadata, "bad_words_token_ids", None) != {}
    ):
        return False

    spec_token_ids = getattr(sampling_metadata, "spec_token_ids", None)
    if spec_token_ids and any(spec_token_ids):
        return False
    holder = getattr(sampling_metadata, "thinking_budget_state_holder", None)
    if holder is not None:
        has_tracked_requests = getattr(holder, "has_tracked_requests", None)
        if not callable(has_tracked_requests) or has_tracked_requests():
            return False
    return _legacy_logits_processors_are_inactive(sampling_metadata)


def can_use_qwen_legacy_unfiltered_gumbel(
    logits: torch.Tensor,
    sampling_metadata: Any,
    logprobs_mode: LogprobsMode,
    predict_bonus_token: bool,
    use_fp64_gumbel: bool,
    is_qwen_family: bool = False,
    sampler: Any | None = None,
) -> bool:
    """Gate raw-logits Gumbel for Qwen models on the legacy GPU runner."""
    if not is_musa_tensor(logits):
        return False
    if not can_use_qwen_legacy_unfiltered_metadata(
        sampling_metadata,
        logprobs_mode,
        predict_bonus_token,
        use_fp64_gumbel,
        is_qwen_family,
    ):
        return False
    if logits.dtype != torch.bfloat16 or logits.shape[0] == 0:
        return False
    if sampler is not None and _is_qwen_sharded_logits(sampler, logits):
        return True
    return _is_qwen_sampler_vocab(logits) and logits.stride(-1) == 1


def _find_logits_processor(model: Any) -> tuple[Any, Any] | None:
    candidate = model
    for _ in range(3):
        processor = getattr(candidate, "logits_processor", None)
        lm_head = getattr(candidate, "lm_head", None)
        if processor is not None and lm_head is not None:
            return processor, lm_head
        candidate = getattr(candidate, "language_model", None)
        if candidate is None:
            break
    return None


def _musa_jit_pair_gather_available() -> bool:
    try:
        from vllm.distributed.parallel_state import get_tp_group

        from vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce import (
            MusaJitCustomAllreduce,
        )

        communicator = getattr(get_tp_group().device_communicator, "ca_comm", None)
        return (
            isinstance(communicator, MusaJitCustomAllreduce)
            and not communicator.disabled
            and bool(getattr(communicator, "_jit_available", False))
        )
    except Exception:
        return False


def musa_compute_logits_if_eligible(
    model: Any,
    hidden_states: torch.Tensor,
    sampling_metadata: Any,
    sampler: Any,
) -> tuple[torch.Tensor | None, bool]:
    """Compute local Qwen logits only for the exact sharded-Gumbel contract."""
    sampler._musa_qwen_sharded_logits = False
    sampler._musa_qwen_global_vocab_size = 0
    sampler._musa_qwen_shard_start_index = 0
    sampler._musa_qwen_tp_size = 0
    metadata_ok = can_use_qwen_legacy_unfiltered_metadata(
        sampling_metadata,
        getattr(sampler, "logprobs_mode", "raw_logprobs"),
        predict_bonus_token=False,
        use_fp64_gumbel=bool(getattr(sampler, "use_fp64_gumbel", False)),
        is_qwen_family=prefers_optimization(
            sampler,
            OptimizationFeature.QWEN_TP4_SHARDED_GUMBEL,
        ),
    )
    rows = int(hidden_states.shape[0]) if hidden_states.ndim > 0 else 0
    if (
        not metadata_ok
        or rows < _MUSA_QWEN_SHARDED_MIN_BATCH
        or rows > 64
        or get_pp_group().world_size != 1
        or not _musa_jit_pair_gather_available()
    ):
        return model.compute_logits(hidden_states), False
    processor_and_head = _find_logits_processor(model)
    if processor_and_head is None:
        return model.compute_logits(hidden_states), False
    processor, _lm_head = processor_and_head
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size != 4:
        return model.compute_logits(hidden_states), False
    global_vocab = int(getattr(processor, "org_vocab_size", 0))
    if (
        global_vocab not in _MUSA_QWEN_SAMPLER_VOCAB_SIZES
        or getattr(processor, "scale", 1.0) != 1.0
        or getattr(processor, "soft_cap", None) is not None
        or not getattr(processor, "use_all_gather", True)
    ):
        return model.compute_logits(hidden_states), False

    old_skip = getattr(processor, "_musa_skip_tp_gather", False)
    processor._musa_skip_tp_gather = True
    try:
        logits = model.compute_logits(hidden_states)
    finally:
        processor._musa_skip_tp_gather = old_skip
    if (
        logits is None
        or not is_musa_tensor(logits)
        or logits.ndim != 2
        or logits.dtype != torch.bfloat16
        or logits.shape[0] != rows
        or logits.shape[1] * tp_size != global_vocab
        or not logits.is_contiguous()
        or logits.stride(-1) != 1
    ):
        return model.compute_logits(hidden_states), False
    sampler._musa_qwen_sharded_logits = True
    sampler._musa_qwen_global_vocab_size = global_vocab
    sampler._musa_qwen_tp_size = tp_size
    sampler._musa_qwen_shard_start_index = (
        get_tensor_model_parallel_rank() * logits.shape[-1]
    )
    return logits, True


def _get_qwen_legacy_unfiltered_generator(
    sampler: Any, logits: torch.Tensor
) -> torch.Generator | None:
    generator = getattr(sampler, "_musa_qwen_unfiltered_generator", None)
    if generator is None:
        seed = int(torch.musa.initial_seed())
        if seed < 0 or seed > np.iinfo(np.int64).max:
            return None
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(seed)
        sampler._musa_qwen_unfiltered_generator = generator
    if getattr(getattr(generator, "device", None), "type", None) != logits.device.type:
        return None
    return generator


def sample_qwen_legacy_unfiltered_gumbel(
    sampler: Any, logits: torch.Tensor
) -> torch.Tensor | None:
    """Sample raw Qwen logits while advancing one private Philox stream."""
    generator = _get_qwen_legacy_unfiltered_generator(sampler, logits)
    if generator is None:
        return None
    generator_state = get_qwen_legacy_generator_state({0: generator}, 1)
    if generator_state is None:
        return None
    seeds_cpu, offsets_cpu = generator_state
    seed = seeds_cpu[0]
    offset = offsets_cpu[0]
    rows = logits.shape[0]
    next_offset = offset + 4 * rows
    if next_offset > np.iinfo(np.int64).max:
        return None

    buffers = getattr(sampler, "_musa_qwen_unfiltered_buffers", None)
    if buffers is None:
        buffers = {}
        sampler._musa_qwen_unfiltered_buffers = buffers
    entry = buffers.get(rows)
    if entry is None:
        entry = (
            torch.zeros(rows, dtype=torch.int32, device=logits.device),
            torch.ones(1, dtype=torch.float32, device=logits.device),
            torch.tensor([seed], dtype=torch.int64, device=logits.device),
            torch.empty(rows, dtype=torch.int64, device=logits.device),
        )
        buffers[rows] = entry
    mapping, temperature, seeds, positions = entry
    torch.arange(offset // 4, offset // 4 + rows, out=positions)
    sharded = _is_qwen_sharded_logits(sampler, logits)
    gumbel_kwargs = {
        "apply_temperature": False,
        "use_fp64": False,
    }
    if sharded:
        gumbel_kwargs.update(
            vocab_start_index=int(getattr(sampler, "_musa_qwen_shard_start_index", 0)),
            return_values=True,
        )
    sampled = vllm_worker_sampler.gumbel_sample(
        logits,
        mapping,
        temperature,
        seeds,
        positions,
        **gumbel_kwargs,
    )
    try:
        generator.set_offset(next_offset)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Failed to advance the legacy-runner unfiltered Gumbel generator"
        ) from error
    if not sharded:
        return sampled

    local_sampled, local_values = sampled
    pair_buffers = getattr(sampler, "_musa_qwen_sharded_pair_buffers", None)
    if pair_buffers is None:
        pair_buffers = {}
        sampler._musa_qwen_sharded_pair_buffers = pair_buffers
    pair = pair_buffers.get(rows)
    if pair is None:
        pair = torch.empty((rows, 4), dtype=torch.float32, device=logits.device)
        pair_buffers[rows] = pair
    pair[:, 0].copy_(local_values)
    pair[:, 1].copy_(local_sampled.to(torch.float32))
    pair[:, 2:].zero_()
    from vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce import (
        maybe_musa_jit_logits_all_gather,
    )

    gathered = maybe_musa_jit_logits_all_gather(pair, dim=-1)
    if gathered is None:
        raise RuntimeError(
            "MUSA sharded Qwen Gumbel lost its IPC pair all-gather capability"
        )
    tp_size = int(getattr(sampler, "_musa_qwen_tp_size", 4))
    gathered = gathered.view(rows, tp_size, 4)
    # MUSA argmax/gather can mis-handle the strided [B, TP, 4] view.  These
    # compact views are only four floats per request and keep the winner
    # bit-exact with the full-vocab path.
    scores = gathered[:, :, 0].contiguous()
    token_ids = gathered[:, :, 1].contiguous()
    winner_rank = scores.argmax(dim=-1)
    sampled = token_ids.gather(1, winner_rank.unsqueeze(-1)).squeeze(-1)
    vllm_topk_topp_sampler.logger.info_once(
        "Using MUSA sharded Qwen Gumbel pair reduction.", scope="global"
    )
    return sampled.to(torch.int64)


def can_use_qwen_legacy_gumbel(
    logits: torch.Tensor,
    sampling_metadata: Any,
    logprobs_mode: LogprobsMode,
    deferred_min_p: torch.Tensor | None,
    use_fp64_gumbel: bool,
    is_qwen_family: bool = False,
) -> bool:
    """Gate MRV1 Qwen sampling to a batched stateless Gumbel handoff."""
    if (
        not is_qwen_family
        or not current_platform.is_musa()
        or not is_musa_tensor(logits)
    ):
        return False
    if (
        logits.ndim != 2
        # The legacy Gumbel seed/mapping setup is not amortized below four
        # rows on S5000. Preserve the seeded-multinomial fallback for those
        # shapes; the first measured positive serving cell starts at four.
        or logits.shape[0] < 4
        or logits.shape[1] != 248320
        or logits.stride(-1) != 1
    ):
        return False
    if logprobs_mode != "raw_logprobs" or deferred_min_p is not None or use_fp64_gumbel:
        return False
    if (
        sampling_metadata.max_num_logprobs is not None
        or sampling_metadata.logprob_token_ids
        or not sampling_metadata.all_random
    ):
        return False
    top_k = sampling_metadata.top_k
    if sampling_metadata.top_p is not None or (
        top_k is not None and (top_k.ndim != 1 or top_k.numel() != logits.shape[0])
    ):
        return False
    if top_k is None and logits.shape[0] < _QWEN_LEGACY_UNFILTERED_GUMBEL_MIN_ROWS:
        return False
    generators = sampling_metadata.generators
    rows = logits.shape[0]
    if not generators or any(row < 0 or row >= rows for row in generators):
        return False
    if top_k is None:
        generator_state = _get_qwen_legacy_generator_state_for_rows(
            generators,
            list(range(rows)),
        )
        if generator_state is None or min(generator_state[1]) < (
            _QWEN_LEGACY_UNFILTERED_GUMBEL_MIN_OFFSET
        ):
            return False

    spec_token_ids = getattr(sampling_metadata, "spec_token_ids", None)
    if spec_token_ids and any(spec_token_ids):
        return False

    return all(
        hasattr(generator, "initial_seed")
        and hasattr(generator, "get_offset")
        and hasattr(generator, "set_offset")
        and getattr(getattr(generator, "device", None), "type", None)
        == logits.device.type
        for generator in generators.values()
    )


def _get_qwen_legacy_generator_state_for_rows(
    generators: dict[int, torch.Generator], row_indices: list[int]
) -> tuple[list[int], list[int]] | None:
    """Read a valid sparse per-row seed/offset snapshot without mutation."""
    seeds = []
    offsets = []
    int64_max = np.iinfo(np.int64).max
    try:
        for row in row_indices:
            generator = generators[row]
            seed = int(generator.initial_seed())
            offset = int(generator.get_offset())
            if seed < 0 or seed > int64_max or offset < 0 or offset % 4:
                return None
            seeds.append(seed)
            offsets.append(offset)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    return seeds, offsets


def get_qwen_legacy_generator_state(
    generators: dict[int, torch.Generator], rows: int
) -> tuple[list[int], list[int]] | None:
    """Read a valid contiguous per-row seed/offset snapshot without mutation."""
    return _get_qwen_legacy_generator_state_for_rows(generators, list(range(rows)))


def sample_qwen_legacy_gumbel(
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    top_k: int | torch.Tensor | None,
    generator_state: tuple[list[int], list[int]],
) -> torch.Tensor:
    """Batch MRV1 seeded rows and preserve one-call generator offsets."""
    logits = _apply_top_k_top_p(logits, top_k, None)
    return _sample_qwen_legacy_gumbel_filtered(logits, generators, generator_state)


def _sample_qwen_legacy_gumbel_filtered(
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    generator_state: tuple[list[int], list[int]],
) -> torch.Tensor:
    """Sample already-filtered logits and advance the remapped generators."""
    seeds_cpu, offsets_cpu = generator_state
    rows = logits.shape[0]
    mapping = torch.arange(rows, dtype=torch.int32, device=logits.device)
    temperature = torch.ones(rows, dtype=torch.float32, device=logits.device)
    seeds = torch.tensor(seeds_cpu, dtype=torch.int64, device=logits.device)
    positions = torch.tensor(
        [offset // 4 for offset in offsets_cpu],
        dtype=torch.int64,
        device=logits.device,
    )
    sampled = vllm_worker_sampler.gumbel_sample(
        logits,
        mapping,
        temperature,
        seeds,
        positions,
        apply_temperature=False,
        use_fp64=False,
    )
    advanced_rows = []
    try:
        for row, offset in enumerate(offsets_cpu):
            generators[row].set_offset(offset + 4)
            advanced_rows.append(row)
    except (RuntimeError, TypeError, ValueError) as error:
        rollback_errors = []
        for row in reversed(advanced_rows):
            try:
                generators[row].set_offset(offsets_cpu[row])
            except (RuntimeError, TypeError, ValueError) as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise RuntimeError(
                "Failed to advance and fully roll back legacy MUSA generators"
            ) from error
        raise RuntimeError("Failed to advance legacy MUSA generators") from error
    return sampled


def sample_qwen_legacy_gumbel_partitioned(
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    top_k: torch.Tensor | None,
) -> torch.Tensor | None:
    """Use Gumbel for seeded rows and preserve multinomial for unseeded rows."""
    rows = logits.shape[0]
    seeded_rows = sorted(generators)
    if not seeded_rows or any(row < 0 or row >= rows for row in seeded_rows):
        return None
    seeded_state = _get_qwen_legacy_generator_state_for_rows(generators, seeded_rows)
    if seeded_state is None:
        return None
    if len(seeded_rows) == rows:
        return sample_qwen_legacy_gumbel(
            logits,
            generators,
            top_k,
            seeded_state,
        )

    filtered_logits = _apply_top_k_top_p(logits, top_k, None)
    seeded_index = torch.tensor(seeded_rows, dtype=torch.long, device=logits.device)
    unseeded_rows = [row for row in range(rows) if row not in generators]
    seeded_logits = filtered_logits.index_select(0, seeded_index)
    remapped_generators = {
        new_row: generators[old_row] for new_row, old_row in enumerate(seeded_rows)
    }
    seeded_sampled = _sample_qwen_legacy_gumbel_filtered(
        seeded_logits,
        remapped_generators,
        seeded_state,
    )

    sampled = torch.empty(rows, dtype=torch.long, device=logits.device)
    sampled.index_copy_(0, seeded_index, seeded_sampled)
    if unseeded_rows:
        unseeded_index = torch.tensor(
            unseeded_rows, dtype=torch.long, device=logits.device
        )
        unseeded_probs = filtered_logits.index_select(0, unseeded_index).softmax(
            dim=-1, dtype=torch.float32
        )
        unseeded_sampled = sample_probs_seeded_multinomial(unseeded_probs, {})
        sampled.index_copy_(0, unseeded_index, unseeded_sampled)
    return sampled


def _sample(
    self: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    logprobs_mode_override: LogprobsMode | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    is_qwen_family = prefers_optimization(
        self,
        OptimizationFeature.QWEN_LEGACY_GUMBEL,
    )
    logprobs_mode = logprobs_mode_override or self.logprobs_mode
    assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
    if sampling_metadata.all_random:
        greedy_sampled = None
    else:
        greedy_sampled = self.greedy_sample(logits)
        if sampling_metadata.all_greedy:
            processed_logprobs = None
            if sampling_metadata.max_num_logprobs is not None:
                if logprobs_mode == "processed_logits":
                    processed_logprobs = logits
                elif logprobs_mode == "processed_logprobs":
                    processed_logprobs = self.compute_logprobs(logits)
            return greedy_sampled, processed_logprobs

    assert sampling_metadata.temperature is not None
    if not _can_skip_legacy_qwen_unit_temperature(self, logits, sampling_metadata):
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

    top_k = (
        _legacy_top_k_with_cpu_hint(
            sampling_metadata,
            sampling_metadata.top_k,
            self.topk_topp_sampler,
            logits,
            logprobs_mode,
        )
        if is_qwen_family
        else sampling_metadata.top_k
    )

    musa_min_p = None
    defer_min_p = (
        getattr(self.topk_topp_sampler.forward, "__name__", "") == "forward_musa"
    )
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        if defer_min_p and should_defer_min_p_processor(
            logits, sampling_metadata.generators, logprobs_mode, processor
        ):
            musa_min_p = _legacy_min_p_with_cpu_hint(processor, logits)
            continue
        logits = processor.apply(logits)

    use_legacy_gumbel = can_use_qwen_legacy_gumbel(
        logits,
        sampling_metadata,
        logprobs_mode,
        musa_min_p,
        self.use_fp64_gumbel,
        is_qwen_family,
    )
    legacy_top_k = (
        _legacy_gumbel_top_k_with_cpu_hint(
            sampling_metadata, logits, sampling_metadata.top_k
        )
        if use_legacy_gumbel
        else sampling_metadata.top_k
    )
    legacy_sampled = (
        sample_qwen_legacy_gumbel_partitioned(
            logits,
            sampling_metadata.generators,
            legacy_top_k,
        )
        if use_legacy_gumbel
        else None
    )
    if legacy_sampled is not None:
        if len(sampling_metadata.generators) < logits.shape[0]:
            vllm_topk_topp_sampler.logger.info_once(
                "Using mixed seeded/unseeded MUSA legacy Gumbel partition.",
                scope="global",
            )
        else:
            vllm_topk_topp_sampler.logger.info_once(
                "Using the gated MUSA legacy Gumbel sampler for Qwen requests.",
                scope="global",
            )
        random_sampled = legacy_sampled
        processed_logprobs = None
    elif musa_min_p is None:
        random_sampled, processed_logprobs = _call_topk_topp_sampler(
            self.topk_topp_sampler,
            logprobs_mode,
            logits,
            sampling_metadata.generators,
            top_k,
            sampling_metadata.top_p,
        )
    else:
        random_sampled, processed_logprobs = _call_topk_topp_sampler(
            self.topk_topp_sampler,
            logprobs_mode,
            logits,
            sampling_metadata.generators,
            top_k,
            sampling_metadata.top_p,
            min_p=musa_min_p,
        )

    if greedy_sampled is None:
        return random_sampled, processed_logprobs

    sampled = torch.where(
        sampling_metadata.temperature < vllm_sample_sampler._SAMPLING_EPS,
        greedy_sampled,
        random_sampled,
        out=greedy_sampled,
    )
    return sampled, processed_logprobs


def _sampler_forward(
    self: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    predict_bonus_token: bool = False,
    logprobs_mode_override: LogprobsMode | None = None,
) -> Any:
    original_forward = vllm_sample_sampler.Sampler._musa_original_forward
    if logprobs_mode_override is None and can_use_qwen_legacy_unfiltered_gumbel(
        logits,
        sampling_metadata,
        self.logprobs_mode,
        predict_bonus_token,
        self.use_fp64_gumbel,
        prefers_optimization(self, OptimizationFeature.QWEN_LEGACY_GUMBEL),
        sampler=self,
    ):
        sampled = sample_qwen_legacy_unfiltered_gumbel(self, logits)
        if sampled is not None:
            vllm_topk_topp_sampler.logger.info_once(
                "Using the gated MUSA legacy-runner unfiltered Gumbel sampler "
                "for Qwen requests.",
                scope="global",
            )
            return vllm_sample_sampler.SamplerOutput(
                sampled_token_ids=sampled.to(torch.int32).unsqueeze(-1),
                logprobs_tensors=None,
            )
    if logprobs_mode_override is None:
        return original_forward(
            self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
        )

    original_logprobs_mode = self.logprobs_mode
    self.logprobs_mode = logprobs_mode_override
    try:
        return original_forward(
            self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
        )
    finally:
        self.logprobs_mode = original_logprobs_mode


def has_worker_user_seed(sampling_states: Any, idx_mapping_np: np.ndarray) -> bool:
    has_user_seed = getattr(sampling_states, "has_user_seed", None)
    if has_user_seed is None:
        return True
    return bool(np.any(has_user_seed[idx_mapping_np]))


def has_worker_all_user_seeds(sampling_states: Any, idx_mapping_np: np.ndarray) -> bool:
    has_user_seed = getattr(sampling_states, "has_user_seed", None)
    if has_user_seed is None or idx_mapping_np.size == 0:
        return False
    return bool(np.all(has_user_seed[idx_mapping_np]))


def can_use_qwen_v2_gumbel(
    sampler: Any,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    return_logprobs: bool,
) -> bool:
    """Gate the pinned V2 Gumbel sampler to one validated Qwen contract."""
    if (
        not prefers_optimization(sampler, OptimizationFeature.QWEN_V2_GUMBEL)
        or not current_platform.is_musa()
        or not is_musa_tensor(logits)
    ):
        return False
    if not _is_qwen_sampler_vocab(logits) or logits.stride(-1) != 1:
        return False
    if idx_mapping_np.ndim != 1 or logits.shape[0] != idx_mapping_np.size:
        return False
    if (
        expanded_idx_mapping.ndim != 1
        or expanded_idx_mapping.numel() != logits.shape[0]
    ):
        return False
    if pos.ndim != 1 or pos.numel() != logits.shape[0]:
        return False
    if getattr(sampler, "num_speculative_tokens", 1) != 1:
        return False
    if sampler.use_fp64_gumbel or sampler.logprobs_mode != "raw_logprobs":
        return False
    if return_logprobs or not has_worker_all_user_seeds(
        sampler.sampling_states, idx_mapping_np
    ):
        return False

    states = sampler.sampling_states
    if not np.all(states.temperature.np[idx_mapping_np] == np.float32(1.0)):
        return False
    if not np.all(states.top_k.np[idx_mapping_np] == 50):
        return False
    if not np.all(states.top_p.np[idx_mapping_np] == np.float32(1.0)):
        return False
    return bool(np.all(states.min_p.np[idx_mapping_np] == np.float32(0.05)))


def can_use_qwen_v2_unfiltered_gumbel(
    sampler: Any,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    return_logprobs: bool,
) -> bool:
    """Gate direct logits-domain Gumbel to an unfiltered Qwen contract."""
    if (
        not prefers_optimization(sampler, OptimizationFeature.QWEN_V2_GUMBEL)
        or not current_platform.is_musa()
        or not is_musa_tensor(logits)
    ):
        return False
    if (
        logits.dtype != torch.bfloat16
        or not _is_qwen_sampler_vocab(logits)
        or logits.stride(-1) != 1
    ):
        return False
    if idx_mapping_np.ndim != 1 or logits.shape[0] != idx_mapping_np.size:
        return False
    if (
        expanded_idx_mapping.ndim != 1
        or expanded_idx_mapping.numel() != logits.shape[0]
    ):
        return False
    if pos.ndim != 1 or pos.numel() != logits.shape[0]:
        return False
    if getattr(sampler, "num_speculative_tokens", 1) != 1:
        return False
    if sampler.use_fp64_gumbel or sampler.logprobs_mode != "raw_logprobs":
        return False
    if return_logprobs:
        return False

    states = sampler.sampling_states
    if has_worker_user_seed(states, idx_mapping_np):
        return False
    if not (
        np.all(states.temperature.np[idx_mapping_np] == np.float32(1.0))
        and np.all(states.top_k.np[idx_mapping_np] == logits.shape[1])
        and np.all(states.top_p.np[idx_mapping_np] == np.float32(1.0))
        and np.all(states.min_p.np[idx_mapping_np] == np.float32(0.0))
    ):
        return False

    use_logit_bias = getattr(sampler.logit_bias_state, "use_logit_bias", None)
    use_penalty = getattr(sampler.penalties_state, "use_penalty", None)
    num_bad_words = getattr(sampler.bad_words_state, "num_bad_words", None)
    if use_logit_bias is None or use_penalty is None or num_bad_words is None:
        return False
    return bool(
        not np.any(use_logit_bias[idx_mapping_np])
        and not np.any(use_penalty[idx_mapping_np])
        and not np.any(num_bad_words.np[idx_mapping_np])
    )


def can_use_worker_seeded_multinomial(
    logits: torch.Tensor,
    logprobs_mode: LogprobsMode,
    sampling_states: Any,
    idx_mapping_np: np.ndarray,
) -> bool:
    if not musa_seeded_multinomial_enabled():
        return False
    if not current_platform.is_musa() or not is_musa_tensor(logits):
        return False
    if logprobs_mode == "processed_logprobs":
        return False
    if not has_worker_user_seed(sampling_states, idx_mapping_np):
        return False
    if np.any(sampling_states.temperature.np[idx_mapping_np] <= _SAMPLING_EPS):
        return False
    if getattr(sampling_states, "musa_generators", None) is None:
        return False
    return True


def can_use_worker_sampler(
    logits: torch.Tensor,
    logprobs_mode: LogprobsMode,
    sampling_states: Any,
    idx_mapping_np: np.ndarray,
) -> bool:
    if not current_platform.is_musa() or not is_musa_tensor(logits):
        return False
    if logprobs_mode == "processed_logprobs":
        return False
    if has_worker_user_seed(sampling_states, idx_mapping_np):
        return False
    if np.any(sampling_states.temperature.np[idx_mapping_np] <= _SAMPLING_EPS):
        return False
    return True


def sample_worker_logits(
    logits: torch.Tensor,
    sampling_states: Any,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    is_qwen_family: bool,
) -> torch.Tensor:
    vocab_size = sampling_states.vocab_size
    top_k_np = sampling_states.top_k.np[idx_mapping_np]
    min_p_np = sampling_states.min_p.np[idx_mapping_np]
    use_top_k = np.any(top_k_np != vocab_size)
    use_top_p = np.any(sampling_states.top_p.np[idx_mapping_np] != 1.0)
    use_min_p = np.any(min_p_np != 0.0)

    # Select the scalar from existing CPU sampling state so the decode path
    # never copies a device tensor to the host merely to choose a kernel.
    if (
        use_top_k
        and is_qwen_family
        and _is_uniform_top_k_50(top_k_np)
        and _is_qwen_sampler_vocab(logits)
        and (not use_top_p or logits.shape[0] >= 4)
    ):
        top_k = 50
    else:
        top_k = sampling_states.top_k.gpu[expanded_idx_mapping] if use_top_k else None
    top_p = sampling_states.top_p.gpu[expanded_idx_mapping] if use_top_p else None
    uniform_min_p = _uniform_active_min_p(min_p_np)
    if uniform_min_p is not None:
        min_p = uniform_min_p
    else:
        min_p = sampling_states.min_p.gpu[expanded_idx_mapping] if use_min_p else None
    return sample_from_logits(logits, top_k, top_p, min_p)


def sample_worker_logits_seeded_multinomial(
    logits: torch.Tensor,
    sampling_states: Any,
    idx_mapping_np: np.ndarray,
) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    generators = {
        row_idx: sampling_states.musa_generators.get(int(req_idx))
        for row_idx, req_idx in enumerate(idx_mapping_np)
    }
    return sample_probs_seeded_multinomial(probs, generators)


def sample_worker_logits_qwen_v2_gumbel(
    sampler: Any,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run pinned V2 Gumbel with CPU-proven uniform Qwen top-k state."""
    processed_logits = sampler.apply_sampling_params(
        logits,
        expanded_idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        skip_top_k_top_p=True,
    )
    processed_logits = _apply_top_k_top_p(processed_logits, 50, None)
    sampled = vllm_worker_sampler.gumbel_sample(
        processed_logits,
        expanded_idx_mapping,
        sampler.sampling_states.temperature.gpu,
        sampler.sampling_states.seeds.gpu,
        pos,
        apply_temperature=False,
        use_fp64=False,
    )
    return sampled, processed_logits


def sample_worker_logits_qwen_v2_unfiltered_gumbel(
    sampler: Any,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample unfiltered categorical Qwen logits without materializing probs."""
    sampled = vllm_worker_sampler.gumbel_sample(
        logits,
        expanded_idx_mapping,
        sampler.sampling_states.temperature.gpu,
        sampler.sampling_states.seeds.gpu,
        pos,
        apply_temperature=False,
        use_fp64=False,
    )
    return sampled, logits


def _sampling_states_init(self: Any, max_num_reqs: int, vocab_size: int):
    original_init = vllm_worker_states.SamplingStates._musa_original_init
    original_init(self, max_num_reqs, vocab_size)
    self.has_user_seed = np.zeros(self.max_num_reqs, dtype=np.bool_)
    self.musa_generators = {}


def _sampling_states_add_request(self: Any, req_idx: int, sampling_params: Any) -> None:
    original_add_request = vllm_worker_states.SamplingStates._musa_original_add_request
    original_add_request(self, req_idx, sampling_params)
    seed = sampling_params.seed
    self.has_user_seed[req_idx] = seed is not None
    if seed is None:
        self.musa_generators.pop(req_idx, None)
    else:
        generator = torch.Generator(device="musa")
        generator.manual_seed(seed)
        self.musa_generators[req_idx] = generator


def _apply_worker_sampling_params_defer_filters(
    sampler: Any,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
) -> torch.Tensor:
    logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
    sampler.logit_bias_state.apply_logit_bias(
        logits, expanded_idx_mapping, idx_mapping_np, pos
    )
    sampler.penalties_state.apply_penalties(
        logits,
        expanded_idx_mapping,
        idx_mapping_np,
        input_ids,
        expanded_local_pos,
    )
    sampler.bad_words_state.apply_bad_words(
        logits,
        expanded_idx_mapping,
        idx_mapping_np,
        input_ids,
        expanded_local_pos,
    )
    sampler.sampling_states.apply_temperature(
        logits, expanded_idx_mapping, idx_mapping_np
    )
    return logits


def _apply_worker_sampling_filters_for_seeded_multinomial(
    sampler: Any,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    preserve_processed_logits: bool = True,
) -> torch.Tensor:
    if preserve_processed_logits:
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

    vocab_size = sampler.sampling_states.vocab_size
    top_k_np = sampler.sampling_states.top_k.np[idx_mapping_np]
    use_top_k = np.any(top_k_np != vocab_size)
    use_top_p = np.any(sampler.sampling_states.top_p.np[idx_mapping_np] != 1.0)
    use_min_p = np.any(sampler.sampling_states.min_p.np[idx_mapping_np] != 0.0)
    if (
        use_top_k
        and prefers_optimization(sampler, OptimizationFeature.QWEN_V2_GUMBEL)
        and _is_uniform_top_k_50(top_k_np)
        and _is_qwen_sampler_vocab(logits)
        and (not use_top_p or logits.shape[0] >= 4)
    ):
        top_k = 50
    else:
        top_k = (
            sampler.sampling_states.top_k.gpu[expanded_idx_mapping]
            if use_top_k
            else None
        )
    top_p = (
        sampler.sampling_states.top_p.gpu[expanded_idx_mapping] if use_top_p else None
    )
    logits = _apply_top_k_top_p(logits, top_k, top_p)
    if use_min_p:
        sampler.sampling_states.apply_min_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )
    return logits


def _worker_sample(
    self: Any,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    return_logprobs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if can_use_qwen_v2_unfiltered_gumbel(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping_np,
        pos,
        return_logprobs,
    ):
        vllm_topk_topp_sampler.logger.info_once(
            "Using the gated MUSA unfiltered Gumbel sampler for Qwen requests.",
            scope="global",
        )
        return sample_worker_logits_qwen_v2_unfiltered_gumbel(
            self,
            logits,
            expanded_idx_mapping,
            pos,
        )

    if can_use_qwen_v2_gumbel(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping_np,
        pos,
        return_logprobs,
    ):
        vllm_topk_topp_sampler.logger.info_once(
            "Using the gated MUSA V2 Gumbel sampler for Qwen requests.",
            scope="global",
        )
        return sample_worker_logits_qwen_v2_gumbel(
            self,
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )

    if logits.shape[0] == idx_mapping_np.shape[0] and can_use_worker_seeded_multinomial(
        logits, self.logprobs_mode, self.sampling_states, idx_mapping_np
    ):
        vllm_topk_topp_sampler.logger.info_once(
            "Using MUSA seeded multinomial sampling for user-seeded requests.",
            scope="global",
        )
        processed_logits = _apply_worker_sampling_params_defer_filters(
            self,
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )
        sampling_logits = _apply_worker_sampling_filters_for_seeded_multinomial(
            self,
            processed_logits,
            expanded_idx_mapping,
            idx_mapping_np,
            preserve_processed_logits=return_logprobs,
        )
        sampled = sample_worker_logits_seeded_multinomial(
            sampling_logits, self.sampling_states, idx_mapping_np
        )
        return sampled, processed_logits

    if can_use_worker_sampler(
        logits, self.logprobs_mode, self.sampling_states, idx_mapping_np
    ):
        processed_logits = _apply_worker_sampling_params_defer_filters(
            self,
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )
        sampled = sample_worker_logits(
            processed_logits,
            self.sampling_states,
            expanded_idx_mapping,
            idx_mapping_np,
            prefers_optimization(self, OptimizationFeature.QWEN_V2_GUMBEL),
        )
        return sampled, processed_logits

    original_sample = vllm_worker_sampler.Sampler._musa_original_sample
    return original_sample(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        return_logprobs=return_logprobs,
    )


def install_hooks() -> None:
    topk_cls = vllm_topk_topp_sampler.TopKTopPSampler
    if not getattr(topk_cls, "_musa_sampling_hooks_installed", False):
        topk_cls._musa_original_init = topk_cls.__init__
        topk_cls.__init__ = _topk_topp_sampler_init
        topk_cls.forward_musa = forward_musa
        topk_cls._musa_sampling_hooks_installed = True

    vllm_topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p

    sample_cls = vllm_sample_sampler.Sampler
    if not getattr(sample_cls, "_musa_sampling_hooks_installed", False):
        sample_cls._musa_original_sample = sample_cls.sample
        sample_cls._musa_original_forward = sample_cls.forward
        sample_cls.forward = _sampler_forward
        sample_cls.sample = _sample
        sample_cls._musa_sampling_hooks_installed = True

    states_cls = vllm_worker_states.SamplingStates
    if not getattr(states_cls, "_musa_sampling_hooks_installed", False):
        states_cls._musa_original_init = states_cls.__init__
        states_cls._musa_original_add_request = states_cls.add_request
        states_cls.__init__ = _sampling_states_init
        states_cls.add_request = _sampling_states_add_request
        states_cls._musa_sampling_hooks_installed = True

    worker_cls = vllm_worker_sampler.Sampler
    if not getattr(worker_cls, "_musa_sampling_hooks_installed", False):
        worker_cls._musa_original_sample = worker_cls.sample
        worker_cls.sample = _worker_sample
        worker_cls._musa_sampling_hooks_installed = True


install_hooks()
