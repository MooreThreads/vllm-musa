# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from typing import Any

import numpy as np
import torch
import vllm.v1.sample.ops.topk_topp_sampler as vllm_topk_topp_sampler
import vllm.v1.sample.sampler as vllm_sample_sampler
import vllm.v1.worker.gpu.sample.sampler as vllm_worker_sampler
import vllm.v1.worker.gpu.sample.states as vllm_worker_states
from vllm.config.model import LogprobsMode
from vllm.platforms import current_platform

from vllm_musa import _custom_ops as _ops

logger = logging.getLogger(__name__)

_SAMPLING_EPS = 1e-5


def sampler_fast_path_enabled() -> bool:
    value = os.environ.get("VLLM_MUSA_SAMPLER_FAST_PATH", "1")
    return value.lower() not in ("0", "false", "no", "off")


def is_musa_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "musa"


def can_use_musa_sampler(
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    logprobs_mode: LogprobsMode,
) -> bool:
    if not sampler_fast_path_enabled():
        return False
    if not current_platform.is_musa() or not is_musa_tensor(logits):
        return False
    if generators:
        return False
    return logprobs_mode not in ("processed_logits", "processed_logprobs")


def _squeeze_filter_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim > 1 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    return value.contiguous()


def _has_active_top_k(top_k: torch.Tensor | int | None, vocab_size: int) -> bool:
    if top_k is None:
        return False
    if isinstance(top_k, torch.Tensor):
        return bool((top_k != vocab_size).any().item())
    return 0 < top_k < vocab_size


def _has_active_top_p(top_p: torch.Tensor | float | None) -> bool:
    if top_p is None:
        return False
    if isinstance(top_p, torch.Tensor):
        return bool((top_p != 1.0).any().item())
    return top_p != 1.0


def _has_active_min_p(min_p: torch.Tensor | float | None) -> bool:
    if min_p is None:
        return False
    if isinstance(min_p, torch.Tensor):
        return bool((min_p != 0.0).any().item())
    return min_p != 0.0


def sample_from_probs(
    probs: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
    min_p: torch.Tensor | float | None = None,
) -> torch.Tensor:
    top_k = _squeeze_filter_tensor(top_k) if isinstance(top_k, torch.Tensor) else top_k
    top_p = _squeeze_filter_tensor(top_p) if isinstance(top_p, torch.Tensor) else top_p
    min_p = _squeeze_filter_tensor(min_p) if isinstance(min_p, torch.Tensor) else min_p

    use_top_k = _has_active_top_k(top_k, probs.shape[-1])
    use_top_p = _has_active_top_p(top_p)
    use_min_p = _has_active_min_p(min_p)

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

    output = logits.new_full(logits.shape, -float("inf"))
    output.scatter_(dim=-1, index=logits_idx, src=logits_sort)
    logits.copy_(output)
    return logits


def _apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    if current_platform.is_musa() and sampler_fast_path_enabled():
        if k is not None and logits.shape[0] >= 16:
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
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    min_p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not can_use_musa_sampler(logits, generators, self.logprobs_mode):
        if generators:
            logger.debug(
                "MUSA native sampling ops do not support per-request generators; "
                "falling back to PyTorch-native sampling."
            )
        return self.forward_native(logits, generators, k, p)

    return sample_from_logits(logits, k, p, min_p), None


def _topk_topp_sampler_init(self: Any, logprobs_mode: LogprobsMode = "raw_logprobs"):
    original_init = vllm_topk_topp_sampler.TopKTopPSampler._musa_original_init
    original_init(self, logprobs_mode)
    if (
        logprobs_mode not in ("processed_logits", "processed_logprobs")
        and current_platform.is_musa()
        and sampler_fast_path_enabled()
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


def _call_topk_topp_sampler(
    sampler: Any,
    logprobs_mode: LogprobsMode,
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    min_p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    original_logprobs_mode = sampler.logprobs_mode
    sampler.logprobs_mode = logprobs_mode
    try:
        if min_p is None:
            return sampler(logits, generators, top_k, top_p)
        return sampler(logits, generators, top_k, top_p, min_p=min_p)
    finally:
        sampler.logprobs_mode = original_logprobs_mode


def _sample(
    self: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    logprobs_mode_override: LogprobsMode | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
    logits = self.apply_temperature(
        logits, sampling_metadata.temperature, sampling_metadata.all_random
    )

    musa_min_p = None
    defer_min_p = (
        getattr(self.topk_topp_sampler.forward, "__name__", "") == "forward_musa"
    )
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        if defer_min_p and should_defer_min_p_processor(
            logits, sampling_metadata.generators, logprobs_mode, processor
        ):
            musa_min_p = get_processor_min_p(processor)
            continue
        logits = processor.apply(logits)

    if musa_min_p is None:
        random_sampled, processed_logprobs = _call_topk_topp_sampler(
            self.topk_topp_sampler,
            logprobs_mode,
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )
    else:
        random_sampled, processed_logprobs = _call_topk_topp_sampler(
            self.topk_topp_sampler,
            logprobs_mode,
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
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


def can_use_worker_sampler(
    logits: torch.Tensor,
    logprobs_mode: LogprobsMode,
    sampling_states: Any,
    idx_mapping_np: np.ndarray,
) -> bool:
    if not sampler_fast_path_enabled():
        return False
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
) -> torch.Tensor:
    vocab_size = sampling_states.vocab_size
    use_top_k = np.any(sampling_states.top_k.np[idx_mapping_np] != vocab_size)
    use_top_p = np.any(sampling_states.top_p.np[idx_mapping_np] != 1.0)
    use_min_p = np.any(sampling_states.min_p.np[idx_mapping_np] != 0.0)

    top_k = sampling_states.top_k.gpu[expanded_idx_mapping] if use_top_k else None
    top_p = sampling_states.top_p.gpu[expanded_idx_mapping] if use_top_p else None
    min_p = sampling_states.min_p.gpu[expanded_idx_mapping] if use_min_p else None
    return sample_from_logits(logits, top_k, top_p, min_p)


def _sampling_states_init(self: Any, max_num_reqs: int, vocab_size: int):
    original_init = vllm_worker_states.SamplingStates._musa_original_init
    original_init(self, max_num_reqs, vocab_size)
    self.has_user_seed = np.zeros(self.max_num_reqs, dtype=np.bool_)


def _sampling_states_add_request(self: Any, req_idx: int, sampling_params: Any) -> None:
    original_add_request = vllm_worker_states.SamplingStates._musa_original_add_request
    original_add_request(self, req_idx, sampling_params)
    self.has_user_seed[req_idx] = sampling_params.seed is not None


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
        sampler.num_speculative_tokens,
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


def _worker_sample(
    self: Any,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
