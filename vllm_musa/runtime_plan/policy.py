# SPDX-License-Identifier: Apache-2.0
"""Typed queries over the single immutable runtime-plan application.

Keeping these small queries here prevents consumers from recreating model and
shape heuristics outside the plan manager while preserving dynamic correctness
guards that are local to each implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .resolver import resolve_runtime_plan
from .types import RuntimeDecision, RuntimePlan

_DEEPSEEK_V4_SPARSE_PADDED_HEADS = 64
_DEEPSEEK_V4_SPARSE_HEAD_DIM = 512
_DEEPSEEK_V4_SPARSE_DTYPE_BYTES = 2
_DEEPSEEK_V4_CUSTOM_AR_ALLOCATOR_MARGIN_BYTES = 512 * 1024 * 1024
_DEEPSEEK_V4_MTP4_TOKENS_PER_REQUEST = 5
# The exact MTP4 graph ladder is 5 * batch_size. The graph-local CAR contract
# covers descriptors whose launch grid stays below the long-descriptor path;
# the 32/64-request descriptors retain the standard collective. Keeping the
# set explicit prevents a future capture ladder from entering an unproved path
# merely because its token count happens to fit the arena.
_DEEPSEEK_V4_MTP_CAR_GRAPH_REQUEST_SIZES = (1, 2, 4, 8, 16)
_DEEPSEEK_V4_MTP_CAR_GRAPH_BUFFER_BYTES = 512 * 1024 * 1024
_DEEPSEEK_V4_MTP_CAR_EAGER_ALIGNMENT_BYTES = 4 * 1024 * 1024
_DEEPSEEK_V4_MTP_CAR_MAX_META_BYTES_PER_SLOT = 16 * 1024


@dataclass(frozen=True, slots=True)
class DeepSeekV4MtpCarGraphStagingPlan:
    """Capture-time resource contract for DSV4 MTP4 JIT CAR graphs."""

    eager_reserve_bytes: int
    capture_descriptors: frozenset[tuple[int, int]]
    car_ops_per_descriptor: int
    bytes_per_token: int
    graph_data_capacity_bytes: int
    graph_meta_capacity_bytes: int
    max_meta_bytes_per_slot: int
    communicator_buffer_bytes: int

    def allows_descriptor(self, descriptor: Any) -> bool:
        if descriptor is None or not bool(getattr(descriptor, "uniform", False)):
            return False
        if bool(getattr(descriptor, "has_lora", False)) or int(
            getattr(descriptor, "num_active_loras", 0) or 0
        ):
            return False
        num_tokens = getattr(descriptor, "num_tokens", None)
        num_reqs = getattr(descriptor, "num_reqs", None)
        if (
            not isinstance(num_tokens, int)
            or isinstance(num_tokens, bool)
            or not isinstance(num_reqs, int)
            or isinstance(num_reqs, bool)
        ):
            return False
        key = (num_tokens, num_reqs)
        return key in self.capture_descriptors

    def expected_descriptor_data_bytes(self, num_tokens: int) -> int:
        return self.car_ops_per_descriptor * num_tokens * self.bytes_per_token


def runtime_plan_enabled(target: Any, decision: RuntimeDecision) -> bool:
    """Return one selected decision through the unified plan entry point.

    Consumers in model/runtime code often hold an owner that has already been
    bound during model construction, while compilation/platform helpers hold
    the ``VllmConfig`` itself.  Both views must read the same immutable plan;
    resolving a second heuristic from an owner would recreate the old
    contract split.  Prefer the bound snapshot and resolve the explicit config
    only when no snapshot exists yet.
    """
    bound = getattr(target, "_musa_runtime_plan", None)
    if isinstance(bound, RuntimePlan):
        return bound.enabled(decision)
    return resolve_runtime_plan(target).enabled(decision)


def deepseek_v4_mtp_car_graph_guard_enabled(vllm_config: Any) -> bool:
    """Guard eligible DSV4 MTP graph CAR even when no arena plan exists."""
    return resolve_runtime_plan(vllm_config).supports(
        RuntimeDecision.DEEPSEEK_V4_MTP_CAR_GRAPH_STAGING_ARENA
    )


def deepseek_v4_mtp_graph_registered_inputs_enabled(vllm_config: Any) -> bool:
    """Return the selected DSV4 MTP graph CAR path decision."""
    return runtime_plan_enabled(
        vllm_config,
        RuntimeDecision.DEEPSEEK_V4_MTP_CAR_GRAPH_REGISTERED_INPUTS,
    )


def deepseek_v4_mtp_car_graph_staging_plan(
    vllm_config: Any,
) -> DeepSeekV4MtpCarGraphStagingPlan | None:
    """Return the bounded DSV4 MTP4 graph staging plan, or fail closed."""
    if not runtime_plan_enabled(
        vllm_config,
        RuntimeDecision.DEEPSEEK_V4_MTP_CAR_GRAPH_STAGING_ARENA,
    ):
        return None
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if (
        getattr(speculative_config, "num_speculative_tokens", None)
        != _DEEPSEEK_V4_MTP4_TOKENS_PER_REQUEST - 1
    ):
        return None
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    max_num_batched_tokens = int(
        getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
    )
    hidden_size = int(getattr(text_config, "hidden_size", 0) or 0)
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    if max_num_batched_tokens <= 0 or hidden_size <= 0 or num_hidden_layers <= 0:
        return None
    required_eager_bytes = max_num_batched_tokens * hidden_size * 2
    alignment = _DEEPSEEK_V4_MTP_CAR_EAGER_ALIGNMENT_BYTES
    eager_reserve_bytes = (
        (required_eager_bytes + alignment - 1) // alignment * alignment
    )
    capture_descriptors = frozenset(
        (
            _DEEPSEEK_V4_MTP4_TOKENS_PER_REQUEST * num_reqs,
            num_reqs,
        )
        for num_reqs in _DEEPSEEK_V4_MTP_CAR_GRAPH_REQUEST_SIZES
    )
    # The exact DeepSeek-V4 graph contains two CAR sites per transformer layer
    # plus the final model-parallel reduction. Every captured CAR ordinal gets
    # a disjoint slice so graph replay never depends on host-side stream/queue
    # ordering to protect shared signal or staging storage. The capture-time
    # manifest checks the operation count and tensor bytes on every rank.
    car_ops_per_descriptor = 2 * num_hidden_layers + 1
    bytes_per_token = hidden_size * 2
    graph_slot_count = car_ops_per_descriptor * len(capture_descriptors)
    graph_data_capacity_bytes = (
        car_ops_per_descriptor
        * sum(num_tokens for num_tokens, _ in capture_descriptors)
        * bytes_per_token
    )
    # DSV4 H=4096 BF16 inputs and the conservative meta bound are both
    # 256-byte aligned, so runtime align(meta_size + input_bytes, 256) is
    # bounded exactly by input_bytes + max_meta_bytes_per_slot.
    graph_meta_capacity_bytes = (
        graph_data_capacity_bytes
        + graph_slot_count * _DEEPSEEK_V4_MTP_CAR_MAX_META_BYTES_PER_SLOT
    )
    required_communicator_buffer_bytes = max(
        eager_reserve_bytes + graph_data_capacity_bytes,
        eager_reserve_bytes
        + _DEEPSEEK_V4_MTP_CAR_MAX_META_BYTES_PER_SLOT
        + graph_meta_capacity_bytes,
    )
    communicator_buffer_bytes = (
        (
            required_communicator_buffer_bytes
            + _DEEPSEEK_V4_MTP_CAR_EAGER_ALIGNMENT_BYTES
            - 1
        )
        // _DEEPSEEK_V4_MTP_CAR_EAGER_ALIGNMENT_BYTES
        * _DEEPSEEK_V4_MTP_CAR_EAGER_ALIGNMENT_BYTES
    )
    if communicator_buffer_bytes > _DEEPSEEK_V4_MTP_CAR_GRAPH_BUFFER_BYTES:
        return None
    return DeepSeekV4MtpCarGraphStagingPlan(
        eager_reserve_bytes=eager_reserve_bytes,
        capture_descriptors=capture_descriptors,
        car_ops_per_descriptor=car_ops_per_descriptor,
        bytes_per_token=bytes_per_token,
        graph_data_capacity_bytes=graph_data_capacity_bytes,
        graph_meta_capacity_bytes=graph_meta_capacity_bytes,
        max_meta_bytes_per_slot=_DEEPSEEK_V4_MTP_CAR_MAX_META_BYTES_PER_SLOT,
        communicator_buffer_bytes=communicator_buffer_bytes,
    )


def deepseek_v4_mtp_async_prefill_queue_fence_enabled(vllm_config: Any) -> bool:
    """Return whether this runtime plan needs the async queue fence."""
    return runtime_plan_enabled(
        vllm_config, RuntimeDecision.DEEPSEEK_V4_TP8_MTP_ASYNC_PREFILL_QUEUE_FENCE
    )


def deepseek_v4_mtp_prefill_step_requires_sync(scheduler_output: Any) -> bool:
    """Return whether a scheduler step contains DSV4 prefill/context work."""

    # New requests always contain context work. Cached requests expose the
    # scheduler's explicit context-phase fact, which remains correct when
    # structured output drops every speculative token from a pure decode step.
    if getattr(scheduler_output, "scheduled_new_reqs", ()):
        return True
    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    return 0 in getattr(cached_reqs, "num_output_tokens", ())


def model_has_routed_experts(model_config: Any | None) -> bool:
    """Return whether a lightweight model config has routed experts.

    The platform no longer owns this predicate; this compatibility query is
    kept for tests and older consumers while the resolver remains the source
    of truth for complete ``VllmConfig`` objects.
    """
    if model_config is None:
        return False
    is_model_moe = getattr(model_config, "is_model_moe", None)
    if callable(is_model_moe):
        try:
            return bool(is_model_moe())
        except (AttributeError, TypeError, ValueError):
            return False
    is_moe = getattr(model_config, "is_moe", None)
    if is_moe is not None:
        return bool(is_moe)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is None:
        hf_config = getattr(model_config, "hf_config", None)
        hf_text_config = getattr(hf_config, "text_config", hf_config)
    expert_values = [
        getattr(hf_text_config, name, None)
        for name in (
            "num_experts",
            "moe_num_experts",
            "n_routed_experts",
            "num_local_experts",
        )
    ]
    if any(value is not None for value in expert_values):
        return any(bool(value) for value in expert_values)
    architectures = getattr(model_config, "architectures", None)
    if architectures is None:
        architectures = getattr(hf_text_config, "architectures", None)
    return any(
        "moe" in str(architecture).lower() for architecture in architectures or ()
    )


def deepseek_v4_flashmla_sparse_page_size(vllm_config: Any) -> int:
    """Return the runtime-plan-selected sparse FlashMLA page size."""
    value = resolve_runtime_plan(vllm_config).value(
        RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE,
        64,
    )
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def deepseek_v4_mtp_sparse_prefill_headroom_bytes(vllm_config: Any) -> int:
    """Return transient H64 query workspace omitted by MTP profiling.

    When JIT custom all-reduce is enabled, its persistent 512-MiB-class
    staging allocation increases fragmentation around the equally large
    padded query. Keep one additional staging-sized margin so the production
    query allocation remains satisfiable at high memory utilization. The
    explicit ``--disable-custom-all-reduce`` path does not need that margin.
    """
    if not runtime_plan_enabled(
        vllm_config,
        RuntimeDecision.DEEPSEEK_V4_TP8_MTP_SPARSE_PREFILL_HEADROOM,
    ):
        return 0
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_batched_tokens = int(
        getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
    )
    if max_num_batched_tokens <= 0:
        return 0
    workspace_bytes = (
        max_num_batched_tokens
        * _DEEPSEEK_V4_SPARSE_PADDED_HEADS
        * _DEEPSEEK_V4_SPARSE_HEAD_DIM
        * _DEEPSEEK_V4_SPARSE_DTYPE_BYTES
    )
    parallel_config = getattr(vllm_config, "parallel_config", None)
    custom_all_reduce_enabled = not bool(
        getattr(parallel_config, "disable_custom_all_reduce", False)
    )
    if custom_all_reduce_enabled:
        workspace_bytes += _DEEPSEEK_V4_CUSTOM_AR_ALLOCATOR_MARGIN_BYTES
    return workspace_bytes
