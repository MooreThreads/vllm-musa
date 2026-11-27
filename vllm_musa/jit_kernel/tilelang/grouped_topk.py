"""TileLang MUSA implementation of grouped softmax top-k routing."""

import functools

import tilelang
import tilelang.language as T
import torch

from vllm_musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS,
    MUSA_COMPILE_FLAGS,
    tilelang_dtype,
)

__all__ = ["grouped_topk_softmax_tilelang"]


_PARALLEL_MAX_EXPERTS = 512
_PARALLEL_MAX_GROUPS = 128
_PARALLEL_MAX_ROUTED_TOPK = 32

_PASS_CONFIGS = dict(MUSA_COMMON_PASS_CONFIGS)
for _unsafe_key in (
    "TL_DISABLE_INDEX_TYPE_PROMOTION",
    "TL_DISABLE_SAFE_MEMORY_ACCESS",
):
    if hasattr(tilelang.PassConfigKey, _unsafe_key):
        _PASS_CONFIGS.pop(getattr(tilelang.PassConfigKey, _unsafe_key), None)


def _check_config(
    num_experts: int,
    num_expert_group: int,
    topk_group: int,
    output_topk: int,
    num_fused_shared_experts: int,
) -> int:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if num_expert_group <= 0 or num_experts % num_expert_group != 0:
        raise ValueError("num_expert_group must be positive and divide num_experts")
    if not 1 <= topk_group <= num_expert_group:
        raise ValueError("topk_group must be in [1, num_expert_group]")
    if num_fused_shared_experts not in (0, 1):
        raise ValueError("TileLang grouped top-k supports zero or one shared expert")

    routed_topk = output_topk - num_fused_shared_experts
    selected_capacity = topk_group * (num_experts // num_expert_group)
    if not 1 <= routed_topk <= selected_capacity:
        raise ValueError(
            "routed top-k must be positive and no larger than the experts "
            "in selected groups"
        )
    return routed_topk


@functools.cache
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _grouped_topk_parallel_kernel(
    input_dtype: str,
    num_experts: int,
    num_expert_group: int,
    topk_group: int,
    output_topk: int,
    num_fused_shared_experts: int,
    renormalize: bool,
    apply_routed_scaling_factor: bool,
):
    num_tokens = T.dynamic("num_tokens")
    stride_m = T.dynamic("stride_m")
    stride_n = T.dynamic("stride_n")
    routed_topk = output_topk - num_fused_shared_experts
    experts_per_group = num_experts // num_expert_group
    threads = 32
    aligned_experts = ((num_experts + threads - 1) // threads) * threads
    aligned_groups = ((num_expert_group + threads - 1) // threads) * threads

    @T.prim_func
    def grouped_topk_parallel(
        gating_output: T.StridedTensor(
            (num_tokens, num_experts),
            (stride_m, stride_n),
            input_dtype,
        ),
        topk_weights: T.Tensor((num_tokens, output_topk), "float32"),
        topk_ids: T.Tensor((num_tokens, output_topk), "int32"),
        routed_scaling_factor: T.float32,
    ):
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            logits = T.alloc_fragment((aligned_experts,), "float32")
            exp_scores = T.alloc_fragment((aligned_experts,), "float32")
            candidate_scores = T.alloc_fragment((aligned_experts,), "float32")
            row_max = T.alloc_fragment((1,), "float32")
            row_sum = T.alloc_fragment((1,), "float32")
            group_scores = T.alloc_fragment((aligned_groups,), "float32")
            group_max = T.alloc_fragment((1,), "float32")
            group_idx = T.alloc_reducer((1,), T.int32, "min", replication="all")
            expert_max = T.alloc_fragment((1,), "float32")
            expert_idx = T.alloc_reducer((1,), T.int32, "min", replication="all")
            group_mask = T.alloc_shared((aligned_groups,), "int32")
            selected_ids = T.alloc_shared((routed_topk,), "int32")
            selected_weights = T.alloc_shared((routed_topk,), "float32")
            selected_sum = T.alloc_shared((1,), "float32")
            output_scale = T.alloc_var("float32")

            for expert in T.Parallel(aligned_experts):
                if expert < num_experts:
                    logits[expert] = T.cast(gating_output[token_id, expert], "float32")
                else:
                    logits[expert] = -T.infinity(T.float32)

            T.reduce_max(logits, row_max)
            if not renormalize:
                for expert in T.Parallel(aligned_experts):
                    if expert < num_experts:
                        exp_scores[expert] = T.exp(logits[expert] - row_max[0])
                    else:
                        exp_scores[expert] = 0.0
                T.reduce_sum(exp_scores, row_sum)

            if topk_group < num_expert_group:
                for group in T.Parallel(aligned_groups):
                    if group < num_expert_group:
                        group_scores[group] = -T.infinity(T.float32)
                        for offset in T.serial(experts_per_group):
                            group_scores[group] = T.max(
                                group_scores[group],
                                T.cast(
                                    gating_output[
                                        token_id,
                                        group * experts_per_group + offset,
                                    ],
                                    "float32",
                                ),
                            )
                    else:
                        group_scores[group] = -T.infinity(T.float32)
                    group_mask[group] = 0
                T.sync_threads()

                for _ in T.serial(topk_group):
                    T.reduce_max(group_scores, group_max)
                    T.fill(group_idx, T.max_value(T.int32))
                    for group in T.Parallel(aligned_groups):
                        if (
                            group < num_expert_group
                            and group_scores[group] == group_max[0]
                        ):
                            group_idx[0] = T.min(group_idx[0], group)
                    T.finalize_reducer(group_idx)
                    for group in T.Parallel(aligned_groups):
                        if group < num_expert_group and group == group_idx[0]:
                            group_mask[group] = 1
                            group_scores[group] = -T.infinity(T.float32)
                    T.sync_threads()

            for expert in T.Parallel(aligned_experts):
                if topk_group == num_expert_group:
                    if expert < num_experts:
                        candidate_scores[expert] = logits[expert]
                    else:
                        candidate_scores[expert] = -T.infinity(T.float32)
                else:
                    if (
                        expert < num_experts
                        and group_mask[expert // experts_per_group] != 0
                    ):
                        candidate_scores[expert] = logits[expert]
                    else:
                        candidate_scores[expert] = -T.infinity(T.float32)

            for kth in T.serial(routed_topk):
                T.reduce_max(candidate_scores, expert_max)
                T.fill(expert_idx, T.max_value(T.int32))
                for expert in T.Parallel(aligned_experts):
                    if topk_group == num_expert_group:
                        if (
                            expert < num_experts
                            and candidate_scores[expert] == expert_max[0]
                        ):
                            expert_idx[0] = T.min(expert_idx[0], expert)
                    else:
                        if (
                            expert < num_experts
                            and group_mask[expert // experts_per_group] != 0
                            and candidate_scores[expert] == expert_max[0]
                        ):
                            expert_idx[0] = T.min(expert_idx[0], expert)
                T.finalize_reducer(expert_idx)
                if tx == 0:
                    selected_ids[kth] = expert_idx[0]
                    selected_weights[kth] = T.exp(expert_max[0] - row_max[0])
                    if not renormalize:
                        selected_weights[kth] /= T.max(row_sum[0], 1e-20)
                T.sync_threads()
                for expert in T.Parallel(aligned_experts):
                    if expert == expert_idx[0]:
                        candidate_scores[expert] = -T.infinity(T.float32)

            if tx == 0:
                selected_sum[0] = 0.0
                for kth in T.serial(routed_topk):
                    selected_sum[0] += selected_weights[kth]
            T.sync_threads()

            output_scale = 1.0
            if renormalize:
                output_scale = 1.0 / T.max(selected_sum[0], 1e-20)
            if apply_routed_scaling_factor:
                output_scale *= routed_scaling_factor

            for kth in T.Parallel(output_topk):
                if kth < routed_topk:
                    topk_ids[token_id, kth] = selected_ids[kth]
                    topk_weights[token_id, kth] = selected_weights[kth] * output_scale
                else:
                    topk_ids[token_id, kth] = num_experts
                    topk_weights[token_id, kth] = (
                        selected_sum[0] / routed_scaling_factor * output_scale
                    )

    return grouped_topk_parallel


@functools.cache
@tilelang.jit(
    out_idx=[],
    target="musa",
    pass_configs=_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _grouped_topk_serial_kernel(
    input_dtype: str,
    num_experts: int,
    num_expert_group: int,
    topk_group: int,
    output_topk: int,
    num_fused_shared_experts: int,
    renormalize: bool,
    apply_routed_scaling_factor: bool,
):
    num_tokens = T.dynamic("num_tokens")
    stride_m = T.dynamic("stride_m")
    stride_n = T.dynamic("stride_n")
    routed_topk = output_topk - num_fused_shared_experts
    experts_per_group = num_experts // num_expert_group

    @T.prim_func
    def grouped_topk_serial(
        gating_output: T.StridedTensor(
            (num_tokens, num_experts),
            (stride_m, stride_n),
            input_dtype,
        ),
        topk_weights: T.Tensor((num_tokens, output_topk), "float32"),
        topk_ids: T.Tensor((num_tokens, output_topk), "int32"),
        routed_scaling_factor: T.float32,
    ):
        with T.Kernel(num_tokens, threads=32) as token_id:
            tx = T.get_thread_binding()
            if tx == 0:
                row_max = T.alloc_local((1,), "float32")
                row_sum = T.alloc_local((1,), "float32")
                group_score = T.alloc_local((1,), "float32")
                previous_group_score = T.alloc_local((1,), "float32")
                previous_group = T.alloc_local((1,), "int32")
                best_group_score = T.alloc_local((1,), "float32")
                best_group = T.alloc_local((1,), "int32")
                previous_expert_score = T.alloc_local((1,), "float32")
                previous_expert = T.alloc_local((1,), "int32")
                best_expert_score = T.alloc_local((1,), "float32")
                best_expert = T.alloc_local((1,), "int32")
                selected_sum = T.alloc_local((1,), "float32")

                row_max[0] = -T.infinity(T.float32)
                for expert in T.serial(num_experts):
                    row_max[0] = T.max(
                        row_max[0],
                        T.cast(gating_output[token_id, expert], "float32"),
                    )
                if not renormalize:
                    row_sum[0] = 0.0
                    for expert in T.serial(num_experts):
                        row_sum[0] += T.exp(
                            T.cast(gating_output[token_id, expert], "float32")
                            - row_max[0]
                        )

                if topk_group == num_expert_group:
                    previous_group_score[0] = -T.infinity(T.float32)
                    previous_group[0] = num_expert_group
                else:
                    previous_group_score[0] = T.infinity(T.float32)
                    previous_group[0] = -1
                    for group_rank in T.serial(topk_group):
                        best_group_score[0] = -T.infinity(T.float32)
                        best_group[0] = num_expert_group
                        for group in T.serial(num_expert_group):
                            group_score[0] = -T.infinity(T.float32)
                            for offset in T.serial(experts_per_group):
                                group_score[0] = T.max(
                                    group_score[0],
                                    T.cast(
                                        gating_output[
                                            token_id,
                                            group * experts_per_group + offset,
                                        ],
                                        "float32",
                                    ),
                                )
                            if (
                                group_rank == 0
                                or group_score[0] < previous_group_score[0]
                                or (
                                    group_score[0] == previous_group_score[0]
                                    and group > previous_group[0]
                                )
                            ) and (
                                group_score[0] > best_group_score[0]
                                or (
                                    group_score[0] == best_group_score[0]
                                    and group < best_group[0]
                                )
                            ):
                                best_group_score[0] = group_score[0]
                                best_group[0] = group
                        previous_group_score[0] = best_group_score[0]
                        previous_group[0] = best_group[0]

                previous_expert_score[0] = T.infinity(T.float32)
                previous_expert[0] = -1
                selected_sum[0] = 0.0
                for kth in T.serial(routed_topk):
                    best_expert_score[0] = -T.infinity(T.float32)
                    best_expert[0] = num_experts
                    for group in T.serial(num_expert_group):
                        group_score[0] = -T.infinity(T.float32)
                        for offset in T.serial(experts_per_group):
                            group_score[0] = T.max(
                                group_score[0],
                                T.cast(
                                    gating_output[
                                        token_id,
                                        group * experts_per_group + offset,
                                    ],
                                    "float32",
                                ),
                            )
                        if group_score[0] > previous_group_score[0] or (
                            group_score[0] == previous_group_score[0]
                            and group <= previous_group[0]
                        ):
                            for offset in T.serial(experts_per_group):
                                expert = group * experts_per_group + offset
                                expert_score = T.cast(
                                    gating_output[token_id, expert], "float32"
                                )
                                if (
                                    kth == 0
                                    or expert_score < previous_expert_score[0]
                                    or (
                                        expert_score == previous_expert_score[0]
                                        and expert > previous_expert[0]
                                    )
                                ) and (
                                    expert_score > best_expert_score[0]
                                    or (
                                        expert_score == best_expert_score[0]
                                        and expert < best_expert[0]
                                    )
                                ):
                                    best_expert_score[0] = expert_score
                                    best_expert[0] = expert

                    previous_expert_score[0] = best_expert_score[0]
                    previous_expert[0] = best_expert[0]
                    topk_ids[token_id, kth] = best_expert[0]
                    topk_weights[token_id, kth] = T.exp(
                        best_expert_score[0] - row_max[0]
                    )
                    if not renormalize:
                        topk_weights[token_id, kth] /= T.max(row_sum[0], 1e-20)
                    selected_sum[0] += topk_weights[token_id, kth]

                if num_fused_shared_experts == 1:
                    topk_ids[token_id, routed_topk] = num_experts
                    topk_weights[token_id, routed_topk] = (
                        selected_sum[0] / routed_scaling_factor
                    )

                if renormalize:
                    output_scale = T.alloc_local((1,), "float32")
                    output_scale[0] = 1.0 / T.max(selected_sum[0], 1e-20)
                    for kth in T.serial(output_topk):
                        topk_weights[token_id, kth] *= output_scale[0]
                if apply_routed_scaling_factor:
                    for kth in T.serial(output_topk):
                        topk_weights[token_id, kth] *= routed_scaling_factor

    return grouped_topk_serial


_grouped_topk_parallel_kernel.mode = "lazy"
_grouped_topk_serial_kernel.mode = "lazy"


def grouped_topk_softmax_tilelang(
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    *,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float | None = None,
    apply_routed_scaling_factor_on_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run grouped softmax top-k with a TileLang MUSA kernel."""

    if gating_output.device.type != "musa":
        raise ValueError("TileLang grouped top-k requires a MUSA tensor")
    if gating_output.dim() != 2:
        raise ValueError("gating_output must be a 2D tensor")
    input_dtype = tilelang_dtype(gating_output.dtype)
    num_tokens, num_experts = gating_output.shape
    routed_topk = _check_config(
        num_experts,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
    )
    if num_fused_shared_experts == 1 and (
        routed_scaling_factor is None or routed_scaling_factor == 0
    ):
        raise ValueError("a non-zero routed_scaling_factor is required")
    if (
        renormalize
        and apply_routed_scaling_factor_on_output
        and routed_scaling_factor is None
    ):
        raise ValueError("cannot apply a missing routed_scaling_factor")

    topk_weights = torch.empty(
        (num_tokens, topk),
        dtype=torch.float32,
        device=gating_output.device,
    )
    topk_ids = torch.empty(
        (num_tokens, topk),
        dtype=torch.int32,
        device=gating_output.device,
    )
    if num_tokens == 0:
        return topk_weights, topk_ids

    use_parallel = (
        num_experts <= _PARALLEL_MAX_EXPERTS
        and num_expert_group <= _PARALLEL_MAX_GROUPS
        and routed_topk <= _PARALLEL_MAX_ROUTED_TOPK
    )
    kernel_factory = (
        _grouped_topk_parallel_kernel if use_parallel else _grouped_topk_serial_kernel
    )
    kernel = kernel_factory(
        input_dtype,
        num_experts,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        renormalize,
        apply_routed_scaling_factor_on_output,
    )
    kernel(
        gating_output,
        topk_weights,
        topk_ids,
        float(routed_scaling_factor if routed_scaling_factor is not None else 1.0),
    )
    return topk_weights, topk_ids
