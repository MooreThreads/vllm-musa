# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MATE Mega MoE as a modular-kernel experts implementation.

MATE's split decode/prefill entry points dispatch tokens, run both FP8
expert GEMMs with the gated activation and combine their weighted results
over an intranode symmetric buffer. They are paired with the pass-through
prepare/finalize.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)

# Limits of mate.mega_moe.MegaMoESymmBuffer and transform_weights_for_mega_moe.
BLOCK_SIZE = 128
# The kernels tile hidden and intermediate sizes in 256-wide N blocks.
SHAPE_ALIGNMENT = 256
TOKEN_ALIGNMENT = 256
# fp8_fp8_mega_moe runs its decode kernels up to this many tokens per rank and
# its split prefill stages above it; every rank of a step must take one path.
DECODE_MAX_TOKENS = 128
MAX_RANKS = 8
MAX_TOPK = 32
MAX_INTERMEDIATE = 8192
MAX_TOKENS_PER_RANK = 65536

_supported_activations: frozenset[MoEActivation] = frozenset((MoEActivation.SILU,))


def mate_mega_moe_available() -> bool:
    if getattr(torch.version, "musa", None) is None:
        return False
    try:
        return importlib.util.find_spec("mate.mega_moe") is not None
    except ModuleNotFoundError:
        return False


def require_mate_mega_moe_api(module: Any) -> None:
    """Check the split entry points before allocating collective workspaces."""
    required = ("_run_decode_mega_moe", "_run_prefill_mega_moe")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            "MATE Mega MoE requires the split decode/prefill API "
            f"(MATE 0.2.8.post1); missing {', '.join(missing)}"
        )


def round_up_tokens(num_tokens: int) -> int:
    return -(-max(num_tokens, 1) // TOKEN_ALIGNMENT) * TOKEN_ALIGNMENT


def max_tokens_across_ranks(num_tokens: int) -> int:
    """The largest token count any data-parallel rank feeds this step."""
    if is_forward_context_available():
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is not None:
            return int(dp_metadata.num_tokens_across_dp_cpu.max())
    return num_tokens


def use_decode_kernels(num_tokens: int) -> bool:
    return max_tokens_across_ranks(num_tokens) <= DECODE_MAX_TOKENS


def mate_mega_moe_shape_reasons(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    ep_size: int,
    max_num_tokens: int,
) -> tuple[str, ...]:
    """Shape and topology limits of the MATE Mega MoE kernels."""
    reasons: list[str] = []
    if hidden_size % SHAPE_ALIGNMENT:
        reasons.append(f"hidden size {hidden_size} not divisible by {SHAPE_ALIGNMENT}")
    if intermediate_size % SHAPE_ALIGNMENT:
        reasons.append(
            f"intermediate size {intermediate_size} not divisible by {SHAPE_ALIGNMENT}"
        )
    if intermediate_size > MAX_INTERMEDIATE:
        reasons.append(f"intermediate size {intermediate_size} > {MAX_INTERMEDIATE}")
    if ep_size < 1 or ep_size > MAX_RANKS:
        reasons.append(
            f"{ep_size} expert-parallel ranks (intranode, at most {MAX_RANKS})"
        )
    elif num_experts % ep_size:
        reasons.append(f"{num_experts} experts not divisible by {ep_size} ranks")
    if top_k > MAX_TOPK:
        reasons.append(f"top-k {top_k} > {MAX_TOPK}")
    if round_up_tokens(max_num_tokens) > MAX_TOKENS_PER_RANK:
        reasons.append(f"{max_num_tokens} tokens per rank > {MAX_TOKENS_PER_RANK}")
    return tuple(reasons)


def mate_mega_moe_unsupported_reasons(moe: FusedMoEConfig) -> tuple[str, ...]:
    """Deployment constraints beyond the framework's quant/activation checks."""
    parallel = moe.moe_parallel_config
    reasons = list(
        mate_mega_moe_shape_reasons(
            hidden_size=moe.hidden_dim,
            intermediate_size=moe.intermediate_size_per_partition,
            num_experts=moe.num_experts,
            top_k=moe.experts_per_token,
            ep_size=parallel.ep_size,
            max_num_tokens=moe.max_num_tokens,
        )
    )
    if moe.in_dtype != torch.bfloat16:
        reasons.append(f"activation dtype {moe.in_dtype}")
    if moe.has_bias:
        reasons.append("expert bias")
    if moe.swiglu_alpha is not None or moe.swiglu_beta is not None:
        reasons.append("custom SwiGLU alpha or beta")
    if moe.swiglu_limit is not None:
        reasons.append("SwiGLU clamp limit")
    if moe.skip_final_all_reduce:
        reasons.append("skip_final_all_reduce")
    if moe.is_lora_enabled:
        reasons.append("LoRA")
    if moe.num_logical_experts != moe.num_experts:
        reasons.append("redundant experts")
    vllm_config = get_current_vllm_config()
    if vllm_config.weight_transfer_config is not None:
        reasons.append("runtime weight transfer")
    if vllm_config.parallel_config.nnodes != 1:
        reasons.append("deployment must use a single node")
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        reasons.append("pipeline parallelism")
    if vllm_config.speculative_config is not None:
        reasons.append("speculative decoding")
    if vllm_config.parallel_config.enable_dbo:
        reasons.append("dual batch overlap")
    if parallel.enable_eplb:
        reasons.append("EPLB")
    return tuple(reasons)


@dataclass(frozen=True)
class _BufferKey:
    group: Any
    num_experts: int
    num_max_tokens_per_rank: int
    top_k: int
    hidden_size: int
    intermediate_size: int


# One symmetric buffer per shape serves every MoE layer: layers run one after
# another on the same stream, and the kernels fully consume a step's inputs.
_SYMM_BUFFERS: dict[_BufferKey, Any] = {}


def destroy_mate_mega_moe_buffers(group: Any) -> None:
    """Collectively close this group's buffers before its process group dies."""
    keys = sorted(
        (key for key in _SYMM_BUFFERS if key.group is group),
        key=lambda key: (
            key.num_experts,
            key.num_max_tokens_per_rank,
            key.top_k,
            key.hidden_size,
            key.intermediate_size,
        ),
    )
    for key in keys:
        _SYMM_BUFFERS[key].destroy()
        del _SYMM_BUFFERS[key]


def _get_symm_buffer(key: _BufferKey):
    buffer = _SYMM_BUFFERS.get(key)
    if buffer is None:
        from mate.mega_moe import get_symm_buffer_for_mega_moe

        buffer = get_symm_buffer_for_mega_moe(
            key.group,
            key.num_experts,
            key.num_max_tokens_per_rank,
            key.top_k,
            key.hidden_size,
            key.intermediate_size,
        )
        _SYMM_BUFFERS[key] = buffer
        logger.info_once(
            "MATE Mega MoE symmetric buffer: %.3f GiB for %d tokens per rank",
            buffer.buffer.nbytes / 2**30,
            key.num_max_tokens_per_rank,
        )
    return buffer


class MateMegaMoEExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self._l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._buffer: Any = None
        self._activation_clamp = moe_config.swiglu_limit

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return False, reason
        reasons = mate_mega_moe_unsupported_reasons(moe_config)
        if reasons:
            return False, f"kernel does not support {', '.join(reasons)}"
        return True, None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return mate_mega_moe_available()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key == kFp8Static128BlockSym and activation_key in (
            kFp8Dynamic128Sym,
            None,
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in _supported_activations

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return (
            moe_parallel_config.use_ep
            and not moe_parallel_config.use_batched_activation_format
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (0,), (0,), (M, K)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Interleave the FP8 expert weights and attach the symmetric buffer."""
        from mate import mega_moe

        require_mate_mega_moe_api(mega_moe)
        strategy = layer.expert_map_manager.placement_strategy
        if strategy != "linear":
            raise ValueError(
                f"MATE Mega MoE requires linear expert placement, got {strategy}"
            )
        scale_name = (
            "weight_scale_inv"
            if hasattr(layer, "w13_weight_scale_inv")
            else "weight_scale"
        )
        w13_scale = getattr(layer, f"w13_{scale_name}")
        w2_scale = getattr(layer, f"w2_{scale_name}")
        l1, l2 = mega_moe.transform_weights_for_mega_moe(
            (layer.w13_weight.data, w13_scale.data.float().contiguous()),
            (layer.w2_weight.data, w2_scale.data.float().contiguous()),
        )
        replace_parameter(layer, "w13_weight", l1[0])
        replace_parameter(layer, f"w13_{scale_name}", l1[1])
        replace_parameter(layer, "w2_weight", l2[0])
        replace_parameter(layer, f"w2_{scale_name}", l2[1])
        self._l1_weights = (layer.w13_weight, getattr(layer, f"w13_{scale_name}"))
        self._l2_weights = (layer.w2_weight, getattr(layer, f"w2_{scale_name}"))

        moe = self.moe_config
        self._buffer = _get_symm_buffer(
            _BufferKey(
                group=get_ep_group().device_group,
                num_experts=moe.num_experts,
                num_max_tokens_per_rank=round_up_tokens(moe.max_num_tokens),
                top_k=moe.experts_per_token,
                hidden_size=moe.hidden_dim,
                intermediate_size=moe.intermediate_size_per_partition,
            )
        )
        check = getattr(mega_moe, "_check_transformed_mega_moe_weights", None)
        if check is not None:
            check(self._l1_weights, self._l2_weights, self._buffer)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self._buffer is None:
            raise RuntimeError("MATE Mega MoE weights have not been processed")
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "MATE Mega MoE applies the router weights in its combine"
            )
        num_tokens = hidden_states.size(0)
        if num_tokens > self._buffer.num_max_tokens_per_rank:
            raise ValueError(
                f"MATE Mega MoE got {num_tokens} tokens; the symmetric buffer "
                f"holds {self._buffer.num_max_tokens_per_rank}"
            )
        buffer = self._buffer
        if num_tokens:
            _, x_sf = per_token_group_quant_fp8(
                hidden_states,
                BLOCK_SIZE,
                column_major_scales=False,
                out_q=buffer.x[:num_tokens],
                use_ue8m0=False,
            )
            buffer.x_sf[:num_tokens].copy_(x_sf)
            buffer.topk_idx[:num_tokens].copy_(topk_ids)
            buffer.topk_weights[:num_tokens].copy_(topk_weights)

        y = output if output.is_contiguous() else torch.empty_like(output)
        self._run(y, num_tokens)
        if y is not output:
            output.copy_(y)

    def _run(self, y: torch.Tensor, num_tokens: int) -> None:
        from mate import mega_moe

        decode = use_decode_kernels(num_tokens)
        run = (
            mega_moe._run_decode_mega_moe if decode else mega_moe._run_prefill_mega_moe
        )
        run(
            y,
            self._l1_weights,
            self._l2_weights,
            self._buffer,
            num_tokens,
            "swiglu",
            self._activation_clamp,
            None,
            None,
            True,
        )
