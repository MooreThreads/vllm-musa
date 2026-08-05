# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA allreduce + RMSNorm fusion pass.

This is the MUSA peer of the platform-specific fusion passes wired by vLLM's
post-grad pass manager. It handles the decomposed v0.22 graph shapes:

    custom_all_reduce(input) -> rms_norm(allreduce_output)
    custom_all_reduce(input) -> add(allreduce_output, residual) -> rms_norm(add)

It also handles the equivalent v0.24 IR form without decomposing the existing
vLLM op:

    custom_all_reduce(input) -> fused_add_rms_norm(allreduce_output, residual)

The replacements are opaque MUSA custom ops. Their graph-level ABI preserves
the tensor consumed by downstream users: all-reduced output for no-residual
graphs and residual output for residual graphs.
"""

from __future__ import annotations

import operator
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.compilation.passes.inductor_pass import enable_fake_mode
from vllm.compilation.passes.vllm_inductor_pass import (
    VllmInductorPass,
    VllmPatternMatcherPass,
)
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_musa.fused_allreduce_rmsnorm_ops import (
    musa_fused_allreduce_residual_rms_norm,
    musa_fused_allreduce_residual_rms_norm_no_raw,
    musa_fused_allreduce_rms_norm,
)

logger = init_logger(__name__)


def _rms_input_weight_supported_dtype(match: pm.Match) -> bool:
    """Allow RMSNorm weight dtypes supported by the unfused semantics.

    vllm.ir.ops.rms_norm upcasts activations to fp32 and multiplies in the
    weight dtype before casting the output back to the activation dtype, so
    input/weight dtype equality is not a semantic requirement. Keep this check
    limited to fused-kernel capability: fp16/bf16 activations with either the
    same dtype or fp32 weights.
    """
    for node in match.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        residual = None
        if "fused_add_rms_norm.default" in target:
            if len(node.args) < 3:
                return True
            x, residual, weight = node.args[0], node.args[1], node.args[2]
            variance_size = (
                node.args[4]
                if len(node.args) > 4
                else node.kwargs.get("variance_size")
            )
        elif "rms_norm.default" in target:
            if len(node.args) < 2:
                return True
            x, weight = node.args[0], node.args[1]
            variance_size = (
                node.args[3]
                if len(node.args) > 3
                else node.kwargs.get("variance_size")
            )
        else:
            continue
        if variance_size is not None:
            return False
        if not isinstance(x, fx.Node) or not isinstance(weight, fx.Node):
            return True
        x_dtype = x.meta["val"].dtype
        weight_dtype = weight.meta["val"].dtype
        supported = x_dtype in (torch.float16, torch.bfloat16) and weight_dtype in (
            x_dtype,
            torch.float32,
        )
        if isinstance(residual, fx.Node):
            supported = supported and residual.meta["val"].dtype == x_dtype
        return supported
    return True


class MusaAllReduceRMSNormPattern:
    """Replace allreduce + RMSNorm with a MUSA fused CAR-RMSNorm op."""

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        group_name: str,
        comm_id: int,
    ) -> None:
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.group_name = group_name
        self.comm_id = comm_id

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=self.dtype, device=self.device, **kwargs)

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty(5, 16), self.empty(16)]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = torch.ops.vllm.all_reduce.default(
                input,
                group_name=self.group_name,
            )
            rms = vllm.ir.ops.rms_norm(allreduce_output, weight, self.epsilon)
            return rms, allreduce_output

        def replacement(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rms, allreduce_output = musa_fused_allreduce_rms_norm(
                input,
                weight,
                self.epsilon,
                self.comm_id,
            )
            return rms, allreduce_output

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_supported_dtype,
        )


class MusaAllReduceResidualRMSNormPattern:
    """Replace custom allreduce + residual add + RMSNorm with fused MUSA op."""

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        jit_comm_id: int,
        fused_comm_id: int,
    ) -> None:
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.jit_comm_id = jit_comm_id
        self.fused_comm_id = fused_comm_id

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=self.dtype, device=self.device, **kwargs)

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty(5, 16), self.empty(5, 16), self.empty(16)]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            allreduce_output = torch.ops.vllm.musa_jit_custom_all_reduce.default(
                input,
                self.jit_comm_id,
            )
            residual_output = torch.add(allreduce_output, residual)
            rms = vllm.ir.ops.rms_norm(residual_output, weight, self.epsilon)
            return rms, residual_output, allreduce_output

        def replacement(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            rms, residual_output, allreduce_output = musa_fused_allreduce_residual_rms_norm(
                input,
                residual,
                weight,
                self.epsilon,
                self.fused_comm_id,
            )
            return rms, residual_output, allreduce_output

        def replacement_no_raw(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rms, residual_output = musa_fused_allreduce_residual_rms_norm_no_raw(
                input,
                residual,
                weight,
                self.epsilon,
                self.fused_comm_id,
            )
            return rms, residual_output

        # Graph fragments that only keep (rms, residual_out) or rms do not need
        # the raw all-reduce tensor. Register these narrower dropped-output
        # patterns before the full 3-return pattern; otherwise Inductor can
        # greedily match the full ABI first and route no-copy candidates through
        # the raw-car fused op.
        first_return_only = lambda fn: lambda a, b, c: fn(a, b, c)[0]
        first_two_returns = lambda fn: lambda a, b, c: fn(a, b, c)[:2]
        pm.register_replacement(
            first_two_returns(pattern),  # type: ignore[no-untyped-call]
            replacement_no_raw,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_supported_dtype,
        )

        pm.register_replacement(
            first_return_only(pattern),  # type: ignore[no-untyped-call]
            first_return_only(lambda a, b, c: replacement_no_raw(a, b, c)),  # type: ignore[no-untyped-call]
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_supported_dtype,
        )

        # Keep the full raw-car ABI last for copy-bearing candidates that still
        # need the original all-reduced tensor (for example CAR -> copy_).
        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_supported_dtype,
        )


class MusaAllReduceRMSNormFusionPass(VllmPatternMatcherPass):
    """MUSA-specific no-residual CAR-RMSNorm pattern matcher pass."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.disabled = True
        self.max_token_num: int | None = None
        self.max_tokens_by_comm: int | None = None
        self.jit_comm_max_size: int | None = None
        if not current_platform.is_musa():
            return

        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size <= 1:
            logger.warning_once(
                "MUSA allreduce-rmsnorm fusion pass is disabled for tp_size <= 1."
            )
            return

        if config.model_config is None:
            logger.warning_once(
                "MUSA allreduce-rmsnorm fusion pass is disabled for missing "
                "model_config."
            )
            return

        self.hidden_dim = config.model_config.get_hidden_size()
        if self.model_dtype not in (torch.float16, torch.bfloat16):
            logger.warning_once(
                "MUSA allreduce-rmsnorm fusion only supports fp16/bf16; got %s.",
                self.model_dtype,
            )
            return

        if self.hidden_dim % 8 != 0 or self.hidden_dim > 16384:
            logger.warning_once(
                "MUSA allreduce-rmsnorm fusion requires hidden_dim %% 8 == 0 "
                "and hidden_dim <= 16384; got %d.",
                self.hidden_dim,
            )
            return

        tp_group = get_tp_group()
        self.group_name = tp_group.unique_name
        device_comm = getattr(tp_group, "device_communicator", None)
        ca_comm = getattr(device_comm, "ca_comm", None)
        if ca_comm is None or getattr(ca_comm, "disabled", False):
            logger.warning_once(
                "MUSA allreduce-rmsnorm fusion disabled: missing enabled ca_comm."
            )
            return

        jit_comm = getattr(ca_comm, "_jit_comm", None)
        self.jit_comm_id = getattr(jit_comm, "_comm_id", None)
        if self.jit_comm_id is None:
            logger.warning_once(
                "MUSA allreduce-rmsnorm fusion disabled: missing JIT comm id."
            )
            return

        jit_comm_max_size = getattr(jit_comm, "max_size", None)
        if not isinstance(jit_comm_max_size, int) or jit_comm_max_size <= 0:
            logger.warning_once(
                "MUSA allreduce-rmsnorm fusion disabled: invalid JIT comm "
                "max_size=%r.",
                jit_comm_max_size,
            )
            return
        element_size = torch.empty((), dtype=self.model_dtype).element_size()
        max_tokens_by_comm = self._max_fusable_tokens(
            None, jit_comm_max_size, self.hidden_dim, element_size
        )
        if max_tokens_by_comm < 1:
            logger.warning_once(
                "MUSA allreduce-rmsnorm fusion disabled: JIT comm max_size=%d "
                "cannot hold one token for hidden_dim=%d dtype=%s.",
                jit_comm_max_size,
                self.hidden_dim,
                self.model_dtype,
            )
            return

        # Reuse the JIT communicator registry that already owns the compiled
        # custom-allreduce ABI and removes the id during communicator close.
        self.comm_id = self.jit_comm_id
        self.patterns = PatternMatcherPass(pass_name="musa_all_reduce_rms_fusion_pass")
        self.jit_comm_max_size = jit_comm_max_size
        self.max_tokens_by_comm = max_tokens_by_comm
        self.max_token_num = self._max_fusable_tokens(
            config.scheduler_config.max_num_batched_tokens,
            jit_comm_max_size,
            self.hidden_dim,
            element_size,
        )
        self.register_patterns()
        self.dump_patterns(config, self.patterns)
        logger.warning_once(
            "MUSA allreduce-rmsnorm fusion pass enabled: group=%s tp_size=%d "
            "hidden_dim=%d max_token_num=%s max_tokens_by_comm=%d comm_id=%d "
            "jit_comm_id=%d.",
            self.group_name,
            self.tp_size,
            self.hidden_dim,
            self.max_token_num,
            self.max_tokens_by_comm,
            self.comm_id,
            self.jit_comm_id,
        )

    @staticmethod
    def _max_fusable_tokens(
        scheduler_limit: int | None,
        max_size: int,
        hidden_dim: int,
        element_size: int,
    ) -> int:
        if max_size <= 0 or hidden_dim <= 0 or element_size <= 0:
            return 0
        max_tokens = max_size // (hidden_dim * element_size)
        if scheduler_limit is None:
            return max_tokens
        return min(scheduler_limit, max_tokens)

    @enable_fake_mode
    def register_patterns(self) -> None:
        for epsilon in [1e-5, 1e-6]:
            MusaAllReduceRMSNormPattern(
                epsilon,
                self.model_dtype,
                self.device,
                self.group_name,
                self.comm_id,
            ).register(self.patterns)
            torch._inductor.pattern_matcher._seen_patterns.clear()

            MusaAllReduceResidualRMSNormPattern(
                epsilon,
                self.model_dtype,
                self.device,
                self.jit_comm_id,
                self.comm_id,
            ).register(self.patterns)
            # Clear the pattern cache so both eps values can register equivalent
            # graph shapes.
            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.disabled = False

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if self.disabled:
            return False
        if self.max_token_num is None:
            return True
        return bool(compile_range.end <= self.max_token_num)

    def uuid(self) -> str:
        """Include runtime rewrite state in the Inductor disk-cache key."""
        state: dict[str, Any] = {
            "source": self.hash_source(
                self,
                MusaAllReduceRMSNormPattern,
                MusaAllReduceResidualRMSNormPattern,
            ),
            "enabled": not self.disabled,
            "max_token_num": self.max_token_num,
            "max_tokens_by_comm": self.max_tokens_by_comm,
            "jit_comm_max_size": self.jit_comm_max_size,
        }
        if not self.disabled:
            state.update(
                {
                    "comm_id": self.comm_id,
                    "jit_comm_id": self.jit_comm_id,
                    "group_name": self.group_name,
                    "tp_size": self.tp_size,
                    "hidden_dim": self.hidden_dim,
                    "model_dtype": str(self.model_dtype),
                }
            )
        return self.hash_dict(state)

    @staticmethod
    def _target_name(node: fx.Node) -> str:
        return str(getattr(node, 'target', ''))

    @classmethod
    def _is_musa_car_node(cls, node: fx.Node) -> bool:
        return (
            node.op == 'call_function'
            and 'musa_jit_custom_all_reduce' in cls._target_name(node)
        )

    @staticmethod
    def _car_comm_id(node: fx.Node) -> Any | None:
        if len(node.args) > 1:
            return node.args[1]
        return node.kwargs.get("comm_id")

    def _is_target_musa_car_node(self, node: fx.Node) -> bool:
        return self._is_musa_car_node(node) and (
            self._car_comm_id(node) == self.jit_comm_id
        )

    @classmethod
    def _is_rms_norm_node(cls, node: fx.Node) -> bool:
        target = cls._target_name(node)
        return (
            node.op == 'call_function'
            and 'rms_norm.default' in target
            and 'fused_add_rms_norm.default' not in target
        )

    @classmethod
    def _is_fused_add_rms_norm_node(cls, node: fx.Node) -> bool:
        return (
            node.op == 'call_function'
            and 'fused_add_rms_norm.default' in cls._target_name(node)
        )

    @classmethod
    def _is_add_node(cls, node: fx.Node) -> bool:
        if (
            node.op != "call_function"
            or node.target is not torch.ops.aten.add.Tensor
            or len(node.args) != 2
            or any(name != "alpha" for name in node.kwargs)
        ):
            return False
        alpha = node.kwargs.get("alpha", 1)
        return isinstance(alpha, (int, float, complex)) and alpha == 1

    @staticmethod
    def _node_tensor_meta(node: Any) -> torch.Tensor | None:
        if not isinstance(node, fx.Node):
            return None
        value = node.meta.get("val")
        return value if isinstance(value, torch.Tensor) else None

    def _manual_residual_inputs_supported(
        self, car: fx.Node, residual: Any, weight: Any
    ) -> bool:
        """Fail closed unless manual-rewrite inputs satisfy the fused ABI."""
        if len(car.args) < 1:
            return False

        input_value = self._node_tensor_meta(car.args[0])
        car_value = self._node_tensor_meta(car)
        residual_value = self._node_tensor_meta(residual)
        weight_value = self._node_tensor_meta(weight)
        if any(
            value is None
            for value in (input_value, car_value, residual_value, weight_value)
        ):
            return False

        assert input_value is not None
        assert car_value is not None
        assert residual_value is not None
        assert weight_value is not None
        try:
            hidden_size = input_value.shape[-1]
            return bool(
                input_value.device.type == "musa"
                and car_value.device == input_value.device
                and residual_value.device == input_value.device
                and weight_value.device == input_value.device
                and input_value.dim() == 2
                and car_value.dim() == 2
                and residual_value.dim() == 2
                and weight_value.dim() == 1
                and car_value.shape == input_value.shape
                and residual_value.shape == input_value.shape
                and hidden_size == self.hidden_dim
                and hidden_size > 0
                and hidden_size % 8 == 0
                and hidden_size <= 16384
                and weight_value.numel() == hidden_size
                and input_value.dtype in (torch.float16, torch.bfloat16)
                and car_value.dtype == input_value.dtype
                and residual_value.dtype == input_value.dtype
                and weight_value.dtype in (input_value.dtype, torch.float32)
                and input_value.is_contiguous()
                and car_value.is_contiguous()
                and residual_value.is_contiguous()
                and weight_value.is_contiguous()
            )
        except (AttributeError, IndexError, RuntimeError, TypeError):
            return False

    @staticmethod
    def _rms_norm_weight(node: fx.Node) -> Any | None:
        if len(node.args) > 1:
            return node.args[1]
        return node.kwargs.get("weight")

    @staticmethod
    def _rms_norm_eps(node: fx.Node) -> Any | None:
        if len(node.args) > 2:
            return node.args[2]
        return node.kwargs.get("epsilon", node.kwargs.get("eps"))

    @staticmethod
    def _rms_norm_variance_size(node: fx.Node) -> Any | None:
        if len(node.args) > 3:
            return node.args[3]
        return node.kwargs.get("variance_size")

    @staticmethod
    def _other_add_arg(add: fx.Node, car: fx.Node) -> Any | None:
        if len(add.args) < 2:
            return None
        lhs, rhs = add.args[0], add.args[1]
        if lhs is car:
            return rhs
        if rhs is car:
            return lhs
        return None

    @staticmethod
    def _fused_add_rms_norm_args(
        node: fx.Node, car: fx.Node
    ) -> tuple[Any, Any, Any, Any] | None:
        """Return residual, weight, epsilon and variance size for vLLM 0.24."""
        if len(node.args) < 4 or node.args[0] is not car:
            return None
        variance_size = (
            node.args[4] if len(node.args) > 4 else node.kwargs.get("variance_size")
        )
        return node.args[1], node.args[2], node.args[3], variance_size

    @staticmethod
    def _getitem_index(node: fx.Node, parent: fx.Node) -> int | None:
        if (
            node.op != "call_function"
            or node.target is not operator.getitem
            or len(node.args) < 2
            or node.args[0] is not parent
            or not isinstance(node.args[1], int)
        ):
            return None
        return int(node.args[1])

    def _manual_rewrite_residual_musa_jit_car_rmsnorm(
        self, graph: fx.Graph
    ) -> tuple[int, int]:
        """Rewrite the 0.22 add IR and 0.24 fused-add IR explicitly.

        PatternMatcher can greedily keep matching the full 3-output ABI even when
        the raw all-reduce value is not used. This rewrite separates the two
        residual cases from actual graph users:
        - copy/other CAR users present: keep raw 3-output fused ABI.
        - only add->RMSNorm needs the CAR result: use the 2-output no-raw ABI.
        """
        no_raw_replaced = 0
        raw_replaced = 0

        for car in list(graph.nodes):
            if not self._is_target_musa_car_node(car):
                continue
            if len(car.args) < 1:
                continue

            # vLLM 0.24 lowers add + RMSNorm to a two-output IR node:
            # fused_add_rms_norm(CAR, residual, weight, eps) -> (rms, residual).
            # Keep the raw/no-raw user routing and adapt only how this combined
            # node's inputs and outputs are accessed.
            rewrote_fused_add = False
            fused_add_users = [
                user
                for user in list(car.users)
                if self._is_fused_add_rms_norm_node(user)
            ]
            for fused_add in fused_add_users:
                fused_add_args = self._fused_add_rms_norm_args(fused_add, car)
                if fused_add_args is None:
                    continue
                residual, weight, eps, variance_size = fused_add_args
                if weight is None or eps is None or variance_size is not None:
                    continue
                if not self._manual_residual_inputs_supported(
                    car, residual, weight
                ):
                    continue

                output_users = list(fused_add.users)
                output_indices = {
                    user: self._getitem_index(user, fused_add)
                    for user in output_users
                }
                if not output_users or any(
                    index not in (0, 1) for index in output_indices.values()
                ):
                    continue

                raw_users = [
                    user for user in list(car.users) if user is not fused_add
                ]
                use_raw = bool(raw_users)

                with graph.inserting_before(fused_add):
                    if use_raw:
                        fused = graph.call_function(
                            torch.ops.vllm.musa_fused_allreduce_residual_rms_norm.default,
                            args=(car.args[0], residual, weight, eps, self.comm_id),
                        )
                        fused_rms = graph.call_function(
                            operator.getitem, args=(fused, 0)
                        )
                        fused_residual = graph.call_function(
                            operator.getitem, args=(fused, 1)
                        )
                        fused_raw = graph.call_function(
                            operator.getitem, args=(fused, 2)
                        )
                    else:
                        fused = graph.call_function(
                            torch.ops.vllm.musa_fused_allreduce_residual_rms_norm_no_raw.default,
                            args=(car.args[0], residual, weight, eps, self.comm_id),
                        )
                        fused_rms = graph.call_function(
                            operator.getitem, args=(fused, 0)
                        )
                        fused_residual = graph.call_function(
                            operator.getitem, args=(fused, 1)
                        )
                        fused_raw = None

                for user, index in output_indices.items():
                    replacement = fused_rms if index == 0 else fused_residual
                    user.replace_all_uses_with(replacement)
                if fused_raw is not None:
                    for user in raw_users:
                        user.replace_input_with(car, fused_raw)

                for user in output_users:
                    if len(user.users) == 0:
                        graph.erase_node(user)
                if len(fused_add.users) == 0:
                    graph.erase_node(fused_add)
                if len(car.users) == 0:
                    graph.erase_node(car)

                if use_raw:
                    raw_replaced += 1
                else:
                    no_raw_replaced += 1
                rewrote_fused_add = True
                break

            if rewrote_fused_add:
                continue

            for add in [user for user in list(car.users) if self._is_add_node(user)]:
                residual = self._other_add_arg(add, car)
                if residual is None:
                    continue

                rms_users = [
                    user
                    for user in list(add.users)
                    if self._is_rms_norm_node(user)
                    and len(user.args) >= 1
                    and user.args[0] is add
                ]
                if not rms_users:
                    continue

                rms = rms_users[0]
                weight = self._rms_norm_weight(rms)
                eps = self._rms_norm_eps(rms)
                variance_size = self._rms_norm_variance_size(rms)
                if weight is None or eps is None or variance_size is not None:
                    continue
                if not self._manual_residual_inputs_supported(
                    car, residual, weight
                ):
                    continue

                raw_users = [user for user in list(car.users) if user is not add]
                use_raw = bool(raw_users)

                with graph.inserting_before(rms):
                    if use_raw:
                        fused = graph.call_function(
                            torch.ops.vllm.musa_fused_allreduce_residual_rms_norm.default,
                            args=(car.args[0], residual, weight, eps, self.comm_id),
                        )
                        fused_rms = graph.call_function(
                            operator.getitem, args=(fused, 0)
                        )
                        fused_residual = graph.call_function(
                            operator.getitem, args=(fused, 1)
                        )
                        fused_raw = graph.call_function(
                            operator.getitem, args=(fused, 2)
                        )
                    else:
                        fused = graph.call_function(
                            torch.ops.vllm.musa_fused_allreduce_residual_rms_norm_no_raw.default,
                            args=(car.args[0], residual, weight, eps, self.comm_id),
                        )
                        fused_rms = graph.call_function(
                            operator.getitem, args=(fused, 0)
                        )
                        fused_residual = graph.call_function(
                            operator.getitem, args=(fused, 1)
                        )
                        fused_raw = None

                rms.replace_all_uses_with(fused_rms)
                for user in list(add.users):
                    if user is not rms:
                        user.replace_input_with(add, fused_residual)
                if fused_raw is not None:
                    for user in raw_users:
                        user.replace_input_with(car, fused_raw)

                graph.erase_node(rms)
                if len(add.users) == 0:
                    graph.erase_node(add)
                if len(car.users) == 0:
                    graph.erase_node(car)

                if use_raw:
                    raw_replaced += 1
                else:
                    no_raw_replaced += 1
                break

        replaced = no_raw_replaced + raw_replaced
        if replaced:
            graph.lint()
            graph.eliminate_dead_code()

        return no_raw_replaced, raw_replaced

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if self.disabled:
            return

        manual_no_raw, manual_raw = (
            self._manual_rewrite_residual_musa_jit_car_rmsnorm(graph)
        )
        manual_count = manual_no_raw + manual_raw
        if manual_count:
            self.matched_count = manual_count
            return

        self.matched_count = self.patterns.apply(graph)
