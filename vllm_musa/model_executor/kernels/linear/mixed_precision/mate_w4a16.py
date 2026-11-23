# SPDX-License-Identifier: Apache-2.0
"""MATE-backed W4A16 mixed-precision linear kernel for vLLM-MUSA (ticket MUSA-100077, stage 3).

Registration happens through this package's ``__init__.py``, which the ``vllm.general_plugins`` entry
point in ``pyproject.toml`` calls; it registers the class below for ``(PlatformEnum.OOT, "mp")`` and
stays importable when MATE is absent. Upstream selects
W4A16 kernels through the ``MPLinearKernel`` ABC (``MPLinearKernel.py:26``), exactly as it already does
for ``TritonW4A16LinearKernel`` (:273), ``RDNA3W4A16LinearKernel`` (:26), ``RDNAHybridW4A16LinearKernel``
(:442) and ``CPUWNA16LinearKernel`` (:20), so no fork-local hook is needed and the reusable upstream
Marlin Python surface (``marlin_utils.py``, the padding/``g_idx`` conventions, ``is_marlin_compatible``)
stays in place.

Contract notes that are load-bearing rather than stylistic:

* ``can_implement`` rejects any group size other than 128. That is not a policy choice: MATE's K256
  mainloop tile carries exactly two K128 weight-scale blocks (asserted in the device header), so other
  groups would need a different tile, not a different argument.
* ``process_weights_after_loading`` *replaces* the packed weight with the repacked buffer instead of
  keeping both. One resident layout, decided inside the op, is what keeps memory honest and keeps the
  serving path free of a second copy (ticket constraint 1).
* ``apply_weights`` has no branch on M. Decode and prefill take the same call; the M-dependence lives
  inside MATE's scheduler, so nothing is frozen at trace time under ``mode=vllm_compile``.
* ``g_idx``/act-order is supported by materializing the permutation into the packed weights at load
  time, because the kernel's smem load copies a K tile contiguously and has no per-element index to hook.

This module owns no Marlin semantics. MATE declares ``prepare_marlin_w4a16`` and
``apply_gptq_marlin_linear`` as "the two names a consumer calls" (``mate/marlin/__init__.py``), and they
absorb everything this module used to hand-roll: the repack, the scale permutation, the zero-point
conversion, the load-time ordering, and the M-on-M scheduling decision. So there are exactly two call
sites here, and the earlier "provisional / blocked on P1" notes are retired: MATE's
``gptq_marlin_repack`` / ``awq_marlin_repack`` / ``marlin_permute_scales`` / ``awq_to_marlin_zero_points``
already exist and are exported.
"""

from __future__ import annotations

import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.scalar_type import scalar_types

from mate.marlin import (  # MATE's declared consumer surface, not its internals
    apply_gptq_marlin_linear,
    prepare_w4a16_from_gptq_frame,
    unpack_zero_points,
)

MARLIN_GROUP_SIZE = 128


def _packed_zero_point_dim(shape, num_groups: int, size_n: int) -> int:
    """State which axis holds the 8 N-nibbles per zero-point word.

    Both orientations occur and MATE refuses to guess on a square tensor: GPTQ checkpoints store
    (num_groups, N/8) while vLLM's AWQ-to-standard conversion produces (N/8, num_groups) (see
    mate/marlin/loader.py:unpack_zero_points). Naming it here, from the shapes, keeps the failure a
    reasoned error rather than a wrong frame.
    """
    if shape == (num_groups, size_n // 8):
        return 1
    if shape == (size_n // 8, num_groups):
        return 0
    raise ValueError(
        f"zero points of shape {shape} are neither (num_groups, N/8)={(num_groups, size_n // 8)} nor "
        f"(N/8, num_groups)={(size_n // 8, num_groups)} for N={size_n}, K/groups={num_groups}"
    )
SUPPORTED_WEIGHT_TYPES = (scalar_types.uint4, scalar_types.uint4b8)


class MateW4A16LinearKernel(MPLinearKernel):
    """W4A16 GEMM served by MATE's dense Marlin-class kernel on mp31."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 31

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if c.act_type != torch.float16:
            return False, "MATE W4A16 needs an fp16 activation: the atom is int4 x fp16 SQMMA"
        if c.weight_type not in SUPPORTED_WEIGHT_TYPES:
            return False, f"unsupported weight type {c.weight_type}"
        if c.group_size != MARLIN_GROUP_SIZE:
            return (
                False,
                "MATE's K256 tile holds two K128 weight-scale blocks, so group size is structurally "
                f"128, not {c.group_size}",
            )
        if c.partition_weight_shape[0] % 256:
            return (
                False,
                "MATE's K256 mainloop tile needs K % 256 == 0 (group 128 x two scale blocks), got "
                f"K={c.partition_weight_shape[0]}",
            )
        if c.out_type not in (None, torch.float16, torch.bfloat16):
            return False, f"unsupported output type {c.out_type}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """One call to MATE's single load-time entry; MATE owns every step inside it."""
        c = self.config
        size_k, size_n = c.partition_weight_shape

        # The layer carries the loader's GPTQ-layout words at this point (vLLM converts an AWQ checkpoint to
        # the standard GPTQ layout during load), so the frame is built with the from-checkpoint entry -- see
        # mate/marlin/__init__.py: prepare_marlin_w4a16 takes upstream's *already repacked* tensors, and
        # handing it checkpoint words makes compat.marlin_repacked_nk read a (K/8, N) tensor as (K/16, 2N),
        # which surfaced as "packed (n=768, k=3072) vs logical (n=1536, k=1536)".
        # Zero points are taken only when the CONFIG declares them, never from whether the layer carries a
        # `qzeros` tensor: auto_gptq registers `qzeros` for every GPTQ layer but builds MPLinearLayerConfig
        # with zero_points=False (auto_gptq.py:347) because its TYPE_MAP only admits symmetric checkpoints
        # (uint4b8, i.e. an implicit bias of 8), and vLLM's own Marlin path likewise applies
        # marlin_zero_points only `if c.zero_points`. Reading qzeros here would feed a symmetric checkpoint's
        # dummy tensor to the kernel as real zero points -- the "already-converted zero points converted a
        # second time" mechanism this ticket already recorded once. AWQ is the asymmetric case (uint4,
        # zero_points=True) and keeps the unpack below unchanged.
        zero_points = None
        if c.zero_points:
            assert self.w_zp_name is not None
            num_groups = size_k // c.group_size
            zp = getattr(layer, self.w_zp_name)
            packed_dim = _packed_zero_point_dim(tuple(zp.shape), num_groups, size_n)
            zero_points = unpack_zero_points(zp, num_groups, size_n, packed_dim)
        # Act order is taken only when the CONFIG declares it. vLLM registers a `g_idx` parameter for every
        # GPTQ layer (auto_gptq.py:398) but a non-desc_act checkpoint does not fill it, so deciding from the
        # parameter's presence would read uninitialized memory: `has_g_idx` (== quant_config.desc_act) is
        # the authoritative flag, and it is what MPLinearKernel.__init__ asserts on. When it is set, MATE's
        # loader entry sorts the k rows itself (resolve_act_order with sort_indices=None) and keeps the
        # permutation in the frame, which apply_gptq_marlin_linear gathers the activation with -- so no
        # g_idx_sort_indices argument is needed here, unlike the marlin-frame entry.
        g_idx = None
        if self.w_gidx_name is not None and c.has_g_idx:
            g_idx = getattr(layer, self.w_gidx_name)
        prepared = prepare_w4a16_from_gptq_frame(
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            zero_points,
            size_k=size_k,
            size_n=size_n,
            group_size=c.group_size,
            g_idx=g_idx,
        )
        self._frame = prepared

        # One resident layout: the layer now carries MATE's frame tensors, not the pre-repack ones.
        setattr(layer, self.w_q_name, Parameter(prepared.weight_b.contiguous(), requires_grad=False))
        setattr(layer, self.w_s_name, Parameter(prepared.scale_b.contiguous(), requires_grad=False))
        if prepared.zero_points_b is not None and self.w_zp_name is not None:
            setattr(
                layer, self.w_zp_name, Parameter(prepared.zero_points_b.contiguous(), requires_grad=False)
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """No branch on M: MATE's entry runs decode and prefill on the same path."""
        pre = self._frame
        out = apply_gptq_marlin_linear(x, pre, bias=bias)
        # Unconditional, per MarlinW4A16KernelFrame.unpad_output: `padded` is a property, and a python-level
        # predicate in the traced path gets constant-folded by Dynamo and then *called*, which fails as
        # "'bool' object is not callable". With logical_n == padded_n the slice is already the identity.
        return pre.unpad_output(out)
