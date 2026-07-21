# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm, RMSNormGated

from vllm_musa.jit_kernel.csrc import norm as musa_jit_norm
from vllm_musa.utils.environ import envs


def _can_use_musa_jit_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    hidden_size = x.shape[-1]
    return (
        x.device.type == "musa"
        and weight.device.type == "musa"
        and x.dim() == 2
        and weight.dim() == 1
        and hidden_size > 0
        and hidden_size == weight.numel()
        and hidden_size <= 32768
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and x.is_contiguous()
        and weight.is_contiguous()
    )


@RMSNorm.register_oot
class MusaRMSNorm(RMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()
            or self.variance_size_override is not None
        ):
            return self.forward_native(x, residual)

        if residual is not None:
            # All residual calls use IR. It validates donation and selects the
            # measured JIT kernel, the broad C-extension kernel, or native.
            return self.forward_native(x, residual)

        weight = self.weight.data
        eps = self.variance_epsilon
        if _can_use_musa_jit_rmsnorm(x, weight):
            return musa_jit_norm.rmsnorm(x, weight, eps)

        return nn.functional.rms_norm(x, (self.hidden_size,), weight, eps)


@GemmaRMSNorm.register_oot
class MusaGemmaRMSNorm(GemmaRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()
            or getattr(self, "variance_size_override", None) is not None
        ):
            return self.forward_native(x, residual)

        if residual is not None:
            weight = self.weight.data
            if (
                _can_use_musa_jit_rmsnorm(x, weight)
                and residual.shape == x.shape
                and residual.dtype == x.dtype
                and residual.is_contiguous()
            ):
                # MUSA: fused residual-add + Gemma RMSNorm in one JIT kernel;
                # weight is the raw zero-centered param, gemma=True applies +1.
                return musa_jit_norm.fused_add_rmsnorm(
                    x, residual, weight, self.variance_epsilon, gemma=True
                )
            return self.forward_native(x, residual)

        weight = self.weight.data
        if x.dim() != 2:
            # MUSA: q/k GemmaRMSNorm gets a 3D [tokens, heads, head_dim] tensor;
            # the fused rmsnorm gate needs 2D, so reshape to 2D, run the fused
            # kernel, and reshape back (mirrors SGLang RMSNorm.forward_cuda).
            x2 = x.contiguous().reshape(-1, x.shape[-1])
            if x2.shape[0] > 0 and _can_use_musa_jit_rmsnorm(x2, weight):
                y = musa_jit_norm.gemma_rmsnorm(x2, weight, self.variance_epsilon)
                return y.reshape(x.shape)
            return self.forward_native(x, residual)

        if _can_use_musa_jit_rmsnorm(x, weight):
            return musa_jit_norm.gemma_rmsnorm(x, weight, self.variance_epsilon)

        return self.forward_native(x, residual)


@RMSNormGated.register_oot
class MusaRMSNormGated(RMSNormGated):
    def forward_oot(self, x, z=None):
        if (
            z is not None
            and self.bias is None
            and (self.group_size is None or self.group_size == x.shape[-1])
            and self.norm_before_gate
            and self.activation in ("silu", "swish")
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.is_contiguous()
            and z.is_contiguous()
        ):
            try:
                from vllm_musa.jit_kernel.tilelang.layernorm_gated import (
                    rms_norm_gated,
                )

                return rms_norm_gated(
                    x=x,
                    weight=self.weight.data,
                    bias=None,
                    z=z,
                    eps=self.eps,
                    group_size=None,
                    norm_before_gate=True,
                    is_rms_norm=True,
                    activation="silu",
                )
            except Exception:
                pass
        return self.forward_native(x, z)
