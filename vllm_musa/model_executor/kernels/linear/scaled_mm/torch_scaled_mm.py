# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
    _get_num_tokens,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonInt8ScaledMMLinearKernel,
)
from vllm.platforms import current_platform


class MUSATorchFP8ScaledMMKernelMixin:
    """Enable torch._scaled_mm-backed FP8 kernels on MUSA.

    The upstream Torch FP8 kernels are written for CUDA/ROCm/CPU and reject low
    CUDA compute capabilities. S5000 exposes FP8 scaled-mm through torch_musa, so
    MUSA needs only a platform-specific support gate; the implementation can use
    the upstream apply/process logic with MUSA-safe scale contiguity.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_musa():
            return False, "requires MUSA."
        return True, None

    def get_output_padding(self) -> int | None:
        # torch_musa/muDNN cannot currently pad FP8 tensors via torch.nn.functional.pad.
        # Padding is only a torch._scaled_mm performance hint, so skip it on MUSA.
        return None


class MUSAPerTensorTorchFP8ScaledMMLinearKernel(
    MUSATorchFP8ScaledMMKernelMixin,
    PerTensorTorchFP8ScaledMMLinearKernel,
):
    pass


class MUSARowWiseTorchFP8ScaledMMLinearKernel(
    MUSATorchFP8ScaledMMKernelMixin,
    RowWiseTorchFP8ScaledMMLinearKernel,
):
    """Fused rowwise ``torch._scaled_mm`` FP8 linear kernel on MUSA (mublasLt).

    Eliminates ``ChannelWiseTorchFP8ScaledMMLinearKernel``'s per-call
    ``torch.ones(1)`` allocation, ``[M, N]`` FP32 intermediate buffer, and three
    elementwise dequant/cast launches for per-token × per-channel FP8 layers.
    """

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        scale_b = (Bs.view(1, -1) if Bs.dim() == 1 else Bs.t()).contiguous()
        scale_a = (As.view(-1, 1) if As.dim() == 1 else As).contiguous()
        output = torch._scaled_mm(
            A,
            B,
            out_dtype=out_dtype,
            scale_a=scale_a,
            scale_b=scale_b,
            bias=bias,
        )
        if type(output) is tuple and len(output) == 2:
            output = output[0]
        num_tokens = _get_num_tokens(output_shape)
        return torch.narrow(output, 0, 0, num_tokens).view(*output_shape)


class MUSAChannelWiseTorchFP8ScaledMMLinearKernel(
    MUSATorchFP8ScaledMMKernelMixin,
    ChannelWiseTorchFP8ScaledMMLinearKernel,
):
    """Fallback for mixed per-tensor/per-channel scales or FP16 outputs."""

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        unit_scale = getattr(self, "_musa_unit_scale", None)
        if unit_scale is None or unit_scale.device != A.device:
            unit_scale = torch.ones(1, dtype=torch.float32, device=A.device)
            self._musa_unit_scale = unit_scale

        output = torch._scaled_mm(
            A,
            B,
            scale_a=unit_scale,
            scale_b=unit_scale,
            out_dtype=torch.float32,
        )
        if type(output) is tuple and len(output) == 2:
            output = output[0]

        num_tokens = _get_num_tokens(output_shape)
        output = torch.narrow(output, 0, 0, num_tokens)
        x_scale = (
            torch.narrow(As, 0, 0, num_tokens)
            if As.dim() > 0 and As.shape[0] >= num_tokens
            else As
        )
        output = output * x_scale * Bs.t()
        if bias is not None:
            output = output + bias
        return output.to(out_dtype).view(*output_shape)


class MUSATorchInt8ScaledMMLinearKernel(TritonInt8ScaledMMLinearKernel):
    """MUSA W8A8 INT8 linear kernel backed by native ``scaled_int8_quant`` + mublasLt.

    ``triton_musa`` (Triton 3.2) lacks ``int8 x int8 -> int32`` SQMMA lowering
    and ``muDNN`` rejects ``torch._int_mm``. Because every ``int8`` integer in
    ``[-128, 127]`` is bit-exact in ``bfloat16`` (7-bit significand + exponent
    range ``3.4e38`` avoiding FP16 accumulator overflow before dequant scaling),
    this kernel keeps weights stored as compact ``[K, N]`` ``int8`` in VRAM and
    executes the unscaled integer dot product on S5000's BF16 SQMMA Tensor Cores
    via ``torch.mm`` with FP32 accumulation, zero-point correction, and scaling.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_musa():
            return False, "requires MUSA."
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)
        output_shape = [*x.shape[:-1], w_q.shape[1]]
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()

        symmetric = azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(x_2d, i_s, i_zp, symmetric=symmetric)

        # Every int8 value [-128, 127] is bit-exact in bfloat16, and bfloat16's
        # exponent range prevents unscaled integer dot sums from overflowing
        # float16 (65504) when x.dtype == torch.float16.
        mm_out = torch.mm(x_q.to(torch.bfloat16), w_q.to(torch.bfloat16)).to(
            torch.float32
        )

        if azp_adj is not None:
            static = i_zp is not None
            if not static and x_zp is not None:
                mm_out = mm_out - (x_zp.to(torch.int32) * azp_adj).to(torch.float32)
            else:
                mm_out = mm_out - azp_adj.to(torch.float32)

        scale_a = x_s.reshape(-1, 1) if x_s.dim() > 0 else x_s
        scale_b = w_s.reshape(1, -1) if w_s.dim() > 0 else w_s
        out = mm_out * scale_a * scale_b
        if bias is not None:
            out = out + bias
        return out.to(x.dtype).view(*output_shape)
