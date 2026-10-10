from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_musa.jit_kernel import rotary_embedding
from vllm_musa.utils.environ import envs


@RotaryEmbedding.register_oot
class MusaRotaryEmbedding(RotaryEmbedding):
    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get():
            return self.forward_native(positions, query, key)

        # do NOT reassign self.cos_sin_cache inside forward.
        # CUDAGraph-aware Dynamo flags `self.cos_sin_cache = ...` as a
        # buffer mutation and refuses to compile (RuntimeError: Assigning
        # / modifying buffers of nn.Module during forward pass is not
        # allowed when using cudagraph). Use a local variable instead;
        # .to() is a no-op when device/dtype already match.
        cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)

        rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            cos_sin_cache,
            self.is_neox_style,
        )
        return query, key


class MusaVisionRotaryPositions(nn.Module):
    """Keep an eager position buffer alive while a captured graph uses it."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_positions", None, persistent=False)
        self._key: tuple[int, int, torch.device] | None = None
        self._graph_pinned = False

    @staticmethod
    def _is_capturing(device: torch.device) -> bool:
        if device.type != "musa":
            return False
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except (AttributeError, RuntimeError):
            return True

    def get(self, batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (batch, seq_len, device)
        capturing = self._is_capturing(device)
        if (
            self._positions is not None
            and self._key == key
            and self._positions.device == device
        ):
            if capturing:
                self._graph_pinned = True
            return self._positions

        positions = torch.arange(seq_len, device=device, dtype=torch.long).repeat(batch)
        # A graph capture must not publish storage into this shared buffer.
        if not capturing and not self._graph_pinned:
            self._positions = positions
            self._key = key
        return positions


class MusaVisionApplyRotaryEmb(ApplyRotaryEmb):
    """Route a selected vision layer's x/cos/sin API to the existing MUSA RoPE."""

    def __init__(
        self,
        *,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
        inplace: bool = False,
        flatten: bool = False,
        positions_cache: MusaVisionRotaryPositions | None = None,
        required_bf16_neox_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            enforce_enable=True,
            is_neox_style=is_neox_style,
            enable_fp32_compute=enable_fp32_compute,
        )
        self.inplace = inplace
        self.flatten = flatten
        self.positions_cache = positions_cache
        self.required_bf16_neox_shape = required_bf16_neox_shape

    def forward_oot(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        if x.device.type != "musa":
            return self.forward_native(x, cos, sin)
        if self.required_bf16_neox_shape is not None:
            head_size, cos_width = self.required_bf16_neox_shape
            if (
                x.dtype != torch.bfloat16
                or x.shape[-1] != head_size
                or cos.ndim != 2
                or cos.shape[-1] != cos_width
                or sin.shape != cos.shape
                or not self.is_neox_style
                or self.enable_fp32_compute
            ):
                return self.forward_native(x, cos, sin)
        x, cos, sin, origin_shape, origin_dtype = self._pre_process(x, cos, sin)
        output = x.contiguous() if self.inplace else x.clone(
            memory_format=torch.contiguous_format
        )
        batch, seq_len, num_heads, head_size = output.shape
        cos_sin_cache = torch.cat((cos, sin), dim=-1).to(output.dtype).contiguous()
        if self.flatten:
            positions = (
                self.positions_cache.get(batch, seq_len, output.device)
                if self.positions_cache is not None
                else torch.arange(seq_len, device=output.device).repeat(batch)
            )
            query = output.reshape(batch * seq_len, num_heads * head_size)
        else:
            positions = torch.arange(
                seq_len, device=output.device, dtype=torch.long
            ).expand(batch, -1)
            query = output
        rotary_embedding(
            positions, query, None, head_size, cos_sin_cache, self.is_neox_style
        )
        return self._post_process(output, origin_shape, origin_dtype)


class MusaMRotaryEmbedding(nn.Module):
    """Select upstream multimodal MRoPE without changing its cached instance."""

    def __init__(
        self, inner: nn.Module, qk_hidden_sizes: tuple[int, int] | None = None
    ) -> None:
        super().__init__()
        self.inner = inner
        self.qk_hidden_sizes = qk_hidden_sizes

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        args = (
            (positions, query, key)
            if offsets is None
            else (positions, query, key, offsets)
        )
        if (
            query.device.type == "musa"
            and positions.ndim == 2
            and key is not None
            and (
                self.qk_hidden_sizes is None
                or (query.shape[-1], key.shape[-1]) == self.qk_hidden_sizes
            )
        ):
            return self.inner.forward_cuda(*args)
        return self.inner(*args)
