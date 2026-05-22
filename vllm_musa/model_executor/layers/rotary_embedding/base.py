import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from vllm_musa.jit_kernel import rotary_embedding


@RotaryEmbedding.register_oot
class MusaRotaryEmbedding(RotaryEmbedding):
    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)

        rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        return query, key
