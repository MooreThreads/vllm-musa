from vllm_musa.v1.sample.qwen_sample_input_views import (
    install_qwen_sample_input_views,
)
from vllm_musa.v1.worker import qwen_fused_next_input_ids as qwen_fused_next_input_ids
from vllm_musa.v1.worker import qwen_identity_logits_view as qwen_identity_logits_view

install_qwen_sample_input_views()

__all__ = ["qwen_fused_next_input_ids", "qwen_identity_logits_view"]
