from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelFamily(str, Enum):
    UNKNOWN = "unknown"
    DEEPSEEK_V4 = "deepseek_v4"
    QWEN2 = "qwen2"
    QWEN3 = "qwen3"
    QWEN35_36 = "qwen3.5_3.6"


class ModelRole(str, Enum):
    UNKNOWN = "unknown"
    TEXT = "text_generation"
    MTP_DRAFT = "mtp_draft"
    COSYVOICE_TALKER = "cosyvoice_talker"


class OptimizationFeature(str, Enum):
    DEEPSEEK_V4_SHARED_MLP_CLAMP_FP8 = "deepseek_v4.shared_mlp_clamp_fp8"
    DEEPSEEK_V4_NATIVE_SPARSE_INDEXER = "deepseek_v4.native_sparse_indexer"
    DEEPSEEK_V4_MATERIALIZED_PREFILL_INDEXER = (
        "deepseek_v4.materialized_prefill_indexer"
    )
    DEEPSEEK_V4_TP8_FLASHMLA_SPARSE_PAGE256 = "deepseek_v4.tp8_flashmla_sparse_page256"
    DEEPSEEK_V4_TP8_MTP_SPARSE_DIRECT_OUT = "deepseek_v4.tp8_mtp_sparse_direct_out"
    DEEPSEEK_V4_TP8_MTP_SPARSE_PREFILL_HEADROOM = (
        "deepseek_v4.tp8_mtp_sparse_prefill_headroom"
    )
    DEEPSEEK_V4_TP8_FUSED_ADD_RMSNORM_BLOCK256 = (
        "deepseek_v4.tp8_fused_add_rmsnorm_block256"
    )
    DEEPSEEK_V4_CAR_GRAPH_INPUT_CAPTURE_GUARD = (
        "deepseek_v4.car_graph_input_capture_guard"
    )
    QWEN_V2_SAMPLING = "qwen.v2_sampling"
    QWEN_LEGACY_SAMPLING = "qwen.legacy_sampling"
    QWEN_FA3_SCHEDULER = "qwen.fa3_scheduler"
    QWEN_FA3_SINGLE_REQUEST_METADATA = "qwen.fa3_single_request_metadata"
    QWEN2_ROPE_KV_PRESPLIT = "qwen2.rope_kv_presplit"
    QWEN3_QK_ROPE_KV_PRESPLIT = "qwen3.qk_rope_kv_presplit"
    QWEN3_DENSE_FP8_POST_GRAD_FUSIONS = "qwen3.dense_fp8_post_grad_fusions"
    QWEN35_GDN_WIDTH4_PREFILL = "qwen3.5_3.6.gdn_width4_prefill"
    QWEN35_MOE_BF16_PREFILL = "qwen3.5_3.6.moe_bf16_prefill"
    QWEN_UNIFORM_DECODE_VIEWS = "qwen.uniform_decode_views"
    QWEN_UNIFORM_SAMPLE_COUNTS = "qwen.uniform_sample_counts"
    QWEN_SAMPLE_INPUT_VIEWS = "qwen.sample_input_views"
    QWEN_LEGACY_GUMBEL = "qwen.legacy_gumbel"
    QWEN_V2_GUMBEL = "qwen.v2_gumbel"
    QWEN_TP_LOGITS_IPC_GATHER = "qwen.tp_logits_ipc_gather"
    QWEN_TP4_SHARDED_GUMBEL = "qwen.tp4_sharded_gumbel"
    QWEN35_SHARED_EXPERT_FOLD = "qwen3.5_3.6.shared_expert_fold"
    QWEN35_INTERLEAVED_MROPE_QK = "qwen3.5_3.6.interleaved_mrope_qk"
    HYBRID_SEPARATE_MAMBA_POOL = "hybrid.separate_mamba_pool"


@dataclass(frozen=True, slots=True)
class ModelSignature:
    family: ModelFamily
    role: ModelRole
    architectures: tuple[str, ...]
    model_type: str | None
    dtype: str | None
    quantization: str | None
    hidden_size: int | None
    intermediate_size: int | None
    num_hidden_layers: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    vocab_size: int | None
    num_experts: int | None
    num_experts_per_tok: int | None
    num_shared_experts: int | None
    moe_intermediate_size: int | None
    expert_dtype: str | None
    hidden_act: str | None
    swiglu_limit: float | None
    gdn_conv_width: int | None
    gdn_conv_dim: int | None
    has_routed_experts: bool | None
    enforce_eager: bool
    outer_architectures: tuple[str, ...] = ()
    text_architectures: tuple[str, ...] = ()
    outer_model_type: str | None = None
    uses_mla: bool | None = None
    index_topk: int | None = None
    quant_block_shape: tuple[int, ...] | None = None
    is_hybrid: bool | None = None


@dataclass(frozen=True, slots=True)
class ExecutionSignature:
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    decode_context_parallel_size: int
    has_speculative_config: bool
    has_quant_config: bool
    is_pooling_model: bool
    has_parallel_config: bool
    cache_dtype: str | None
    cache_block_size: int | None
    max_num_seqs: int | None
    attention_backend: str | None = None
    compilation_mode: str | None = None
    cudagraph_mode: str | None = None
    batch_invariant_enabled: bool = False
    speculative_method: str | None = None


@dataclass(frozen=True, slots=True)
class MusaOptimizationContract:
    model: ModelSignature
    execution: ExecutionSignature
    profile: str
    supported_features: frozenset[OptimizationFeature]
    preferred_features: frozenset[OptimizationFeature]

    def supports(self, feature: OptimizationFeature) -> bool:
        return feature in self.supported_features

    def prefers(self, feature: OptimizationFeature) -> bool:
        return feature in self.preferred_features
