"""The single vLLM-MUSA runtime-plan entry point.

Consumers should import plan types and the resolver from this package rather
than reaching into a model-specific provider.  Provider modules remain an
implementation detail of plan materialization.
"""

from .catalog import (
    RUNTIME_DECISION_SPECS,
    RuntimeDecisionKind,
    RuntimeDecisionSpec,
    RuntimeDecisionTunability,
    RuntimePlanPhase,
    list_runtime_decision_specs,
    runtime_decision_spec,
)
from .declarative import (
    DeclarativeProfileError,
    builtin_declarative_profiles,
    declarative_profile_catalog,
    declarative_profile_identity,
    load_declarative_profile,
    parse_declarative_profile,
)
from .policy import (
    DeepSeekV4MtpCarGraphStagingPlan,
    deepseek_v4_flashmla_sparse_page_size,
    deepseek_v4_mtp_async_prefill_queue_fence_enabled,
    deepseek_v4_mtp_car_graph_guard_enabled,
    deepseek_v4_mtp_car_graph_staging_plan,
    deepseek_v4_mtp_graph_registered_inputs_enabled,
    deepseek_v4_mtp_prefill_step_requires_sync,
    deepseek_v4_mtp_sparse_prefill_headroom_bytes,
    model_has_routed_experts,
    runtime_plan_enabled,
)
from .qwen import (
    matches_qwen35_moe_bf16_decode_gemv_layer,
    matches_qwen35_moe_bf16_prefill_layer,
)
from .resolver import (
    RUNTIME_PLAN_TRANSPORT_KEY,
    bind_runtime_plan,
    publish_runtime_plan_transport,
    resolve_runtime_plan,
)
from .types import (
    ExecutionSignature,
    ModelFamily,
    ModelRole,
    ModelSignature,
    RuntimeDecision,
    RuntimeDecisionError,
    RuntimeDecisionResolution,
    RuntimeDecisionValue,
    RuntimePlan,
    canonical_fingerprint,
)

__all__ = [
    "DeepSeekV4MtpCarGraphStagingPlan",
    "DeclarativeProfileError",
    "ExecutionSignature",
    "ModelFamily",
    "ModelRole",
    "ModelSignature",
    "RuntimeDecision",
    "RuntimeDecisionKind",
    "RuntimeDecisionError",
    "RuntimeDecisionResolution",
    "RuntimeDecisionSpec",
    "RuntimeDecisionTunability",
    "RuntimeDecisionValue",
    "RuntimePlan",
    "RuntimePlanPhase",
    "RUNTIME_DECISION_SPECS",
    "RUNTIME_PLAN_TRANSPORT_KEY",
    "bind_runtime_plan",
    "builtin_declarative_profiles",
    "canonical_fingerprint",
    "deepseek_v4_flashmla_sparse_page_size",
    "deepseek_v4_mtp_async_prefill_queue_fence_enabled",
    "deepseek_v4_mtp_car_graph_guard_enabled",
    "deepseek_v4_mtp_car_graph_staging_plan",
    "deepseek_v4_mtp_graph_registered_inputs_enabled",
    "deepseek_v4_mtp_prefill_step_requires_sync",
    "deepseek_v4_mtp_sparse_prefill_headroom_bytes",
    "declarative_profile_catalog",
    "declarative_profile_identity",
    "load_declarative_profile",
    "matches_qwen35_moe_bf16_decode_gemv_layer",
    "matches_qwen35_moe_bf16_prefill_layer",
    "model_has_routed_experts",
    "parse_declarative_profile",
    "publish_runtime_plan_transport",
    "list_runtime_decision_specs",
    "resolve_runtime_plan",
    "runtime_decision_spec",
    "runtime_plan_enabled",
]
