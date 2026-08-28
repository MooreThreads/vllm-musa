# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code inside this file can safely assume musa platform, e.g. importing
pymtml. However, it should not initialize musa context.
"""

import json
import os
import sys
from bisect import bisect_left
from collections.abc import Callable
from functools import cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from packaging.version import Version

# isort: off
import torchada  # noqa: F401
import torch

# isort: on
from typing_extensions import ParamSpec
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None
    CacheDType = None

import pymtml as pynvml

from vllm_musa.runtime_plan import (
    RUNTIME_PLAN_TRANSPORT_KEY,
    ModelFamily,
    RuntimeDecision,
    publish_runtime_plan_transport,
    resolve_runtime_plan,
)
from vllm_musa.runtime_plan import policy as plan_policy
from vllm_musa.tuning import (
    DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS,
    configure_fused_add_rmsnorm_min_rows,
    is_fused_add_rmsnorm_tuned_hidden_size,
)

logger = init_logger(__name__)
engine_plan_logger = init_logger("vllm.engine_plan")

_QWEN3_VL_ARCHITECTURES = {"Qwen3VLForConditionalGeneration"}
_MUSA_FUSED_MOE_INTERNAL_POLICY_ENV = "VLLM_MUSA_INTERNAL_FUSED_MOE_DISPATCH_POLICY"
_MUSA_FUSED_MOE_INTERNAL_PLAN_ID_ENV = "VLLM_MUSA_INTERNAL_FUSED_MOE_PLAN_ID"
_MUSA_FUSED_MOE_INTERNAL_FINGERPRINT_ENV = (
    "VLLM_MUSA_INTERNAL_FUSED_MOE_PLAN_FINGERPRINT"
)
_MUSA_FUSED_MOE_INTERNAL_PROFILE_ENV = "VLLM_MUSA_INTERNAL_FUSED_MOE_PROFILE"


def _is_torch_211_or_newer() -> bool:
    return Version(torch.__version__.split("+", 1)[0]) >= Version("2.11")


_P = ParamSpec("_P")
_R = TypeVar("_R")


@cache
def _get_backend_priorities(
    use_mla: bool,
    device_capability: DeviceCapability,
    num_heads: int | None = None,
) -> list[AttentionBackendEnum]:
    """Get backend priorities with lazy import to avoid circular dependency."""
    if use_mla:
        return [
            AttentionBackendEnum.FLASHMLA,
            AttentionBackendEnum.FLASHMLA_SPARSE,
            AttentionBackendEnum.TRITON_MLA,
        ]
    else:
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TURBOQUANT,
        ]


def with_mtml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            # Note: We intentionally do NOT call nvmlShutdown() here because
            # pymtml has a bug where nvmlInit() fails after nvmlShutdown()
            # has been called. The library handles cleanup on process exit.
            pass

    return wrapper


def register_attention_backends() -> None:
    # Pre-register all attention backends
    register_backend(
        AttentionBackendEnum.FLASHMLA,
        class_path="vllm_musa.v1.attention.backends.mla.flashmla.MUSAFlashMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA_SPARSE,
        class_path=(
            "vllm_musa.v1.attention.backends.mla.flashmla_sparse."
            "MUSAFlashMLASparseBackend"
        ),
    )
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        class_path="vllm_musa.v1.attention.backends.flash_attn.MUSAFlashAttentionBackend",
    )
    register_backend(
        AttentionBackendEnum.TURBOQUANT,
        class_path=(
            "vllm_musa.v1.attention.backends.turboquant."
            "MUSATurboQuantAttentionBackend"
        ),
    )
    tree_attn_backend = getattr(AttentionBackendEnum, "TREE_ATTN", None)
    if tree_attn_backend is not None:
        # tree drafting via a MUSA-routed TreeAttention backend
        # (Triton unified_attention is already MUSA-patched; reshape_and_cache_flash
        # is wired through fa_utils.reshape_and_cache_flash). v0.22 removed this
        # enum member, so only register it on older upstream snapshots.
        register_backend(
            tree_attn_backend,
            class_path=(
                "vllm_musa.v1.attention.backends.tree_attn.MUSATreeAttentionBackend"
            ),
        )


def _will_capture_piecewise_cudagraph(vllm_config: Any) -> bool:
    """Whether this config will do PIECEWISE CUDAGraph capture under torch.compile.

    Called from ``check_and_update_config`` (``VllmConfig.__post_init__``), which runs
    BEFORE vLLM finalizes ``compilation_config.backend``/``mode`` and the
    ``cudagraph_mode`` default -- so we cannot gate on ``backend == "inductor"`` (unset
    here). Exclude only the cases that never capture piecewise: ``enforce_eager``,
    ``mode == NONE``, or an explicitly non-piecewise ``cudagraph_mode``. A ``None``
    ``cudagraph_mode`` means "not yet defaulted"; the v1 default is FULL_AND_PIECEWISE
    (piecewise), so None counts as piecewise.
    """
    from vllm.config.compilation import CompilationMode

    if getattr(vllm_config.model_config, "enforce_eager", False):
        return False
    comp = vllm_config.compilation_config
    if getattr(comp, "mode", None) == CompilationMode.NONE:
        return False
    cg_mode = getattr(comp, "cudagraph_mode", None)
    if cg_mode is not None and not cg_mode.has_piecewise_cudagraphs():
        return False
    return True


def _should_route_quantized_piecewise_ops_native(vllm_config: Any) -> bool:
    """Whether the quantized PIECEWISE correctness route applies to this model."""
    return getattr(
        vllm_config, "quant_config", None
    ) is not None and _will_capture_piecewise_cudagraph(vllm_config)


def _configure_fused_add_rmsnorm_compile_range(
    vllm_config: Any,
    *,
    native_custom_ops: bool,
    min_rows: int = DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS,
) -> bool:
    """Split compilation at the measured fused-add RMSNorm profit boundary."""
    ir_priority = getattr(
        getattr(vllm_config, "kernel_config", None),
        "ir_op_priority",
        None,
    )
    fused_priority = getattr(ir_priority, "fused_add_rms_norm", None)
    native_custom_ops = native_custom_ops or bool(
        fused_priority and fused_priority[0] == "native"
    )
    model_config = getattr(vllm_config, "model_config", None)
    get_hidden_size = getattr(model_config, "get_hidden_size", None)
    max_tokens = getattr(
        getattr(vllm_config, "scheduler_config", None),
        "max_num_batched_tokens",
        0,
    )
    if (
        native_custom_ops
        or model_config is None
        or not callable(get_hidden_size)
        or not is_fused_add_rmsnorm_tuned_hidden_size(get_hidden_size())
        or getattr(model_config, "dtype", None) != torch.bfloat16
        or getattr(model_config, "enforce_eager", False)
        or max_tokens is None
        or max_tokens < min_rows
    ):
        return False

    comp = vllm_config.compilation_config
    endpoints = list(comp.compile_ranges_endpoints or [])
    fallback_endpoint = min_rows - 1
    if fallback_endpoint <= 0:
        return False
    if fallback_endpoint in endpoints:
        return False
    comp.compile_ranges_endpoints = sorted([*endpoints, fallback_endpoint])
    return True


def _configure_fused_moe_compile_ranges(
    vllm_config: Any,
    *,
    boundaries: tuple[int, ...],
) -> bool:
    """Split dynamic compile ranges where a sealed MoE backend can change."""

    if not boundaries or getattr(
        getattr(vllm_config, "model_config", None), "enforce_eager", False
    ):
        return False
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        return False
    unsupported = tuple(boundary for boundary in boundaries if boundary <= 1)
    if unsupported:
        raise RuntimeError(
            "MUSA fused-MoE RuntimePlan cannot represent a backend transition "
            "immediately after token 1 with this vLLM compile-range lifecycle: "
            f"boundaries={unsupported}. Retune or smooth the policy so tokens 1 "
            "and 2 use the same backend."
        )
    endpoints = list(
        getattr(compilation_config, "compile_ranges_endpoints", None) or ()
    )
    updated = sorted(set(endpoints).union(boundaries))
    if updated == endpoints:
        return False
    compilation_config.compile_ranges_endpoints = updated
    return True


def _validate_fused_moe_compile_ranges(
    vllm_config: Any,
    *,
    boundaries: tuple[int, ...],
) -> None:
    """Validate RuntimePlan boundaries after vLLM finalizes compile ranges.

    RuntimePlan owns tactic selection, not the graph-memory budget. This
    validation only verifies that vLLM preserved every plan transition when it
    constructed the final compile ranges.
    """

    if not boundaries or getattr(
        getattr(vllm_config, "model_config", None), "enforce_eager", False
    ):
        return
    compilation_config = getattr(vllm_config, "compilation_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if compilation_config is None or scheduler_config is None:
        return

    max_tokens = getattr(scheduler_config, "max_num_batched_tokens", None)
    if max_tokens is None or int(max_tokens) <= 0:
        return
    max_tokens = int(max_tokens)
    finalized_endpoints = tuple(
        int(endpoint)
        for endpoint in (
            getattr(compilation_config, "compile_ranges_endpoints", None) or ()
        )
    )
    expected = tuple(boundary for boundary in boundaries if 0 < boundary < max_tokens)
    missing = tuple(
        boundary for boundary in expected if boundary not in finalized_endpoints
    )
    if missing:
        raise RuntimeError(
            "vLLM dropped MUSA fused-MoE RuntimePlan compile boundaries during "
            f"finalization: missing={missing}, final={finalized_endpoints}. "
            "Refusing to compile a range that can bake more than one tactic."
        )


def _fused_moe_capture_policy_entries(
    value: object,
) -> tuple[tuple[dict[str, object], tuple[dict[str, object], ...]], ...]:
    policy = _runtime_plan_mapping(value)
    entries = policy.get("entries", ())
    if not isinstance(entries, tuple):
        return ()
    decoded = []
    for raw_entry in entries:
        entry = _runtime_plan_mapping(raw_entry)
        shape = _runtime_plan_mapping(entry.get("shape"))
        ranges = entry.get("ranges", ())
        if shape.get("graph_mode") != "capture" or not isinstance(ranges, tuple):
            continue
        decoded.append(
            (
                shape,
                tuple(_runtime_plan_mapping(raw_range) for raw_range in ranges),
            )
        )
    return tuple(decoded)


def _fused_moe_policy_backend(
    ranges: tuple[dict[str, object], ...],
    tokens: int,
) -> str | None:
    for token_range in ranges:
        minimum = token_range.get("min_tokens")
        maximum = token_range.get("max_tokens")
        backend = token_range.get("backend")
        if (
            isinstance(minimum, int)
            and not isinstance(minimum, bool)
            and isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and isinstance(backend, str)
            and minimum <= tokens <= maximum
        ):
            return backend
    return None


def _validate_fused_moe_cudagraph_padding(
    vllm_config: Any,
    *,
    policy: object,
    uniform_decode_query_len: int,
) -> None:
    """Reject graph padding that crosses a RuntimePlan tactic transition."""

    capture_entries = _fused_moe_capture_policy_entries(policy)
    if not capture_entries:
        return
    if int(uniform_decode_query_len) != 1:
        raise RuntimeError(
            "Contextual fused-MoE CUDAGraph RuntimePlans do not yet support "
            "speculative decode or draft graph domains; "
            f"uniform_decode_query_len={uniform_decode_query_len}. Retune "
            "without speculative decode or disable the contextual capture "
            "policy."
        )
    compilation_config = getattr(vllm_config, "compilation_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if compilation_config is None or scheduler_config is None:
        return

    from vllm.config import CUDAGraphMode

    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
    if cudagraph_mode is None:
        return
    if cudagraph_mode == CUDAGraphMode.NONE:
        raise RuntimeError(
            "MUSA fused-MoE RuntimePlan contains capture policy entries, but "
            "the resolved CUDAGraph mode is NONE"
        )
    capture_sizes = tuple(
        sorted(
            {
                int(size)
                for size in (
                    getattr(compilation_config, "cudagraph_capture_sizes", None) or ()
                )
                if int(size) > 0
            }
        )
    )
    if not capture_sizes:
        return

    mixed_mode = cudagraph_mode.mixed_mode()
    if mixed_mode != CUDAGraphMode.NONE:
        reachable = range(1, capture_sizes[-1] + 1)
        graph_capture_sizes = frozenset(capture_sizes)
    elif cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
        query_len = max(1, int(uniform_decode_query_len))
        max_num_seqs = int(getattr(scheduler_config, "max_num_seqs", 0) or 0)
        max_decode_tokens = query_len * max_num_seqs
        reachable = range(query_len, max_decode_tokens + 1, query_len)
        graph_capture_sizes = frozenset(
            size for size in capture_sizes if query_len <= size <= max_decode_tokens
        )
    else:
        return

    for shape, ranges in capture_entries:
        for actual_tokens in reachable:
            index = bisect_left(capture_sizes, actual_tokens)
            if index == len(capture_sizes):
                continue
            padded_tokens = capture_sizes[index]
            if padded_tokens not in graph_capture_sizes:
                continue
            actual_backend = _fused_moe_policy_backend(ranges, actual_tokens)
            padded_backend = _fused_moe_policy_backend(ranges, padded_tokens)
            if actual_backend == padded_backend:
                continue
            raise RuntimeError(
                "MUSA fused-MoE RuntimePlan CUDAGraph padding crosses a tactic "
                "transition: "
                f"actual_tokens={actual_tokens}, padded_tokens={padded_tokens}, "
                f"actual_backend={actual_backend or 'unplanned'}, "
                f"padded_backend={padded_backend or 'unplanned'}, "
                f"capture_sizes={capture_sizes}, graph_mode={cudagraph_mode}, "
                f"shape=(E={shape.get('local_experts')},"
                f"N={shape.get('w1_output_size')},K={shape.get('hidden_size')},"
                f"topk={shape.get('top_k')}). Configure an aligned capture "
                "size or rebuild the capture policy from reachable padded graph "
                "keys. RuntimePlan will not expand the graph-memory budget."
            )


def _json_runtime_plan_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_runtime_plan_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_runtime_plan_value(item) for key, item in value.items()}
    return value


def _runtime_plan_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, tuple):
        return {}
    return {
        item[0]: item[1]
        for item in value
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
    }


def _fused_moe_policy_boundaries(value: object) -> tuple[int, ...]:
    policy = _runtime_plan_mapping(value)
    entries = policy.get("entries", ())
    boundaries: set[int] = set()
    if not isinstance(entries, tuple):
        return ()
    for raw_entry in entries:
        entry = _runtime_plan_mapping(raw_entry)
        ranges = entry.get("ranges", ())
        if not isinstance(ranges, tuple):
            continue
        for raw_range in ranges:
            token_range = _runtime_plan_mapping(raw_range)
            maximum = token_range.get("max_tokens")
            if isinstance(maximum, int) and not isinstance(maximum, bool):
                boundaries.add(maximum)
    return tuple(sorted(boundaries))


def _materialize_fused_moe_runtime_policy(plan: Any) -> tuple[int, ...]:
    """Transport the validated policy without importing the heavy MoE module."""

    value = plan.value(RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY, ())
    resolution = plan.decision_resolution
    receipt = (
        resolution.plan_id if resolution is not None else "",
        resolution.fingerprint if resolution is not None else plan.fingerprint,
        plan.profile,
    )
    names = (
        _MUSA_FUSED_MOE_INTERNAL_POLICY_ENV,
        _MUSA_FUSED_MOE_INTERNAL_PLAN_ID_ENV,
        _MUSA_FUSED_MOE_INTERNAL_FINGERPRINT_ENV,
        _MUSA_FUSED_MOE_INTERNAL_PROFILE_ENV,
    )
    if value:
        os.environ[names[0]] = json.dumps(
            _json_runtime_plan_value(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        os.environ[names[1]], os.environ[names[2]], os.environ[names[3]] = receipt
    else:
        for name in names:
            os.environ.pop(name, None)
    module = sys.modules.get(
        "vllm_musa.model_executor.layers.fused_moe.dispatch_policy"
    )
    configure = getattr(module, "configure_fused_moe_runtime_policy", None)
    if callable(configure):
        configure(
            value,
            plan_id=receipt[0],
            plan_fingerprint=receipt[1],
            profile=receipt[2],
        )
    return _fused_moe_policy_boundaries(value)


def materialize_fused_moe_runtime_policy_for_worker(
    vllm_config: Any,
    *,
    worker_rank: int,
) -> tuple[int, ...] | None:
    """Install contextual fused-MoE decisions in a spawned TP worker.

    The dispatch-policy module keeps process-local state. A worker may be
    spawned by a process that was started before the API parent materialized
    its RuntimePlan environment, so reconstruct the cached, fingerprinted plan
    from the serialized ``vllm_config`` instead of relying on inheritance.
    """

    plan = resolve_runtime_plan(vllm_config)
    boundaries = _materialize_fused_moe_runtime_policy(plan)
    if boundaries is None:
        return None
    resolution = plan.decision_resolution
    policy = plan.value(RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY, ())
    source = plan.decision_source(RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY)
    if source != "engine_plan" or resolution is None or not resolution.plan_id:
        return boundaries
    engine_plan_logger.info(
        "Materialized MUSA fused-MoE dispatch RuntimePlan policy: "
        "profile=%s profile_config_id=%s profile_config_fingerprint=%s "
        "entries=%d source=%s plan_id=%s plan_fingerprint=%s worker_rank=%d",
        plan.profile,
        plan.profile_config_id,
        plan.profile_config_fingerprint,
        len(policy),
        source,
        resolution.plan_id,
        resolution.fingerprint,
        worker_rank,
    )
    return boundaries


class MUSAPlatformBase(Platform):
    _enum = PlatformEnum.OOT  # Out-of-tree platform
    device_name: str = "musa"
    device_type: str = "musa"
    dispatch_key: str = "MUSA"
    ray_device_key: str = "GPU"
    dist_backend: str = "mccl"  # MUSA's NCCL equivalent
    device_control_env_var: str = "MUSA_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES",
    ]

    @classmethod
    def get_pass_manager_cls(cls) -> str:
        return "vllm_musa.compilation.passes.MusaPostGradPassManager"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        # MUSA GPUs support BF16 and FP16
        return [torch.bfloat16, torch.float16, torch.float32]

    def is_cuda_alike(self) -> bool:
        """MUSA is CUDA-alike for compatibility purposes."""
        return True

    def is_musa(self) -> bool:
        """This is the MUSA platform."""
        return True

    def is_sleep_mode_available(self) -> bool:
        """MUSA supports sleep mode."""
        return True

    @classmethod
    def import_kernels(cls) -> None:
        """Import upstream vLLM kernels, including v0.22 stable ABI ops."""
        super().import_kernels()
        try:
            import vllm._C_stable_libtorch  # noqa: F401
        except ImportError as e:
            logger.warning("Failed to import from vllm._C_stable_libtorch: %r", e)

    @classmethod
    def check_runner_kv_caches_multi_layer(cls) -> None:
        """Allow multiple attention caches sharing one decoder layer index.

        The MUSA V2 runner uses the same ordered ``runner_kv_caches`` binding
        contract as CUDA, ROCm, and XPU. DeepSeek-V4 exercises this with its
        sparse indexer and MLA caches in the same decoder block.
        """
        pass

    @classmethod
    def import_ir_kernels(cls) -> None:
        """Import upstream and MUSA-OOT IR-op providers.

        Order matters: upstream first (registers `native` / `vllm_c` /
        `oink` / etc.), then OOT MUSA providers so they appear in the
        registry alongside upstream impls. Used by
        `vllm.config.kernel.KernelConfig.set_priority()` to ensure
        every provider mentioned in `ir_op_priority` is registered
        before the dispatcher needs it.
        """
        super().import_ir_kernels()
        try:
            import vllm_musa.kernels  # noqa: F401
        except ImportError as exc:
            from vllm.logger import init_logger

            init_logger(__name__).info(
                "vllm_musa.kernels unavailable (%s); MUSA IR providers "
                "will not be registered. Upstream providers remain "
                "available.",
                exc,
            )

    @classmethod
    def get_default_ir_op_priority(cls, vllm_config):
        """Platform-default priority list for vllm.ir.ops on MUSA.

        When compiling with Inductor, prefer the `native` (pure-PyTorch)
        rms_norm implementation. Prefer the measured in-place `musa`
        fused_add_rms_norm implementation in both compiled and eager paths.
        That single provider dispatches internally between the measured JIT
        kernel and the broad pre-existing C-extension kernel, then falls back
        to native through normal IR priority; IR functionalization owns
        activation donation.
        This mirrors the upstream `cuda.py` pattern
        (`default = ["native"] if using_inductor else ["vllm_c", "native"]`):
        under Inductor the native rms_norm is a handful of
        elementwise/reduction ops the compiler can fuse with its
        neighbours, whereas a `torch.ops._C.*` custom op is an opaque
        fusion barrier; in eager mode there is no fusion to lose so the
        kernel provider is taken directly.

        The non-fused rms_norm choice remains upstream-consistent; the fused
        provider is independently gated by its measured supports_args scope.
        """
        from vllm.config.compilation import CompilationMode
        from vllm.config.kernel import IrOpPriorityConfig

        from vllm_musa.engine_plugins import get_engine_plugin_application

        application = get_engine_plugin_application(vllm_config)
        if application is not None:
            engine_plan_logger.info(
                "Active vLLM-MUSA engine plan: plugin=%s plan=%s variant=%s "
                "context=%s tactics=%s fallback=%s",
                application.plugin_name,
                application.plan_id,
                application.selected_variant or "none",
                application.context_fingerprint or "none",
                application.selected_tactics,
                application.fallback_reason or "none",
            )
        cc = vllm_config.compilation_config
        using_inductor = cc.backend == "inductor" and cc.mode != CompilationMode.NONE
        if using_inductor:
            # Let Inductor fuse the native reference impl. Keep the
            # `musa` custom-op provider out of the priority here — it is
            # a fusion barrier (no measurable batch1 effect, but native
            # is the upstream-aligned default).
            default = ["native"]
            rms_norm = ["native"]
        else:
            # Eager path: no Inductor fusion to lose, so take the rms_norm
            # kernel. Keep other IR ops native by default.
            default = ["native"]
            rms_norm = ["musa", "native"]
        # VLLM_MUSA_CUSTOM_OP_USE_NATIVE is a correctness route for quantized
        # models under piecewise CUDAGraph capture. It must remain authoritative
        # for the IR provider too; unlike supports_args, this is global provider
        # enablement and therefore belongs in the priority policy.
        from vllm_musa.utils.environ import envs as musa_envs

        native_custom_ops = musa_envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()
        if not musa_envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.is_set():
            # VllmConfig installs platform IR priorities before invoking
            # check_and_update_config(), where the process-wide safety route is
            # normally published for custom-op forward paths. Derive the same
            # policy from this config here so quantized PIECEWISE compilation
            # cannot retain an unsafe fused-add provider in its frozen priority.
            native_custom_ops = _should_route_quantized_piecewise_ops_native(
                vllm_config
            )
        fused_add_rms_norm = ["native"] if native_custom_ops else ["musa", "native"]
        # The direct Triton HOP is profitable for every validated dense shape
        # family; supports_args remains the semantic/shape guard. Keep routed
        # MoE on native by default because a 100-prompt Qwen3.6 TP8 sweep
        # regressed 2.81% as eager-faithful BF16 rounding changed expert routes.
        # Users can still explicitly select the generic provider for A/B work.
        plan = resolve_runtime_plan(vllm_config)
        gated_qkv_rms_norm_rope = (
            ["musa_inductor", "native"]
            if using_inductor and plan.model.has_routed_experts is not True
            else ["native"]
        )
        return IrOpPriorityConfig.with_default(
            default,
            rms_norm=rms_norm,
            fused_add_rms_norm=fused_add_rms_norm,
            gated_qkv_rms_norm_rope=gated_qkv_rms_norm_rope,
        )

    @classmethod
    def support_deep_gemm(cls) -> bool:
        """
        Returns if DeepGEMM is supported by the current platform.
        """
        return True

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)
        # With this trick we can force the device to be set eagerly
        # see https://github.com/pytorch/pytorch/issues/155668
        # for why and when it is needed
        _ = torch.zeros(1, device=device)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.musa.manual_seed_all(seed)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def log_warnings(cls):
        pass

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        # Engine-plan plugins must run before vLLM freezes IR provider priority
        # and derives fusion defaults from it.  With no enabled plugin this is
        # an exact no-op.
        from vllm_musa.engine_plugins import apply_engine_plugin_defaults

        engine_application = apply_engine_plugin_defaults(vllm_config)
        if engine_application is not None:
            engine_plan_logger.info(
                "Applied vLLM-MUSA engine plan defaults: plugin=%s plan=%s "
                "fingerprint=%s variant=%s tactics=%s fallback=%s settings=%s",
                engine_application.plugin_name,
                engine_application.plan_id,
                engine_application.plan_fingerprint,
                engine_application.selected_variant or "legacy",
                engine_application.selected_tactics,
                engine_application.fallback_reason or "none",
                engine_application.applied_settings,
            )

            # EngineCore/TP workers can be spawned before vLLM reaches its
            # late ``finalize_config`` hook.  Publish the exact validated
            # decision projection immediately after plugin selection so the
            # worker's serialized VllmConfig is self-contained.  The late
            # hook below re-resolves and verifies the same immutable decision
            # projection after vLLM has finalized its execution defaults.
            publish_runtime_plan_transport(
                vllm_config,
                resolve_runtime_plan(vllm_config),
            )

        # Ensure custom ops are enabled for MUSA platform so that
        # OOT forward implementations (forward_oot) are dispatched.
        # This must be set here (before VllmConfig.__post_init__ defaults)
        # to prevent custom_ops from being defaulted to ['none'] when
        # the inductor backend is active.
        compilation_config = vllm_config.compilation_config
        if all(s not in compilation_config.custom_ops for s in ("all", "none")):
            compilation_config.custom_ops.append("all")

        # torch 2.11's Inductor tiling heuristic turns Qwen3-VL's decode-time
        # clone/index-select/split kernel into a slower 2D grid on MUSA. The
        # torch 2.9 1D grid is restored by limiting this model's pointwise
        # kernels to one tile dimension. Preserve an explicit user override.
        architectures = set(
            getattr(
                getattr(vllm_config.model_config, "hf_config", None),
                "architectures",
                None,
            )
            or ()
        )
        inductor_config = compilation_config.inductor_compile_config
        if (
            _is_torch_211_or_newer()
            and architectures & _QWEN3_VL_ARCHITECTURES
            and "triton.max_tiles" not in inductor_config
        ):
            inductor_config["triton.max_tiles"] = 1
            logger.info(
                "Using triton.max_tiles=1 for Qwen3-VL on torch >= 2.11 "
                "to preserve the MUSA decode pointwise layout"
            )

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        # when dflash spec-decode is active, coerce the draft-loop
        # CUDAGraph capture to block-aligned FULL sizes (see below). The dflash
        # source patch (DFlashProposer.dummy_run signature) is applied at BUILD
        # time to the cloned vLLM (setup.py -> series/0030 dflash.patch), so the
        # installed dflash.py the spawn workers import is already patched — no
        # runtime source-patching is needed here.
        spec_config = getattr(vllm_config, "speculative_config", None)
        # Detect dflash via the STABLE `use_dflash()` API (the same predicate
        # gpu_model_runner branches on) rather than the `method` string, whose
        # representation can change across vLLM versions; fall back to the string
        # only if use_dflash() is unavailable.
        _use_dflash = getattr(spec_config, "use_dflash", None)
        if spec_config is not None and (
            _use_dflash()
            if callable(_use_dflash)
            else getattr(spec_config, "method", None) == "dflash"
        ):
            # the dflash draft-loop FULL CUDAGraph capture is
            # default-on (opt out with VLLM_MUSA_DFLASH_FULL_WRAP=0). It makes
            # the draft forward process a (1 + num_speculative_tokens)-token
            # verify block, so every captured cudagraph size must be a whole
            # multiple of that block. vLLM's default capture sizes
            # ([1, 2, 4, 8, 16, ...]) are not block-aligned, and capturing the
            # draft at a sub-block size raises "MUSA error: an illegal memory
            # access". Coerce to pure FULL + the block-aligned subset of the
            # configured sizes (default: the single BS=1 block). Multi-block
            # (BS>1) draft capture is tracked separately.
            import os as _df_os

            if _df_os.environ.get("VLLM_MUSA_DFLASH_FULL_WRAP", "1") != "0":
                _comp = getattr(vllm_config, "compilation_config", None)
                _nspec = getattr(spec_config, "num_speculative_tokens", None)
                if _comp is not None and _nspec:
                    from vllm.config import CUDAGraphMode as _CGM

                    _block = 1 + int(_nspec)
                    # Capture exactly the single BS=1 verify block — the proven,
                    # fast size (no padding). vLLM's default capture sizes are
                    # either not block-aligned (1, 2, 4, 8, 16, ...) or far too
                    # large (the only multiples of the block in the default set
                    # are 72, 144, ... = BS>=8, which leave a BS=1 9-token decode
                    # padded to 72 and running ~35 tok/s vs 48 at the exact
                    # block). extends this to BS=1,2,4,8 = block-aligned
                    # [9,18,36,72]; a BS=N decode uses the size-N*block graph exactly.
                    _maxseq = (
                        getattr(
                            getattr(vllm_config, "scheduler_config", None),
                            "max_num_seqs",
                            8,
                        )
                        or 8
                    )
                    _comp.cudagraph_capture_sizes = [
                        _block * k for k in (1, 2, 4, 8) if k <= max(1, _maxseq)
                    ]
                    _comp.max_cudagraph_capture_size = _comp.cudagraph_capture_sizes[-1]
                    _comp.cudagraph_mode = _CGM.FULL
                    logger.info(
                        "dflash draft capture default-on; coerced "
                        "cudagraph_mode=FULL, capture_sizes=%s (block=%d)",
                        _comp.cudagraph_capture_sizes,
                        _block,
                    )

        # FP8 correctness under torch.compile + PIECEWISE CUDAGraph capture: the
        # MUSA RMSNorm / SiluAndMul / per-token-group FP8-quant custom-op kernels
        # (forward_oot) are opaque to Inductor's quant fusion and leave the quant's
        # (q, scale) in buffers it cannot lifetime-track, so the captured piecewise
        # graph reads a stale scale and the FP8 forward emits degenerate output.
        # Route them to the native (decomposable, Inductor-fused, CUDAGraph-safe)
        # path while piecewise capture is active -- no perf loss (Inductor fuses the
        # quant). Eager / FULL_DECODE_ONLY and bf16 are unaffected, as is the
        # deep_gemm linear path (it fuses quant+GEMM, so its buffers never escape).
        from vllm_musa.utils.environ import envs as musa_envs

        # Only quantized models can enter the FP8 quant path described above.
        # Keeping this process-wide route enabled for unquantized BF16 models
        # unnecessarily disables their safe normalization kernels.
        route_native = _should_route_quantized_piecewise_ops_native(vllm_config)
        logger.debug(
            "MUSA custom-op native-routing: quantized_piecewise=%s", route_native
        )
        # Set the flag process-wide (not on the config): the op forward_oot paths read
        # this env live and spawn workers inherit os.environ. First-writer-wins (the
        # is_set() guard) is safe -- the native path is correct in every cudagraph mode,
        # so an inherited value can never yield wrong output, only the native path.
        if route_native and not musa_envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.is_set():
            musa_envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.set(True)
            logger.info(
                "Routing MUSA RMSNorm/SiluAndMul/QuantFP8 to the native "
                "(Inductor-fused) path: the custom-op kernels corrupt FP8 output "
                "under piecewise CUDAGraph capture. Set "
                "VLLM_MUSA_CUSTOM_OP_USE_NATIVE=0 to force the kernels."
            )

        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        # vLLM picks IR implementations once per dynamic compile range and
        # drops shape guards. Materialize the RuntimePlan crossover before
        # compilation, then split immediately below it so every range observes
        # one stable runner choice.
        plan = resolve_runtime_plan(vllm_config)
        if (
            plan.profile == "deepseek_v4.tp8_flash_base_mtp"
            and spec_config is not None
            and getattr(spec_config, "attention_backend", None) is None
        ):
            # vLLM normally auto-selects the draft backend independently.  The
            # validated DeepSeek-V4 MTP profile is narrower: target and draft
            # share the same fixed-page FlashMLA sparse layout, and clearing the
            # backend while cloning the draft config makes the RuntimePlan
            # regress to ``deepseek_v4.unvalidated``.  Materialize only this
            # exact profile; explicit draft choices and every other model keep
            # upstream's independent-selection behavior.
            spec_config.attention_backend = AttentionBackendEnum.FLASHMLA
            logger.info(
                "Materialized FLASHMLA for the validated DeepSeek-V4 MTP "
                "draft RuntimePlan profile"
            )
        selected_min_rows = int(
            plan.value(
                RuntimeDecision.MUSA_FUSED_ADD_RMSNORM_MIN_ROWS,
                DEFAULT_FUSED_ADD_RMSNORM_MIN_ROWS,
            )
        )
        # The fused-MoE policy is a compile/capture-static decision. Transport
        # it exactly once here, before vLLM creates compiled ranges or captures
        # a graph; the heavy dispatcher decodes it on module import (or uses an
        # already loaded immutable table) and request dispatch only reads that
        # table.
        moe_policy = plan.value(RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY, ())
        resolution = plan.decision_resolution
        moe_boundaries = _materialize_fused_moe_runtime_policy(plan)
        if moe_policy:
            logger.info(
                "Materialized MUSA fused-MoE dispatch RuntimePlan policy "
                "profile=%s plan_id=%s fingerprint=%s before compilation/capture",
                plan.profile,
                resolution.plan_id if resolution is not None else "none",
                resolution.fingerprint if resolution is not None else plan.fingerprint,
            )
        if _configure_fused_moe_compile_ranges(
            vllm_config,
            boundaries=moe_boundaries,
        ):
            logger.info(
                "Splitting MUSA fused-MoE compile ranges at token boundaries=%s",
                moe_boundaries,
            )
        configure_fused_add_rmsnorm_min_rows(selected_min_rows)
        logger.info(
            "Materialized MUSA fused-add RMSNorm threshold=%d from runtime "
            "plan profile=%s",
            selected_min_rows,
            plan.profile,
        )
        if _configure_fused_add_rmsnorm_compile_range(
            vllm_config,
            native_custom_ops=musa_envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get(),
            min_rows=selected_min_rows,
        ):
            logger.info(
                "Splitting MUSA fused-add RMSNorm compile ranges at %d rows",
                selected_min_rows - 1,
            )

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_musa.worker.MTGPUWorker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        # Note: block_size is initialized in
        # HybridAttentionMambaModelConfig.verify_and_update_config
        # for models with both attention and mamba,
        # and doesn't need to be reinitialized here
        if (
            model_config is not None
            and model_config.use_mla
            and cache_config.block_size is not None
        ):
            use_sparse = hasattr(vllm_config.model_config.hf_config, "index_topk")
            # If `--attention-config.backend` is not set and we are using MLA,
            # then we default to FlashMLA backend.
            use_flashmla = False
            use_flashmla_sparse = False

            from vllm_musa.v1.attention.ops.flashmla import is_flashmla_dense_supported

            if vllm_config.attention_config.backend is None:
                # Default case: use FlashMLA if supported
                if is_flashmla_dense_supported()[0]:
                    use_flashmla = True
            else:
                # Forced case
                backend = vllm_config.attention_config.backend
                use_flashmla = backend == AttentionBackendEnum.FLASHMLA
                use_flashmla_sparse = backend == AttentionBackendEnum.FLASHMLA_SPARSE

            if (
                use_flashmla
                and is_flashmla_dense_supported()[0]
                and cache_config.block_size % 64 != 0
            ):
                cache_config.block_size = 64
                logger.info("Forcing kv cache block size to 64 for FlashMLA backend.")

            if use_sparse:
                if not use_flashmla_sparse:
                    use_flashmla_sparse = True

                sparse_block_size = plan_policy.deepseek_v4_flashmla_sparse_page_size(
                    vllm_config
                )
                if use_flashmla_sparse and cache_config.block_size != sparse_block_size:
                    dsv4_page_decision = (
                        RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE
                    )
                    if (
                        plan.model.family is ModelFamily.DEEPSEEK_V4
                        and plan.supports(dsv4_page_decision)
                        and getattr(cache_config, "user_specified_block_size", False)
                    ):
                        raise RuntimeError(
                            "The explicit DeepSeek-V4 KV block size "
                            f"{cache_config.block_size} conflicts with the "
                            f"RuntimePlan layout {sparse_block_size}"
                        )
                    cache_config.block_size = sparse_block_size
                    logger.info(
                        "Forcing kv cache block size to %d for FlashMLASparse backend.",
                        sparse_block_size,
                    )

        scheduler_config = vllm_config.scheduler_config
        # Note: model_config may be None during testing
        if (
            model_config is not None
            and model_config.is_mm_prefix_lm
            and scheduler_config.is_multimodal_model
            and not scheduler_config.disable_chunked_mm_input
        ):
            logger.warning(
                "Forcing --disable_chunked_mm_input for models "
                "with multimodal-bidirectional attention."
            )
            scheduler_config.disable_chunked_mm_input = True

        # Validate after every MUSA platform override has been applied.  The
        # plugin receipt is provenance for the resolved runtime configuration,
        # not merely for the input JSON document.
        from vllm_musa.engine_plugins import validate_engine_plugin_runtime

        engine_receipt = validate_engine_plugin_runtime(vllm_config)
        if engine_receipt is not None:
            engine_plan_logger.info(
                "Validated vLLM-MUSA engine plan: plugin=%s version=%s plan=%s "
                "fingerprint=%s variant=%s tactics=%s context=%s fallback=%s "
                "settings=%s",
                engine_receipt.plugin_name,
                engine_receipt.plugin_version,
                engine_receipt.plan_id,
                engine_receipt.plan_fingerprint,
                engine_receipt.selected_variant or "legacy",
                engine_receipt.selected_tactics,
                engine_receipt.context_fingerprint or "legacy",
                engine_receipt.fallback_reason or "none",
                engine_receipt.validated_settings,
            )

    @classmethod
    def finalize_config(cls, vllm_config: "VllmConfig") -> None:
        """Validate and publish plan state after vLLM finalizes config."""

        plan = resolve_runtime_plan(vllm_config)
        moe_policy = plan.value(
            RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY,
            (),
        )
        _validate_fused_moe_compile_ranges(
            vllm_config,
            boundaries=_fused_moe_policy_boundaries(moe_policy),
        )
        # Verify the transport after the plugin and every plan-dependent
        # final-config invariant have been validated. TP workers rehydrate its
        # immutable decision projection from VllmConfig.additional_config;
        # they never reopen the plan path or depend on inherited environment.
        publish_runtime_plan_transport(vllm_config, plan)

    @classmethod
    def validate_cudagraph_config(
        cls,
        vllm_config: "VllmConfig",
        *,
        uniform_decode_query_len: int,
    ) -> None:
        """Validate the resolved graph padding domain against RuntimePlan."""

        plan = resolve_runtime_plan(vllm_config)
        from vllm_musa.engine_plugins import (
            validate_engine_plugin_cudagraph_runtime,
        )

        validate_engine_plugin_cudagraph_runtime(
            vllm_config,
            required=plan.decision_resolution is not None,
            serialized_transport=bool(
                getattr(vllm_config, "additional_config", {}).get(
                    RUNTIME_PLAN_TRANSPORT_KEY
                )
            ),
        )
        policy = plan.value(
            RuntimeDecision.MUSA_FUSED_MOE_DISPATCH_POLICY,
            (),
        )
        _validate_fused_moe_cudagraph_padding(
            vllm_config,
            policy=policy,
            uniform_decode_query_len=uniform_decode_query_len,
        )
        if _fused_moe_capture_policy_entries(policy):
            compilation_config = vllm_config.compilation_config
            resolution = plan.decision_resolution
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else -1
            )
            logger.info(
                "MUSA RuntimePlan CUDAGraph padding validated: plan_id=%s "
                "plan_fingerprint=%s rank=%d graph_mode=%s capture_sizes=%s "
                "uniform_decode_query_len=%d",
                resolution.plan_id if resolution is not None else plan.profile,
                resolution.fingerprint if resolution is not None else plan.fingerprint,
                rank,
                compilation_config.cudagraph_mode,
                compilation_config.cudagraph_capture_sizes,
                uniform_decode_query_len,
            )

    @classmethod
    def _find_all_non_ssm_backends(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[type[Any], ...]:
        """Return every distinct live attention backend participating in KV layout."""
        from vllm.config.vllm import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )

        attention_layers = get_layers_from_vllm_config(
            vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        backends: dict[type[Any], None] = {}
        for layer in attention_layers.values():
            backend_cls = layer.get_attn_backend()
            if not backend_cls.is_ssm():
                backends.setdefault(backend_cls, None)
        return tuple(backends)

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        # The separate Mamba pools have independent physical pages and block-id
        # spaces. In the no-prefix-cache mode used by Qwen3.5/Qwen3.6 serving,
        # do not let upstream inflate the attention block to the full recurrent
        # state page (1056 tokens for Qwen3.6-35B-A3B). That inflation wastes
        # attention KV capacity and forces avoidable request preemption.
        separate_mamba_pages = (
            model_config is not None
            and cache_config is not None
            and model_config.is_hybrid
            and cache_config.mamba_cache_mode == "none"
            and resolve_runtime_plan(vllm_config).selected(
                RuntimeDecision.HYBRID_KV_CACHE_POOL_LAYOUT,
                "separate",
            )
        )
        if separate_mamba_pages:
            backend_cls = cls._find_non_ssm_backend(vllm_config)
            if backend_cls is not None:
                from vllm.config.vllm import set_current_vllm_config

                target_block_size = (
                    cache_config.block_size
                    if cache_config.user_specified_block_size
                    else 64
                )
                with set_current_vllm_config(vllm_config):
                    supported = backend_cls.supports_block_size(target_block_size)
                if supported:
                    cache_config.block_size = target_block_size
                    cache_config.mamba_page_size_padded = None
                    logger.info(
                        "[MUSA]Keeping separate-pool hybrid attention block "
                        "size at %d for %s backend.",
                        target_block_size,
                        backend_cls.get_name(),
                    )
                    return

        # MUSA: 64 is the only optimal KV page here (paged FMHA/MLA decode takes
        # the TME bulk-gather path). Let upstream pick and mamba-align the page
        # first, then pin every non-hybrid backend that supports a 64 page to 64,
        # whatever super() picked. A user --block-size and fixed-page kernels that
        # cannot take 64 (sparse MLA at 256) are left as super() resolved them.
        super().update_block_size_for_backend(vllm_config)
        if model_config is None or cache_config is None:
            return
        plan = resolve_runtime_plan(vllm_config)
        if plan.model.family is ModelFamily.DEEPSEEK_V4:
            decision = RuntimeDecision.DEEPSEEK_V4_FLASHMLA_SPARSE_PAGE_SIZE
            if not plan.supports(decision):
                return
            backend_classes = cls._find_all_non_ssm_backends(vllm_config)
            if not backend_classes:
                raise RuntimeError(
                    "Cannot validate the DeepSeek-V4 RuntimePlan KV layout "
                    "because no live attention backends were discovered"
                )
            planned_block_size = int(plan.value(decision))
            from vllm.config.vllm import set_current_vllm_config

            unsupported_backends: list[str] = []
            for backend_cls in backend_classes:
                with set_current_vllm_config(vllm_config):
                    backend_supports_plan = backend_cls.supports_block_size(
                        planned_block_size
                    )
                if not backend_supports_plan:
                    unsupported_backends.append(backend_cls.get_name())
            if unsupported_backends:
                raise RuntimeError(
                    "DeepSeek-V4 RuntimePlan KV block size "
                    f"{planned_block_size} is unsupported by the live "
                    f"backends {', '.join(sorted(unsupported_backends))}"
                )
            backend_names = ",".join(
                sorted(backend_cls.get_name() for backend_cls in backend_classes)
            )
            if cache_config.block_size != planned_block_size:
                raise RuntimeError(
                    "DeepSeek-V4 RuntimePlan KV block size drifted after live "
                    f"backend discovery: planned={planned_block_size}, "
                    f"actual={cache_config.block_size}, "
                    f"backends={backend_names}"
                )
            logger.info(
                "Validated final DeepSeek-V4 RuntimePlan KV block size=%d: "
                "backends=%s profile=%s.",
                planned_block_size,
                backend_names,
                plan.profile,
            )
            return
        if cache_config.user_specified_block_size or model_config.is_hybrid:
            return
        backend_cls = cls._find_non_ssm_backend(vllm_config)
        if backend_cls is None:
            return
        from vllm.config.vllm import set_current_vllm_config

        with set_current_vllm_config(vllm_config):
            supports_64 = backend_cls.supports_block_size(64)
        if supports_64 and cache_config.block_size != 64:
            logger.info(
                "[MUSA]Setting attention block size to 64 for %s backend.",
                backend_cls.get_name(),
            )
            cache_config.block_size = 64

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", list[str]],
    ]:
        valid_backends_priorities = []
        invalid_reasons = {}

        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            device_capability,
            num_heads,
        )
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = backend.get_class()
                invalid_reasons_i = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = invalid_reasons_i
            else:
                valid_backends_priorities.append((backend, priority))

        return valid_backends_priorities, invalid_reasons

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        register_attention_backends()
        device_capability = cls.get_device_capability()
        assert device_capability is not None

        attn_selector_config = attn_selector_config._replace(block_size=None)
        # First try checking just the selected backend, if there is one.
        if selected_backend is not None:
            try:
                backend_class = selected_backend.get_class()
                invalid_reasons = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons = ["ImportError"]
            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                logger.info("Using %s backend.", selected_backend)
                return selected_backend.get_path()

        # No selected backend or the selected backend is invalid,
        # so we try finding a valid backend.
        valid_backends_priorities, invalid_reasons = cls.get_valid_backends(
            device_capability=device_capability,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )
        reasons_str = (
            "{"
            + ", ".join(
                f"{backend.name}: [{', '.join(reasons)}]"
                for backend, reasons in invalid_reasons.items()
            )
            + "}"
        )
        config_str = attn_selector_config.__repr__()
        logger.debug_once(
            f"Some attention backends are not valid for {cls.device_name} with "
            f"{config_str}. Reasons: {reasons_str}."
        )
        if len(valid_backends_priorities) == 0:
            raise ValueError(
                f"No valid attention backend found for {cls.device_name} "
                f"with {config_str}. Reasons: {reasons_str}."
            )

        # We have found some valid backends. Select the one with the
        # highest priority.
        sorted_indices = sorted(
            range(len(valid_backends_priorities)),
            key=lambda i: valid_backends_priorities[i][1],
        )
        selected_index = sorted_indices[0]
        selected_backend = valid_backends_priorities[selected_index][0]
        logger.info_once(
            "Using %s attention backend out of potential backends: %s.",
            selected_backend.name,
            "[" + ", ".join(f"'{b[0].name}'" for b in valid_backends_priorities) + "]",
            scope="local",
        )

        return selected_backend.get_path()

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
        ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: "AttentionBackendEnum | None" = None,
    ) -> "AttentionBackendEnum":
        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention. "
                f"Supported backends are: {cls.get_supported_vit_attn_backends()}"
            )
            logger.info_once(f"Using backend {backend} for vit attention")
            return backend

        cc = cls.get_device_capability()
        for vit_attn_backend in cls.get_supported_vit_attn_backends():
            if vit_attn_backend == AttentionBackendEnum.TORCH_SDPA:
                continue
            try:
                backend_class = vit_attn_backend.get_class()
                is_backend_supported = backend_class.supports_head_size(
                    head_size
                ) and backend_class.supports_dtype(dtype)
                if cc is not None:
                    is_backend_supported = (
                        is_backend_supported
                        and backend_class.supports_compute_capability(cc)
                    )
                if is_backend_supported:
                    logger.info_once(
                        f"Using backend {vit_attn_backend} for vit attention"
                    )
                    return vit_attn_backend
            except ImportError:
                pass

        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa
        )

    @classmethod
    def supports_fp8(cls) -> bool:
        return cls.has_device_capability((3, 1))

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return True

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def device_count(cls) -> int:
        return torch.cuda.device_count()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        # MUSA devices support bfloat16 natively, no capability check needed.
        pass

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache on GPU."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from GPU to host (CPU)."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def num_compute_units(cls, device_id=0):
        return torch.cuda.get_device_properties(device_id).multi_processor_count


# MTML utils
# Note that MTML is not affected by `MUSA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using MTML is that it will not initialize MUSA
class MtmlMUSAPlatform(MUSAPlatformBase):
    @classmethod
    @cache
    @with_mtml_context
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        try:
            # XXX (MUSA): CUDA uses physical device ids (cls.device_id_to_physical_device_id(device_id)),
            # but torch.musa.get_device_capability uses logical device ids when MUSA_VISIBLE_DEVICES is set.
            # Since pymtml doesn't do remapping, so we can only use device_id here.
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @with_mtml_context
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        try:
            return super().has_device_capability(capability, device_id)
        except RuntimeError:
            return False

    @classmethod
    @with_mtml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @with_mtml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return pynvml.nvmlDeviceGetUUID(handle)

    @classmethod
    @with_mtml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_mtml_context
    def get_device_numa_node(cls, device_id: int = 0) -> int | None:
        """Get the NUMA node ID for a GPU device."""
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)

        try:
            numa_node = pynvml.nvmlDeviceGetNumaNodeId(handle)
            if cls._numa_node_has_cpus(numa_node):
                return numa_node
            logger.debug(
                "NUMA node %d for GPU %d has no CPUs, falling back to "
                "CPU-affinity-based detection",
                numa_node,
                device_id,
            )
        except Exception:
            pass

        try:
            cpu_ids = cls._get_device_cpu_affinity(handle)
            if cpu_ids:
                numa_node = cls._get_numa_node_for_cpu(cpu_ids[0])
                if numa_node is not None:
                    logger.debug(
                        "Determined NUMA node %d for GPU %d via CPU affinity",
                        numa_node,
                        device_id,
                    )
                    return numa_node
        except Exception as e:
            logger.warning("Failed to get NUMA node for GPU %d: %s", device_id, e)

        return None

    @classmethod
    def _numa_node_has_cpus(cls, node_id: int) -> bool:
        """Check whether a NUMA node has any CPUs assigned to it."""
        cpulist_file = Path(f"/sys/devices/system/node/node{node_id}/cpulist")
        try:
            return cpulist_file.read_text().strip() != ""
        except (OSError, ValueError):
            return False

    @classmethod
    def _get_device_cpu_affinity(cls, handle) -> list[int]:
        """Get the list of CPU IDs associated with a GPU via MTML."""
        cpu_count = os.cpu_count()
        if cpu_count is None:
            return []

        cpu_set_size = (cpu_count + 63) // 64
        cpu_affinity_mask = pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_set_size)

        cpu_ids = []
        for i, mask in enumerate(cpu_affinity_mask):
            for bit in range(64):
                cpu_id = i * 64 + bit
                if cpu_id >= cpu_count:
                    break
                if mask & (1 << bit):
                    cpu_ids.append(cpu_id)
        return cpu_ids

    @classmethod
    def _get_numa_node_for_cpu(cls, cpu_id: int) -> int | None:
        """Determine which NUMA node a CPU belongs to."""
        node_path = Path("/sys/devices/system/node")
        if not node_path.exists():
            return None

        for node_dir in node_path.iterdir():
            if not node_dir.name.startswith("node"):
                continue
            try:
                node_id = int(node_dir.name[4:])
                cpulist_file = node_dir / "cpulist"
                if cpulist_file.exists():
                    cpulist = cpulist_file.read_text().strip()
                    if cls._cpu_in_cpulist(cpu_id, cpulist):
                        return node_id
            except (ValueError, OSError):
                continue
        return None

    @classmethod
    def _cpu_in_cpulist(cls, cpu_id: int, cpulist: str) -> bool:
        """Check if a CPU ID is in a cpulist string such as '0-3,8-11'."""
        for part in cpulist.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                if int(start) <= cpu_id <= int(end):
                    return True
            elif part.isdigit() and int(part) == cpu_id:
                return True
        return False

    @classmethod
    @with_mtml_context
    def get_all_device_numa_nodes(cls) -> list[int] | None:
        """Get NUMA nodes for all visible GPU devices."""
        try:
            numa_nodes = []
            for device_id in range(cls.device_count()):
                numa_node = cls.get_device_numa_node(device_id)
                if numa_node is None:
                    logger.warning(
                        "Could not detect NUMA node for GPU %d, "
                        "disabling automatic NUMA binding",
                        device_id,
                    )
                    return None
                numa_nodes.append(numa_node)
            return numa_nodes
        except Exception as e:
            logger.warning("Failed to get NUMA nodes for GPUs: %s", e)
            return None

    @classmethod
    @with_mtml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        query if the set of gpus are fully connected by mtlink (1 hop)
        """
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception(
                            "MtLink detection failed. This is normal if"
                            " your machine has no MtLink equipped."
                        )
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetName(handle)

    @classmethod
    @with_mtml_context
    def log_warnings(cls):
        device_ids: int = pynvml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [cls._get_physical_device_name(i) for i in range(device_ids)]
            if (
                len(set(device_names)) > 1
                and os.environ.get("MUSA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                logger.warning(
                    "Detected different devices in the system: %s. Please"
                    " make sure to set `MUSA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )


class NonMtmlMUSAPlatform(MUSAPlatformBase):
    @classmethod
    @cache
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        logger.exception(
            "MtLink detection not possible, as context support was"
            " not found. Assuming no MtLink available."
        )
        return False


# Autodetect either MTML-enabled or non-MTML platform
# based on whether MTML is available.
mtml_available = False
try:
    try:
        pynvml.nvmlInit()
        mtml_available = True
    except Exception:
        # MTML may not be supported on all systems.
        mtml_available = False
finally:
    # Note: We intentionally do NOT call nvmlShutdown() here because
    # pymtml has a bug where nvmlInit() fails after nvmlShutdown()
    # has been called. The library handles cleanup on process exit.
    pass

MUSAPlatform = MtmlMUSAPlatform if mtml_available else NonMtmlMUSAPlatform

MUSAPlatform.log_warnings()

__all__ = [
    "MUSAPlatform",
    "MUSAPlatformBase",
    "MtmlMUSAPlatform",
    "NonMtmlMUSAPlatform",
    "mtml_available",
    "with_mtml_context",
]
