# SPDX-License-Identifier: Apache-2.0
"""Pin how ``optimization_level`` drives the CAR-RMSNorm default-on rule.

``optimization_level`` is a real ``VllmConfig`` field, not a derived property:

    optimization_level: OptimizationLevel = OptimizationLevel.O2   # == 2

so the ``>= 2`` half of the default-on rule holds for a plain ``vllm serve``.
What makes the rule load-bearing is that upstream already claims the same field:
``OPTIMIZATION_LEVEL_TO_CONFIG`` sets
``compilation_config.pass_config.fuse_allreduce_rms`` from
``enable_allreduce_rms_fusion`` on O2 and to ``False`` on O1/O0, and that
predicate is CUDA-only, so on MUSA it answers ``False``.

The platform hook runs before the optimization-level presets and
``_set_config_default`` only writes a still-``None`` field, so the hook wins on
O2 and must yield on O1/O0. These tests fix both halves, and pin that carrying
the level in the execution signature moves no other provider's verdict.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_musa.optimization_contract.car_rmsnorm import (
    car_rmsnorm_default_on,
    resolve_car_rmsnorm_enabled,
)
from vllm_musa.optimization_contract.resolver import resolve_optimization_contract
from vllm_musa.optimization_contract.types import ExecutionSignature

# The architecture strings are what each provider keys its family off, so a real
# family resolution needs them present rather than monkeypatched.
QWEN35_ARCH = "Qwen3_5ForConditionalGeneration"
QWEN35_MOE_ARCH = "Qwen3_5MoeForConditionalGeneration"
DEEPSEEK_V4_ARCH = "DeepseekV4ForCausalLM"
GLM_ARCH = "GlmMoeDsaForCausalLM"


def _config(
    *,
    optimization_level: int | None,
    pass_value: bool | None = None,
    tp_size: int = 2,
    hidden_size: int = 5120,
    dtype: torch.dtype | None = torch.bfloat16,
    architectures: tuple[str, ...] = (QWEN35_ARCH,),
) -> SimpleNamespace:
    return SimpleNamespace(
        optimization_level=optimization_level,
        model_config=SimpleNamespace(
            architectures=list(architectures),
            dtype=dtype,
            get_hidden_size=lambda: hidden_size,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(fuse_allreduce_rms=pass_value)
        ),
    )


def test_optimization_level_is_a_real_field_that_defaults_to_o2() -> None:
    """The `>= 2` gate is satisfied by default, so it is not an opt-in gate."""
    from vllm.config.vllm import OptimizationLevel, VllmConfig

    default = VllmConfig.__dataclass_fields__["optimization_level"].default
    assert default is OptimizationLevel.O2
    assert int(OptimizationLevel.O2) == 2
    assert int(OptimizationLevel.O1) == 1
    assert int(OptimizationLevel.O0) == 0


def test_upstream_owns_and_forbids_the_same_pass_field() -> None:
    """Upstream already decides `fuse_allreduce_rms` from the O level.

    O2 delegates to a CUDA-only predicate; O1 and O0 hard-code ``False``. The
    MUSA hook therefore has to win on O2 and yield on O1/O0 -- which is exactly
    the shape of the ``optimization_level >= 2`` gate.
    """
    from vllm.config.vllm import (
        OPTIMIZATION_LEVEL_00,
        OPTIMIZATION_LEVEL_01,
        OPTIMIZATION_LEVEL_02,
        enable_allreduce_rms_fusion,
    )

    key = "fuse_allreduce_rms"
    assert OPTIMIZATION_LEVEL_00["compilation_config"]["pass_config"][key] is False
    assert OPTIMIZATION_LEVEL_01["compilation_config"]["pass_config"][key] is False
    assert (
        OPTIMIZATION_LEVEL_02["compilation_config"]["pass_config"][key]
        is enable_allreduce_rms_fusion
    )


def test_upstream_predicate_is_false_on_musa() -> None:
    """`enable_allreduce_rms_fusion` is CUDA/90/100-only, so MUSA gets False.

    Without the platform hook the O2 preset would leave ``fuse_allreduce_rms``
    at ``False`` on every MUSA host.
    """
    from vllm.config.vllm import enable_allreduce_rms_fusion
    from vllm.platforms import current_platform

    assert not current_platform.is_cuda(), current_platform
    stub = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2)
    )
    assert enable_allreduce_rms_fusion(stub) is False


def test_platform_hook_runs_before_the_o_level_presets() -> None:
    """If upstream reorders these, the hook silently stops applying.

    Pairs with ``_set_config_default`` only filling a still-``None`` field: that
    rule is what makes the earlier write stick. Checked by source order because
    the alternative is starting a real engine per assertion.
    """
    import inspect

    from vllm.config.vllm import VllmConfig

    source = inspect.getsource(VllmConfig.__post_init__)
    hook = source.index("apply_config_platform_defaults")
    preset = source.index("_apply_optimization_level_defaults")
    assert hook < preset, "O-level defaults now run before the platform hook"
    assert "if getattr(config_obj, key) is None:" in inspect.getsource(
        VllmConfig._set_config_default
    )


@pytest.mark.parametrize(
    ("optimization_level", "expected"),
    [(0, False), (1, False), (2, True), (3, True)],
)
def test_default_on_gate_tracks_the_upstream_o1_o2_boundary(
    optimization_level: int,
    expected: bool,
) -> None:
    config = _config(optimization_level=optimization_level)
    assert car_rmsnorm_default_on(config) is expected
    assert resolve_car_rmsnorm_enabled(config) is expected


def test_default_on_requires_a_policy_cell_and_a_supported_dtype() -> None:
    assert car_rmsnorm_default_on(_config(optimization_level=2))
    assert not car_rmsnorm_default_on(_config(optimization_level=2, tp_size=8))
    assert not car_rmsnorm_default_on(_config(optimization_level=2, hidden_size=4096))
    assert not car_rmsnorm_default_on(
        _config(optimization_level=2, dtype=torch.float32)
    )
    assert not car_rmsnorm_default_on(
        _config(optimization_level=2, architectures=(DEEPSEEK_V4_ARCH,))
    )


def test_a_missing_level_fails_closed_rather_than_reading_as_o0() -> None:
    """``None`` means "no vLLM config"; it must not behave like a real level."""
    assert resolve_car_rmsnorm_enabled(_config(optimization_level=None)) is False


def test_an_explicit_pass_value_stays_authoritative() -> None:
    assert resolve_car_rmsnorm_enabled(
        _config(optimization_level=0, pass_value=True)
    )
    assert not resolve_car_rmsnorm_enabled(
        _config(optimization_level=3, pass_value=False)
    )


def test_car_gate_reads_the_level_from_the_execution_signature() -> None:
    """The gate and the level justifying it must come from one resolution."""
    for level in (0, 1, 2, 3):
        contract = resolve_optimization_contract(_config(optimization_level=level))
        assert isinstance(contract.execution, ExecutionSignature)
        assert contract.execution.optimization_level == level


@pytest.mark.parametrize(
    "architectures",
    [
        (QWEN35_ARCH,),
        (QWEN35_MOE_ARCH,),
        (DEEPSEEK_V4_ARCH,),
        (GLM_ARCH,),
        ("SomeOtherForCausalLM",),
    ],
)
def test_carrying_the_level_moves_no_provider_verdict(architectures) -> None:
    """Adding a signature field is only safe while no provider reads it.

    ``optimization_level`` is always present on a real config and defaults to
    ``O2``, so recording it cannot change an existing verdict unless a provider
    starts branching on it. Pin that: every level must resolve identically.
    """
    verdicts = []
    for level in (0, 1, 2, 3):
        contract = resolve_optimization_contract(
            _config(optimization_level=level, architectures=architectures)
        )
        verdicts.append(
            (contract.model.family, contract.supported_features, contract.preferred_features)
        )
    assert all(v == verdicts[0] for v in verdicts), architectures

