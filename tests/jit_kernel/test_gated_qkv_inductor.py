"""Tests for the Inductor-visible gated-QKV MUSA Triton provider."""

import copy
import os

import pytest
import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.fixture(scope="module", autouse=True)
def _musa_device():
    pytest.importorskip("torch_musa")
    pytest.importorskip("torchada")
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.musa.set_device(0)
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability((3, 1)):
        pytest.skip("the gated-QKV Triton provider requires an S5000/mp31 device")


def _inputs(
    num_tokens: int,
    num_q_heads: int,
    *,
    seed: int | None = None,
    randomized_ir_inputs: bool = False,
) -> tuple:
    device = torch.device("musa:0")
    head_dim = 256
    rotary_dim = 64
    torch.manual_seed(
        seed if seed is not None else 1234 + num_tokens + 100 * num_q_heads
    )
    packed = torch.randn(
        (num_tokens, (2 * num_q_heads + 2) * head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    weight_scale = 1.0 if randomized_ir_inputs else 0.02
    q_weight = weight_scale * torch.randn(
        (head_dim,), device=device, dtype=torch.bfloat16
    )
    k_weight = weight_scale * torch.randn(
        (head_dim,), device=device, dtype=torch.bfloat16
    )
    if randomized_ir_inputs:
        cache = torch.randn((4096, rotary_dim), device=device, dtype=torch.bfloat16)
    else:
        inv_freq = 1.0 / (
            10000000.0
            ** (
                torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                / rotary_dim
            )
        )
        frequencies = torch.outer(
            torch.arange(4096, device=device, dtype=torch.float32), inv_freq
        )
        cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(
            torch.bfloat16
        )
    position_storage = torch.empty(
        (3 * (num_tokens + 17),), device=device, dtype=torch.int64
    )
    positions = torch.as_strided(
        position_storage,
        (3, num_tokens),
        (num_tokens + 17, 1),
    )
    if randomized_ir_inputs:
        positions.copy_(
            torch.randint(
                4096,
                (3, num_tokens),
                device=device,
                dtype=torch.int64,
            )
        )
    else:
        token = torch.arange(num_tokens, device=device, dtype=torch.int64)
        positions[0].copy_(token % 4096)
        positions[1].copy_((token * 3 + 7) % 4096)
        positions[2].copy_((token * 5 + 11) % 4096)
    return (
        packed,
        q_weight,
        k_weight,
        cache,
        positions,
        1e-6,
        num_q_heads,
        1,
        head_dim,
        rotary_dim,
        [11, 11, 10],
        True,
        True,
        1.0,
    )


@pytest.mark.parametrize("num_q_heads", [2, 3])
@pytest.mark.parametrize("num_tokens", [1, 64, 2500])
def test_gated_qkv_inductor_matches_native(
    num_q_heads: int,
    num_tokens: int,
) -> None:
    from vllm import ir

    from vllm_musa.kernels.gated_qkv import gated_qkv_rms_norm_rope

    args = _inputs(num_tokens, num_q_heads)
    expected = ir.ops.gated_qkv_rms_norm_rope.impls["native"].impl_fn(*args)
    actual = gated_qkv_rms_norm_rope.impl_fn(*args)
    torch.musa.synchronize()

    assert actual[0].is_contiguous()
    assert actual[1].is_contiguous()
    assert (
        actual[0].untyped_storage().data_ptr() != actual[1].untyped_storage().data_ptr()
    )
    tolerance = ir.ops.gated_qkv_rms_norm_rope.get_tolerance(torch.bfloat16)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor.float(),
            expected_tensor.float(),
            **tolerance,
        )


@pytest.mark.parametrize("num_q_heads", [2, 3])
@pytest.mark.parametrize("seed", [15, 17, 1234, 9001])
def test_gated_qkv_inductor_matches_randomized_ir_inputs(
    num_q_heads: int,
    seed: int,
) -> None:
    """Cover the IR generator's unconstrained BF16 weights and RoPE cache."""
    from vllm import ir

    from vllm_musa.kernels.gated_qkv import gated_qkv_rms_norm_rope

    args = _inputs(
        2500,
        num_q_heads,
        seed=seed,
        randomized_ir_inputs=True,
    )
    expected = ir.ops.gated_qkv_rms_norm_rope.impls["native"].impl_fn(*args)
    actual = gated_qkv_rms_norm_rope.impl_fn(*args)
    torch.musa.synchronize()

    tolerance = ir.ops.gated_qkv_rms_norm_rope.get_tolerance(torch.bfloat16)
    assert tolerance == {"atol": 2e-2, "rtol": 1.6e-2}
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor.float(), expected_tensor.float(), **tolerance
        )


def test_gated_qkv_vllm_ir_lowers_to_triton_hop() -> None:
    from torch import fx
    from torch._inductor.compile_fx import compile_fx
    from vllm import ir
    from vllm.compilation.passes.inductor_pass import pass_context
    from vllm.compilation.passes.ir.inplace_functionalization import (
        VllmIRInplaceFunctionalizationPass,
    )
    from vllm.compilation.passes.ir.lowering_pass import VllmIRLoweringPass
    from vllm.config import VllmConfig
    from vllm.config.utils import Range
    from vllm.platforms import current_platform

    current_platform.import_ir_kernels()
    args_64 = _inputs(64, 3)
    args_2500 = _inputs(2500, 3)
    for args in (args_64, args_2500):
        torch._dynamo.mark_dynamic(args[0], 0, min=64, max=8192)
        torch._dynamo.mark_dynamic(args[4], 1, min=64, max=8192)

    def model(packed, q_weight, k_weight, cache, positions):
        return ir.ops.gated_qkv_rms_norm_rope(
            packed,
            q_weight,
            k_weight,
            cache,
            positions,
            1e-6,
            3,
            1,
            256,
            64,
            [11, 11, 10],
            True,
            True,
            1.0,
        )

    vllm_config = VllmConfig()
    lowering = VllmIRLoweringPass(vllm_config)
    graph_targets = []
    compile_count = 0

    def post_grad_pass(graph: fx.Graph) -> None:
        lowering(graph)
        graph.owning_module.recompile()
        graph_targets.extend(str(node.target) for node in graph.nodes)

    inductor_config = copy.deepcopy(
        vllm_config.compilation_config.inductor_compile_config
    )
    inductor_config["force_disable_caches"] = True
    inductor_config["pre_grad_custom_pass"] = VllmIRInplaceFunctionalizationPass(
        vllm_config
    )
    inductor_config["post_grad_custom_post_pass"] = post_grad_pass

    def backend(graph: fx.GraphModule, example_inputs):
        nonlocal compile_count
        compile_count += 1
        with pass_context(Range(64, 8192)):
            return compile_fx(
                graph,
                example_inputs,
                config_patches=inductor_config,
            )

    with (
        ir.ops.gated_qkv_rms_norm_rope.set_priority(["musa_inductor", "native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
        actuals = [compiled(*args[:5]) for args in (args_64, args_2500)]

    assert compile_count == 1
    assert (
        sum("triton_kernel_wrapper_mutation" in target for target in graph_targets) == 1
    )
    assert str(torch.ops.aten.clone.default) not in graph_targets
    assert set(lowering.selected_impls["gated_qkv_rms_norm_rope"].values()) == {
        "musa_inductor"
    }

    tolerance = ir.ops.gated_qkv_rms_norm_rope.get_tolerance(torch.bfloat16)
    for args, actual in zip((args_64, args_2500), actuals):
        expected = ir.ops.gated_qkv_rms_norm_rope.impls["native"].impl_fn(*args)
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(
                actual_tensor.float(), expected_tensor.float(), **tolerance
            )


def test_gated_qkv_inductor_support_scope(monkeypatch) -> None:
    from vllm_musa.kernels import gated_qkv

    _supports_gated_qkv = gated_qkv._supports_gated_qkv

    args = _inputs(64, 3)
    assert _supports_gated_qkv(*args)

    unsupported = list(args)
    unsupported[10] = [16, 8, 8]
    assert not _supports_gated_qkv(*unsupported)

    unsupported = list(args)
    unsupported[13] = 0.0
    assert not _supports_gated_qkv(*unsupported)

    unsupported = list(args)
    unsupported[0] = unsupported[0].cpu()

    def fail_capability_query(*args, **kwargs):
        pytest.fail("non-MUSA inputs must not query MUSA device capability")

    # The capability query the kernel makes is `current_platform.is_validated_musa_target`;
    # `current_platform` here left the mock unconsulted.
    monkeypatch.setattr(
        gated_qkv.current_platform, "is_validated_musa_target", fail_capability_query
    )
    assert not _supports_gated_qkv(*unsupported)

    if torch.musa.device_count() > 1:
        unsupported = list(args)
        unsupported[1] = unsupported[1].to("musa:1")
        assert not _supports_gated_qkv(*unsupported)
