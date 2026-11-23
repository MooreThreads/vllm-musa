# SPDX-License-Identifier: Apache-2.0
"""Gates for the MATE-backed W4A16 linear kernel.

Two layers, because they fail for different reasons:

* the config decisions (``can_implement``, zero-point orientation) are pure logic and run anywhere;
* the numeric gate needs a MUSA device and compares the kernel against an fp32 dequantised reference,
  which is the only reference that can catch a frame that is internally consistent but wrong.

Every numeric arm here exists because the corresponding mistake was actually made:

* a zero point read from a parameter the config does not declare,
* a ``g_idx`` read from a parameter a non-``desc_act`` checkpoint never fills,
* a reference dequantised with the wrong scale convention, which turned a correct kernel into a
  19.5 % "failure".
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("mate.marlin", reason="MATE is not installed in this environment")

K, N, GROUP, M = 1024, 512, 128, 8

requires_musa = pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    reason="the numeric arms need a MUSA device",
)


def _config(has_g_idx: bool, *, group_size: int = GROUP, act_type=torch.float16, size_k: int = K):
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    return MPLinearLayerConfig(
        full_weight_shape=(size_k, N),
        partition_weight_shape=(size_k, N),
        weight_type=scalar_types.uint4b8,
        act_type=act_type,
        group_size=group_size,
        zero_points=False,
        has_g_idx=has_g_idx,
        out_type=None,
    )


def _kernel():
    from vllm_musa.model_executor.kernels.linear.mixed_precision.mate_w4a16 import (
        MateW4A16LinearKernel,
    )

    return MateW4A16LinearKernel


# --------------------------------------------------------------------------------------
# config decisions: no device needed
# --------------------------------------------------------------------------------------


def test_accepts_the_serving_shape():
    ok, reason = _kernel().can_implement(_config(has_g_idx=False))
    assert ok, reason


def test_accepts_act_order_because_it_is_servable():
    """``desc_act`` is not refused: the permutation cancels and the numeric gate below proves it.

    ``can_implement`` sees only ``has_g_idx: bool`` and cannot inspect the tensor, so refusing here
    would also refuse a configuration the kernel serves correctly.
    """
    ok, reason = _kernel().can_implement(_config(has_g_idx=True))
    assert ok, reason


@pytest.mark.parametrize("group_size", [32, 64, 256])
def test_refuses_other_group_sizes(group_size):
    ok, reason = _kernel().can_implement(_config(False, group_size=group_size))
    assert not ok
    assert "group size" in reason


def test_refuses_k_not_multiple_of_256():
    ok, reason = _kernel().can_implement(_config(False, size_k=K + 128))
    assert not ok
    assert "K % 256" in reason


def test_refuses_non_fp16_activation():
    ok, reason = _kernel().can_implement(_config(False, act_type=torch.bfloat16))
    assert not ok
    assert "fp16" in reason


def test_zero_point_orientation_is_named_from_the_shape():
    from vllm_musa.model_executor.kernels.linear.mixed_precision.mate_w4a16 import (
        _packed_zero_point_dim,
    )

    groups = K // GROUP
    # GPTQ checkpoints store (num_groups, N/8); vLLM's AWQ conversion produces (N/8, num_groups).
    assert _packed_zero_point_dim((groups, N // 8), groups, N) == 1
    assert _packed_zero_point_dim((N // 8, groups), groups, N) == 0
    with pytest.raises(ValueError, match="neither"):
        _packed_zero_point_dim((groups + 1, N // 8), groups, N)


# --------------------------------------------------------------------------------------
# numeric gate: needs a device
# --------------------------------------------------------------------------------------


def _quantize_symmetric(w: torch.Tensor, group: int):
    """(K, N) fp32 -> codes in [1, 15] + one scale per group, uint4b8 convention (zero code = 8)."""
    g = w.shape[0] // group
    wg = w.reshape(g, group, w.shape[1])
    amax = wg.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
    q = torch.clamp(torch.round(wg / scale), -7, 7).to(torch.int64) + 8
    return q.reshape(w.shape), scale.reshape(g, w.shape[1])


def _pack_rows(u: torch.Tensor) -> torch.Tensor:
    """(K, N) nibbles -> (K/8, N) int32, the GPTQ input layout: 8 consecutive k per word."""
    k, n = u.shape
    u = u.to(torch.int64).reshape(k // 8, 8, n)
    words = torch.zeros((k // 8, n), dtype=torch.int64, device=u.device)
    for j in range(8):
        words |= (u[:, j, :] & 0xF) << (4 * j)
    return words.to(torch.int32)



@requires_musa
def test_garbage_g_idx_is_ignored_when_the_config_has_no_act_order():
    dev = "musa"
    torch.manual_seed(0)
    w = (torch.randn(K, N) * 0.05).float()
    codes, scales = _quantize_symmetric(w, GROUP)
    wdq = ((codes.reshape(K // GROUP, GROUP, N) - 8) * scales.reshape(-1, 1, N)).reshape(K, N)

    words = _pack_rows(codes).to(dev)
    sc = scales.half().to(dev)
    x = (torch.randn(M, K, device=dev) * 0.1).half()
    ref = (x.float() @ wdq.to(dev).float()).half()

    cls = _kernel()
    ok, reason = cls.can_implement(_config(False))
    assert ok, reason
    cfg = _config(False)
    # A checkpoint with desc_act=False never fills g_idx, so the layer holds uninitialised memory.
    # Feeding it to the kernel must change nothing, and must not read as a permutation.
    garbage = torch.randint(0, 999, (K,), dtype=torch.int32, device=dev)
    # Likewise qzeros: auto_gptq registers it for every layer while the config says zero_points=False.
    garbage_zp = torch.full((K // GROUP, N // 8), 7, dtype=torch.int32, device=dev)

    out = []
    for _ in range(2):
        kernel = cls(
            cfg,
            w_q_param_name="qweight",
            w_s_param_name="scales",
            w_zp_param_name="qzeros",
            w_gidx_param_name="g_idx",
        )
        layer = _dummy_layer(words.clone(), sc.clone(), garbage.clone(), garbage_zp.clone())
        kernel.process_weights_after_loading(layer)
        out.append(kernel.apply_weights(layer, x))

    err = (out[0].float() - ref.float()).abs()
    rms = (err.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item()
    assert rms < 5e-3, f"rms_rel={rms:.3e}"
    assert torch.equal(out[0], out[1]), "two identical calls disagreed: nonzero state leaked between them"


def _dummy_layer(words, scales, gidx, qzeros):
    layer = torch.nn.Module()
    layer.register_parameter("qweight", torch.nn.Parameter(words, requires_grad=False))
    layer.register_parameter("scales", torch.nn.Parameter(scales, requires_grad=False))
    layer.register_parameter("qzeros", torch.nn.Parameter(qzeros, requires_grad=False))
    layer.register_parameter("g_idx", torch.nn.Parameter(gidx, requires_grad=False))
    return layer


@requires_musa
def test_act_order_cancels_instead_of_shifting_the_result():
    dev = "musa"
    torch.manual_seed(0)
    w = (torch.randn(K, N) * 0.05).float()
    codes, scales = _quantize_symmetric(w, GROUP)

    # A group-reversed g_idx: rows are stored block-reversed, so the k-th row belongs to group
    # g_idx[k] and the dequantised weight is (u - 8) * SCALE[g_idx[k]] -- not SCALE[k // group].
    # Sorting by the permutation restores canonical order; gathering x with the same permutation
    # cancels, so the kernel's output equals x @ W_act. Using canonical scales here instead was the
    # 19.5 % "failure" that was entirely the reference's fault.
    g = torch.arange(K, dtype=torch.int64) // GROUP
    pg = g.reshape(K // GROUP, GROUP).flip(0).reshape(K).to(torch.int32)
    perm = torch.argsort(pg.to(torch.int64), stable=True)
    assert torch.equal(pg[perm].to(torch.int64), torch.arange(K) // GROUP)

    w_act = ((codes - 8).float() * scales.float()[pg.to(torch.int64)]).to(dev)
    words = _pack_rows(codes).to(dev)
    sc = scales.half().to(dev)
    x = (torch.randn(M, K, device=dev) * 0.1).half()
    ref = (x.float() @ w_act.float()).half()

    cls = _kernel()
    cfg = _config(True)
    ok, reason = cls.can_implement(cfg)
    assert ok, reason
    kernel = cls(
        cfg,
        w_q_param_name="qweight",
        w_s_param_name="scales",
        w_zp_param_name="qzeros",
        w_gidx_param_name="g_idx",
    )
    garbage_zp = torch.full((K // GROUP, N // 8), 7, dtype=torch.int32, device=dev)
    layer = _dummy_layer(words, sc, pg, garbage_zp)
    kernel.process_weights_after_loading(layer)
    out = kernel.apply_weights(layer, x)

    err = (out.float() - ref.float()).abs()
    rms = (err.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item()
    assert rms < 5e-3, f"rms_rel={rms:.3e}"

    # The signature of a skipped gather: the kernel would have returned x @ W_act[perm] instead.
    skipped = (x.float() @ w_act.float()[perm].float()).half()
    assert not torch.allclose(out.float(), skipped.float(), atol=1e-2), (
        "act-order output is indistinguishable from a skipped gather, so the test proves nothing"
    )
