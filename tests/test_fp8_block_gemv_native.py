from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEMV = ROOT / "vllm_musa" / "kernels" / "fp8_block_gemv.py"
NATIVE = ROOT / "vllm_musa" / "jit_kernel" / "csrc" / "fp8_block_gemv.py"
KERNEL = ROOT / "vllm_musa" / "jit_kernel" / "csrc" / "gemv" / "fp8_block_gemv.mu"
LINEAR = (
    ROOT
    / "vllm_musa"
    / "model_executor"
    / "kernels"
    / "linear"
    / "scaled_mm"
    / "block_gemv.py"
)
FUSED_MOE = (
    ROOT / "vllm_musa" / "model_executor" / "layers" / "fused_moe" / "fused_moe.py"
)


def _function(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(source, node)


def _literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def test_native_tables_cover_eight_rows_with_known_configs() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    known = {int(c) for c in re.findall(r"case (\d+): launch_dense", kernel)}
    for name in ("_NATIVE_BF16_CFG", "_NATIVE_FP8_CFG"):
        table = _literal(GEMV, name)
        assert (1792, 5120) in table
        for cfgs in table.values():
            assert len(cfgs) == 8
            assert {c for c in cfgs if c is not None} <= known


def test_native_gemv_needs_32x32_fp32_scales_and_has_a_kill_switch() -> None:
    source = _function(GEMV, "_native_cfg")
    assert "_NATIVE_GEMV" in source
    assert "_weight_groups(weight, weight_scale) != (32, 32)" in source
    assert "torch.float32" in source
    assert 'os.environ.get("VLLM_MUSA_FP8_NATIVE_GEMV", "1")' in GEMV.read_text(
        encoding="utf-8"
    )


def test_quantizing_gemv_keeps_swiglu_on_the_triton_kernel() -> None:
    source = _function(GEMV, "fp8_block_gemv_quant")
    assert "swiglu_limit is None and group_size == 32" in source
    assert "native.act_quant(x)" in source


def test_native_moe_gate_matches_the_instantiated_down_shape() -> None:
    gate = _function(FUSED_MOE, "_native_fp8_block_moe_gemv_supported")
    assert "topk_ids.shape[1] in (3, 6)" in gate
    assert "w2.shape[2] == 288" in gate
    assert "_NATIVE_FP8_BLOCK_MOE_GEMV_MIN_TOKENS" in gate
    assert "gemm1_clamp_limit is not None" in gate
    assert "list(block_shape) == [32, 32]" in gate
    kernel = KERNEL.read_text(encoding="utf-8")
    assert "MOE_W2_CASE(6, 288)" in kernel and "MOE_W2_CASE(3, 288)" in kernel
    body = _function(FUSED_MOE, "_musa_fp8_block_moe_gemv")
    assert body.index("_native_fp8_block_moe_gemv_supported") < body.index(
        "\n    fp8_block_moe_gemv("
    )
    assert "expert_ids.to(torch.int32)" in body
    assert "to(torch.float32)" in body


def test_swiglu_down_projection_past_gemv_rows_quantizes_once() -> None:
    source = _function(LINEAR, "apply_swiglu_clamp_weights")
    assert "rows > _gemv_max_m(params.weight)" in source
    helper = _function(LINEAR, "_swiglu_clamp_block_mm")
    assert "native.act_quant(gate_up.view(rows, -1), swiglu_limit)" in helper
    assert "fp8_block_scaled_mm(" in helper


def test_activation_quant_divides_like_per_token_group_quant() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    assert "#pragma float_control(precise, on, push)" in kernel
    assert "div_rn(v[i * 4], s)" in kernel
    assert "__musa_cvt_float4_to_fp8x4(f4, __MT_SATFINITE, __MT_E4M3)" in kernel


def test_custom_ops_mutate_only_their_outputs() -> None:
    source = NATIVE.read_text(encoding="utf-8")
    for op, mutated in (
        ("musa_fp8_gemv_act_quant", '["xq", "xs"]'),
        ("musa_fp8_dense_gemv", '["out"]'),
        ("musa_fp8_moe_w1_gemv", '["out"]'),
        ("musa_fp8_moe_w2_gemv", '["out"]'),
    ):
        block = source[source.index(f'op_name="{op}"') :]
        assert f"mutates_args={mutated}" in block[: block.index(")")]


try:
    import torch
except ImportError:  # source checks above still run
    torch = None

requires_mp31 = pytest.mark.skipif(
    torch is None
    or not hasattr(torch, "musa")
    or not torch.musa.is_available()
    or tuple(torch.musa.get_device_capability()) != (3, 1),
    reason="requires an mp31 MUSA device",
)


def _block_fp8_weight(n: int, k: int, experts: int | None = None):
    shape = (n, k) if experts is None else (experts, n, k)
    w = torch.randn(*shape) * 0.02
    blocks = w.view(*shape[:-2], n // 32, 32, k // 32, 32)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(dim=(-3, -1)) / 448.0)))
    full = scale.repeat_interleave(32, -2).repeat_interleave(32, -1)
    weight = (w / full).clamp(-448, 448).to(torch.float8_e4m3fn)
    return weight.to("musa"), scale.float().contiguous().to("musa")


def _dequant(weight, scale):
    return weight.float() * scale.repeat_interleave(32, -2).repeat_interleave(32, -1)


@requires_mp31
@pytest.mark.parametrize("shape", [(1792, 5120), (1024, 4096), (576, 5120)])
@pytest.mark.parametrize("rows", [1, 3, 5, 8])
def test_native_dense_gemv_matches_triton(shape, rows) -> None:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    import vllm_musa.kernels.fp8_block_gemv as gemv

    torch.manual_seed(rows)
    n, k = shape
    weight, scale = _block_fp8_weight(n, k)
    x = torch.randn(rows, k).to(torch.bfloat16).to("musa")
    xq, xs = per_token_group_quant_fp8(x, 32, use_ue8m0=False)
    ref = (xq.float() * xs.repeat_interleave(32, -1)) @ _dequant(weight, scale).T
    for fn, args in (
        (gemv.fp8_block_gemv_quant, (x, weight, scale)),
        (gemv.fp8_block_gemv, (xq, xs, weight, scale)),
    ):
        outs = []
        for native in (False, True):
            gemv._NATIVE_GEMV = native
            out = torch.empty((rows, n), dtype=torch.bfloat16, device="musa")
            outs.append(
                fn(*args, out, 32)
                if fn is gemv.fp8_block_gemv_quant
                else fn(*args, out)
            )
        gemv._NATIVE_GEMV = True
        torch.testing.assert_close(
            outs[1].float(), outs[0].float(), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(outs[1].float(), ref, rtol=3e-2, atol=3e-2)


@requires_mp31
def test_native_activation_quant_is_bit_exact() -> None:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    from vllm_musa.jit_kernel.csrc import fp8_block_gemv as native

    torch.manual_seed(7)
    x = (torch.randn(6, 5120) * 4).to(torch.bfloat16).to("musa")
    xq, xs = native.act_quant(x)
    rq, rs = per_token_group_quant_fp8(x, 32, use_ue8m0=False)
    assert torch.equal(xq.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(xs, rs)

    gate_up = (torch.randn(5, 576) * 3).to(torch.bfloat16).to("musa")
    aq, a_scale = native.act_quant(gate_up, 10.0)
    gate = gate_up.float()[:, :288].clamp(max=10.0)
    up = gate_up.float()[:, 288:].clamp(-10.0, 10.0)
    silu = (gate / (1 + torch.exp(-gate))).to(torch.bfloat16).float()
    rq, rs = per_token_group_quant_fp8(
        (silu * up).to(torch.bfloat16), 32, use_ue8m0=False
    )
    assert torch.equal(aq.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(a_scale, rs)


@requires_mp31
@pytest.mark.parametrize("tokens", [1, 2, 5])
def test_native_moe_gemv_matches_triton(tokens) -> None:
    from vllm_musa.kernels import fp8_moe_mma
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    torch.manual_seed(11 + tokens)
    experts, top_k, hidden, inter = 64, 6, 5120, 288
    w1, w1s = _block_fp8_weight(2 * inter, hidden, experts)
    w2, w2s = _block_fp8_weight(hidden, inter, experts)
    x = torch.randn(tokens, hidden).to(torch.bfloat16).to("musa")
    ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)])
    ids = ids.to(torch.int32).to("musa")
    weights = torch.softmax(torch.randn(tokens, top_k), -1).to("musa")
    outs = []
    fp8_moe_mma._ENABLED = False
    try:
        for native in (False, True):
            fused_moe._NATIVE_FP8_BLOCK_MOE_GEMV = native
            outs.append(
                fused_moe._musa_fp8_block_moe_gemv(
                    x, w1, w2, weights, ids, w1s, w2s, [32, 32], 10.0
                )
            )
    finally:
        fused_moe._NATIVE_FP8_BLOCK_MOE_GEMV = True
        fp8_moe_mma._ENABLED = True
    torch.testing.assert_close(outs[1].float(), outs[0].float(), rtol=1e-2, atol=1e-2)


SPLITK = ROOT / "vllm_musa" / "kernels" / "fp8_block_splitk.py"
SPLITK_TABLE = ROOT / "vllm_musa" / "tactics" / "fp8_block_splitk.json"
BF16 = ROOT / "vllm_musa" / "kernels" / "fp8_block_bf16.py"
MOE_MMA = ROOT / "vllm_musa" / "kernels" / "fp8_moe_mma.py"


def _splitk_rows() -> list[dict]:
    """The split-K table as rows: shape, capture mode, band and the values."""
    raw = json.loads(SPLITK_TABLE.read_text(encoding="utf-8"))
    rows = []
    for item in raw["tactics"]:
        match = item["match"]
        low, high = match["m_range"]
        rows.append(
            {
                "shape": tuple(match["shape"]),
                "capturing": match["graph_capture"],
                "low": low,
                "high": high,
                **item["values"],
            }
        )
    return rows


def test_splitk_configs_tile_k_and_keep_one_row_on_the_gemv() -> None:
    rows = _splitk_rows()
    for row in rows:
        n, k = row["shape"]
        assert k % (row["split"] * 32 * row["unroll"]) == 0
        assert 2 <= row["low"] <= 16 <= row["high"]
        assert row["tile_m"] in (16, 32)
    # Every projection is measured in both capture modes: an eager launch admits
    # more rows than a captured one, so one row cannot serve both.
    by_shape: dict = {}
    for row in rows:
        by_shape.setdefault(row["shape"], set()).add(row["capturing"])
    assert all(modes == {False, True} for modes in by_shape.values())
    # The dispatch reads the table instead of two dicts, and the capture question
    # is asked once, where the problem is built.
    source = _function(SPLITK, "fp8_block_splitk_config")
    assert "config_for(" in source
    assert "torch.cuda.is_current_stream_capturing()" in source


def test_splitk_row_tiles_add_no_padding() -> None:
    namespace = {
        "BLOCK_M": _literal(SPLITK, "BLOCK_M"),
        "triton": SimpleNamespace(cdiv=lambda a, b: -(-a // b)),
    }
    exec(_function(SPLITK, "_tile_rows"), namespace)
    tile_rows = namespace["_tile_rows"]
    rows = (1, 16, 17, 32, 40, 48, 64, 80, 96, 160)
    assert [tile_rows(r, 32) for r in rows] == [16, 16, 32, 32, 16, 16, 32, 16, 32, 32]
    assert {tile_rows(r, 16) for r in range(1, 321)} == {16}


def test_bf16_gemm_rows_start_past_the_splitk_rows() -> None:
    rows = _splitk_rows()
    min_rows = _literal(BF16, "MIN_ROWS")
    by_shape = _literal(BF16, "_MIN_ROWS_BY_SHAPE")
    captured_max = {
        row["shape"]: row["high"] for row in rows if row["capturing"]
    }
    eager_max = {row["shape"]: row["high"] for row in rows if not row["capturing"]}
    assert set(by_shape) <= set(captured_max)
    # Inside a capture the two paths meet: bf16 starts one row past split-K.
    for shape, high in captured_max.items():
        assert by_shape.get(shape, min_rows) > high
    # Eagerly, the split-K rows end well below where bf16 takes over.
    assert min_rows > max(eager_max.values())
    assert all(rows <= min_rows for rows in by_shape.values())
    source = _function(BF16, "fp8_block_bf16_supported")
    assert "_MIN_ROWS_BY_SHAPE.get((n, k), MIN_ROWS)" in source
    for path, name in (
        (BF16, "fp8_block_bf16_supported"),
        (SPLITK, "fp8_block_splitk_config"),
    ):
        assert "torch.cuda.is_current_stream_capturing()" in _function(path, name)
    body = _function(LINEAR, "apply_weights")
    assert (
        body.index("_small_m_path(")
        < body.index("fp8_block_bf16_supported(")
        < body.index("super().apply_weights(")
    )


@pytest.mark.parametrize(
    "shape",
    sorted({row["shape"] for row in _splitk_rows()}),
    ids=lambda s: f"{s[0]}x{s[1]}",
)
@pytest.mark.parametrize("capturing", [False, True])
def test_measured_rows_apply_only_in_graph_capture(
    monkeypatch, capturing, shape
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm_musa.kernels import fp8_block_bf16 as bf
    from vllm_musa.kernels import fp8_block_splitk as sk

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    weight = torch.empty(shape, dtype=torch.float8_e4m3fn)
    scale = torch.empty((shape[0] // 32, shape[1] // 32), dtype=torch.float32)
    band = {row["capturing"]: row for row in _splitk_rows() if row["shape"] == shape}
    splitk_max = band[capturing]["high"]
    bf16_min = bf._MIN_ROWS_BY_SHAPE[shape] if capturing else bf.MIN_ROWS
    assert sk.fp8_block_splitk_config(weight, scale, splitk_max) is not None
    assert sk.fp8_block_splitk_config(weight, scale, splitk_max + 1) is None
    assert not bf.fp8_block_bf16_supported(weight, scale, bf16_min - 1)
    assert bf.fp8_block_bf16_supported(weight, scale, bf16_min)


def test_splitk_reduction_sums_partials_in_a_fixed_order() -> None:
    source = SPLITK.read_text(encoding="utf-8")
    reduce = source[source.index("def _fp8_block_splitk_reduce_kernel") :]
    assert "for s in tl.static_range(SPLIT)" in reduce
    assert "atomic" not in source


def test_dense_entry_points_try_splitk_first() -> None:
    for name in ("fp8_block_gemv", "fp8_block_gemv_quant"):
        body = _function(GEMV, name)
        assert body.index("fp8_block_splitk_config") < body.index("_native_cfg")
    assert "_small_m_path(params.weight, params.block_scale, rows)" in LINEAR.read_text(
        encoding="utf-8"
    )


def test_moe_dot_path_reads_each_expert_once_and_bounds_rows() -> None:
    source = MOE_MMA.read_text(encoding="utf-8")
    assert "_moe_route_map_kernel" in source
    supported = _function(MOE_MMA, "fp8_moe_mma_supported")
    assert "tokens * top_k <= _MAX_ROUTED" in supported
    assert "tokens <= 16" in supported
    body = _function(FUSED_MOE, "_musa_fp8_block_moe_gemv")
    assert body.index("_fp8_moe_mma_eligible") < body.index("native.act_quant")
    gate = _function(FUSED_MOE, "_fp8_moe_mma_eligible")
    assert "fp8_moe_mma_supported" in gate
    assert "gemm1_clamp_limit is None" in gate
    assert "MIN_TOKENS" in gate
    assert "_fp8_moe_mma_eligible" in _function(
        FUSED_MOE, "_can_use_musa_fp8_block_moe_gemv"
    )


@requires_mp31
@pytest.mark.parametrize("tokens", [4, 5])
def test_moe_dot_path_matches_triton(tokens) -> None:
    from vllm_musa.kernels import fp8_moe_mma
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    torch.manual_seed(23 + tokens)
    experts, top_k, hidden, inter = 64, 6, 5120, 288
    w1, w1s = _block_fp8_weight(2 * inter, hidden, experts)
    w2, w2s = _block_fp8_weight(hidden, inter, experts)
    x = torch.randn(tokens, hidden).to(torch.bfloat16).to("musa")
    ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)])
    ids[1:, :3] = ids[:-1, :3]
    ids = ids.to(torch.int32).to("musa")
    weights = torch.softmax(torch.randn(tokens, top_k), -1).to("musa")
    assert fp8_moe_mma.fp8_moe_mma_supported(x, w1, w2, w1s, w2s, ids)
    fused_moe._NATIVE_FP8_BLOCK_MOE_GEMV = False
    fp8_moe_mma._ENABLED = False
    try:
        ref = fused_moe._musa_fp8_block_moe_gemv(
            x, w1, w2, weights, ids, w1s, w2s, [32, 32], 10.0
        )
    finally:
        fused_moe._NATIVE_FP8_BLOCK_MOE_GEMV = True
        fp8_moe_mma._ENABLED = True
    got = fp8_moe_mma.fp8_moe_mma(x, w1, w2, weights, ids, w1s, w2s, 10.0)
    torch.testing.assert_close(got.float(), ref.float(), rtol=3e-2, atol=3e-2)


@requires_mp31
@pytest.mark.parametrize("shape", [(1792, 5120), (4096, 1280), (1024, 4096)])
def test_splitk_row_tiles_keep_each_row_result(shape) -> None:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    from vllm_musa.kernels import fp8_block_splitk as sk

    torch.manual_seed(shape[0])
    n, k = shape
    weight, scale = _block_fp8_weight(n, k)
    max_rows = sk._EAGER_MAX_ROWS[shape]
    x = torch.randn(max_rows, k).to(torch.bfloat16).to("musa")
    xq, xs = per_token_group_quant_fp8(x, 32, use_ue8m0=False)
    ref = (xq.float() * xs.repeat_interleave(32, -1)) @ _dequant(weight, scale).T
    for rows in [r for r in (17, 40, 64) if r < max_rows] + [max_rows]:
        config = sk.fp8_block_splitk_config(weight, scale, rows)
        assert config is not None
        out = torch.empty((rows, n), dtype=torch.bfloat16, device="musa")
        sk.fp8_block_splitk_mm(x[:rows], None, weight, scale, out, config)
        torch.testing.assert_close(out.float(), ref[:rows], rtol=3e-2, atol=3e-2)
        for row in (0, rows // 2, rows - 1):
            single = torch.empty((1, n), dtype=torch.bfloat16, device="musa")
            sk.fp8_block_splitk_mm(
                x[row : row + 1], None, weight, scale, single, config
            )
            assert torch.equal(single[0], out[row])
    assert sk.fp8_block_splitk_config(weight, scale, max_rows + 1) is None


@requires_mp31
@pytest.mark.parametrize("shape", [(1792, 5120), (1024, 4096)])
def test_bf16_gemm_dequantizes_power_of_two_scales_exactly(shape) -> None:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    from vllm_musa.kernels import fp8_block_bf16 as bf

    torch.manual_seed(shape[1])
    n, k = shape
    weight, scale = _block_fp8_weight(n, k)
    dense = _dequant(weight, scale)
    assert torch.equal(bf._dequant(weight, scale, 32, 32).float(), dense)
    min_rows = bf.MIN_ROWS
    rows = min_rows + 88
    assert not bf.fp8_block_bf16_supported(weight, scale, min_rows - 1)
    assert bf.fp8_block_bf16_supported(weight, scale, rows)
    x = torch.randn(rows, k).to(torch.bfloat16).to("musa")
    out = bf.fp8_block_bf16_linear(x, weight, scale)
    torch.testing.assert_close(out.float(), x.float() @ dense.T, rtol=2e-2, atol=2e-2)
    xq, xs = per_token_group_quant_fp8(x, 128, use_ue8m0=False)
    out = bf.fp8_block_bf16_linear(xq, weight, scale, xs)
    ref = (xq.float() * xs.repeat_interleave(128, -1)) @ dense.T
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
