# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MOE_MMA = ROOT / "vllm_musa" / "kernels" / "fp8_moe_mma.py"


def _reference_chunked_route_map(
    ids: list[int], num_experts: int, block_m: int = 16
) -> list[tuple[int, list[int]]]:
    """Reference first-use chunked 16-row expert work-units."""
    seen_per_expert: dict[int, int] = {}
    unit_for_chunk: dict[tuple[int, int], int] = {}
    units: list[tuple[int, list[int]]] = []
    for r, e in enumerate(ids):
        if e < 0 or e >= num_experts:
            continue
        rank = seen_per_expert.get(e, 0)
        seen_per_expert[e] = rank + 1
        key = (e, rank // block_m)
        if rank % block_m == 0:
            unit_for_chunk[key] = len(units)
            units.append((e, [r]))
        else:
            units[unit_for_chunk[key]][1].append(r)
    return units


def test_max_work_units_pigeonhole_bound_covers_all_distributions() -> None:
    tree = ast.parse(MOE_MMA.read_text(encoding="utf-8"))
    fn_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_max_work_units"
    )
    ns: dict[str, object] = {}
    exec(ast.get_source_segment(MOE_MMA.read_text(encoding="utf-8"), fn_node), ns)
    max_work_units = ns["_max_work_units"]

    for num_experts in (4, 16, 64, 129, 385):
        for rows in (1, 15, 16, 17, 64, 70, 130, 224, 256, 384):
            bound = max_work_units(rows, num_experts)
            # Adversarial 1: spread across min(rows, E) experts
            spread = [(r % num_experts) for r in range(rows)]
            # Adversarial 2: all rows on one shared expert
            hot = [num_experts - 1] * rows
            # Adversarial 3: min(E, rows) experts get 1 row, rest on expert 0
            k = min(rows, num_experts)
            skewed = list(range(k)) + [0] * (rows - k)
            # Adversarial 4: padding rows mixed in
            padded = [(-1 if r % 5 == 0 else (r % num_experts)) for r in range(rows)]
            for dist in (spread, hot, skewed, padded):
                units = _reference_chunked_route_map(dist, num_experts)
                assert len(units) <= bound
                covered = sorted(r for _, u_rows in units for r in u_rows)
                expected_valid = [r for r, e in enumerate(dist) if 0 <= e < num_experts]
                assert covered == expected_valid
                assert all(1 <= len(u_rows) <= 16 for _, u_rows in units)


try:
    import torch
except ImportError:
    torch = None

requires_mp31 = pytest.mark.skipif(
    torch is None
    or not hasattr(torch, "musa")
    or not torch.musa.is_available()
    or tuple(torch.musa.get_device_capability()) != (3, 1),
    reason="requires an mp31 MUSA device",
)


@requires_mp31
@pytest.mark.parametrize(
    ("tokens", "top_k", "num_experts"),
    [
        (4, 7, 385),
        (10, 7, 385),
        (20, 7, 385),
        (32, 7, 385),
        (24, 4, 129),
        (48, 4, 129),
    ],
)
def test_native_and_triton_chunked_route_maps_are_bit_identical(
    tokens: int,
    top_k: int,
    num_experts: int,
) -> None:
    from vllm_musa.jit_kernel.csrc import fp8_block_gemv as native
    from vllm_musa.kernels import fp8_moe_mma

    torch.manual_seed(100 + tokens * 13 + top_k)
    routed_e = num_experts - 1
    routed = torch.stack([torch.randperm(routed_e)[: top_k - 1] for _ in range(tokens)])
    # Force hot collisions (>16 rows on expert 0 AND all tokens on folded shared expert)
    routed[:, 0] = 0
    shared = torch.full((tokens, 1), routed_e, dtype=torch.long)
    ids_2d = torch.cat([routed, shared], dim=1).to(torch.int32)
    # Mark one padding token to verify invalid-id filtering
    ids_2d[-1, :] = -1
    ids = ids_2d.reshape(-1).contiguous().to("musa")
    rows = ids.numel()
    map_slots = fp8_moe_mma._map_slots(rows)

    x = (torch.randn(tokens, 5120, device="musa") * 2.0).to(torch.bfloat16)
    cpp_map, x_quant = native.moe_route_prep(ids, x, num_experts, map_slots)
    assert x_quant is not None
    ref_xq, ref_xs = native.act_quant(x)
    assert torch.equal(x_quant[0].view(torch.uint8), ref_xq.view(torch.uint8))
    assert torch.equal(x_quant[1], ref_xs)

    ref_units = _reference_chunked_route_map(ids.cpu().tolist(), num_experts)
    n_units = int(cpp_map[map_slots * 18].item())
    assert n_units == len(ref_units)
    assert n_units <= fp8_moe_mma._max_work_units(rows, num_experts)
    for u, (exp_e, exp_rows) in enumerate(ref_units):
        slot = cpp_map[u * 18 : u * 18 + 2 + len(exp_rows)].cpu().tolist()
        assert slot == [exp_e, len(exp_rows), *exp_rows]

    if map_slots <= 128:
        tri_map = torch.empty_like(cpp_map)
        fp8_moe_mma._moe_route_map_kernel[(1,)](
            ids,
            tri_map,
            rows,
            num_experts,
            MAX_R=map_slots,
            BLOCK_M=16,
            num_warps=max(4, map_slots // 32),
        )
        assert int(tri_map[map_slots * 18].item()) == n_units
        for u, (_, exp_rows) in enumerate(ref_units):
            n_col = 2 + len(exp_rows)
            assert torch.equal(
                tri_map[u * 18 : u * 18 + n_col],
                cpp_map[u * 18 : u * 18 + n_col],
            )


@requires_mp31
@pytest.mark.parametrize(
    ("tokens", "top_k", "experts"),
    [
        (10, 7, 65),
        (20, 7, 65),
        (32, 7, 65),
        (24, 4, 65),
    ],
)
def test_extended_fp8_moe_mma_matches_reference_and_replays_cudagraph(
    tokens: int,
    top_k: int,
    experts: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_fp8_block_gemv_native import _block_fp8_weight
    from vllm_musa.kernels import fp8_moe_mma
    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    monkeypatch.setattr(fp8_moe_mma, "_MAX_TOKENS", max(fp8_moe_mma._MAX_TOKENS, 48))
    monkeypatch.setattr(fp8_moe_mma, "_MAX_ROUTED", max(fp8_moe_mma._MAX_ROUTED, 256))

    torch.manual_seed(300 + tokens * 7 + top_k)
    hidden, inter = 5120, 288
    w1, w1s = _block_fp8_weight(2 * inter, hidden, experts)
    w2, w2s = _block_fp8_weight(hidden, inter, experts)
    x = torch.randn(tokens, hidden, device="musa", dtype=torch.bfloat16)
    routed = torch.stack(
        [torch.randperm(experts - 1)[: top_k - 1] for _ in range(tokens)]
    )
    routed[:, 0] = 0  # >16 rows on expert 0 when tokens >= 17
    ids = (
        torch.cat([routed, torch.full((tokens, 1), experts - 1)], dim=1)
        .to(torch.int32)
        .to("musa")
    )
    weights = torch.cat(
        [
            torch.softmax(torch.randn(tokens, top_k - 1), dim=-1) * 1.5,
            torch.ones(tokens, 1),
        ],
        dim=1,
    ).to("musa")

    assert fp8_moe_mma.fp8_moe_mma_supported(x, w1, w2, w1s, w2s, ids)
    got_native = fp8_moe_mma.fp8_moe_mma(x, w1, w2, weights, ids, w1s, w2s, 10.0)

    ref = torch.zeros(tokens, hidden, device="musa", dtype=torch.float32)
    for t in range(tokens):
        ref[t] = fused_moe._musa_fp8_block_moe_gemv(
            x[t : t + 1],
            w1,
            w2,
            weights[t : t + 1].contiguous(),
            ids[t : t + 1].contiguous(),
            w1s,
            w2s,
            [32, 32],
            10.0,
        )[0].float()
    cos = math.isfinite(float(got_native.float().sum().item())) and (
        torch.nn.functional.cosine_similarity(
            got_native.float().flatten(), ref.flatten(), dim=0
        ).item()
    )
    assert cos > 0.9995
    torch.testing.assert_close(got_native.float(), ref, rtol=3e-2, atol=3e-2)

    if tokens * top_k <= 128:
        monkeypatch.setattr(fp8_moe_mma, "_NATIVE_ROUTE_PREP", False)
        got_triton = fp8_moe_mma.fp8_moe_mma(x, w1, w2, weights, ids, w1s, w2s, 10.0)
        monkeypatch.setattr(fp8_moe_mma, "_NATIVE_ROUTE_PREP", True)
        torch.testing.assert_close(
            got_native.float(), got_triton.float(), rtol=2e-2, atol=2e-2
        )

    # CUDAGraph 8-replay bitwise parity across changing routing & activations
    static_x = x.clone()
    static_ids = ids.clone()
    static_w = weights.clone()
    _ = fp8_moe_mma.fp8_moe_mma(static_x, w1, w2, static_w, static_ids, w1s, w2s, 10.0)
    torch.musa.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g):
        graph_out = fp8_moe_mma.fp8_moe_mma(
            static_x, w1, w2, static_w, static_ids, w1s, w2s, 10.0
        )
    torch.cuda.current_stream().wait_stream(s)
    torch.musa.synchronize()
    for replay_idx in range(8):
        torch.manual_seed(900 + replay_idx)
        static_x.copy_(torch.randn_like(static_x))
        new_routed = torch.stack(
            [torch.randperm(experts - 1)[: top_k - 1] for _ in range(tokens)]
        )
        if replay_idx % 2 == 0:
            new_routed[:, :2] = replay_idx % (experts - 1)
        new_ids = torch.cat(
            [new_routed, torch.full((tokens, 1), experts - 1)], dim=1
        ).to(torch.int32)
        if replay_idx == 7:
            new_ids[-2:, :] = -1
        static_ids.copy_(new_ids.to("musa"))
        eager_ref = fp8_moe_mma.fp8_moe_mma(
            static_x, w1, w2, static_w, static_ids, w1s, w2s, 10.0
        )
        g.replay()
        torch.musa.synchronize()
        assert torch.equal(graph_out, eager_ref), f"Replay {replay_idx} mismatch"
