# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BENCHMARK_DIR = ROOT / "benchmarks/kernel_tactics"
sys.path.insert(0, str(BENCHMARK_DIR))

import run_mp_tactic_campaign as runner  # noqa: E402
import summarize_mp_tactic_campaign as summary  # noqa: E402
from _benchmark_utils import effective_gemv_block, source_identity  # noqa: E402
from benchmark_dsv4_mhc_jit import production_split_for_path  # noqa: E402


def test_campaign_manifest_has_executable_reviewed_cells() -> None:
    campaign = runner.load_campaign(BENCHMARK_DIR / "mp_tactic_campaign.json")
    assert campaign["schema"] == "vllm-musa-mp-tactic-campaign.v2"
    assert "hardware" not in campaign
    assert campaign["methodology"]["inner_iters"] == 1
    assert campaign["methodology"]["cache_policy"] == "cold-l2-per-sample"
    assert campaign["methodology"]["isolated_device_required_for_promotion"] is True

    enabled = [cell for cell in campaign["cells"] if cell["enabled"]]
    assert {cell["id"] for cell in enabled} >= {
        "qwen35-folded-bf16-moe-blocks",
        "dsv4-fp8-moe-blocks",
        "dsv4-fused-add-rmsnorm-aot",
        "dsv4-mhc-jit-fuse",
        "dsv4-mhc-jit-standalone",
    }
    for cell in enabled:
        assert (ROOT / cell["script"]).is_file()
        assert set(cell["modes"]) == {"quick", "full"}
        assert "prediction" not in cell


def test_disabled_cells_cannot_be_selected() -> None:
    campaign = runner.load_campaign(BENCHMARK_DIR / "mp_tactic_campaign.json")
    with pytest.raises(ValueError, match="requested disabled"):
        runner.selected_cells(campaign, ["dsv4-dense-fp8-gemv"])


def test_visible_device_contract_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert runner.verify_visible_device_contract() == {
        "MUSA_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="must both be set"):
        runner.verify_visible_device_contract()


def test_campaign_appends_lease_device_fence() -> None:
    command = runner.add_device_fence_args(
        ["bench.py", "--output", "x.json"], 0, "uuid"
    )
    assert command[-4:] == [
        "--expected-physical-device",
        "0",
        "--expected-device-uuid",
        "uuid",
    ]
    with pytest.raises(ValueError, match="require --expected-physical-device"):
        runner.add_device_fence_args(["bench.py"], None, None)


def test_amdahl_prediction_uses_time_share_and_hit_rate() -> None:
    predicted = summary.amdahl_prediction(0.8, 0.25, 0.5)
    assert predicted == pytest.approx((1 / 0.975 - 1) * 100)
    assert summary.amdahl_prediction(0.8, None, 0.5) is None


def test_mp_only_key_drops_route_for_moe() -> None:
    schema = "musa-fused-gemv-moe-aot-paired-ab.v2"
    assert summary.mp_only_key(schema, ("qwen", 4, "hot", "w1")) == (
        "qwen",
        4,
        "w1",
    )


def test_dsv4_one_token_special_arm_is_not_mislabeled() -> None:
    assert effective_gemv_block("dsv4_fp8", 1, "w1", (32, 4)) == (
        (4, 32),
        False,
        "dsv4-one-token-split-tile",
    )
    assert effective_gemv_block("dsv4_fp8", 1, "w2", (32, 4))[1] is True


def test_mhc_schema_has_pairable_key_and_tactic() -> None:
    schema = "musa-dsv4-mhc-jit-aot-paired-ab.v1"
    row = {
        "production_path": "fused_post_prenorm",
        "tokens": 1,
        "split_k": 4,
        "threads": 128,
        "hidden_block": 512,
        "pass_config": "burst",
    }
    assert summary.row_key(schema, row) == ("fused_post_prenorm", 1, 4)
    assert summary.tactic(schema, row) == "128x512xburst"
    assert summary.key_dict(schema, ("fused_post_prenorm", 1, 4)) == {
        "production_path": "fused_post_prenorm",
        "tokens": 1,
        "split_k": 4,
    }
    row.update(
        {
            "median_ratio": 0.9,
            "ratio_p95": 0.95,
            "correctness_pass": True,
            "poison_output": False,
            "is_production_split": True,
        }
    )
    assert summary.promotion_pass(
        row, {"median_ratio_max": 0.98, "ratio_p95_max": 1.02}
    )
    row["is_production_split"] = False
    assert not summary.promotion_pass(
        row, {"median_ratio_max": 0.98, "ratio_p95_max": 1.02}
    )


@pytest.mark.parametrize(
    ("tokens", "expected"), [(1, 8), (2, 8), (4, 8), (8, 4), (16, 4)]
)
def test_fused_mhc_production_split(tokens: int, expected: int) -> None:
    assert (
        production_split_for_path("fused_post_prenorm", tokens, 4096, lambda *_: -1)
        == expected
    )


def test_standalone_mhc_production_split_uses_resolver() -> None:
    calls: list[tuple[int, int]] = []

    def resolve(tokens: int, hc_hidden_size: int) -> int:
        calls.append((tokens, hc_hidden_size))
        return 64

    assert production_split_for_path("standalone", 1, 4096, resolve) == 64
    assert calls == [(1, 16384)]


def test_summarizer_accepts_mhc_result_bundle(tmp_path: Path) -> None:
    campaign_path = BENCHMARK_DIR / "mp_tactic_campaign.json"
    campaign_bytes = campaign_path.read_bytes()
    (tmp_path / "campaign.json").write_bytes(campaign_bytes)
    result = {
        "schema": "musa-dsv4-mhc-jit-aot-paired-ab.v1",
        "multiprocessor_count": 56,
        "provenance": {"hostname": "mhc-host"},
        "rows": [
            {
                "production_path": "standalone",
                "tokens": 1,
                "split_k": 1,
                "threads": 128,
                "hidden_block": 512,
                "pass_config": "safe",
                "median_ms": 1.0,
                "p90_ms": 1.0,
                "p99_ms": 1.0,
                "baseline_median_ms": 1.0,
                "median_ratio": 1.0,
                "ratio_p95": 1.0,
                "ratio_p99": 1.0,
                "speedup_pct": 0.0,
                "correctness_pass": True,
                "poison_output": False,
                "is_production_config": True,
            },
            {
                "production_path": "standalone",
                "tokens": 1,
                "split_k": 1,
                "threads": 256,
                "hidden_block": 512,
                "pass_config": "safe",
                "median_ms": 0.8,
                "p90_ms": 0.9,
                "p99_ms": 0.9,
                "baseline_median_ms": 1.0,
                "median_ratio": 0.8,
                "ratio_p95": 0.9,
                "ratio_p99": 0.9,
                "speedup_pct": 25.0,
                "correctness_pass": True,
                "poison_output": False,
                "is_production_config": False,
            },
        ],
    }
    result_path = tmp_path / "mhc.json"
    result_path.write_text(json.dumps(result))
    (tmp_path / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema": "vllm-musa-mp-tactic-run.v1",
                "campaign_sha256": hashlib.sha256(campaign_bytes).hexdigest(),
                "expected_mp": 56,
                "lease_isolated": True,
                "exploratory": False,
                "runs": [
                    {
                        "run_id": "mhc",
                        "cell": "dsv4-mhc-jit-fuse",
                        "variant": "seed20260827",
                        "result": str(result_path),
                        "returncode": 0,
                        "error": None,
                    }
                ],
            }
        )
    )
    output = tmp_path / "summary.json"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_mp_tactic_campaign.py", str(tmp_path), "--output", str(output)],
    )
    try:
        assert summary.main() == 0
    finally:
        monkeypatch.undo()
    payload = json.loads(output.read_text())
    assert payload["observation_count"] == 1
    assert payload["mp_only_candidates"][0]["winner"] == "256x512xsafe"


def test_gpu_benches_reject_warm_inner_loop_evidence() -> None:
    for filename in (
        "benchmark_fused_gemv_moe_blocks.py",
        "benchmark_fused_add_rmsnorm_paired_ab.py",
    ):
        source = (BENCHMARK_DIR / filename).read_text()
        assert "cold-L2 evidence requires --inner-iters 1" in source
        assert "flush_buffer.zero_()" in source
        assert '"cache_policy": "cold-l2-per-sample"' in source
        assert '"--expected-physical-device"' in source
        assert '"--expected-device-uuid"' in source
        assert "verify_lease_device_fence" in source
        assert '"lease_device_fence"' in source
        assert (
            "variance is not a correctness oracle" in source
            or "constant normalized row" in source
        )
        if "gemv" in filename:
            assert "requested_block_applied" in source
            assert '"--baseline-block"' in source


def test_campaign_json_is_canonical_object() -> None:
    payload = json.loads((BENCHMARK_DIR / "mp_tactic_campaign.json").read_text())
    assert payload["schema"] == "vllm-musa-mp-tactic-campaign.v2"
    assert "hardware" not in payload
    assert "historical_seeds" not in payload
    assert "generated/MUSA-" not in json.dumps(payload)


def test_source_identity_accepts_archive_revision_marker(tmp_path: Path) -> None:
    script = tmp_path / "benchmarks/kernel_tactics/bench.py"
    script.parent.mkdir(parents=True)
    script.write_text("# synthetic\n")
    (tmp_path / ".source-revision").write_text("deadbeef\n")
    identity = source_identity(script)
    assert identity["head"] == "deadbeef"
    assert identity["archive_marker"] == "deadbeef"
    assert identity["dirty"] is False


def test_source_identity_accepts_shallow_script_path(tmp_path: Path) -> None:
    script = tmp_path / "bench.py"
    script.write_text("# synthetic\n")
    (tmp_path / ".source-revision").write_text("abc123\n")
    assert source_identity(script)["head"] == "abc123"


def test_summarizer_requires_same_winner_across_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path = BENCHMARK_DIR / "mp_tactic_campaign.json"
    campaign_bytes = campaign_path.read_bytes()
    (tmp_path / "campaign.json").write_bytes(campaign_bytes)
    result_path = tmp_path / "result.json"
    rows = []
    for route in ("balanced", "hot"):
        for block, ratio in (((16, 8), 0.90), ((32, 4), 0.95)):
            rows.append(
                {
                    "family": "qwen35_folded_bf16",
                    "tokens": 4,
                    "route": route,
                    "stage": "w1",
                    "candidate_block": list(block),
                    "median_ratio": ratio,
                    "ratio_p95": ratio + 0.01,
                    "speedup_pct": (1 / ratio - 1) * 100,
                    "correctness_pass": True,
                    "poison_output": False,
                }
            )
    result_path.write_text(
        json.dumps(
            {
                "schema": "musa-fused-gemv-moe-aot-paired-ab.v2",
                "multiprocessor_count": 56,
                "provenance": {"hostname": "mp56-host"},
                "rows": rows,
            }
        )
    )
    (tmp_path / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema": "vllm-musa-mp-tactic-run.v1",
                "campaign_sha256": hashlib.sha256(campaign_bytes).hexdigest(),
                "expected_mp": 56,
                "lease_isolated": True,
                "exploratory": False,
                "runs": [
                    {
                        "run_id": "synthetic",
                        "cell": "qwen35-folded-bf16-moe-blocks",
                        "variant": "seed7",
                        "result": str(result_path),
                        "returncode": 0,
                        "error": None,
                    }
                ],
            }
        )
    )
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_mp_tactic_campaign.py", str(tmp_path), "--output", str(output)],
    )
    assert summary.main() == 0
    payload = json.loads(output.read_text())
    assert payload["mp_only_candidates"][0]["winner"] == "16x8"
    assert payload["mp_only_candidates"][0]["promotion_ready"] is True

    manifest_path = tmp_path / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["lease_isolated"] = False
    manifest["exploratory"] = True
    manifest_path.write_text(json.dumps(manifest))
    exploratory_output = tmp_path / "exploratory-summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_mp_tactic_campaign.py",
            str(tmp_path),
            "--output",
            str(exploratory_output),
        ],
    )
    assert summary.main() == 0
    exploratory = json.loads(exploratory_output.read_text())
    assert exploratory["route_candidates"][0]["evidence_eligible"] is False
    assert exploratory["mp_only_candidates"][0]["promotion_ready"] is False


def test_fused_moe_gemv_benchmark_requires_lease_device_fence() -> None:
    source = (BENCHMARK_DIR / "benchmark_fused_gemv_moe_blocks.py").read_text(
        encoding="utf-8"
    )
    assert '"--expected-physical-device", type=int, required=True' in source
    assert '"--expected-device-uuid", required=True' in source
    assert '"lease_device_fence": lease_device_fence' in source
    assert 'choices=("eager", "graph"), default="eager"' in source
    assert "graph = torch.cuda.CUDAGraph()" in source
    assert '"execution_mode": args.execution_mode' in source
    assert '"qwen_bf16": Family("qwen_bf16", 32,' in source
    assert '"qwen35_folded_bf16", 33,' in source


def test_mhc_pre_benchmark_requires_lease_device_fence() -> None:
    source = (BENCHMARK_DIR / "benchmark_dsv4_mhc_jit.py").read_text(encoding="utf-8")
    assert '"--expected-physical-device", type=int, required=True' in source
    assert '"--expected-device-uuid", required=True' in source
    assert '"lease_device_fence": lease_device_fence' in source
