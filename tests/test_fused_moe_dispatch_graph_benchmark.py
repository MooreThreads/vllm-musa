from pathlib import Path

BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "fused_moe"
    / "benchmark_dispatch_graph.py"
)


def _source() -> str:
    return BENCHMARK.read_text(encoding="utf-8")


def test_graph_harness_supports_bf16_without_fp8_scales() -> None:
    source = _source()
    assert '"--weight-dtype"' in source
    assert 'choices=("fp8", "bf16")' in source
    assert "_make_bf16_weights(" in source
    assert 'kwargs["global_num_experts"] = args.experts' in source
    assert 'kwargs["apply_router_weight_on_input"] = False' in source
    assert "w1_scale = None" in source
    assert "w2_scale = None" in source
    assert "block_size = None" in source
    assert "args.graph_bucket_num_reqs not in (1, 2, 4, 8)" in source
    assert "args.graph_bucket_num_reqs != args.tokens" in source


def test_graph_harness_fails_closed_on_lease_device_mismatch() -> None:
    source = _source()
    assert '"--expected-physical-device", type=int, required=True' in source
    assert '"--expected-device-uuid", required=True' in source
    assert '"--expected-multiprocessor-count", type=int, required=True' in source
    assert '"--graph-bucket-num-reqs"' in source
    assert 'runtime_mode="FULL"' in source
    assert "lease_device_fence = verify_lease_device_fence(" in source
    assert '"lease_device_fence": lease_device_fence' in source
    assert "primed_hardware = prime_musa_kernel_hardware(0)" in source
    assert '"primed_hardware": {' in source
    assert '"source": source_identity(Path(__file__))' in source


def test_graph_harness_checks_production_backend_and_replays_routes() -> None:
    source = _source()
    assert 'choices=("gemv", "upstream")' in source
    assert 'route_modes = ("balanced", "unique_random", "hot")' in source
    assert (
        "production_dispatch = fused_moe._upstream_fused_moe.fused_experts_impl"
        in source
    )
    assert "captured_output = production_dispatch(**kwargs)" in source
    assert (
        "fused_moe._upstream_fused_moe.fused_experts_impl = counted_upstream"
        not in source
    )
    assert "capture_calls == expected_capture_calls" in source
    assert 'captured_gemv_launch_kwargs.get("_gemv_blocks")' in source
    assert '"gemv_stage_tactic_passed": gemv_stage_tactic_passed' in source
    assert "and gemv_stage_tactic_passed" in source
    assert 'item["bitwise_equal"] and item["upstream_reference_passed"]' in source


def test_graph_harness_records_real_bf16_gates_and_upstream_quality() -> None:
    source = _source()
    assert "_can_use_qwen35_bf16_moe_decode_gemv = traced_qwen35_gate" in source
    assert "_can_use_musa_native_bf16_moe_gemv = traced_native_bf16_gate" in source
    assert '"capability_trace": capability_trace' in source
    assert '"device_fingerprint_trace": device_fingerprint_trace' in source
    assert '"backend_call_trace": backend_call_trace' in source
    assert '"policy_trace": policy_trace' in source
    assert '"selection_observed": selection_observed' in source
    assert '"capability_passed": capability_passed' in source
    assert '"multiprocessor_count_passed": multiprocessor_count_passed' in source
    assert "and multiprocessor_count_passed" in source
    assert "**captured_gemv_launch_kwargs" in source
    assert "upstream_reference = original_upstream(**kwargs).detach().clone()" in source
    assert 'item["upstream_reference_passed"]' in source
