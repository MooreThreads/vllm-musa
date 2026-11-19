from __future__ import annotations

import argparse
import ast
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

BENCHMARK = (
    Path(__file__).resolve().parents[1] / "benchmarks/benchmark_deepselect_topk.py"
)


def _benchmark_function(name: str, namespace: dict):
    function = next(
        node
        for node in ast.parse(BENCHMARK.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(BENCHMARK), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("dtype_name, itemsize", [("float32", 4), ("bfloat16", 2)])
@pytest.mark.parametrize("include_torch", [False, True])
def test_benchmark_dtype_reaches_inputs_oracle_kernel_timing_and_report(
    tmp_path: Path, dtype_name: str, itemsize: int, include_torch: bool
) -> None:
    dtype = SimpleNamespace(itemsize=itemsize)
    torch = Mock(**{dtype_name: dtype})
    torch.randn.return_value.dtype = dtype
    torch.randn.return_value.numel.return_value = 1024
    torch.randn.return_value.element_size.return_value = itemsize
    torch.full.return_value.numel.return_value = 512

    def time_runner(runner, *args, **kwargs):
        runner()
        return 1e-6

    def event_time_runner(runner, **kwargs):
        runner()
        return [1.0, 2.0]

    kernel_timer = Mock(side_effect=time_runner)
    custom_ops = Mock()
    namespace = {
        "torch": torch,
        "_custom_ops": custom_ops,
        "check_output": Mock(),
        "bench_gpu_time_with_musa_event": Mock(side_effect=event_time_runner),
        "bench_kineto": kernel_timer,
        "statistics": statistics,
        "percentile": lambda samples, quantile: 1.0,
    }
    args = SimpleNamespace(
        dtype=dtype_name,
        lengths="full",
        topk=512,
        include_torch=include_torch,
        rounds=2,
        iters=2,
        output=tmp_path / "report.json",
    )
    result = _benchmark_function("run_case", namespace)(1, 1024, args, profile=False)
    assert torch.randn.call_args.kwargs["dtype"] is dtype
    assert torch.empty.call_args_list[0].kwargs["dtype"] is dtype
    assert result["dtype"] == dtype_name
    assert kernel_timer.call_args_list[1].args[1] == (
        "topk_bf16_kernel" if dtype_name == "bfloat16" else "topk_fp32_kernel"
    )
    logits = torch.randn.return_value
    lengths = torch.full.return_value
    radix = custom_ops.deepseek_v4_sparse_indexer_topk_decode
    assert radix.call_count == 4
    radix.assert_called_with(logits, lengths, torch.full.return_value, 512)
    deepselect = torch.ops.deep_select.topk
    assert deepselect.call_count == 4
    assert deepselect.call_args.args[:4] == (logits, 512, None, lengths)
    assert torch.topk.call_count == (3 if include_torch else 1)
    torch.musa.empty_cache.assert_not_called()
    assert result["metrics"]["deepselect"]["effective_tb_s"] == pytest.approx(
        (1024 * itemsize + 512 * 4) / 0.0015 / 1e12
    )


@pytest.mark.parametrize(
    "alignment, dtype_name, width, valid",
    [
        (32, None, 520, True),
        (32, "float32", 520, True),
        (32, "bfloat16", 520, False),
        (32, "bfloat16", 528, True),
        (1024, None, 520, False),
        (1024, "float32", 520, False),
        (1024, "float32", 768, True),
        (1024, "bfloat16", 528, False),
        (1024, "bfloat16", 768, False),
        (1024, "bfloat16", 1024, True),
    ],
)
def test_benchmark_cli_validates_alignment_for_selected_dtype(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    alignment: int,
    dtype_name: str | None,
    width: int,
    valid: bool,
) -> None:
    argv = [
        str(BENCHMARK),
        "--batches",
        "1",
        "--widths",
        str(width),
        "--output",
        str(tmp_path / "report.json"),
    ]
    if dtype_name is not None:
        argv.extend(["--dtype", dtype_name])
    monkeypatch.setattr(sys, "argv", argv)
    runner = Mock(
        return_value={
            "metrics": {
                name: {"median_us": 1.0} for name in ("vllm_radix", "deepselect")
            },
            "speedup_vs_vllm_radix": 1.0,
        }
    )
    namespace = {
        "__doc__": "Benchmark",
        "argparse": argparse,
        "sys": sys,
        "Path": Path,
        "torch": SimpleNamespace(
            float32=SimpleNamespace(itemsize=4),
            bfloat16=SimpleNamespace(itemsize=2),
            musa=SimpleNamespace(empty_cache=Mock()),
        ),
        "logging": Mock(),
        "vllm_musa": Mock(),
        "get_deep_select_stride_requirement": Mock(return_value=(alignment, 32)),
        "provenance": Mock(return_value={}),
        "run_case": runner,
        "json": SimpleNamespace(dumps=lambda *args, **kwargs: "{}"),
        "LOGGER": Mock(),
    }
    main = _benchmark_function("main", namespace)
    if valid:
        main()
        assert runner.call_args.args[2].dtype == (dtype_name or "float32")
        namespace["provenance"].assert_called_once_with(dtype_name or "float32")
        namespace["torch"].musa.empty_cache.assert_called_once_with()
    else:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
        assert f"{alignment}-byte aligned input rows" in capsys.readouterr().err
        runner.assert_not_called()
        namespace["torch"].musa.empty_cache.assert_not_called()
    namespace["get_deep_select_stride_requirement"].assert_called_once_with()
