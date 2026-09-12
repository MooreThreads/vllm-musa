# SPDX-License-Identifier: Apache-2.0
"""Keep torch_musa Inductor benchmarking compatible with Triton 3.6."""

import functools
import importlib.util

from vllm.logger import init_logger

logger = init_logger(__name__)
PATCHES: list = []


def apply() -> None:
    try:
        from torch_musa._inductor import utils as musa_utils
    except Exception as e:
        logger.debug("Skipping torch_musa Inductor compatibility patch: %s", e)
        return

    try:
        has_legacy_bench = importlib.util.find_spec(
            "triton.backends.mtgpu.musa_testing"
        ) is not None
    except (ImportError, ModuleNotFoundError):
        has_legacy_bench = False
    if has_legacy_bench:
        return

    benchmark_cls = musa_utils.TritonBenchmarker
    if getattr(benchmark_cls, "_musa_triton36_compat", False):
        return

    def triton_do_bench(self):
        from triton.testing import do_bench

        def do_bench_compat(*args, **kwargs):
            result = do_bench(*args, device_type="musa", **kwargs)
            if kwargs.get("quantiles") is not None and not isinstance(
                result, (list, tuple)
            ):
                return [result]
            return result

        return do_bench_compat

    benchmarker = functools.cached_property(triton_do_bench)
    benchmarker.__set_name__(benchmark_cls, "triton_do_bench")
    benchmark_cls.triton_do_bench = benchmarker

    benchmark_cls._musa_triton36_compat = True
    logger.info("Applied Triton 3.6 torch_musa Inductor benchmark compatibility")
