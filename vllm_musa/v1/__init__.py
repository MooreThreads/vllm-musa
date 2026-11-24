# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA v1 runtime extensions loaded by ``vllm_musa.register_out_of_tree``.

vLLM imports this package from inside ``vllm/v1/worker/gpu/model_runner.py``,
so importing ``vllm_musa.v1.worker`` at module scope would re-enter that
partially initialized module. Both v1 compatibility adapters and worker hooks
are installed explicitly and idempotently by the general plugin.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

_V1_ADAPTERS_REGISTERED = False


def register_v1_adapters() -> None:
    """Install idempotent MUSA v1 attention, executor, sampler, and spec-decode hooks."""
    global _V1_ADAPTERS_REGISTERED
    if _V1_ADAPTERS_REGISTERED:
        logger.debug("vllm_musa.v1 adapters already registered; skipping")
        return

    import vllm_musa.v1.attention.backends.mla.flashmla  # noqa: F401
    from vllm_musa.v1.attention.backends import turboquant
    from vllm_musa.v1.attention.backends.mla import common as mla_common
    from vllm_musa.v1.attention.backends.mla import flashmla_sparse
    from vllm_musa.v1.executor import multiproc_executor
    from vllm_musa.v1.sample import topk_topp_sampler, uniform_sample_counts
    from vllm_musa.v1.spec_decode import utils as spec_decode_utils

    adapters = (
        ("mla.common", mla_common),
        ("mla.flashmla_sparse", flashmla_sparse),
        ("turboquant", turboquant),
        ("multiproc_executor", multiproc_executor),
        ("topk_topp_sampler", topk_topp_sampler),
        ("uniform_sample_counts", uniform_sample_counts),
        ("spec_decode.utils", spec_decode_utils),
    )
    for name, module in adapters:
        module.install_hooks()
        logger.debug("Installed v1 adapter hooks: %s", name)
    _V1_ADAPTERS_REGISTERED = True
