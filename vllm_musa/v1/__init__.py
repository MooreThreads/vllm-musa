# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA v1 runtime extensions loaded by ``vllm_musa.register_out_of_tree``.

vLLM imports this package from inside ``vllm/v1/worker/gpu/model_runner.py``,
so importing ``vllm_musa.v1.worker`` at module scope would re-enter that
partially initialized module. Both v1 compatibility adapters and worker hooks
are installed explicitly and idempotently by the general plugin.
"""

_V1_ADAPTERS_REGISTERED = False


def register_v1_adapters() -> None:
    """Install idempotent MUSA v1 attention, executor, sampler, and spec-decode hooks."""
    global _V1_ADAPTERS_REGISTERED
    if _V1_ADAPTERS_REGISTERED:
        return

    import vllm_musa.v1.attention.backends.mla.flashmla  # noqa: F401
    from vllm_musa.v1.attention.backends.mla import (
        common as mla_common,
    )
    from vllm_musa.v1.attention.backends.mla import (
        flashmla_sparse,
    )
    from vllm_musa.v1.executor import multiproc_executor
    from vllm_musa.v1.sample import topk_topp_sampler, uniform_sample_counts
    from vllm_musa.v1.spec_decode import utils as spec_decode_utils

    mla_common.install_hooks()
    flashmla_sparse.install_hooks()
    multiproc_executor.install_hooks()
    topk_topp_sampler.install_hooks()
    uniform_sample_counts.install_hooks()
    spec_decode_utils.install_hooks()
    _V1_ADAPTERS_REGISTERED = True
