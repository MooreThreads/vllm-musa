# vLLM imports this package from inside vllm/v1/worker/gpu/model_runner.py, so
# importing vllm_musa.v1.worker here would re-enter that partially initialized
# module. The Qwen runner hooks are installed by the general plugin instead.
#
# These imports exist for their import-time registration, not for their names.
import vllm_musa.v1.attention.backends.mla.common  # noqa: F401
import vllm_musa.v1.attention.backends.mla.flashmla  # noqa: F401
import vllm_musa.v1.attention.backends.mla.flashmla_sparse  # noqa: F401
import vllm_musa.v1.executor  # noqa: F401
import vllm_musa.v1.sample  # noqa: F401
import vllm_musa.v1.spec_decode.utils  # noqa: F401
