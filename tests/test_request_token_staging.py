import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm_musa/v1/worker/request_token_staging.py"
)
SPEC = spec_from_file_location("request_token_staging", MODULE_PATH)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
preallocate_deepseek_request_token_staging = (
    MODULE.preallocate_deepseek_request_token_staging
)


@pytest.fixture
def pool_factory(monkeypatch):
    factory = Mock(return_value=sentinel.staging_pool)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu.buffer_utils",
        SimpleNamespace(UvaBufferPool=factory),
    )
    return factory


@pytest.mark.parametrize(
    "model_type", ["deepseek_v4", "deepseek_v41", "deepseek_v41_text"]
)
@pytest.mark.parametrize("capacity", [1048576, 4 * 131072])
def test_preallocates_every_staging_slot(pool_factory, model_type, capacity):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        )
    )
    token_ids = SimpleNamespace(
        gpu=SimpleNamespace(numel=lambda: capacity),
        dtype=sentinel.dtype,
        max_concurrency=2,
        write_contents=sentinel.original_pool,
    )
    runner = SimpleNamespace(req_states=SimpleNamespace(all_token_ids=token_ids))

    preallocate_deepseek_request_token_staging(config, runner)

    pool_factory.assert_called_once_with(capacity, sentinel.dtype, max_concurrency=2)
    assert token_ids.write_contents is sentinel.staging_pool


@pytest.mark.parametrize("model_type", ["other_model", None])
def test_other_models_do_not_access_request_state(model_type):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        )
    )

    preallocate_deepseek_request_token_staging(config, object())


def test_missing_model_config_is_unchanged():
    preallocate_deepseek_request_token_staging(
        SimpleNamespace(model_config=None), object()
    )


def test_dense_token_storage_is_unchanged(pool_factory):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v41")
        )
    )
    token_ids = SimpleNamespace(write_contents=None)
    runner = SimpleNamespace(req_states=SimpleNamespace(all_token_ids=token_ids))

    preallocate_deepseek_request_token_staging(config, runner)

    pool_factory.assert_not_called()
    assert token_ids.write_contents is None
