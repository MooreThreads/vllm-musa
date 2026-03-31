# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import msgspec
import torch
import zmq
import zmq.asyncio
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TpKVTopology,
    get_current_attn_backend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeXferMetadata,
    MooncakeXferResponse,
    ReqId,
    SendBlockMeta,
    TransferId,
    _async_loop,
    get_mooncake_bootstrap_addr,
    should_launch_bootstrap_server,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip
from vllm.v1.attention.backends.utils import get_kv_cache_layout

try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    raise ImportError(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
        "to run VLLM with MooncakeTransferEngine."
    ) from e

logger = init_logger(__name__)


def get_rdma_devices():
    env_device_list = os.getenv("MOONCAKE_RDMA_DEVICES")

    if env_device_list:
        device_list = env_device_list
        logger.info(f"mooncake engine use the setup device: {device_list}")

    else:
        device_list = "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
        logger.warning(
            f"unset MOONCAKE_RDMA_DEVICES environment, mooncake engine use default device: {device_list}"
        )

    return device_list


def __init__(self, vllm_config: VllmConfig, engine_id: str):
    logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

    self.vllm_config = vllm_config

    self.engine = TransferEngine()
    self.hostname = get_ip()
    # ==================== MUSA ADAPTATION ====================
    device_list = get_rdma_devices()
    ret_value = self.engine.initialize(
        self.hostname, "P2PHANDSHAKE", "rdma", device_list
    )
    # ========================== END ==========================

    assert (kv_transfer_config := vllm_config.kv_transfer_config)
    self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
    self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
    self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
        "num_workers", 10
    )
    # Create more tasks than workers to keep the thread pool saturated.
    # Tasks can await async events, so a surplus (2x is a robust heuristic)
    # prevents workers from idling.
    self.num_sender_tasks = self.num_sender_workers * 2
    protocol = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
        "mooncake_protocol", "rdma"
    )
    logger.info("The Mooncake Transfer Engine is using %s as its protocol.", protocol)
    ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")
    if ret_value != 0:
        raise RuntimeError("Mooncake Transfer Engine initialization failed.")

    self.rpc_port = self.engine.get_rpc_port()

    logger.debug(
        "Mooncake Transfer Engine initialized at %s:%d",
        self.hostname,
        self.rpc_port,
    )

    self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
    self._pending_bootstrap_querys: dict[str, asyncio.Event] = {}
    self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
    self.engine_id: EngineId = engine_id
    self.tp_rank = get_tensor_model_parallel_rank()
    self.tp_size = get_tensor_model_parallel_world_size()
    self.num_blocks = 0

    assert (parallel_config := vllm_config.parallel_config)
    dp_rank = parallel_config.data_parallel_index
    dp_local_rank = parallel_config.data_parallel_rank_local
    self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    if pp_size > 1:
        raise ValueError(
            "Mooncake Transfer Engine does not support pipeline parallelism yet."
        )
    self.pp_rank = get_pp_group().rank_in_group

    self.kv_caches_base_addr: list[int] = []
    self.device_kv_caches: dict[str, torch.Tensor] = {}
    self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}

    # For kv_both, we will act both prefiller and decoder.
    if not self.is_kv_consumer:
        # Background threads for sending kvcaches to D.
        self._sender_executor = ThreadPoolExecutor(
            max_workers=self.num_sender_workers,
            thread_name_prefix="vllm-mooncake-sender",
        )
        logger.debug(
            "Mooncake Prefiller: use %d workers to send kvcaches",
            self.num_sender_workers,
        )
        # An asyncio queue to buffer incoming requests for the sender
        self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
        self.sender_loop = asyncio.new_event_loop()
        # Background thread for processing new sending requests.
        self._sender_listener_t = threading.Thread(
            target=_async_loop, args=(self.sender_loop,), daemon=True
        )
        self._sender_listener_t.start()

        # Start bootstrap server on global rank 0.
        if should_launch_bootstrap_server(vllm_config):
            _, port = get_mooncake_bootstrap_addr(vllm_config)
            self.bootstrap_server = MooncakeBootstrapServer(
                vllm_config, "0.0.0.0", port
            )
            self.bootstrap_server.start()

    if not self.is_kv_producer:
        self.receiver_loop = asyncio.new_event_loop()
        self._mooncake_receiver_t = threading.Thread(
            target=_async_loop, args=(self.receiver_loop,), daemon=True
        )
        self._mooncake_receiver_t.start()
        logger.debug("Mooncake Decoder: start receiver thread")

    self.finished_sending_reqs: set[ReqId] = set()
    self.finished_recving_reqs: set[ReqId] = set()

    self.block_size = vllm_config.cache_config.block_size
    self.model_config = vllm_config.model_config
    self.cache_config = vllm_config.cache_config
    self.use_mla = self.model_config.use_mla

    # Get the attention backend from the first layer
    # NOTE (NickLucche) models with multiple backends are not supported yet
    backend = get_current_attn_backend(vllm_config)
    self.backend_name = backend.get_name()
    self.kv_cache_layout = get_kv_cache_layout()
    logger.debug("Detected attention backend %s", self.backend_name)
    logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

    self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
    self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}
    self.kv_topo = TpKVTopology(
        tp_rank=self.tp_rank,
        engine_id=self.engine_id,
        remote_tp_size=self._tp_size,  # shared state
        remote_block_size=self._block_size,  # shared state
        is_mla=self.use_mla,
        total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
        attn_backend=backend,
    )

    self.async_zmq_ctx = zmq.asyncio.Context()
    self._encoder = msgspec.msgpack.Encoder()
    self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
    self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)


from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorWorker,
)

MooncakeConnectorWorker.__init__ = __init__
