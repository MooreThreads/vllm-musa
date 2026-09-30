# DeepSeek-V4-Flash-0731

## Overview

DeepSeek-V4-Flash-0731 (FP8 checkpoint) is served with tensor parallelism on
eight S5000 GPUs per node and DSpark speculative decoding with four draft
tokens. This recipe covers two deployment shapes:

- **Single node**: one TP8 server handles prefill and decode.
- **1P1D**: one prefill node and one decode node. Decode pulls the prompt KV
  from prefill over RDMA through `MooncakeConnector`. Prefill also saves the
  prompt KV to a standalone Mooncake store (host DRAM plus an SSD tier), so a
  repeated prefix is loaded instead of recomputed. A Mooncake proxy fronts
  both nodes.

> [!TIP]
> Use the single-node profile for short and medium contexts. Use 1P1D for long
> contexts: the profile below is sized for 128K-input/1K-output traffic up to
> 16 concurrent requests.

## At a glance

| Setting | Single node | 1P1D prefill | 1P1D decode |
|---|---|---|---|
| Hardware | 8x S5000 | 8x S5000 | 8x S5000 |
| Tensor parallelism | TP8 | TP8 | TP8 |
| Attention backend | FlashMLA | FlashMLA | FlashMLA |
| KV cache | FP8 | FP8 | FP8 |
| Speculative decoding | DSpark, 4 tokens | DSpark, 4 tokens | DSpark, 4 tokens |
| Maximum context | 8,192 tokens | 131,072 tokens | 131,072 tokens |
| Maximum batched tokens | 8,192 | 8,192 | 8,192 |
| Maximum sequences | 64 | 16 | 64 |
| GPU memory utilization | 0.90 | 0.80 | 0.95 |
| GPU prefix cache | Off | Off (Mooncake store) | On |
| CUDA graphs | `FULL_DECODE_ONLY` | Off | `FULL_DECODE_ONLY` |

## Prerequisites

- vLLM-MUSA v0.28.0-dev with its patched vLLM installed.
- Eight S5000 GPUs per node: one node for the single-node profile, two nodes
  for 1P1D.
- The checkpoint at the same path on every node. The commands read it from
  `MODEL_PATH`.
- For 1P1D:
  - RoCE between the prefill and decode nodes, with one HCA per GPU.
  - The `mooncake-transfer-engine-musa` wheel, which provides
    `mooncake_master` and `mooncake_client`.
  - An existing, writable SSD directory for the store's offload tier.
  - About 400 GiB of free host DRAM on the store node for the 128K profile.
  - Unlimited locked memory (`ulimit -l unlimited`) for RDMA registration.

Common environment for every vLLM process in this recipe:

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MODEL_PATH=/models/DeepSeek-V4-Flash-0731-FP8-mt
```

## Single node

```bash
CUDAGRAPH_CAPTURE_SIZES="1,2,3,4,5,8,16,24,32,40,48,56,64"

vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name DeepSeek-V4-Flash \
  --port 8000 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --attention-backend FLASHMLA \
  --speculative-config '{"method":"dspark","num_speculative_tokens":4}' \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CUDAGRAPH_CAPTURE_SIZES}]}" \
  --async-scheduling \
  --hf-overrides '{"expert_dtype":"fp8"}'
```

## 1P1D with a Mooncake store

### Topology and addresses

| Component | Host | Port |
|---|---|---|
| Mooncake master | `MOONCAKE_MASTER_IP` | 50051 (RPC), 9003 (metrics/admin) |
| Mooncake client (store segment + SSD tier) | `MOONCAKE_CLIENT_IP` | 50052 (RPC), 50053 (P2P handshake) |
| Prefill vLLM | `PREFILL_IP` | 28500 (HTTP), 8998 (Mooncake bootstrap) |
| Decode vLLM | `DECODE_IP` | 8025 (HTTP) |
| Proxy | `PROXY_IP` | 8000 (HTTP) |

The master, the store client, and the proxy can share the prefill node. Set
the addresses and devices on every host before running the commands below:

```bash
export PREFILL_IP="<prefill-node-address>"
export DECODE_IP="<decode-node-address>"
export PROXY_IP="<proxy-host-address>"
export MOONCAKE_MASTER_IP="<mooncake-master-address>"
export MOONCAKE_CLIENT_IP="<mooncake-store-client-address>"

# One RoCE HCA per GPU, listed in GPU order (see `ibdev2netdev` or
# `ls /sys/class/infiniband`); use the same order on both nodes.
export RDMA_DEVICES="<hca0>,<hca1>,<hca2>,<hca3>,<hca4>,<hca5>,<hca6>,<hca7>"
export MOONCAKE_SSD_DIR="<existing-ssd-directory>"
```

### 1. Mooncake master

```bash
mooncake_master \
  --rpc_port=50051 \
  --metrics_port=9003 \
  --enable_offload=true \
  --offload_on_evict=false \
  --logtostderr=true
```

### 2. Mooncake store client

The client owns the store's host-DRAM segment and offloads KV blocks to
`MOONCAKE_SSD_DIR`.

```bash
export MC_MS_FILTERS="${RDMA_DEVICES}"
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="${MOONCAKE_SSD_DIR}"
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=34359738368

mooncake_client \
  --master_server_address="${MOONCAKE_MASTER_IP}:50051" \
  --metadata_server=P2PHANDSHAKE \
  --host="${MOONCAKE_CLIENT_IP}:50053" \
  --port=50052 \
  --protocol=rdma \
  --device_names="${RDMA_DEVICES}" \
  --global_segment_size=400GB \
  --local_buffer_size=512MB \
  --enable_offload=true \
  --logtostderr=true
```

### 3. vLLM Mooncake configuration

Create the same file on the prefill and decode nodes. vLLM runs the store in
`standalone-store` mode: it contributes no segment of its own and uses the
client's segment.

```bash
cat > mooncake_config.json <<EOF
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "${MOONCAKE_MASTER_IP}:50051",
  "global_segment_size": 0,
  "local_buffer_size": "5GB",
  "protocol": "rdma",
  "device_name": "${RDMA_DEVICES}",
  "enable_offload": true
}
EOF
```

### 4. Prefill node

```bash
export MOONCAKE_CONFIG_PATH="${PWD}/mooncake_config.json"
export MOONCAKE_PREFERRED_SEGMENT="${MOONCAKE_CLIENT_IP}:50053"
export MC_TE_FILTERS="${RDMA_DEVICES}"
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=34359738368

vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name DeepSeek-V4-Flash \
  --port 28500 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --attention-backend FLASHMLA \
  --enable-chunked-prefill \
  --long-prefill-token-threshold 8192 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":4}' \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"NONE"}' \
  --async-scheduling \
  --disable-log-stats \
  --hf-overrides '{"expert_dtype":"fp8"}' \
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}]}}'
```

### 5. Decode node

```bash
export MOONCAKE_CONFIG_PATH="${PWD}/mooncake_config.json"
export MOONCAKE_PREFERRED_SEGMENT="${MOONCAKE_CLIENT_IP}:50053"
export MC_TE_FILTERS="${RDMA_DEVICES}"
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998

CUDAGRAPH_CAPTURE_SIZES="1,2,3,4,5,8,15,16,24,32,40,48,56,64,80"

vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --served-model-name DeepSeek-V4-Flash \
  --port 8025 \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --attention-backend FLASHMLA \
  --speculative-config '{"method":"dspark","num_speculative_tokens":4}' \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CUDAGRAPH_CAPTURE_SIZES}]}" \
  --async-scheduling \
  --hf-overrides '{"expert_dtype":"fp8"}' \
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}]}}'
```

### 6. Proxy

The proxy is `mooncake_connector_proxy.py` from vLLM's
`examples/disaggregated/mooncake_connector/`. It tags each request with a
`transfer_id` and the prefill bootstrap address, sends the prompt to prefill,
then streams the generation from decode. The bootstrap address is built from
the host in the `--prefill` URL, so pass an address the decode node can reach.

```bash
python3 mooncake_connector_proxy.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefill "http://${PREFILL_IP}:28500" 8998 \
  --decode "http://${DECODE_IP}:8025"
```

The proxy returns 503 until it has queried the prefill bootstrap server. Send
traffic after its log shows `All prefiller instances are ready`.

Start the components in the order shown: master, store client, prefill and
decode, then the proxy.

## Verifying the server

Both deployment shapes expose the same served model name. Point the client at
`http://localhost:8000/v1` for the single node, or at
`http://${PROXY_IP}:8000/v1` for 1P1D.

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="DeepSeek-V4-Flash",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0,
)
print(response.choices[0].message.content)
```

## Configuration notes

- Keep TP8 on every node.
- `--hf-overrides '{"expert_dtype":"fp8"}'` selects the FP8 routed-expert
  weights of this checkpoint.
- Prefill uses `gpu-memory-utilization 0.80` to leave room for the FlashMLA
  sparse-prefill workspace.
- Prefill keeps the GPU prefix cache off and relies on the Mooncake store for
  prefix reuse. It does not capture CUDA graphs because it only runs prefill.
- Decode capture sizes cover DSpark-4 steps up to 16 concurrent requests
  (16 × 5 = 80 tokens).
- `--global_segment_size=400GB` sizes the store for 128K-input traffic up to
  16 concurrent requests. Mooncake starts evicting at 90% of the segment.
- `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` is the SSD staging buffer on the
  store client and prefill. The client `--local_buffer_size` and the JSON
  `local_buffer_size` are RDMA buffers, not SSD staging.
- On a rail-optimized RoCE fabric, HCA *i* of each node sits on its own
  subnet:
  - List the HCAs in GPU order in `RDMA_DEVICES`.
  - Set `MC_ENABLE_DEST_DEVICE_AFFINITY=1` on the store client, prefill, and decode, so HCA *i* transfers only to HCA *i* of the peer. Mooncake enables affinity whenever the variable is set.
- Store keys are vLLM block hashes, which are seeded per process when
  `PYTHONHASHSEED` is unset, as in the commands above:
  - Decode receives KV over the RDMA P2P path.
  - After a prefill restart, entries written before the restart are no longer reused.
  - Restart the store client together with prefill to reclaim its segment.
- If a node has several network interfaces, set `VLLM_HOST_IP` to the address
  used in this recipe so vLLM advertises the intended interface.

Return to the [cookbook index](../README.md).
