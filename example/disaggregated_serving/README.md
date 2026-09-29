# Examples

Supplementary examples for running vLLM on MTGPU. For general vLLM usage, refer to the upstream `vllm/examples` directory.

## Disaggregated Serving

Demonstrates disaggregated prefill/decode serving and KV-cache offload using
the Mooncake KV-transfer connectors.

- **`disaggregated_serving.sh`** – launches one prefiller and one decoder on two
  logical MUSA GPUs, starts the proxy shipped by the pinned upstream vLLM
  checkout, and validates two completion requests. Cleanup targets only the
  processes started by the script.
- **`mooncake_store_serving.sh`** – runs one vLLM instance with
  `MooncakeStoreConnector` against a standalone Mooncake store (memory segment
  plus SSD offload) and checks that a repeated prompt is served from the store.
  See [Mooncake store examples](#mooncake-store-examples).
- **`mooncake_store_pd_serving.sh`** – runs 1P1D on one host on top of the same
  store: prefill and decode each use `MultiConnector`
  (`MooncakeConnector` + `MooncakeStoreConnector`), fronted by the upstream
  Mooncake proxy. See [Mooncake store examples](#mooncake-store-examples).

### Quick Start

```bash
cd example/disaggregated_serving
# Default model: Qwen/Qwen3-8B
bash disaggregated_serving.sh

# Or specify a model:
bash disaggregated_serving.sh meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Container and RDMA prerequisites

The single-node example uses P2P handshake endpoints and dynamic Mooncake RPC
ports. When it runs in a container, use the host network namespace and expose
the host RoCE devices. `--network host` alone is not sufficient; the container
also needs the verbs and `rdma_cm` character devices plus locked-memory access.

The following non-privileged container shape is intended for S5000 with RoCE
on a host whose Docker daemon provides the MUSA runtime by default. It does
not select a runtime by name; configure the host runtime before using it.
The registry image is maintained independently of this branch; use the
source-built image when the exact branch dependency stack is required.

```bash
IMAGE=registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.28.0
# Run this block in bash; the device numbers are hexadecimal in stat output.
read -r VERBS_MAJOR_HEX _ < \
  <(stat -c '%t %T' /dev/infiniband/uverbs0)
read -r RDMA_CM_MAJOR_HEX RDMA_CM_MINOR_HEX < \
  <(stat -c '%t %T' /dev/infiniband/rdma_cm)
VERBS_MAJOR=$((16#${VERBS_MAJOR_HEX}))
RDMA_CM_MAJOR=$((16#${RDMA_CM_MAJOR_HEX}))
RDMA_CM_MINOR=$((16#${RDMA_CM_MINOR_HEX}))
# Optional: export MC_TE_FILTERS=mlx5_<n>,mlx5_<m> to restrict HCA selection.
docker run --rm --name vllm-musa-mooncake \
  --detach \
  --network host \
  --shm-size 256g \
  --env MUSA_VISIBLE_DEVICES=0,1 \
  --env MTHREADS_VISIBLE_DEVICES=0,1 \
  --env MC_FORCE_HCA=1 \
  --env MC_TE_FILTERS \
  --volume /dev/infiniband:/dev/infiniband \
  --mount type=bind,src=/sys/class/infiniband,dst=/sys/class/infiniband,readonly \
  --mount type=bind,src=/sys/class/net,dst=/sys/class/net,readonly \
  --device-cgroup-rule="c ${VERBS_MAJOR}:* rmw" \
  --device-cgroup-rule="c ${RDMA_CM_MAJOR}:${RDMA_CM_MINOR} rmw" \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 \
  --entrypoint /bin/bash \
  "${IMAGE}" -lc 'sleep infinity'
```

Run the example commands with `docker exec vllm-musa-mooncake ...`, then stop
the container so it releases the GPUs and `--rm` removes it:

```bash
docker stop vllm-musa-mooncake
```

Use `ls /sys/class/infiniband` or `ibdev2netdev` on the host to obtain the
available HCA names; do not copy a node-specific list. The `stat` expressions
derive the verbs and `rdma_cm` device numbers for the current host, so the
command does not depend on a particular device minor.
Unset `MC_TE_FILTERS` to let Mooncake discover all available HCAs, or export it
before the command with a comma-separated HCA allow-list. `MC_FORCE_HCA=1`
makes an RDMA setup fail instead of silently falling back to TCP.

Before starting vLLM, verify that the same HCA devices are visible inside the
detached container:

```bash
docker exec vllm-musa-mooncake ls /dev/infiniband /sys/class/infiniband
docker exec vllm-musa-mooncake python -c \
  'from mooncake.engine import TransferEngine; print(TransferEngine().get_local_topology(""))'
```

`MOONCAKE_RDMA_DEVICES` remains a deprecated vLLM-MUSA compatibility alias. It
is mapped to `MC_TE_FILTERS` before the upstream connector is constructed;
when both variables are present, the official `MC_TE_FILTERS` value wins.

By default, the example leaves the normal compiled serving path enabled and
limits each server to 16 concurrent sequences. Set `VLLM_ENFORCE_EAGER=1` for a
functional diagnostic that isolates Mooncake from compilation. Logs are written
under `/tmp/vllm-musa-mooncake-example-<pid>`; set `LOG_DIR` to retain them
elsewhere. `PREFILL_GPU`, `DECODE_GPU`, service ports, `MAX_MODEL_LEN`,
`MAX_NUM_SEQS`, and `STARTUP_TIMEOUT` can also be overridden through the
environment.

### Mooncake store examples

Both scripts start a standalone Mooncake store — `mooncake_master` plus a
`mooncake_client` that owns the memory segment and offloads KV blocks to SSD —
and stop every process they started on exit, including on Ctrl-C.

**Prerequisites**

- The [container and RDMA prerequisites](#container-and-rdma-prerequisites)
  above; `mooncake_master` and `mooncake_client` come with the
  `mooncake-transfer-engine-musa` wheel.
- An SSD/NVMe directory for offloaded KV blocks. Mooncake requires it to be an
  existing, writable, absolute path; the scripts check this before starting and
  never create it. The default is `/nvme/mooncake_cache/`.

**Quick start**

```bash
# Single instance on card 5 (default model: /home/dist/models/Qwen3-8B-FP8)
MUSA_VISIBLE_DEVICES=5 bash example/disaggregated_serving/mooncake_store_serving.sh

# 1P1D: prefill on card 5, decode on card 6
MUSA_VISIBLE_DEVICES=5,6 bash example/disaggregated_serving/mooncake_store_pd_serving.sh

# Other model or SSD directory
OFFLOAD_STORAGE_PATH=/data/mooncake_cache/ MUSA_VISIBLE_DEVICES=5 \
  bash example/disaggregated_serving/mooncake_store_serving.sh /path/to/model
```

The scripts can be run from any directory. Each ends with `PASS ... logs=<dir>`;
`master.log`, `client.log` and one log per vLLM instance (plus `proxy.log` for
1P1D) are written under `/tmp/vllm-musa-mooncake-store[-pd]-example-<pid>`, or
`LOG_DIR` if set. The two scripts share Mooncake ports, so run one at a time;
each refuses to start while any of its ports is in use.

**What they check**

- `mooncake_store_serving.sh` sends two short prompts, then a unique long
  prompt, resets the GPU prefix cache, and resends it; the run fails unless
  vLLM's external prefix-cache hits grow, i.e. the KV came back from Mooncake.
  The reset uses a dev-mode endpoint, so the script sets
  `VLLM_SERVER_DEV_MODE=1`; `VERIFY_STORE=0` skips the check and dev mode.
- `mooncake_store_pd_serving.sh` sends the same prompts through the proxy; the
  run fails unless decode loads the long prompt's KV from prefill instead of
  recomputing it (`VERIFY_KV=0` skips the check).

**Configuration** (environment variables; defaults in parentheses)

| Variable | Scripts | Meaning |
|---|---|---|
| `MUSA_VISIBLE_DEVICES` | both | Card for the single instance (`0`); `prefill,decode` for 1P1D (`0,1`). |
| `HOST_IP` | 1P1D | Address every service advertises (source address of the default route). |
| `SERVED_MODEL_NAME` | both | Served model name (basename of the model path). |
| `RDMA_DEVICE` | both | HCA for the client and vLLM (`mlx5_2`); sets `--device_names`, `MC_MS_FILTERS` and `MC_TE_FILTERS`. |
| `OFFLOAD_STORAGE_PATH` | both | SSD directory for offloaded KV (`/nvme/mooncake_cache/`). |
| `OFFLOAD_STORAGE_BACKEND` | both | Mooncake offload backend (`bucket_storage_backend`). |
| `GLOBAL_SEGMENT_SIZE` | both | Client in-memory store capacity (`5GB`). |
| `CLIENT_LOCAL_BUFFER_SIZE` / `VLLM_LOCAL_BUFFER_SIZE` | both | RDMA staging buffers (`512MB` / `1GB`). |
| `MOONCAKE_CONFIG_PATH` | both | vLLM-side Mooncake config; generated in `LOG_DIR` in `standalone-store` mode when unset. |
| `MASTER_PORT` / `MASTER_METRICS_PORT` | both | Master RPC and metrics/admin ports (`50051` / `9003`). |
| `CLIENT_PORT` / `CLIENT_HANDSHAKE_PORT` | both | Client RPC and P2P handshake ports (`50052` / `50053`). |
| `VLLM_PORT` | single | vLLM port (`8024`). |
| `PREFILL_PORT` / `DECODE_PORT` / `PROXY_PORT` | 1P1D | Service ports (`8024` / `8025` / `8026`). |
| `BOOTSTRAP_PORT` | 1P1D | Prefill Mooncake bootstrap port (`8998`). |
| `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION` | both | vLLM limits (`8192`, `32`, `8192`, `0.85`). |
| `STARTUP_TIMEOUT` / `REQUEST_TIMEOUT` | both / 1P1D | Seconds to wait for startup / per request (`1200` / `300`). |

**How 1P1D moves KV**

`mooncake_connector_proxy.py` tags each request with a `transfer_id` and the
prefill bootstrap address. Prefill computes the prompt and keeps its blocks;
decode pulls them over RDMA through `MooncakeConnector`, which `MultiConnector`
consults first. Prefill also saves the prompt KV to the Mooncake store
(`kv_both`), so later requests that share the prefix can be served from the
store. The proxy answers 503 until it has queried the prefill bootstrap server;
the script waits for `All prefiller instances are ready` in `proxy.log`.

Running `MooncakeConnector` inside `MultiConnector` requires the vLLM series
patch `0172-MUSA-skip-Prometheus-observe-for-connectors-without-.patch`.
Without it, the first P2P transfer stops the API server with
`MooncakeConnector is not contained in the list of registered connectors with
Prometheus metrics support`.
