# Examples

Supplementary examples for running vLLM on MTGPU. For general vLLM usage, refer to the upstream `vllm/examples` directory.

## Disaggregated Serving

Demonstrates disaggregated prefill/decode serving using the Mooncake KV-transfer
connector.

- **`disaggregated_serving.sh`** – launches one prefiller and one decoder on two
  logical MUSA GPUs, starts the proxy shipped by the pinned upstream vLLM
  checkout, and validates two completion requests. Cleanup targets only the
  processes started by the script.

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

```bash
IMAGE=registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.24.0
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
