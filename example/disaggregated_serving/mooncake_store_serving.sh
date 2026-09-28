#!/usr/bin/env bash
# Demonstrate vLLM KV-cache offload to a standalone Mooncake store on one MUSA GPU.
#
# Starts three processes and tears them down on exit:
#   mooncake_master  - store metadata service, with SSD offload enabled
#   mooncake_client  - standalone store process that owns the memory segment
#                      and offloads KV blocks to SSD under OFFLOAD_STORAGE_PATH
#   vllm serve       - MooncakeStoreConnector (kv_both): saves prompt KV to the
#                      store and loads it back on a prefix hit
#
# Usage: [VAR=value ...] mooncake_store_serving.sh [MODEL_PATH]
# Every setting below is an environment variable; logs go to LOG_DIR.

set -euo pipefail

# Model and device. vLLM runs on the first card in MUSA_VISIBLE_DEVICES.
MODEL_PATH="${1:-/home/dist/models/Qwen3-8B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"

# Ports. The client's P2P handshake address (127.0.0.1:CLIENT_HANDSHAKE_PORT)
# names its segment, which vLLM targets via MOONCAKE_PREFERRED_SEGMENT.
VLLM_PORT="${VLLM_PORT:-8024}"
MASTER_PORT="${MASTER_PORT:-50051}"
MASTER_METRICS_PORT="${MASTER_METRICS_PORT:-9003}"
CLIENT_PORT="${CLIENT_PORT:-50052}"
CLIENT_HANDSHAKE_PORT="${CLIENT_HANDSHAKE_PORT:-50053}"

# RDMA NIC used by both the client and vLLM's transfer engine.
RDMA_DEVICE="${RDMA_DEVICE:-mlx5_2}"

# GLOBAL_SEGMENT_SIZE is the client's in-memory store capacity; the local
# buffers are per-process staging for RDMA transfers.
GLOBAL_SEGMENT_SIZE="${GLOBAL_SEGMENT_SIZE:-5GB}"
CLIENT_LOCAL_BUFFER_SIZE="${CLIENT_LOCAL_BUFFER_SIZE:-512MB}"
VLLM_LOCAL_BUFFER_SIZE="${VLLM_LOCAL_BUFFER_SIZE:-1GB}"

# SSD offload directory and backend. Unset OFFLOAD_STORAGE_PATH falls back to
# the default, which must already exist on this host.
DEFAULT_OFFLOAD_STORAGE_PATH=/nvme/mooncake_cache/
OFFLOAD_STORAGE_PATH_SOURCE="OFFLOAD_STORAGE_PATH"
if [[ -z "${OFFLOAD_STORAGE_PATH:-}" ]]; then
    OFFLOAD_STORAGE_PATH="${DEFAULT_OFFLOAD_STORAGE_PATH}"
    OFFLOAD_STORAGE_PATH_SOURCE="default"
fi
OFFLOAD_STORAGE_BACKEND="${OFFLOAD_STORAGE_BACKEND:-bucket_storage_backend}"

# vLLM serving limits.
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# VERIFY_STORE=1 checks that a repeated prompt is served from the Mooncake store.
VERIFY_STORE="${VERIFY_STORE:-1}"
LOG_DIR="${LOG_DIR:-/tmp/vllm-musa-mooncake-store-example-$$}"

CLIENT_HOST="127.0.0.1:${CLIENT_HANDSHAKE_PORT}"

case "${VERIFY_STORE}" in
    0) SERVER_DEV_MODE=0 ;;
    # The store check drops the GPU prefix cache via a dev-mode endpoint.
    1) SERVER_DEV_MODE=1 ;;
    *)
        echo "VERIFY_STORE must be 0 or 1" >&2
        exit 1
        ;;
esac

mkdir -p "${LOG_DIR}"

# Mooncake requires MOONCAKE_OFFLOAD_FILE_STORAGE_PATH to be an existing,
# absolute directory; it holds the KV blocks offloaded to SSD.
offload_path_error() {
    echo "Mooncake SSD offload storage ${1}: ${OFFLOAD_STORAGE_PATH}" >&2
    if [[ "${OFFLOAD_STORAGE_PATH_SOURCE}" == default ]]; then
        echo "The default ${DEFAULT_OFFLOAD_STORAGE_PATH} is unavailable on this host." >&2
    fi
    echo "Set OFFLOAD_STORAGE_PATH to an existing, writable, absolute directory" \
        "on SSD/NVMe storage (passed as MOONCAKE_OFFLOAD_FILE_STORAGE_PATH)." >&2
    exit 1
}
if [[ "${OFFLOAD_STORAGE_PATH}" != /* ]]; then
    offload_path_error "must be an absolute path"
elif [[ ! -e "${OFFLOAD_STORAGE_PATH}" ]]; then
    offload_path_error "does not exist"
elif [[ ! -d "${OFFLOAD_STORAGE_PATH}" ]]; then
    offload_path_error "is not a directory"
elif [[ ! -w "${OFFLOAD_STORAGE_PATH}" ]]; then
    offload_path_error "is not writable"
fi
echo "Mooncake SSD offload storage: ${OFFLOAD_STORAGE_PATH}" \
    "(${OFFLOAD_STORAGE_PATH_SOURCE}, backend ${OFFLOAD_STORAGE_BACKEND})"

for bin in mooncake_master mooncake_client vllm; do
    if ! command -v "${bin}" >/dev/null 2>&1; then
        echo "Missing required command: ${bin}" >&2
        exit 1
    fi
done

# An explicit MOONCAKE_CONFIG_PATH wins; otherwise point vLLM at the local
# master in standalone-store mode (the segment lives in mooncake_client).
if [[ -z "${MOONCAKE_CONFIG_PATH:-}" ]]; then
    MOONCAKE_CONFIG_PATH="${LOG_DIR}/mooncake_config.json"
    cat >"${MOONCAKE_CONFIG_PATH}" <<EOF
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:${MASTER_PORT}",
  "global_segment_size": 0,
  "local_buffer_size": "${VLLM_LOCAL_BUFFER_SIZE}",
  "protocol": "rdma",
  "device_name": "${RDMA_DEVICE}",
  "enable_offload": true
}
EOF
fi
if [[ ! -f "${MOONCAKE_CONFIG_PATH}" ]]; then
    echo "MOONCAKE_CONFIG_PATH does not exist: ${MOONCAKE_CONFIG_PATH}" >&2
    exit 1
fi

PIDS=()

# Stop a process group and wait until every member has exited, escalating to
# SIGKILL after 30 seconds.
stop_group() {
    local pgid=$1
    local deadline=$((SECONDS + 30))

    kill -TERM -- "-${pgid}" 2>/dev/null || return 0
    while kill -0 -- "-${pgid}" 2>/dev/null; do
        if ((SECONDS >= deadline)); then
            kill -KILL -- "-${pgid}" 2>/dev/null || true
            break
        fi
        sleep 1
    done
    wait "${pgid}" 2>/dev/null || true
}

cleanup() {
    local status=${1:-$?}
    # Ignore further signals so a repeated Ctrl-C cannot orphan a service.
    trap '' INT TERM HUP
    trap - EXIT
    if ((${#PIDS[@]} > 0)); then
        echo "Stopping services..." >&2
    fi

    # Each service leads its own process group; signal the whole group so the
    # native mooncake binaries behind the Python entry points stop too.
    # Stop in reverse start order: vLLM, then the client, then the master.
    local i
    for ((i = ${#PIDS[@]} - 1; i >= 0; i--)); do
        stop_group "${PIDS[i]}"
    done

    exit "${status}"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM
trap 'cleanup 129' HUP

# Run a readiness probe until it succeeds, the process dies, or the timeout hits.
wait_until() {
    local name=$1
    local pid=$2
    shift 2
    local start_time=${SECONDS}

    until "$@" >/dev/null 2>&1; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "${name} exited before becoming ready; see ${LOG_DIR}" >&2
            return 1
        fi
        if ((SECONDS - start_time >= STARTUP_TIMEOUT)); then
            echo "Timed out waiting for ${name}" >&2
            return 1
        fi
        sleep 2
    done
}

# True when something is listening on the local TCP port.
port_open() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

# Refuse to start if a port is taken, so a stale service cannot pass a probe.
for port in "${MASTER_PORT}" "${MASTER_METRICS_PORT}" "${CLIENT_PORT}" \
    "${CLIENT_HANDSHAKE_PORT}" "${VLLM_PORT}"; do
    if port_open "${port}"; then
        echo "Port ${port} is already in use" >&2
        exit 1
    fi
done

echo "Starting Mooncake master on port ${MASTER_PORT}"
setsid mooncake_master \
    --rpc_port="${MASTER_PORT}" \
    --metrics_port="${MASTER_METRICS_PORT}" \
    --enable_offload=true \
    --offload_on_evict=false \
    --logtostderr=true \
    >"${LOG_DIR}/master.log" 2>&1 &
PIDS+=("$!")
wait_until "mooncake_master" "${PIDS[0]}" port_open "${MASTER_PORT}"

# MC_MS_FILTERS pins the client to RDMA_DEVICE; the two offload variables
# select where and how KV blocks are offloaded to SSD.
echo "Starting Mooncake client on port ${CLIENT_PORT} (${RDMA_DEVICE})"
MC_MS_FILTERS="${RDMA_DEVICE}" \
MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="${OFFLOAD_STORAGE_PATH}" \
MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR="${OFFLOAD_STORAGE_BACKEND}" \
setsid mooncake_client \
    --master_server_address="127.0.0.1:${MASTER_PORT}" \
    --metadata_server=P2PHANDSHAKE \
    --host="${CLIENT_HOST}" \
    --port="${CLIENT_PORT}" \
    --protocol=rdma \
    --device_names="${RDMA_DEVICE}" \
    --global_segment_size="${GLOBAL_SEGMENT_SIZE}" \
    --local_buffer_size="${CLIENT_LOCAL_BUFFER_SIZE}" \
    --enable_offload=true \
    --logtostderr=true \
    >"${LOG_DIR}/client.log" 2>&1 &
PIDS+=("$!")
wait_until "mooncake_client" "${PIDS[1]}" port_open "${CLIENT_PORT}"

# PYTHONHASHSEED=0 keeps block hashes, and so store keys, stable across restarts.
# MC_TE_FILTERS pins vLLM's transfer engine to RDMA_DEVICE.
echo "Starting vLLM with MooncakeStoreConnector on MUSA device ${MUSA_VISIBLE_DEVICES}"
PYTHONHASHSEED=0 \
VLLM_SERVER_DEV_MODE="${SERVER_DEV_MODE}" \
MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH}" \
MOONCAKE_PREFERRED_SEGMENT="${CLIENT_HOST}" \
MC_TE_FILTERS="${RDMA_DEVICE}" \
setsid vllm serve "${MODEL_PATH}" \
    --tensor-parallel-size 1 \
    --port "${VLLM_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --no-enable-chunked-prefill \
    --enable-prefix-caching \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}' \
    --compilation-config \
    '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}' \
    >"${LOG_DIR}/vllm.log" 2>&1 &
PIDS+=("$!")
wait_until "vllm" "${PIDS[2]}" \
    curl --fail --silent "http://127.0.0.1:${VLLM_PORT}/health"

# Send one greedy completion and print its text; fail on an empty reply.
complete() {
    local prompt=$1
    local response
    response="$(curl \
        --fail-with-body \
        --silent \
        --show-error \
        -X POST "http://127.0.0.1:${VLLM_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"${prompt}\",\"max_tokens\":10,\"temperature\":0}")"
    printf '%s\n' "${response}" | python3 -c \
        'import json, sys; data = json.load(sys.stdin); text = data["choices"][0]["text"]; assert text.strip(), data; print(text)'
}

# Total prompt tokens vLLM has loaded from the KV connector.
external_hits() {
    curl --fail --silent "http://127.0.0.1:${VLLM_PORT}/metrics" |
        awk '/^vllm:external_prefix_cache_hits(_total)?[{ ]/ {sum += $NF} END {printf "%d\n", sum}'
}

for prompt in "San Francisco is a" "Santa Clara is a"; do
    complete "${prompt}"
done

if [[ "${VERIFY_STORE}" == 1 ]]; then
    # A unique multi-block prompt: the first request saves its KV to Mooncake;
    # after the GPU prefix cache is dropped the repeat must load it back.
    long_prompt="$(python3 -c \
        'import sys; print(f"Run {sys.argv[1]}: " + " ".join(f"Record {i} is stored." for i in range(200)))' \
        "$$-${RANDOM}-$(date +%s)")"
    complete "${long_prompt}" >/dev/null
    hits_before="$(external_hits)"

    loaded=0
    for _ in $(seq 15); do
        sleep 2
        curl --fail --silent -X POST \
            "http://127.0.0.1:${VLLM_PORT}/reset_prefix_cache" >/dev/null
        complete "${long_prompt}" >/dev/null
        hits_after="$(external_hits)"
        if ((hits_after > hits_before)); then
            loaded=1
            break
        fi
    done
    if ((loaded == 0)); then
        echo "Mooncake store returned no KV for a repeated prompt; see ${LOG_DIR}" >&2
        exit 1
    fi
    echo "Mooncake store hit: $((hits_after - hits_before)) external prefix-cache tokens"
fi

echo "PASS vllm-musa-mooncake-store logs=${LOG_DIR}"
