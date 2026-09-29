#!/usr/bin/env bash
# Demonstrate 1P1D disaggregated serving backed by a Mooncake store on one host.
#
# Starts five processes and tears them down on exit:
#   mooncake_master  - store metadata service, with SSD offload enabled
#   mooncake_client  - standalone store process that owns the memory segment
#                      and offloads KV blocks to SSD under OFFLOAD_STORAGE_PATH
#   prefill vllm     - MultiConnector: MooncakeConnector (kv_producer) for
#                      P2P transfer + MooncakeStoreConnector (kv_both)
#   decode vllm      - MultiConnector: MooncakeConnector (kv_consumer) +
#                      MooncakeStoreConnector (kv_consumer)
#   mooncake proxy   - tags each request with a transfer_id, runs it on
#                      prefill, then on decode, which pulls the KV from
#                      prefill over RDMA via the prefill bootstrap server
#
# Usage: [VAR=value ...] mooncake_store_pd_serving.sh [MODEL_PATH]
# Every setting below is an environment variable; logs go to LOG_DIR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROXY_SCRIPT="${REPO_ROOT}/third_party/vllm/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"

# Model and devices. MUSA_VISIBLE_DEVICES lists two cards: prefill runs on
# the first, decode on the second.
MODEL_PATH="${1:-/home/dist/models/Qwen3-8B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
IFS=, read -r -a MUSA_DEVICES <<<"${MUSA_VISIBLE_DEVICES:-0,1}"

# Address every service advertises and is probed on. Defaults to the source
# address of the default route, the same one vLLM's get_ip() picks.
HOST_IP="${HOST_IP:-$(python3 -c 'import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
print(s.getsockname()[0])' 2>/dev/null || true)}"

# Ports. The client's P2P handshake address (HOST_IP:CLIENT_HANDSHAKE_PORT)
# names its segment, which both vLLM instances target via
# MOONCAKE_PREFERRED_SEGMENT. Only the prefiller serves BOOTSTRAP_PORT.
PREFILL_PORT="${PREFILL_PORT:-8024}"
DECODE_PORT="${DECODE_PORT:-8025}"
PROXY_PORT="${PROXY_PORT:-8026}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
MASTER_PORT="${MASTER_PORT:-50051}"
MASTER_METRICS_PORT="${MASTER_METRICS_PORT:-9003}"
CLIENT_PORT="${CLIENT_PORT:-50052}"
CLIENT_HANDSHAKE_PORT="${CLIENT_HANDSHAKE_PORT:-50053}"

# RDMA NIC used by the client and both vLLM transfer engines.
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

# vLLM serving limits, shared by both instances.
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# VERIFY_KV=1 checks that decode receives a long prompt's KV from prefill
# instead of recomputing it.
VERIFY_KV="${VERIFY_KV:-1}"
LOG_DIR="${LOG_DIR:-/tmp/vllm-musa-mooncake-store-pd-example-$$}"

if ((${#MUSA_DEVICES[@]} != 2)); then
    echo "MUSA_VISIBLE_DEVICES must list exactly two cards (prefill,decode)," \
        "got: ${MUSA_VISIBLE_DEVICES:-}" >&2
    exit 1
fi
PREFILL_DEVICE="${MUSA_DEVICES[0]}"
DECODE_DEVICE="${MUSA_DEVICES[1]}"
if [[ -z "${HOST_IP}" ]]; then
    echo "Could not detect the host IP; set HOST_IP" >&2
    exit 1
fi
if [[ ! -f "${PROXY_SCRIPT}" ]]; then
    echo "Missing the pinned upstream Mooncake proxy: ${PROXY_SCRIPT}" >&2
    exit 1
fi
case "${VERIFY_KV}" in
    0 | 1) ;;
    *)
        echo "VERIFY_KV must be 0 or 1" >&2
        exit 1
        ;;
esac

CLIENT_HOST="${HOST_IP}:${CLIENT_HANDSHAKE_PORT}"

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
echo "Host IP: ${HOST_IP}"
echo "Mooncake SSD offload storage: ${OFFLOAD_STORAGE_PATH}" \
    "(${OFFLOAD_STORAGE_PATH_SOURCE}, backend ${OFFLOAD_STORAGE_BACKEND})"

for bin in mooncake_master mooncake_client vllm; do
    if ! command -v "${bin}" >/dev/null 2>&1; then
        echo "Missing required command: ${bin}" >&2
        exit 1
    fi
done

# An explicit MOONCAKE_CONFIG_PATH wins; otherwise point both vLLM instances at
# the master in standalone-store mode (the segment lives in mooncake_client).
if [[ -z "${MOONCAKE_CONFIG_PATH:-}" ]]; then
    MOONCAKE_CONFIG_PATH="${LOG_DIR}/mooncake_config.json"
    cat >"${MOONCAKE_CONFIG_PATH}" <<EOF
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "${HOST_IP}:${MASTER_PORT}",
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
    # Stop in reverse start order: proxy and vLLM first, the store last.
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

# True when something is listening on HOST_IP:port.
port_open() {
    (exec 3<>"/dev/tcp/${HOST_IP}/$1") 2>/dev/null
}

# Refuse to start if a port is taken, so a stale service cannot pass a probe.
for port in "${MASTER_PORT}" "${MASTER_METRICS_PORT}" "${CLIENT_PORT}" \
    "${CLIENT_HANDSHAKE_PORT}" "${PREFILL_PORT}" "${DECODE_PORT}" \
    "${PROXY_PORT}" "${BOOTSTRAP_PORT}"; do
    if port_open "${port}" || (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
        echo "Port ${port} is already in use" >&2
        exit 1
    fi
done

echo "Starting Mooncake master on ${HOST_IP}:${MASTER_PORT}"
setsid mooncake_master \
    --rpc_port="${MASTER_PORT}" \
    --metrics_port="${MASTER_METRICS_PORT}" \
    --enable_offload=true \
    --offload_on_evict=false \
    --logtostderr=true \
    >"${LOG_DIR}/master.log" 2>&1 &
PIDS+=("$!")
wait_until "mooncake_master" "${PIDS[-1]}" port_open "${MASTER_PORT}"

# MC_MS_FILTERS pins the client to RDMA_DEVICE; the two offload variables
# select where and how KV blocks are offloaded to SSD.
echo "Starting Mooncake client on ${HOST_IP}:${CLIENT_PORT} (${RDMA_DEVICE})"
MC_MS_FILTERS="${RDMA_DEVICE}" \
MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="${OFFLOAD_STORAGE_PATH}" \
MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR="${OFFLOAD_STORAGE_BACKEND}" \
setsid mooncake_client \
    --master_server_address="${HOST_IP}:${MASTER_PORT}" \
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
wait_until "mooncake_client" "${PIDS[-1]}" port_open "${CLIENT_PORT}"

# Environment shared by both vLLM instances. PYTHONHASHSEED=0 gives both the
# same block hashes, so the store keys prefill writes are the ones decode reads.
# MC_TE_FILTERS pins each transfer engine to RDMA_DEVICE; VLLM_HOST_IP makes
# the P2P connector advertise HOST_IP.
VLLM_ENV=(
    PYTHONHASHSEED=0
    VLLM_HOST_IP="${HOST_IP}"
    MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH}"
    MOONCAKE_PREFERRED_SEGMENT="${CLIENT_HOST}"
    MC_TE_FILTERS="${RDMA_DEVICE}"
    VLLM_MOONCAKE_BOOTSTRAP_PORT="${BOOTSTRAP_PORT}"
)
VLLM_ARGS=(
    --tensor-parallel-size 1
    --served-model-name "${SERVED_MODEL_NAME}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --no-enable-chunked-prefill
    --trust-remote-code
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
)

# Prefill runs eager without a local prefix cache and saves every prompt's KV
# to the store.
echo "Starting prefill vLLM on MUSA device ${PREFILL_DEVICE}, port ${PREFILL_PORT}"
env "${VLLM_ENV[@]}" MUSA_VISIBLE_DEVICES="${PREFILL_DEVICE}" \
    setsid vllm serve "${MODEL_PATH}" \
    "${VLLM_ARGS[@]}" \
    --port "${PREFILL_PORT}" \
    --no-enable-prefix-caching \
    --enforce-eager \
    --kv-transfer-config '{
      "kv_connector": "MultiConnector",
      "kv_role": "kv_producer",
      "kv_connector_extra_config": {
        "connectors": [
          {"kv_connector": "MooncakeConnector", "kv_role": "kv_producer"},
          {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}
        ]
      }
    }' \
    >"${LOG_DIR}/prefill.log" 2>&1 &
PIDS+=("$!")
PREFILL_PID="${PIDS[-1]}"

# Decode keeps a local prefix cache, captures decode-only CUDA graphs, and only
# reads from the store.
echo "Starting decode vLLM on MUSA device ${DECODE_DEVICE}, port ${DECODE_PORT}"
env "${VLLM_ENV[@]}" MUSA_VISIBLE_DEVICES="${DECODE_DEVICE}" \
    setsid vllm serve "${MODEL_PATH}" \
    "${VLLM_ARGS[@]}" \
    --port "${DECODE_PORT}" \
    --enable-prefix-caching \
    --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config '{
      "kv_connector": "MultiConnector",
      "kv_role": "kv_consumer",
      "kv_connector_extra_config": {
        "connectors": [
          {"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer"},
          {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_consumer"}
        ]
      }
    }' \
    >"${LOG_DIR}/decode.log" 2>&1 &
PIDS+=("$!")
DECODE_PID="${PIDS[-1]}"

wait_until "prefill vllm" "${PREFILL_PID}" \
    curl --fail --silent "http://${HOST_IP}:${PREFILL_PORT}/health"
wait_until "decode vllm" "${DECODE_PID}" \
    curl --fail --silent "http://${HOST_IP}:${DECODE_PORT}/health"

# The proxy answers 503 until it has queried the prefill bootstrap server.
echo "Starting Mooncake proxy on ${HOST_IP}:${PROXY_PORT}"
setsid python3 "${PROXY_SCRIPT}" \
    --host "${HOST_IP}" \
    --port "${PROXY_PORT}" \
    --prefill "http://${HOST_IP}:${PREFILL_PORT}" "${BOOTSTRAP_PORT}" \
    --decode "http://${HOST_IP}:${DECODE_PORT}" \
    >"${LOG_DIR}/proxy.log" 2>&1 &
PIDS+=("$!")
wait_until "proxy" "${PIDS[-1]}" \
    grep -q "All prefiller instances are ready" "${LOG_DIR}/proxy.log"

# Send one greedy completion through the proxy and print its text; fail on an
# empty reply.
complete() {
    local prompt=$1
    local response
    response="$(curl \
        --fail-with-body \
        --silent \
        --show-error \
        --max-time "${REQUEST_TIMEOUT}" \
        -X POST "http://${HOST_IP}:${PROXY_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"${prompt}\",\"max_tokens\":10,\"temperature\":0}")"
    printf '%s\n' "${response}" | python3 -c \
        'import json, sys; data = json.load(sys.stdin); text = data["choices"][0]["text"]; assert text.strip(), data; print(text)'
}

# Total prompt tokens the decoder has loaded from its KV connectors.
decode_external_hits() {
    curl --fail --silent "http://${HOST_IP}:${DECODE_PORT}/metrics" |
        awk '/^vllm:external_prefix_cache_hits(_total)?[{ ]/ {sum += $NF} END {printf "%d\n", sum}'
}

for prompt in "San Francisco is a" "Santa Clara is a"; do
    complete "${prompt}"
done

if [[ "${VERIFY_KV}" == 1 ]]; then
    # A unique multi-block prompt the decoder has never seen: its KV must come
    # from prefill through the connectors, not from a decode-side recompute.
    long_prompt="$(python3 -c \
        'import sys; print(f"Run {sys.argv[1]}: " + " ".join(f"Record {i} is stored." for i in range(200)))' \
        "$$-${RANDOM}-$(date +%s)")"
    hits_before="$(decode_external_hits)"
    complete "${long_prompt}" >/dev/null
    hits_after="$(decode_external_hits)"
    if ((hits_after <= hits_before)); then
        echo "Decode recomputed the prompt instead of loading its KV; see ${LOG_DIR}" >&2
        exit 1
    fi
    echo "Decode loaded $((hits_after - hits_before)) prompt tokens of KV from prefill"
fi

echo "PASS vllm-musa-mooncake-store-pd logs=${LOG_DIR}"
