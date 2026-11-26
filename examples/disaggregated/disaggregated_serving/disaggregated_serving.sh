#!/usr/bin/env bash
# Demonstrate disaggregated prefill/decode serving with Mooncake on two MUSA GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROXY_RELATIVE_PATH="examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"
PROXY_SCRIPT="${REPO_ROOT}/${PROXY_RELATIVE_PATH}"
if [[ ! -f "${PROXY_SCRIPT}" ]]; then
    PROXY_SCRIPT="${REPO_ROOT}/third_party/vllm/${PROXY_RELATIVE_PATH}"
fi

MODEL_NAME="${1:-Qwen/Qwen3-8B}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
PROXY_PORT="${PROXY_PORT:-8000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
LOG_DIR="${LOG_DIR:-/tmp/vllm-musa-mooncake-example-$$}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${PROXY_SCRIPT}" ]]; then
    echo "Missing the pinned upstream Mooncake proxy: ${PROXY_SCRIPT}" >&2
    exit 1
fi

PIDS=()
SERVE_ARGS=()

case "${VLLM_ENFORCE_EAGER}" in
    0) ;;
    1) SERVE_ARGS+=(--enforce-eager) ;;
    *)
        echo "VLLM_ENFORCE_EAGER must be 0 or 1" >&2
        exit 1
        ;;
esac

# This is a wrapper-only switch; do not make vLLM treat it as an unknown
# environment variable in each server process.
unset VLLM_ENFORCE_EAGER

cleanup() {
    local status=${1:-$?}
    trap - EXIT INT TERM

    for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "${PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done

    exit "${status}"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

wait_for_url() {
    local name=$1
    local url=$2
    local pid=$3
    local start_time=${SECONDS}

    until curl --fail --silent --show-error "${url}" >/dev/null 2>&1; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "${name} exited before becoming ready; see ${LOG_DIR}" >&2
            return 1
        fi
        if ((SECONDS - start_time >= STARTUP_TIMEOUT)); then
            echo "Timed out waiting for ${name} at ${url}" >&2
            return 1
        fi
        sleep 2
    done
}

echo "Starting Mooncake prefiller on logical MUSA GPU ${PREFILL_GPU}"
VLLM_MOONCAKE_BOOTSTRAP_PORT="${BOOTSTRAP_PORT}" \
MUSA_VISIBLE_DEVICES="${PREFILL_GPU}" \
vllm serve "${MODEL_NAME}" \
    --port "${PREFILL_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    "${SERVE_ARGS[@]}" \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}' \
    >"${LOG_DIR}/prefill.log" 2>&1 &
PIDS+=("$!")

echo "Starting Mooncake decoder on logical MUSA GPU ${DECODE_GPU}"
MUSA_VISIBLE_DEVICES="${DECODE_GPU}" \
vllm serve "${MODEL_NAME}" \
    --port "${DECODE_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    "${SERVE_ARGS[@]}" \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}' \
    >"${LOG_DIR}/decode.log" 2>&1 &
PIDS+=("$!")

wait_for_url "prefiller" "http://127.0.0.1:${PREFILL_PORT}/health" "${PIDS[0]}"
wait_for_url "decoder" "http://127.0.0.1:${DECODE_PORT}/health" "${PIDS[1]}"

python3 "${PROXY_SCRIPT}" \
    --prefill "http://127.0.0.1:${PREFILL_PORT}" "${BOOTSTRAP_PORT}" \
    --decode "http://127.0.0.1:${DECODE_PORT}" \
    --port "${PROXY_PORT}" \
    >"${LOG_DIR}/proxy.log" 2>&1 &
PIDS+=("$!")

wait_for_url "proxy" "http://127.0.0.1:${PROXY_PORT}/docs" "${PIDS[2]}"

for prompt in "San Francisco is a" "Santa Clara is a"; do
    response="$(curl \
        --fail-with-body \
        --silent \
        --show-error \
        --retry 30 \
        --retry-all-errors \
        --retry-delay 2 \
        -X POST "http://127.0.0.1:${PROXY_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"${prompt}\",\"max_tokens\":10,\"temperature\":0}")"
    printf '%s\n' "${response}" | python3 -c \
        'import json, sys; data = json.load(sys.stdin); text = data["choices"][0]["text"]; assert text.strip(), data; print(text)'
done

echo "PASS vllm-musa-mooncake-disaggregated logs=${LOG_DIR}"
