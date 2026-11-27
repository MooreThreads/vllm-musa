#!/usr/bin/env bash
# smoke_one_model.sh — one vllm-musa functional server smoke, pinned to ONE MUSA device.
#
# Runs INSIDE the MUSA container (expects `vllm`, `curl`, `mthreads-gmi` on PATH and
# the repo at the working dir / $PWD = /workspace). Invoked once per (model, device) so several
# can run concurrently on different devices via MUSA_VISIBLE_DEVICES.
#
# SHARED-BOX SAFETY (the box may host other jobs):
#   * If the target device does not have >= $MIN_FREE_MIB free, it SKIPs — it never
#     kills another job to make room, and never OOMs by launching anyway.
#   * On exit it tears down ONLY its own process group (setsid), never a blanket
#     `pkill -f vllm`.
#
# Usage:
#   tools/musa_verify/smoke_one_model.sh <family> <model_path> <device> <port> <tp> <expect>
# Example:
#   tools/musa_verify/smoke_one_model.sh deepseek_v2_lite /path/to/models/DeepSeek-V2-Lite 0 8000 1 beijing
#
# Output contract (exactly one line, last line of stdout):
#   PASS <family> device=<d> backend=<b> contains_<expect>=True
#   FAIL <family> device=<d> reason=<...>
#   SKIP <family> device=<d> reason=<...>
set -uo pipefail

# Activate the pinned test venv if the caller exported MUSA_VENV.
if [ -n "${MUSA_VENV:-}" ] && [ -f "${MUSA_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${MUSA_VENV}/bin/activate"
fi

FAMILY="${1:?family}"; MODEL="${2:?model_path}"; DEV="${3:?device}"; PORT="${4:?port}"
TP="${5:-1}"; EXPECT="${6:-beijing}"; MODE="${7:-chat}"   # chat | completion (base models)

MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"      # require >=40 GiB free on the target device
READY_TIMEOUT="${READY_TIMEOUT:-900}"      # seconds to wait for /health
REQ_TIMEOUT="${REQ_TIMEOUT:-120}"          # seconds for the chat request
REQ_MAX_TOKENS="${REQ_MAX_TOKENS:-256}"    # enough for reasoning models (Qwen3 <think>) to reach the answer
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
LOGDIR="${LOGDIR:-/tmp/vllm_omni_musa_logs}"; mkdir -p "$LOGDIR"
OUTDIR="${OUTDIR:-/tmp/vllm_omni_musa_outputs}"; mkdir -p "$OUTDIR"
LOG="$LOGDIR/smoke_${FAMILY}_dev${DEV}.log"

fail(){ echo "FAIL $FAMILY device=$DEV reason=$*"; exit 1; }
skip(){ echo "SKIP $FAMILY device=$DEV reason=$*"; exit 0; }

# --- 1) device-busy guard: parse "<used>MiB(<total>MiB)" for this physical device id ---
gpu_line="$(mthreads-gmi 2>/dev/null | grep -E "^${DEV}[[:space:]]+MTT" | head -1 || true)"
if [ -n "$gpu_line" ]; then
  mem="$(echo "$gpu_line" | grep -oE '[0-9]+MiB\([0-9]+MiB\)' | head -1)"
  used="$(echo "$mem" | grep -oE '^[0-9]+')"
  total="$(echo "$mem" | sed -E 's/^[0-9]+MiB\(([0-9]+)MiB\)/\1/')"
  if [ -n "${used:-}" ] && [ -n "${total:-}" ]; then
    free=$(( total - used ))
    [ "$free" -lt "$MIN_FREE_MIB" ] && skip "device busy (free=${free}MiB < ${MIN_FREE_MIB}MiB; not stealing GPU from another job)"
  fi
else
  echo "WARN $FAMILY device=$DEV: mthreads-gmi gave no line for device $DEV; proceeding without busy-guard" >&2
fi

# --- 2) launch server in its own session (so cleanup kills only this tree) ---
export TORCHDYNAMO_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
# NOTE: --enforce-eager => functional smoke. Do NOT use these numbers for perf claims.

setsid env MUSA_VISIBLE_DEVICES="$DEV" \
  vllm serve "$MODEL" \
    --enforce-eager \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --port "$PORT" >"$LOG" 2>&1 &
SRV=$!   # session leader; its PGID == $SRV

cleanup(){
  kill -TERM -- "-$SRV" 2>/dev/null || true
  for _ in 1 2 3 4 5; do kill -0 "$SRV" 2>/dev/null || return; sleep 2; done
  kill -KILL -- "-$SRV" 2>/dev/null || true
}
trap cleanup EXIT

# --- 3) wait for readiness, fail fast if the server dies ---
t=0
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  kill -0 "$SRV" 2>/dev/null || fail "server exited during startup; log tail: $(tail -n 3 "$LOG" | tr '\n' ' ')"
  sleep 5; t=$((t+5))
  [ "$t" -ge "$READY_TIMEOUT" ] && fail "startup timeout ${READY_TIMEOUT}s (log: $LOG)"
done

# --- 4) semantic request (server startup alone is NOT a pass) ---
if [ "$MODE" = "completion" ]; then
  # base (non-instruct) models: /v1/completions with a completion-style prompt
  resp="$(curl -sS -m "$REQ_TIMEOUT" "http://127.0.0.1:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of China is\",\"max_tokens\":${REQ_MAX_TOKENS},\"temperature\":0}" 2>&1)"
else
  resp="$(curl -sS -m "$REQ_TIMEOUT" "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of China? Answer in one word.\"}],\"max_tokens\":${REQ_MAX_TOKENS},\"temperature\":0}" 2>&1)"
fi
echo "$resp" > "$OUTDIR/smoke_${FAMILY}_dev${DEV}.resp.json"
echo "$resp" | grep -iq "$EXPECT" || fail "expected '$EXPECT' missing in response: $(echo "$resp" | head -c 300)"

# --- 5) backend + hygiene checks from the log ---
backend="$(grep -oiE 'FLASH_MLA|FLASH_ATTN|SDPA' "$LOG" | head -1 || true)"
grep -qi "No module named 'flash_attn'" "$LOG" && fail "flash_attn ImportError (mate path not active)"
grep -qiE "Double|float64" "$LOG" && echo "WARN $FAMILY device=$DEV: float64/Double mention in log" >&2

echo "PASS $FAMILY device=$DEV backend=${backend:-unknown} contains_${EXPECT}=True"
