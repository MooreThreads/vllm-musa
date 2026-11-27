#!/usr/bin/env bash
# unit_tests.sh — run the vllm-musa patch unit tests INSIDE the MUSA container.
#
# test_patches.py is the guard for the patch mechanism:
# it asserts patch anchors/behaviour for the 51 runtime patches. It imports
# vllm/vllm_musa, so it needs a MUSA device — pin it to a spare one ($UNIT_TEST_DEVICE)
# so it does not contend with the model smokes on devices 0/1.
#
# Usage (from /workspace inside the container):
#   UNIT_TEST_DEVICE=7 tools/musa_verify/unit_tests.sh [extra pytest args]
#
# Output contract (last line):
#   PASS unit_test_patches tests=<n>
#   FAIL unit_test_patches rc=<rc>
set -uo pipefail

# Activate the pinned test venv if the caller exported MUSA_VENV.
if [ -n "${MUSA_VENV:-}" ] && [ -f "${MUSA_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${MUSA_VENV}/bin/activate"
fi

WS="${WS:-$PWD}"   # default to caller's cwd (orchestrator cds into the repo under test)
cd "$WS" || { echo "FAIL unit_test_patches reason=no_ws($WS)"; exit 1; }

export TORCHDYNAMO_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export MUSA_VISIBLE_DEVICES="${UNIT_TEST_DEVICE:-7}"

LOGDIR="${LOGDIR:-/tmp/vllm_omni_musa_logs}"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/unit_test_patches.log"

# test_patches.py is not in-process isolated (the disk-rewrite patch
# mechanism + os.environ/sys.modules state leak across tests), so failures are
# order-dependent in a single process. Run each test in its own fork (--forked,
# from pytest-forked) for deterministic results. Until a mechanism+test
# refactor removes the shared-state dependency, --forked is the
# supported way to run this suite.
python -m pip show pytest-forked >/dev/null 2>&1 || python -m pip install -q pytest-forked
python -m pytest tests/test_patches.py -q --forked "$@" >"$LOG" 2>&1
rc=$?
tail -n 25 "$LOG"

summary="$(grep -oE '[0-9]+ (passed|failed|error|skipped)[^,]*' "$LOG" | tr '\n' ' ' || true)"
if [ "$rc" -eq 0 ]; then
  echo "PASS unit_test_patches tests=[${summary}]"
else
  echo "FAIL unit_test_patches rc=$rc tests=[${summary}]"
fi
exit "$rc"
