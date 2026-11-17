# MCCL grouped-collective hang

## Scope and observed failure

MCCL 2.30.4 (`mcclGetVersion() == 23004`) on MTT S5000, MUSA 5.2.0 and
driver 5.2.0-server can hang when a group mixes differently sized small
collectives. `mcclGroupEnd()` returns, but the GPU stream never completes.
The first iteration is sufficient. This reproduces without vLLM, PyTorch,
Torchada, a model or Mega MoE, on 2, 4 and 8 ranks.

For the two-all-gather matrix, both operations use the same dtype and each
rank sends the indicated byte counts:

- 4096 + 64, 2048 + 64, or 4096 + 1024 bytes: timeout.
- 1024 + 64, 4096 + 2048, or 4096 + 4096 bytes: completes.
- 4096 + 64 bytes without a group: completes, with byte-exact output.
- Reversing operation order or using uint8 for both operations does not fix
  the failing case.

These are observed boundaries of this particular reproducer, not a claimed
universal size threshold or a diagnosis of MCCL's internal root cause.
Grouped per-root broadcasts for uneven all-gather also hang in the tested
vLLM-shaped workload, even when that workload does not meet the exact
byte-boundary description above.

## Minimal standalone reproducer

[mccl_group_hang.c](../benchmarks/distributed/mccl_group_hang.c) calls only
MCCL and the MUSA runtime. It declares the small MCCL 2.30.4 ABI subset
because the test image does not ship `mccl.h`. Each rank has a setup timeout
and a 20-second stream timeout. A hang exits 124; an API or data error exits 1.

Run only in a task-owned container on leased GPUs. Runtime visibility must
map the assigned devices to local ordinals 0 through N-1. Use a fresh shared
ID file for each test, never one from a previous communicator.

```bash
gcc -O2 -Wall -Wextra -o /tmp/mccl_group_hang \
  benchmarks/distributed/mccl_group_hang.c -I/usr/local/musa/include \
  -L/usr/local/musa/lib -l:libmccl.so.2 -lmusart \
  -Wl,-rpath,/usr/local/musa/lib

# Run with Bash. Two ranks are sufficient; use N=4 or N=8 to expand.
N=2
D=$(mktemp -d)
pids=()
for ((rank=0; rank<N; rank++)); do
  /tmp/mccl_group_hang "$rank" "$N" "$D/id" group 4096 64 \
    >"$D/rank-$rank.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if wait "$pid"; then printf 'unexpected grouped completion\n';
  else printf 'grouped exit=%s (124 is the observed hang)\n' "$?"; fi
done
cat "$D"/rank-*.log

# Byte-exact control: same payload, same rank count, no group.
D=$(mktemp -d)
pids=()
for ((rank=0; rank<N; rank++)); do
  /tmp/mccl_group_hang "$rank" "$N" "$D/id" nogroup 4096 64 \
    >"$D/rank-$rank.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit "$?"; done
cat "$D"/rank-*.log
```

The failure was measured with MCCL 2.30.4 in
`v0.32.0-ph1-5.2.0-torch2.11.0.post2-20261106`
(`sha256:65d5e96b5027c796494a7ad6597c6267953d60b96e2436c344760b243c346bc1`).
Do not infer that another MCCL or driver version has the same defect without
rerunning the failing case and its passing controls.

## Why DP+EP encounters it

The regular allgather/reduce-scatter MoE transport groups several dispatch
tensors: hidden activations, router weights, expert IDs and block scales.
For Qwen3-30B-A3B at two tokens per rank their payloads are 4096, 64, 64 and
128 bytes. Variable-rank all-gather/reduce-scatter also groups a broadcast
or reduction for every peer. Both patterns encounter the hang.

Normal library support is independent of this defect: `find_nccl_library()`
selects `libmccl.so.2` beside the existing CUDA/NCCL and ROCm/RCCL cases;
`PyNcclCommunicator` and all collective call sites remain the upstream APIs.

## Workaround and removal condition

Only the MUSA branch changes the collective implementation:

- Pack a batch of tensors into one byte matrix and perform one all-gather,
  rather than grouping one collective per tensor.
- Pad uneven rows to the largest rank size, run one all-gather or
  reduce-scatter, then compact the valid rows. Unused padding is zeroed.
- Validate row counts before communication; never let one rank silently
  choose a per-tensor fallback while peers choose the packed path.

The affected branches carry `# XXX (MUSA):` comments. CUDA and ROCm retain
their existing grouped implementations, including their public method
signatures. CPU contract tests execute these branch decisions; they do not
claim NVIDIA/AMD hardware coverage.

The workaround is not a fix inside MCCL. Re-enable grouping only after the
fixed MCCL version passes the failing and control matrices, uneven/idle-rank
cases and graph replay, followed by same-build end-to-end validation.

## Settings that did not avoid the failure

On the recorded test stack, forcing Simple/LL/LL128, Ring, lazy connection
settings, GROUP/PARALLEL launch modes, channel counts, symmetric-kernel
settings, and disabling MTLink or SHM did not avoid the failing case.
Disabling P2P failed communicator initialization, so it is inconclusive.
The MCCL debug log contains enqueue records for the grouped operations but
no algorithm/protocol/channel-selection record for them. That is a symptom,
not proof of an internal scheduling or channel-assignment defect.
