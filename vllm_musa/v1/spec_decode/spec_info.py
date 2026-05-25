# MUSA-0090 step 2: Pre-allocated buffers + capture-context dataclasses for
# the Eagle3 full-loop runner.
#
# All fields are tensors (not Python lists), so the captured cudagraph can
# mutate them in-place via tensor ops without Python re-entry. Buffers are
# sized for the MAX captured batch size; smaller batches use the leading
# prefix.
#
# Design ref: generated/musa0090_impl/step1-design-doc.md §3.1.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EagleDraftBuffers:
    """Pre-allocated buffers used by EagleFullLoopRunner across all N steps.

    Allocated once per (batch_size, num_speculative_tokens, hidden_size). During
    capture/replay, the runner indexes these buffers using static integer
    indices — no Python list growth, no per-step allocation.

    Shape conventions:
      max_bs: the largest cudagraph_capture_size for this runner
      num_steps: num_speculative_tokens (=N from the EagleProposer)
      hidden_size: draft model's hidden_size (M2.5 Eagle3 = LLaMA-shape)
      topk: 1 for chain drafting; >1 for tree drafting (deferred)
    """

    # --- Per-step output buffers (indexed by step_idx in [0..N-1]) ---
    # int32 [N, max_bs]: the chosen draft token id at each step
    input_ids_per_step: torch.Tensor
    # int64 [N, max_bs]: absolute sequence position used by attention at each step
    positions_per_step: torch.Tensor
    # int64 [N, max_bs]: KV cache slot index where this step's k/v is written
    slot_mapping_per_step: torch.Tensor
    # int32 [N, max_bs]: running seq_len at each step (= base_seq_len + step_idx)
    seq_lens_per_step: torch.Tensor

    # --- Per-step hidden states (the carry between draft forward passes) ---
    # bfloat16 [N, max_bs, hidden_size]: output of draft fwd at step i, used as
    # input at step i+1
    hidden_states_per_step: torch.Tensor

    # --- Per-step topk probs/indices (for sampling) ---
    # float32 [N, max_bs, topk]: softmax probs at each step
    topk_p_per_step: torch.Tensor
    # int32 [N, max_bs, topk]: token indices at each step
    topk_index_per_step: torch.Tensor

    # --- Output of the full-loop replay ---
    # int32 [max_bs, num_speculative_tokens]: the N drafted tokens per request,
    # in order. This is what the patched EagleProposer.propose() returns.
    draft_token_ids_out: torch.Tensor

    # --- Input carry-over from the previous spec round (target's verify output) ---
    # int32 [max_bs]: bonus token id from the previous verify pass (the target
    # model's accepted last token)
    bonus_token_ids_in: torch.Tensor
    # bfloat16 [max_bs, hidden_size]: target model's last hidden state, used as
    # input to step 0 of the draft loop
    target_hidden_states_in: torch.Tensor
    # MUSA-0090 step 5k (2026-05-16): step-0 input metadata seeded from the
    # caller's CommonAttentionMetadata. Without these, positions_per_step[0]
    # is zero (allocation default), and the captured graph reads zeros at
    # step 0 — RoPE applies at position 0 regardless of actual sequence
    # position. Downstream steps' positions/slot_mapping are then off by an
    # absolute offset (P-0), which is why MUSA-0090 step 5h showed
    # acceptance positions 3-6 = 0.00 (drafts at wrong positions).
    # int64 [max_bs]: actual position of the BONUS token (target's last
    # accepted output). Step 0 of the draft loop runs at this position.
    positions_in: torch.Tensor
    # int64 [max_bs]: KV cache slot for the BONUS token. Step 0 writes its
    # draft KV here.
    slot_mapping_in: torch.Tensor
    # int32 [max_bs]: current sequence length INCLUDING the bonus token.
    # Attention at step 0 attends to seq_lens_in tokens of context.
    seq_lens_in: torch.Tensor

    # --- Static / read-only references (allocated once, not per-step) ---
    # MUSA-0090 step 5l.3: block_table_tensor is OWNED by the buffer
    # (allocated once, copied-into per replay). NOT a reference to the
    # caller's block_table — that pointer becomes stale across spec rounds
    # as vllm reuses tensors, causing 'MUSA error: unknown error' during
    # captured graph replay.
    # Shape: [max_bs, max_blocks_per_req]. The eagle_step kernel reads from
    # this; the runner's replay() copies the current request's block_table
    # into it.
    block_table_tensor: torch.Tensor

    # --- Bookkeeping ---
    max_bs: int
    num_steps: int
    hidden_size: int
    topk: int
    device: torch.device

    @classmethod
    def allocate(
        cls,
        max_bs: int,
        num_steps: int,
        hidden_size: int,
        topk: int,
        block_table_tensor: torch.Tensor,
        device: torch.device,
    ) -> EagleDraftBuffers:
        """One-time allocation at runner setup (before any graph capture).

        MUSA-0090 step 5l.3: block_table_tensor is OWNED by the buffer
        (allocated once at max shape, copied-into per replay). The caller's
        reference is used only to derive the shape (max_blocks_per_req)
        and dtype. The captured graph reads from the buffer's
        block_table_tensor; the runner's replay() copies the current
        request's block_table into it before graph.replay().

        Sizes are stored on the device. The (N, max_bs) leading dims allow
        per-step slicing as a contiguous view (no copy).
        """
        bf16 = torch.bfloat16
        i32 = torch.int32
        i64 = torch.int64
        f32 = torch.float32

        # Allocate a runner-owned block_table buffer at the SHAPE of the
        # caller's template. The caller's block_table_tensor has shape
        # [num_reqs, max_blocks_per_req] (typically int32 — page table
        # entries). We allocate [max_bs, max_blocks_per_req].
        max_blocks_per_req = (
            block_table_tensor.shape[-1] if block_table_tensor.ndim >= 1 else 1
        )
        owned_block_table = torch.zeros(
            (max_bs, max_blocks_per_req),
            dtype=block_table_tensor.dtype,
            device=device,
        )

        return cls(
            input_ids_per_step=torch.zeros(
                (num_steps, max_bs), dtype=i32, device=device
            ),
            positions_per_step=torch.zeros(
                (num_steps, max_bs), dtype=i64, device=device
            ),
            slot_mapping_per_step=torch.zeros(
                (num_steps, max_bs), dtype=i64, device=device
            ),
            seq_lens_per_step=torch.zeros(
                (num_steps, max_bs), dtype=i32, device=device
            ),
            hidden_states_per_step=torch.zeros(
                (num_steps, max_bs, hidden_size), dtype=bf16, device=device
            ),
            topk_p_per_step=torch.zeros(
                (num_steps, max_bs, topk), dtype=f32, device=device
            ),
            topk_index_per_step=torch.zeros(
                (num_steps, max_bs, topk), dtype=i32, device=device
            ),
            draft_token_ids_out=torch.zeros(
                (max_bs, num_steps), dtype=i32, device=device
            ),
            bonus_token_ids_in=torch.zeros((max_bs,), dtype=i32, device=device),
            target_hidden_states_in=torch.zeros(
                (max_bs, hidden_size), dtype=bf16, device=device
            ),
            # MUSA-0090 step 5k: step-0 input metadata buffers.
            positions_in=torch.zeros((max_bs,), dtype=i64, device=device),
            slot_mapping_in=torch.zeros((max_bs,), dtype=i64, device=device),
            seq_lens_in=torch.zeros((max_bs,), dtype=i32, device=device),
            block_table_tensor=owned_block_table,
            max_bs=max_bs,
            num_steps=num_steps,
            hidden_size=hidden_size,
            topk=topk,
            device=device,
        )

    def reset(self) -> None:
        """Zero out per-step state. Called before each capture/replay round.

        Does NOT zero the input carry-over buffers (bonus_token_ids_in,
        target_hidden_states_in); those are populated by the caller via
        copy_() just before replay.
        """
        self.input_ids_per_step.zero_()
        self.positions_per_step.zero_()
        self.slot_mapping_per_step.zero_()
        self.seq_lens_per_step.zero_()
        # hidden_states / topk_p / topk_index are overwritten in-graph,
        # no need to zero them — but doing so makes debug-print outputs
        # cleaner. Skip for perf.
        self.draft_token_ids_out.zero_()

    def memory_footprint_bytes(self) -> int:
        """Sum of all owned tensors' sizes in bytes. Useful for runner sizing
        decisions (e.g., 'is this runner's footprint too big for the GPU?')."""
        tensors = [
            self.input_ids_per_step,
            self.positions_per_step,
            self.slot_mapping_per_step,
            self.seq_lens_per_step,
            self.hidden_states_per_step,
            self.topk_p_per_step,
            self.topk_index_per_step,
            self.draft_token_ids_out,
            self.bonus_token_ids_in,
            self.target_hidden_states_in,
            # MUSA-0090 step 5k: step-0 input metadata buffers.
            self.positions_in,
            self.slot_mapping_in,
            self.seq_lens_in,
            # MUSA-0090 step 5l.3: block_table_tensor is now owned.
            self.block_table_tensor,
        ]
        return sum(t.numel() * t.element_size() for t in tensors)


@dataclass
class EagleFullLoopCaptureContext:
    """Per-batch-size capture state. One per cudagraph_capture_size.

    The runner builds one of these per `cudagraph_capture_sizes` entry at
    setup, then dispatches by batch_size at runtime via a dict lookup.
    """

    batch_size: int
    """The exact batch size this graph was captured for."""

    graph: Any
    """The torch.cuda.CUDAGraph (or torch_musa.MUSAGraph on MUSA) holding the
    captured N-step draft loop. Replay via graph.replay()."""

    pool: Any
    """The memory pool handle returned by torch.cuda.graph_pool_handle(),
    shared with the target model's captured graphs to reduce memory."""

    buffers: EagleDraftBuffers
    """The buffers this graph reads/writes during replay. Persisted as long
    as the graph exists."""

    attn_metadata_array: list
    """List of N CommonAttentionMetadata objects (one per step), pre-built at
    capture time. Each holds READ-ONLY VIEWS into `buffers.{slot_mapping,
    seq_lens}_per_step[i]` so the captured graph's tensor mutations are
    reflected without rebuild."""

    def memory_footprint_bytes(self) -> int:
        """Memory used by this context's buffers (excludes graph memory pool —
        that's shared)."""
        return self.buffers.memory_footprint_bytes()


@dataclass
class EagleFullLoopReplayResult:
    """Returned from `EagleFullLoopRunner.replay()`.

    The runner deliberately returns a thin object (vs a bare tensor) so the
    patched EagleProposer.propose() has a stable shape to consume even if we
    later add fields (e.g., per-step accept probabilities for telemetry).
    """

    draft_token_ids: torch.Tensor
    """int32 [batch_size, num_speculative_tokens] — the N drafted token ids
    per request. View into `EagleDraftBuffers.draft_token_ids_out[:batch_size]`.
    Caller must NOT hold this view past the next replay (the underlying
    buffer is overwritten)."""

    batch_size: int
    """The actual batch size (≤ context.batch_size). Useful for the caller
    to slice correctly."""
