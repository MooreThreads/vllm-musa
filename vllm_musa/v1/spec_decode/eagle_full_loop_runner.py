# MUSA-0090 step 3: EagleFullLoopRunner — captures vLLM's N-step Eagle3 draft
# loop as ONE cudagraph per batch size (vs vLLM's iterative per-step PIECEWISE
# dispatch). Adapted from SGLang's EAGLEDraftCudaGraphRunner shape to vLLM types.
#
# Design ref: ../../../../generated/musa0090_impl/step1-design-doc.md §3.3
# Q1 cudagraph smoke (proved correctness): ../../../../generated/musa0090_impl/
#     step1_5-q1-cudagraph-smoke-result.md
#
# This file is the skeleton. The hot-path inner loop (_capture_one_batch_size)
# has fully-typed signatures + explicit step-by-step pseudocode for the
# step-4 implementation to fill in. Step 3's deliverable is the structural
# scaffolding so the patch can hook in.

from __future__ import annotations

import logging
from typing import Any

import torch

from .attn_backend_array import (
    StepMetadataIndexing,
    build_per_step_attn_metadata_array,
)
from .spec_info import (
    EagleDraftBuffers,
    EagleFullLoopCaptureContext,
    EagleFullLoopReplayResult,
)

logger = logging.getLogger(__name__)


class EagleFullLoopRunner:
    """Captures the full N-step Eagle3 draft loop as one cudagraph per batch size.

    Public surface:
        __init__(proposer, num_speculative_tokens, cudagraph_capture_sizes,
                 hidden_size, device)
        capture()                           # one-time at server boot
        can_run(batch_size) -> bool         # dispatcher predicate
        replay(target_hidden_states, next_token_ids, common_attn_metadata,
               batch_size) -> EagleFullLoopReplayResult

    Architecture (mirrors SGLang's EAGLEDraftCudaGraphRunner):
      1. At init: hold reference to vLLM's EagleProposer. Don't capture yet.
      2. On first proposer.propose() call (lazy): the patch invokes
         runner.capture() to build one graph per cudagraph_capture_sizes
         entry. Each graph captures the full N-step draft loop using buffers
         from spec_info.py + per-step metadata from attn_backend_array.py.
      3. At runtime: patched propose() checks runner.can_run(bs); if true,
         dispatch to runner.replay() — single graph.replay() + 2 .copy_()
         calls (the input carry-over). No Python loop.

    Critical invariants (from design doc §6):
      - All per-step state mutations as in-graph tensor ops only.
      - Attn-metadata array tensors are VIEWS into buffers (not rebuilt).
      - forward_batch is reused, not rebuilt.
      - Graph capture before any spec request (warm boot expected).
      - MUSA-only: no-op on CUDA (gated at the patch level).
    """

    # ---- construction ----

    def __init__(
        self,
        proposer: Any,  # vllm.v1.spec_decode.eagle.EagleProposer
        num_speculative_tokens: int,
        cudagraph_capture_sizes: list[int],
        hidden_size: int,
        device: torch.device,
        topk: int = 1,  # chain drafting; tree path is MUSA-0094's scope
    ):
        if num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}"
            )
        if not cudagraph_capture_sizes:
            raise ValueError("cudagraph_capture_sizes must not be empty")
        if topk != 1:
            # Tree drafting is a follow-up ticket (MUSA-0094); chain-only here.
            raise NotImplementedError(
                f"topk={topk} (tree drafting) is out of scope for MUSA-0090; "
                "chain-only path (topk=1). Tree path is MUSA-0094."
            )

        self.proposer = proposer
        self.num_steps = num_speculative_tokens
        self.capture_sizes: list[int] = sorted(set(cudagraph_capture_sizes))
        self.max_bs: int = max(self.capture_sizes)
        self.hidden_size = hidden_size
        self.topk = topk
        self.device = device

        # Per-batch-size capture state. Populated by capture(); read by replay().
        self.contexts: dict[int, EagleFullLoopCaptureContext] = {}
        self._captured: bool = False

        logger.info(
            "MUSA-0090 EagleFullLoopRunner: configured for "
            "num_speculative_tokens=%d, capture_sizes=%s, max_bs=%d, "
            "hidden_size=%d, topk=%d",
            self.num_steps,
            self.capture_sizes,
            self.max_bs,
            self.hidden_size,
            self.topk,
        )

    # ---- dispatcher predicate ----

    def can_run(self, batch_size: int) -> bool:
        """Whether there's a captured graph compatible with this batch size.

        Strict equality match for now (no padding to next capture size).
        Step 4 may add padding logic if batch size variance is high; for
        BS=1 4k/1k workloads (the /goal shape), exact match suffices.
        """
        if not self._captured:
            return False
        return batch_size in self.contexts

    # ---- one-time capture (called lazily on first propose()) ----

    def capture(self, base_metadata: Any) -> None:
        """Capture one cudagraph per entry in self.capture_sizes.

        Called lazily on the first propose() call. `base_metadata` is the
        CommonAttentionMetadata vLLM would otherwise pass to the iterative
        propose loop; used as the template for per-step metadata.

        After this returns successfully, self._captured is True and
        self.contexts[bs] is populated for every bs in self.capture_sizes.

        Idempotent: subsequent calls are no-ops if already captured.
        """
        if self._captured:
            return
        if not hasattr(self.proposer, "model"):
            raise RuntimeError(
                "EagleProposer does not expose `model` attribute; check vLLM "
                "version compatibility (expected v0.20.1.dev0 shape — the "
                "draft model is exposed as `proposer.model`, NOT "
                "`proposer.draft_model`)."
            )

        # MUSA-0109 spike: use the platform's GLOBAL graph pool (lazily
        # created, shared across target + draft + spec captures). This
        # mirrors sglang's `device_module.graph_pool_handle()` shared-pool
        # pattern, which is the working reference on MUSA. The prior
        # per-runner `torch.cuda.graph_pool_handle()` created a SEPARATE
        # pool, and replay across distinct pools triggered the
        # torch_musa 2.9.0 allocator bug (`MUSA error: unknown error`
        # at capture/replay). Reusing vllm's existing global pool avoids
        # cross-pool interactions during replay.
        pool = None
        try:
            from vllm.platforms import current_platform

            pool = current_platform.get_global_graph_pool()
        except Exception as exc:
            logger.warning(
                "MUSA-0109: failed to acquire vllm global graph pool (%s); "
                "will fall back to a fresh per-runner pool.",
                exc,
            )
        # Validate the returned handle BEFORE using it for capture/replay.
        # A platform that doesn't actually implement get_global_graph_pool
        # may return None (or an invalid sentinel). Passing None through to
        # torch.cuda.graph(..., pool=...) means the capture allocates from
        # the default pool — silently breaking the shared-pool guarantee
        # that the cross-pool replay allocator bug (PR #41 review comment).
        if pool is None:
            pool = torch.cuda.graph_pool_handle()
            logger.warning(
                "MUSA-0109: get_global_graph_pool() returned None; "
                "falling back to per-runner pool (may trigger torch_musa "
                "allocator bug). pool=%s",
                pool,
            )
        else:
            logger.info(
                "MUSA-0109: EagleFullLoopRunner using vllm GLOBAL graph pool "
                "(shared with target model captures); pool=%s",
                pool,
            )

        for bs in self.capture_sizes:
            try:
                ctx = self._capture_one_batch_size(bs, base_metadata, pool)
                self.contexts[bs] = ctx
                logger.info(
                    "MUSA-0090 captured Eagle3 full-loop graph for bs=%d "
                    "(buffer footprint %.2f MiB)",
                    bs,
                    ctx.memory_footprint_bytes() / 1024 / 1024,
                )
            except Exception as exc:
                logger.exception(
                    "MUSA-0090 graph capture FAILED at bs=%d: %s. "
                    "Falling back to iterative path for this size.",
                    bs,
                    exc,
                )
                # Don't propagate — the patch will fall back to vLLM's
                # iterative path when can_run(bs) returns False for this size.

        self._captured = True

    def _capture_one_batch_size(
        self,
        batch_size: int,
        base_metadata: Any,
        pool: Any,
    ) -> EagleFullLoopCaptureContext:
        """Capture the N-step draft loop for one batch size.

        STEP 4 implementation target. Step 3 leaves this as a structured
        skeleton with explicit pseudocode for what each phase must do.

        Phase 1: allocate buffers
        Phase 2: build per-step metadata array (views into buffers)
        Phase 3: warm-up the draft model on this bs (kernels must be
                 compiled before graph capture per torch.cuda.graph docs)
        Phase 4: open torch.cuda.graph context, drive the N-step loop with
                 in-place tensor ops only
        Phase 5: wrap and return context
        """
        if batch_size <= 0 or batch_size > self.max_bs:
            raise ValueError(f"batch_size {batch_size} out of range [1, {self.max_bs}]")

        # Phase 1: allocate buffers for this batch size.
        # block_table_tensor is held as a reference (not copied); we read it
        # via the slot-mapping kernel inside the captured graph.
        block_table_tensor = getattr(base_metadata, "block_table_tensor", None)
        if block_table_tensor is None:
            raise RuntimeError(
                "base_metadata lacks block_table_tensor; cannot capture without it"
            )
        buffers = EagleDraftBuffers.allocate(
            max_bs=self.max_bs,
            num_steps=self.num_steps,
            hidden_size=self.hidden_size,
            topk=self.topk,
            block_table_tensor=block_table_tensor,
            device=self.device,
        )

        # Phase 2: pre-build per-step metadata array (views into buffers).
        # Pass `proposer` so the helper can call
        # proposer.build_per_group_and_layer_attn_metadata() to produce
        # per-layer backend-specific metadata that has the use_cascade /
        # common_prefix_len / etc. fields the compiled draft model expects
        # (see attn_backend_array.py docstring for the rationale).
        attn_metadata_array = build_per_step_attn_metadata_array(
            base_metadata=base_metadata,
            buffers=buffers,
            batch_size=batch_size,
            proposer=self.proposer,
        )
        # Verify the indexing math before committing to graph capture.
        # Skip verify_strict for per-layer dict metadata (StepMetadataIndexing
        # reads `.slot_mapping.data_ptr()` which is a CommonAttentionMetadata
        # field, not a per-layer-dict attribute).
        if attn_metadata_array and not isinstance(attn_metadata_array[0], dict):
            StepMetadataIndexing.inspect(attn_metadata_array).verify_strict()

        # Phase 3: warm-up. STEP 4 TODO — run one eager iteration to compile
        # kernels and exercise allocator hot-paths. Per torch.cuda.graph docs,
        # this is required to avoid stream/allocator surprises during capture.
        # Pattern (from Q1 smoke):
        #   s = torch.cuda.Stream()
        #   s.wait_stream(torch.cuda.current_stream())
        #   with torch.cuda.stream(s):
        #       self._run_one_step_eager(buffers, attn_metadata_array, 0,
        #                                 batch_size)
        #   torch.cuda.current_stream().wait_stream(s)
        #   torch.cuda.synchronize()
        self._warmup_eager(buffers, attn_metadata_array, batch_size)

        # Phase 4: capture the N-step loop. STEP 4 TODO — this is the
        # core deliverable. Pseudocode:
        #
        #   graph = torch.cuda.CUDAGraph()
        #   with torch.cuda.graph(graph, pool=pool):
        #       # Step 0: seed from input carry-over (target last hidden).
        #       # Copy bonus_token_ids_in -> input_ids_per_step[0]
        #       # Copy target_hidden_states_in -> hidden_states_per_step[0]
        #       buffers.input_ids_per_step[0, :batch_size].copy_(
        #           buffers.bonus_token_ids_in[:batch_size]
        #       )
        #       buffers.hidden_states_per_step[0, :batch_size].copy_(
        #           buffers.target_hidden_states_in[:batch_size]
        #       )
        #
        #       # The N-step loop body.
        #       for step_idx in range(self.num_steps):
        #           # Set forward_batch state from buffers[step_idx].
        #           # vLLM passes forward_batch through set_forward_context;
        #           # we DON'T use set_forward_context here because we
        #           # want the loop INSIDE the graph (no Python re-entry).
        #           forward_batch.input_ids = buffers.input_ids_per_step[step_idx, :batch_size]
        #           forward_batch.positions = buffers.positions_per_step[step_idx, :batch_size]
        #           forward_batch.slot_mapping = ... (from metadata_array[step_idx])
        #           forward_batch.attn_metadata = attn_metadata_array[step_idx]
        #
        #           # Call draft model forward (captures all its kernels).
        #           logits, hidden = self.proposer.draft_model(forward_batch)
        #           buffers.hidden_states_per_step[step_idx + 1].copy_(hidden)
        #
        #           # Sample topk=1 (chain). Captured in-graph.
        #           probs = torch.softmax(logits, dim=-1)
        #           topk_p, topk_idx = torch.topk(probs, k=self.topk)
        #           buffers.topk_p_per_step[step_idx + 1].copy_(topk_p)
        #           buffers.topk_index_per_step[step_idx + 1].copy_(topk_idx)
        #           buffers.input_ids_per_step[step_idx + 1, :batch_size].copy_(
        #               topk_idx.flatten()
        #           )
        #
        #           # Update positions + slot_mapping for next step via the
        #           # existing fused Triton kernel (already MUSA-adapted
        #           # for the .pyc-cache issue we hit in MUSA-0089).
        #           eagle_step_update_slot_mapping_and_metadata(
        #               positions_1d=buffers.positions_per_step[step_idx + 1],
        #               block_table_tensor=buffers.block_table_tensor,
        #               seq_lens=buffers.seq_lens_per_step[step_idx + 1],
        #               out_slot_mapping=buffers.slot_mapping_per_step[step_idx + 1],
        #               input_batch_size=batch_size,
        #               ...
        #           )
        #
        #           # Write the step's chosen token to the output tensor.
        #           buffers.draft_token_ids_out[:batch_size, step_idx].copy_(
        #               buffers.input_ids_per_step[step_idx + 1, :batch_size]
        #           )
        #
        #   # End of context manager = graph capture complete.
        graph = self._capture_n_step_loop(
            buffers,
            attn_metadata_array,
            batch_size,
            pool,
        )

        # Phase 5: wrap up and return.
        return EagleFullLoopCaptureContext(
            batch_size=batch_size,
            graph=graph,
            pool=pool,
            buffers=buffers,
            attn_metadata_array=attn_metadata_array,
        )

    # ---- replay (hot path) ----

    def replay(
        self,
        target_hidden_states: torch.Tensor,  # bf16 [bs, hidden_size] OR [num_tokens, hidden_size]
        next_token_ids: torch.Tensor,  # int32 [bs] (the bonus from verify)
        common_attn_metadata: Any,  # for assertion + step-0 metadata seeding
        batch_size: int,
        target_positions: (
            torch.Tensor | None
        ) = None,  # int64 [num_tokens] (the verify pass's positions)
        token_indices_to_sample: (
            torch.Tensor | None
        ) = None,  # int [bs] (idx in num_tokens of last token per seq)
    ) -> EagleFullLoopReplayResult:
        """Replay the captured graph for this batch size.

        Single .replay() call + N .copy_() for input carry-over (5 buffers:
        hidden_states, bonus_token_ids, positions, slot_mapping, seq_lens).
        Versus vLLM's iterative path which pays N × (set_forward_context +
        metadata rebuild + dispatcher dispatch) per spec round.

        Args:
            target_hidden_states: target model's hidden states, shape
                [num_tokens, hidden_size] OR [batch_size, hidden_size].
                The runner selects the per-sequence-last-token slice via
                token_indices_to_sample.
            next_token_ids: bonus token ids from the previous verify pass,
                shape [batch_size]. The "free" token that target accepted.
            common_attn_metadata: source of step-0 slot_mapping and seq_lens
                seeding. The runner reads .slot_mapping[token_indices_to_sample]
                and .seq_lens directly.
            batch_size: actual batch size; must match a captured size.
            target_positions: optional [num_tokens] tensor of the verify pass's
                positions. The runner reads target_positions[token_indices_to_sample]
                to seed step-0 positions. If None, step-0 positions are left
                at the buffer's current value (legacy behavior — wrong, but
                preserved for backward compat during step 5k transition).
            token_indices_to_sample: optional [bs] tensor naming the index
                (within num_tokens) of the last token per sequence. If None,
                computed from common_attn_metadata.query_start_loc[1:] - 1.

        Returns:
            EagleFullLoopReplayResult with .draft_token_ids of shape
            [batch_size, num_speculative_tokens].
        """
        if not self._captured:
            raise RuntimeError("replay() called before capture()")
        if batch_size not in self.contexts:
            raise ValueError(
                f"no captured graph for batch_size={batch_size}; "
                f"capture_sizes={self.capture_sizes}"
            )

        ctx = self.contexts[batch_size]

        # MUSA-0090 step 5k: derive token_indices_to_sample if not provided.
        # Pattern matches vllm/v1/spec_decode/llm_base_proposer.py:set_inputs_first_pass.
        if token_indices_to_sample is None:
            query_start_loc = getattr(common_attn_metadata, "query_start_loc", None)
            if query_start_loc is not None:
                token_indices_to_sample = query_start_loc[1:] - 1
            else:
                # Fall back to "[bs-1, ..., bs-1]" which is correct ONLY when
                # num_tokens == batch_size (one-token-per-seq decode case).
                # The runner's eager path produces this from the proposer.
                token_indices_to_sample = torch.arange(
                    batch_size, dtype=torch.int64, device=target_hidden_states.device
                )

        # Hidden state selection: pick the LAST token per sequence.
        # Shape handling:
        #   target_hidden_states is [num_tokens, h] in the general case.
        #   For BS=1 prefill 4k, num_tokens=4096 and we need the [4095]-th row.
        # MUSA-0090 step 5l.2: Eagle3 uses aux hidden states from N layers
        # (e.g., 3 layers for M2.5-Eagle3: ids 1, 30, 58), concatenated to
        # shape [num_tokens, N * hidden_size] = [num_tokens, 9216] for the
        # 3072-hidden M2.5 draft. vLLM's propose() calls
        # self.model.combine_hidden_states(target_hidden_states) to reduce
        # the concat to [num_tokens, hidden_size] before the draft forward.
        # We replicate that here OUTSIDE the captured graph.
        if (
            self.proposer.method in ("eagle3", "dflash")
            and target_hidden_states.shape[-1] != self.hidden_size
            and hasattr(self.proposer.model, "combine_hidden_states")
        ):
            target_hidden_states = self.proposer.model.combine_hidden_states(
                target_hidden_states
            )
        if (
            target_hidden_states.dim() >= 1
            and target_hidden_states.shape[0] != batch_size
        ):
            selected_hidden = target_hidden_states[token_indices_to_sample]
        else:
            selected_hidden = target_hidden_states[:batch_size]
        ctx.buffers.target_hidden_states_in[:batch_size].copy_(selected_hidden)
        ctx.buffers.bonus_token_ids_in[:batch_size].copy_(next_token_ids)

        # MUSA-0090 step 5k: seed step-0 positions, slot_mapping, seq_lens
        # from the verify pass's metadata. Without these, the captured graph
        # reads zeros at step 0 — RoPE wrong, KV writes to wrong slots,
        # attention attends to wrong context length.
        if target_positions is not None:
            # The bonus token's position is the position AFTER the last verified
            # position. The first draft (step 0) is run AS the bonus token (its
            # input_ids = bonus_token_id, its hidden_states = target_hidden), so
            # step-0 positions = target_positions[last_idx_per_seq].
            # (Per vllm/v1/spec_decode/llm_base_proposer.py line 502:
            #  `positions = self.positions[token_indices_to_sample]`)
            if target_positions.dim() == 2:
                # M-RoPE case: positions shape [3, num_tokens]; take dim 0.
                step0_positions = target_positions[0][token_indices_to_sample]
            else:
                step0_positions = target_positions[token_indices_to_sample]
            ctx.buffers.positions_in[:batch_size].copy_(step0_positions.to(torch.int64))

        # slot_mapping for step 0: the slot of the bonus token. The verify
        # pass's common_attn_metadata.slot_mapping has shape [num_tokens] and
        # tells us where each input token's K/V is written. The bonus token
        # corresponds to the last input position of each sequence.
        cm_slot_mapping = getattr(common_attn_metadata, "slot_mapping", None)
        if cm_slot_mapping is not None and cm_slot_mapping.numel() > 0:
            if cm_slot_mapping.shape[0] != batch_size:
                step0_slot = cm_slot_mapping[token_indices_to_sample]
            else:
                step0_slot = cm_slot_mapping[:batch_size]
            ctx.buffers.slot_mapping_in[:batch_size].copy_(step0_slot.to(torch.int64))

        # seq_lens for step 0: same as the verify pass's seq_lens (the
        # context length BEFORE the draft adds a new token).
        cm_seq_lens = getattr(common_attn_metadata, "seq_lens", None)
        if cm_seq_lens is not None and cm_seq_lens.numel() > 0:
            ctx.buffers.seq_lens_in[:batch_size].copy_(
                cm_seq_lens[:batch_size].to(torch.int32)
            )

        # MUSA-0090 step 5l.3: copy the current request's block_table into
        # the runner-owned buffer. The captured graph reads from
        # buffers.block_table_tensor (a stable pointer), not from the
        # caller's transient tensor (which may be freed/repurposed between
        # spec rounds, causing 'MUSA error: unknown error').
        cm_block_table = getattr(common_attn_metadata, "block_table_tensor", None)
        if cm_block_table is not None and cm_block_table.numel() > 0:
            # Slice to actual shape, pad with zeros if needed.
            src_bs = min(cm_block_table.shape[0], batch_size)
            src_n_blocks = min(
                cm_block_table.shape[1], ctx.buffers.block_table_tensor.shape[1]
            )
            ctx.buffers.block_table_tensor[:src_bs, :src_n_blocks].copy_(
                cm_block_table[:src_bs, :src_n_blocks]
            )

        # Single graph replay. All N draft forwards + sampling + slot-mapping
        # updates happen inside this one call.
        ctx.graph.replay()

        # MUSA-0090 layer-2 fix (2026-05-17): clone the output OUT of the
        # CUDAGraph memory pool. vllm's _copy_draft_token_ids_to_cpu does the
        # H2D copy on a dedicated `draft_token_ids_copy_stream` (not the default
        # stream), and MUSA's MUDNN fails ("err 999 = unknown error") when
        # copying from pool memory across streams. Cloning into a fresh
        # allocation outside the pool fixes this. The clone is small
        # (`[bs, num_steps]` int32 = ~12 bytes for bs=1) and runs on the
        # default stream so subsequent cross-stream sync works.
        out = ctx.buffers.draft_token_ids_out[:batch_size].clone()
        return EagleFullLoopReplayResult(
            draft_token_ids=out,
            batch_size=batch_size,
        )

    # ---- helpers (step 4: real implementations) ----

    def _warmup_eager(
        self,
        buffers: EagleDraftBuffers,
        attn_metadata_array: list,
        batch_size: int,
    ) -> None:
        """Run TWO eager passes through the N-step loop with full sync between,
        BEFORE graph capture.

        MUSA-0109 layer-4 fix (2026-05-17): match SGLang's `_capture_init`
        pattern (sglang/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py:217)
        which is the difference between "MUSA-0090 always crashes" and
        "SGLang works on similar hardware". Specifically:

          1. torch.cuda.synchronize() — drain all pending GPU work
          2. tp_group.barrier() — cross-rank sync; ensures all TP ranks
             see the same allocator state before next warmup
          3. run_once_fn() — actually exercise the workload
          4. on_after_cuda_graph_warmup hook — backend-specific cleanup
          5. REPEAT 2x — settle JIT/Inductor compiles, allocator pool
             pages, and TP comm channels fully BEFORE capture

        Hypothesis: torch_musa's CUDAGraph + allocator interaction bug
        triggers when capture happens with un-settled allocator state.
        SGLang's 2x warmup + TP barrier pattern ensures the state is
        stable before capture opens.
        """
        from vllm.distributed import get_tp_group

        try:
            tp_group = get_tp_group()
            tp_barrier = lambda: tp_group.barrier()
        except Exception as exc:
            logger.debug("TP group unavailable for warmup barrier: %s", exc)
            tp_barrier = lambda: None

        # Side-stream warmup pattern (preserved from Q1 smoke)
        s = torch.cuda.Stream()
        for warmup_iter in range(2):
            torch.cuda.synchronize()
            tp_barrier()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._run_n_step_inner(buffers, attn_metadata_array, batch_size)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            # Per-attn-backend hook (SGLang's on_after_cuda_graph_warmup)
            hook = getattr(
                getattr(self.proposer, "draft_attn_backend", None),
                "on_after_cuda_graph_warmup",
                None,
            )
            if hook is not None:
                try:
                    hook()
                except Exception as exc:
                    logger.debug("on_after_cuda_graph_warmup raised: %s", exc)
        # Final barrier + sync to ensure all ranks see settled state
        torch.cuda.synchronize()
        tp_barrier()

    def _capture_n_step_loop(
        self,
        buffers: EagleDraftBuffers,
        attn_metadata_array: list,
        batch_size: int,
        pool: Any,
    ) -> Any:
        """Open the torch.cuda.graph context and drive the N-step loop with
        in-place tensor ops only. Returns the captured CUDAGraph object.

        Critical invariants enforced inside `_run_n_step_inner`:
          - No Python list growth (no .append, no torch.cat).
          - No new tensor allocations (everything writes into `buffers`).
          - No tensor.item() / .tolist() / .cpu() calls (would CUDA-sync).
          - set_forward_context is allowed (pure Python state mutation; the
            actual GPU work it wraps is captured by torch.cuda.graph).
        """
        # torch.cuda.CUDAGraph is routed to torch_musa.musa_graph.MUSAGraph on
        # MUSA via torchada — proven correctness-wise by the Q1 smoke.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            self._run_n_step_inner(buffers, attn_metadata_array, batch_size)
        return graph

    def _run_n_step_inner(
        self,
        buffers: EagleDraftBuffers,
        attn_metadata_array: list,
        batch_size: int,
    ) -> None:
        """The N-step draft loop body. Called both by _warmup_eager (outside
        graph) and _capture_n_step_loop (inside graph). MUST use only
        in-place tensor ops + tensor-view assignments that the graph can
        capture.

        This mirrors vLLM's `EagleProposer.propose()` loop body
        (vllm/v1/spec_decode/llm_base_proposer.py:554-651) but
        statically-allocates everything ahead of time.
        """
        # Lazy imports to avoid pulling vLLM internals at module-load time.
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.spec_decode.utils import (
            eagle_step_update_slot_mapping_and_metadata,
        )

        proposer = self.proposer
        # Step-0 seed: copy the input carry-over into the per-step buffers.
        # bonus_token_ids_in is the bonus token from the previous verify pass
        # (the target's accepted last token); it's the input to the first
        # draft forward.
        buffers.input_ids_per_step[0, :batch_size].copy_(
            buffers.bonus_token_ids_in[:batch_size]
        )
        buffers.hidden_states_per_step[0, :batch_size].copy_(
            buffers.target_hidden_states_in[:batch_size]
        )
        # MUSA-0090 step 5k (2026-05-16): seed step-0 metadata from input
        # buffers populated by replay(). Without these copies, the captured
        # graph reads zeros at step 0 — wrong RoPE positions, wrong KV slot,
        # wrong attention seq_len. The eagle_step kernel below propagates
        # positions[step] -> positions[step+1] by adding 1; so position 0
        # wrong cascades through all downstream steps. This was the root
        # cause of acceptance=0.00 at positions 3-6 in step 5h bench.
        buffers.positions_per_step[0, :batch_size].copy_(
            buffers.positions_in[:batch_size]
        )
        buffers.slot_mapping_per_step[0, :batch_size].copy_(
            buffers.slot_mapping_in[:batch_size]
        )
        buffers.seq_lens_per_step[0, :batch_size].copy_(
            buffers.seq_lens_in[:batch_size]
        )

        # The N-step loop. Inside torch.cuda.graph(), every GPU op below is
        # captured. The Python for-loop unrolls at trace time — N is a static
        # int, not a tensor.
        for step in range(self.num_steps):
            # Views into this step's slice of the buffers. .narrow-style
            # indexing on a contiguous tensor returns a view; the captured
            # graph captures the kernel launches that consume these views.
            input_ids_view = buffers.input_ids_per_step[step, :batch_size]
            positions_view = buffers.positions_per_step[step, :batch_size]
            hidden_view = buffers.hidden_states_per_step[step, :batch_size]
            slot_mapping_view = buffers.slot_mapping_per_step[step, :batch_size]

            # Build the model kwargs. vLLM's draft model signature:
            #   model(input_ids=..., positions=..., inputs_embeds=None,
            #         hidden_states=...)
            # Returns either (last_hidden, hidden) tuple or just last_hidden.
            model_kwargs = {
                "input_ids": input_ids_view,
                "positions": positions_view,
                "inputs_embeds": None,
            }
            if getattr(proposer, "pass_hidden_states_to_model", True):
                # Eagle3 always passes hidden states from the previous step.
                model_kwargs["hidden_states"] = hidden_view

            # set_forward_context sets vLLM's thread-local context (attn
            # metadata, slot_mapping, num_tokens) for the model call.
            # Critical: cudagraph_runtime_mode=NONE — we're capturing OUR
            # OWN graph; vLLM's PIECEWISE shouldn't re-fire.
            # NOTE: do NOT pass slot_mapping=tensor here — vllm's forward_context
            # internally does `slot_mapping or {}` which raises on multi-element
            # Tensors (`Boolean value of Tensor with more than one value is
            # ambiguous`). The slot_mapping_view is already carried by
            # attn_metadata_array[step].slot_mapping; the kwarg is the per-attn-group
            # dict that vllm uses for cross-group dispatch, NOT the single tensor.
            with set_forward_context(
                attn_metadata_array[step],
                proposer.vllm_config,
                num_tokens=batch_size,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                ret_hidden_states = proposer.model(**model_kwargs)
                if isinstance(ret_hidden_states, tuple):
                    last_hidden_states, next_hidden = ret_hidden_states
                else:
                    last_hidden_states = ret_hidden_states
                    next_hidden = ret_hidden_states

            # Sample greedy via the proposer's helper (lm_head + argmax).
            # Result dtype is int64 by default; vllm casts to int32 because
            # "tensor.argmax() returns int64 by default" and Eagle compile
            # requires int32 (per vllm comment line 556).
            draft_token_ids = proposer._greedy_sample(
                last_hidden_states[:batch_size]
            ).to(torch.int32)

            # Write the drafted token to the output tensor.
            buffers.draft_token_ids_out[:batch_size, step].copy_(draft_token_ids)

            # Update state for step+1 if not the last step.
            if step + 1 < self.num_steps:
                # Stage next step's input.
                buffers.input_ids_per_step[step + 1, :batch_size].copy_(draft_token_ids)
                buffers.hidden_states_per_step[step + 1, :batch_size].copy_(
                    next_hidden[:batch_size]
                )

                # MUSA-0090 step 5k: propagate seq_lens forward before the
                # kernel. The kernel does in-place +1 on its `seq_lens` arg
                # (reads, adds 1, writes back). If we passed
                # seq_lens_per_step[step+1] directly without seeding, the
                # kernel reads 0 and writes 1 — instead of reading
                # actual_seq_lens+step and writing actual_seq_lens+step+1.
                # Copy step's seq_lens to step+1's, then the kernel's
                # in-place +1 produces the correct step+1 value.
                buffers.seq_lens_per_step[step + 1, :batch_size].copy_(
                    buffers.seq_lens_per_step[step, :batch_size]
                )
                # Increment positions + recompute slot_mapping for step+1.
                # The fused kernel writes positions, seq_lens, slot_mapping
                # all in one launch — exactly what vLLM's iterative path
                # does at line 569-578 of llm_base_proposer.py.
                eagle_step_update_slot_mapping_and_metadata(
                    positions_1d=positions_view,
                    block_table_tensor=buffers.block_table_tensor,
                    seq_lens=buffers.seq_lens_per_step[step + 1, :batch_size],
                    block_size=getattr(proposer, "block_size", 16),
                    max_model_len=getattr(proposer, "max_model_len", 196_608),
                    out_clamped_positions=buffers.positions_per_step[
                        step + 1, :batch_size
                    ],
                    out_slot_mapping=buffers.slot_mapping_per_step[
                        step + 1, :batch_size
                    ],
                    input_batch_size=batch_size,
                )

    # ---- memory accounting ----

    def memory_footprint_bytes(self) -> int:
        """Total memory used by all captured contexts (buffer-only, excludes
        the shared graph memory pool)."""
        return sum(c.memory_footprint_bytes() for c in self.contexts.values())

    # ---- introspection ----

    def __repr__(self) -> str:
        return (
            f"EagleFullLoopRunner("
            f"N={self.num_steps}, "
            f"capture_sizes={self.capture_sizes}, "
            f"captured={self._captured}, "
            f"num_contexts={len(self.contexts)}, "
            f"hidden_size={self.hidden_size}, "
            f"topk={self.topk})"
        )
