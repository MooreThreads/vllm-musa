# MUSA-0090 step 2: Pre-allocated per-step attention metadata array.
#
# The TRICK that lets vLLM's iterative Eagle3 draft loop be captured as one
# cudagraph: pre-build N copies of CommonAttentionMetadata at capture time
# (one per spec step), wired so each step's tensors are VIEWS into the
# pre-allocated `EagleDraftBuffers`. The captured graph mutates the buffers
# via in-place kernels, and the metadata views see those mutations without
# any Python rebuild.
#
# Reference SGLang pattern: `draft_attn_backend.attn_backends[i]` —
# sglang/python/sglang/srt/speculative/eagle_worker.py:904
#
# Design ref: generated/musa0090_impl/step1-design-doc.md §3.2.

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .spec_info import EagleDraftBuffers


def build_per_step_attn_metadata_array(
    base_metadata: Any,  # CommonAttentionMetadata; typed Any to avoid hard dep at module-load
    buffers: EagleDraftBuffers,
    batch_size: int,
    proposer: Any = None,  # vllm.v1.spec_decode.eagle.EagleProposer (optional)
) -> list[Any]:
    """Pre-allocate N attn-metadata objects whose tensors point into `buffers`.

    The arithmetic that vLLM does in Python at each loop iteration in
    `EagleProposer.propose()` (max_seq_len += 1, slot_mapping recompute,
    seq_lens += 1) is unrolled here:

      - max_seq_len: bumped statically by step index at metadata-build time
      - slot_mapping: VIEW into buffers.slot_mapping_per_step[step_idx, :batch_size]
                      (mutated by the eagle_step_update_slot_mapping_and_metadata
                      kernel inside the captured graph)
      - seq_lens:     VIEW into buffers.seq_lens_per_step[step_idx, :batch_size]
                      (mutated by the same kernel)

    Returns a list of `buffers.num_steps` CommonAttentionMetadata-shaped objects.
    The runner indexes this list by step number inside its capture loop:
      forward_batch.attn_metadata = metadata_array[step_idx]

    Args:
        base_metadata: The first-step attention metadata that vLLM normally
            passes to EagleProposer.propose(). Used as the template; per-step
            metadata objects copy its non-mutable fields (block_table_tensor,
            cu_seq_lens base, query_start_loc, etc.) and override the
            mutable ones (slot_mapping, seq_lens, max_seq_len).
        buffers: The pre-allocated draft buffers. Per-step views are taken
            into `slot_mapping_per_step` and `seq_lens_per_step`.
        batch_size: The actual batch size for this capture context (≤ buffers.max_bs).

    Returns:
        A list of length buffers.num_steps. Each element is a metadata object
        whose tensor fields point into `buffers`. Order matches the spec
        step order: metadata_array[0] is for the first draft fwd, [N-1] is
        for the last.
    """
    if buffers.num_steps <= 0:
        return []

    # MUSA-0090 step 5e: when the proposer is provided, use its
    # build_per_group_and_layer_attn_metadata() to produce per-layer
    # backend-specific metadata (e.g., FlashAttentionMetadata with the
    # use_cascade, common_prefix_len, etc. fields the compiled draft
    # model's forward expects). The compiled forward reads
    # get_forward_context().attn_metadata and accesses .use_cascade
    # unconditionally; passing the raw CommonAttentionMetadata fails
    # with AttributeError.
    # MUSA-0090 step 5g: disable the per-layer build path. Although it
    # provides use_cascade etc., the resulting metadata's seq_lens has
    # a shape that flash_attn rejects ('seqused_k must have shape
    # (batch_size,)'). Instead, use the simpler dataclasses.replace path
    # below and setattr() the flag-fields the compiled forward expects.
    use_per_layer = False  # was: proposer is not None and hasattr(...)
    if use_per_layer:
        metadata_array: list[Any] = []
        for step_idx in range(buffers.num_steps):
            # Mutate base_metadata in place to reflect the step's state
            # (slot_mapping, seq_lens, max_seq_len), then build per-layer
            # metadata. The per-layer call returns a dict-shaped object
            # that the model reads via get_forward_context().
            base_metadata.slot_mapping = buffers.slot_mapping_per_step[
                step_idx, :batch_size
            ]
            base_metadata.seq_lens = buffers.seq_lens_per_step[step_idx, :batch_size]
            base_metadata.max_seq_len = int(
                getattr(base_metadata, "max_seq_len", 0)
            ) + (1 if step_idx > 0 else 0)
            try:
                _, per_layer = proposer.build_per_group_and_layer_attn_metadata(
                    base_metadata, draft_index=step_idx
                )
                metadata_array.append(per_layer)
            except Exception as exc:
                raise RuntimeError(
                    f"step {step_idx}: proposer.build_per_group_and_layer_attn_metadata "
                    f"failed: {type(exc).__name__}: {exc}"
                ) from exc
        return metadata_array

    # MUSA-0090 reopen (2026-05-16): port SGLang's pattern of pre-allocating
    # backend-specific metadata at capture time, mutating in-place at replay
    # time. Per `[[sglang-eagle3-musa-30pct-confirmed]]` memory + research
    # in `sglang/python/sglang/srt/layers/attention/flashattention_backend.py`
    # at `FlashAttentionMultiStepBackend.init_forward_metadata_capture_cuda_graph`
    # (line 2653), SGLang builds N-1 fully-instantiated backend objects, one
    # per draft step beyond step 0, each with its own pre-allocated metadata.
    #
    # vLLM's equivalent is to call `FlashAttentionMetadataBuilder.build(
    # common_prefix_len=0, common_attn_metadata=<step-specific cam>,
    # fast_build=True)` once per step. The `fast_build=True` flag disables
    # AOT scheduling (the comment in vllm/v1/attention/backends/flash_attn.py
    # line 388 says: "Disables AOT scheduling, used when there will be few
    # iterations i.e. spec-decode"), which keeps the metadata cudagraph-safe
    # by skipping the scheduler-metadata path that requires host-side
    # reductions.
    #
    # The previous setattr+dataclasses.replace workaround (commit f1d09d7)
    # is structurally wrong because `CommonAttentionMetadata` is missing the
    # backend-specific fields (`block_table`, `seqused_k`, etc.) that the
    # compiled draft forward unconditionally reads. The right answer is to
    # produce real `FlashAttentionMetadata` via the builder.
    metadata_array: list[Any] = []
    base_max_seq_len = int(getattr(base_metadata, "max_seq_len", 0))
    base_max_model_len = int(
        getattr(base_metadata, "max_model_len", base_max_seq_len + buffers.num_steps)
    )

    # MUSA-0090 step 5l (2026-05-16): use vllm's first-party
    # `proposer.build_per_group_and_layer_attn_metadata(common_attn_metadata,
    # draft_index=step_idx)` API (vllm/v1/spec_decode/llm_base_proposer.py:822).
    # This calls `attn_group.get_metadata_builder().build_for_drafting()` per
    # draft_attn_group and produces a per-layer dict that the captured model
    # forward reads via set_forward_context.
    #
    # The OLD step 5h path searched 4 candidate attribute paths
    # (attn_metadata_builders, attn_metadata_builder,
    # runner.attn_metadata_builders, runner.attn_metadata_builder) — NONE
    # of these exist in vllm v0.20.1.dev0. The builder is at
    # `proposer.draft_attn_groups[0].get_metadata_builder()`. The
    # `build_per_group_and_layer_attn_metadata` helper exposes the
    # right entry point.
    if proposer is None or not hasattr(
        proposer, "build_per_group_and_layer_attn_metadata"
    ):
        raise RuntimeError(
            "build_per_step_attn_metadata_array: proposer is None or lacks "
            "build_per_group_and_layer_attn_metadata. Expected vllm "
            "v0.20.1.dev0 EagleProposer with draft_attn_groups initialized."
        )

    for step_idx in range(buffers.num_steps):
        # Per-step views into the pre-allocated buffers. The captured graph
        # writes to these slices via the
        # eagle_step_update_slot_mapping_and_metadata kernel; the view (a
        # contiguous tensor) reflects the writes without rebuild because
        # FlashAttentionMetadata stores tensor *references*, not copies.
        slot_mapping_view = buffers.slot_mapping_per_step[step_idx, :batch_size]
        seq_lens_view = buffers.seq_lens_per_step[step_idx, :batch_size]

        # max_seq_len is a static int (not a tensor); bump by step_idx so the
        # attention backend sizes its scratch correctly for each step. The
        # +1 between steps is the spec-decode invariant: each draft token
        # adds one position to the sequence.
        step_max_seq_len = min(base_max_seq_len + step_idx, base_max_model_len)

        # Construct a step-specific CommonAttentionMetadata that the builder
        # will translate into a per-layer attention metadata dict.
        # `dataclasses.replace` produces a new CommonAttentionMetadata with
        # step-specific tensor views; the per-group builder then produces
        # the backend-specific metadata (FlashAttentionMetadata) with all
        # required fields (block_table, use_cascade, seqused_k, etc.).
        #
        # MUSA-0090 step 5l.3: also point block_table_tensor at the
        # runner-owned buffer (not the caller's transient one). The owned
        # buffer's contents are updated by replay() before each graph
        # invocation; the captured graph reads from a stable pointer.
        # The owned tensor is sized [max_bs, max_blocks_per_req]; slice
        # to [:batch_size, :] to match the request.
        step_cam = replace(
            base_metadata,
            slot_mapping=slot_mapping_view,
            seq_lens=seq_lens_view,
            max_seq_len=step_max_seq_len,
            block_table_tensor=buffers.block_table_tensor[:batch_size],
        )

        try:
            _per_group, per_layer = proposer.build_per_group_and_layer_attn_metadata(
                step_cam, draft_index=step_idx
            )
        except Exception as exc:
            raise RuntimeError(
                f"step {step_idx}: proposer."
                f"build_per_group_and_layer_attn_metadata() failed: "
                f"{type(exc).__name__}: {exc}. This path was added in "
                "MUSA-0090 step 5l (2026-05-16) to use vllm's first-party "
                "API. The builder must succeed on a CommonAttentionMetadata "
                "with step-specific slot_mapping/seq_lens/max_seq_len views."
            ) from exc

        # per_layer is the dict {layer_name: attn_metadata} that
        # set_forward_context expects. Store the dict as the step's metadata
        # entry — _run_n_step_inner passes attn_metadata_array[step] into
        # set_forward_context which dispatches per layer.
        metadata_array.append(per_layer)

    return metadata_array


@dataclass
class StepMetadataIndexing:
    """Records how each step's metadata reads from the buffers.

    Used by the capture path to verify that the indexing math is correct
    before committing the graph. Construct from a built metadata_array via
    the inspect() helper.
    """

    num_steps: int
    base_max_seq_len: int
    per_step_max_seq_lens: list[int]
    slot_mapping_data_ptrs: list[int]
    seq_lens_data_ptrs: list[int]

    @classmethod
    def inspect(cls, metadata_array: list[Any]) -> StepMetadataIndexing:
        """Read back the structure of a built metadata array for verification.

        Used in tests / debug logging — not on the hot path.
        """
        if not metadata_array:
            return cls(
                num_steps=0,
                base_max_seq_len=0,
                per_step_max_seq_lens=[],
                slot_mapping_data_ptrs=[],
                seq_lens_data_ptrs=[],
            )
        return cls(
            num_steps=len(metadata_array),
            base_max_seq_len=int(getattr(metadata_array[0], "max_seq_len", 0)),
            per_step_max_seq_lens=[
                int(getattr(m, "max_seq_len", 0)) for m in metadata_array
            ],
            slot_mapping_data_ptrs=[
                int(m.slot_mapping.data_ptr()) for m in metadata_array
            ],
            seq_lens_data_ptrs=[int(m.seq_lens.data_ptr()) for m in metadata_array],
        )

    def verify_strict(self) -> None:
        """Raise if the metadata array isn't shaped correctly for the captured
        loop. Sanity-check before passing to graph capture."""
        if self.num_steps <= 0:
            raise RuntimeError("metadata array is empty")
        if len(set(self.slot_mapping_data_ptrs)) != self.num_steps:
            raise RuntimeError(
                f"slot_mapping views are not all distinct: "
                f"{self.slot_mapping_data_ptrs}"
            )
        if len(set(self.seq_lens_data_ptrs)) != self.num_steps:
            raise RuntimeError(
                f"seq_lens views are not all distinct: " f"{self.seq_lens_data_ptrs}"
            )
        # max_seq_lens should be monotonically non-decreasing across steps
        for i in range(1, self.num_steps):
            if self.per_step_max_seq_lens[i] < self.per_step_max_seq_lens[i - 1]:
                raise RuntimeError(
                    f"max_seq_len decreased at step {i}: "
                    f"{self.per_step_max_seq_lens}"
                )
