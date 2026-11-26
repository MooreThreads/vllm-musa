# SPDX-License-Identifier: Apache-2.0
"""MCCL launch ordering for MRv2 pipeline parallelism."""

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.distributed.parallel_state import Handle
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pp_utils import (
    PendingRecv,
    PPHandler,
    compute_need_sampled_mask,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


def select_pp_handler(
    config: "ParallelConfig", num_speculative_steps: int
) -> type[PPHandler]:
    mode = os.environ.get("VLLM_MUSA_MRV2_PP_HANDLER", "upstream")
    if mode not in ("upstream", "musa"):
        raise ValueError(
            f"Invalid VLLM_MUSA_MRV2_PP_HANDLER={mode!r}; expected upstream or musa"
        )
    if mode == "musa" and (
        config.pipeline_parallel_size != 2 or num_speculative_steps != 0
    ):
        logger.warning_once(
            "MusaPPHandler is selected for PP=%s with %s speculative steps; "
            "this topology is experimental and is not covered by the current "
            "MUSA MRv2 ordering validation",
            config.pipeline_parallel_size,
            num_speculative_steps,
        )
    handler = MusaPPHandler if mode == "musa" else PPHandler
    logger.info_once(
        "MRv2 PP handler: %s (VLLM_MUSA_MRV2_PP_HANDLER=%s)", handler.__name__, mode
    )
    return handler


@dataclass
class _DeferredRecv(PendingRecv):
    pending_counts: torch.Tensor | None = None


class MusaPPHandler(PPHandler):
    """Post P0 feedback before its next activation send or slot consumption.

    P1 sends feedback off the output path and joins it at the next activation recv.
    """

    _feedback_sent: torch.cuda.Event | None = None

    def _make_broadcast_stream(self) -> torch.cuda.Stream:
        if self.is_last_rank:
            return super()._make_broadcast_stream()
        return self.main_stream

    def receive(self, input_batch: InputBatch) -> bool:
        assert not self.is_last_rank
        mask = compute_need_sampled_mask(input_batch)
        if mask is None:
            return False
        with torch.cuda.stream(self.main_stream):
            tokens = torch.empty(
                input_batch.num_reqs,
                self.max_sample_len,
                dtype=torch.int64,
                device=self.device,
            )
            counts = torch.empty(
                2, input_batch.num_reqs, dtype=torch.int32, device=self.device
            )
            sampled, rejected = counts.unbind(dim=0)
        self.queue[-1] = _DeferredRecv(
            event=torch.cuda.Event(),
            sampled_tokens=tokens,
            num_sampled=sampled,
            num_rejected=rejected,
            idx_mapping=input_batch.idx_mapping,
            idx_mapping_np=input_batch.idx_mapping_np,
            need_sampled_mask=mask,
            gen_at_receive_np=self.req_idx_gen_np[input_batch.idx_mapping_np],
            pending_counts=counts,
        )
        return bool(mask.all())

    def _submit_slot(
        self, slot: PendingRecv | None, sends: Iterable[Handle] = ()
    ) -> None:
        if not isinstance(slot, _DeferredRecv) or slot.pending_counts is None:
            return
        try:
            with torch.cuda.stream(self.main_stream):
                for work in sends:
                    work.wait()
                torch.distributed.broadcast(
                    slot.sampled_tokens, src=self.last_rank, group=self.broadcast_group
                )
                torch.distributed.broadcast(
                    slot.pending_counts, src=self.last_rank, group=self.broadcast_group
                )
                slot.event.record(self.main_stream)
        except Exception:
            self.queue.clear()
            raise
        slot.pending_counts = None

    def submit_pending_feedback(self, previous_sends: Sequence[Handle] = ()) -> None:
        sends = iter(previous_sends)
        for slot in self.queue:
            self._submit_slot(slot, sends)

    def get_prev_sampled_outputs(
        self, draft_tokens_to_update: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor] | None:
        if self.queue:
            self._submit_slot(self.queue[0])
        return super().get_prev_sampled_outputs(draft_tokens_to_update)

    def _after_feedback_broadcast(self) -> None:
        self._feedback_sent = self.broadcast_stream.record_event()

    def before_activation_recv(self) -> None:
        if self._feedback_sent is not None:
            torch.cuda.current_stream(self.device).wait_event(self._feedback_sent)
            self._feedback_sent = None
