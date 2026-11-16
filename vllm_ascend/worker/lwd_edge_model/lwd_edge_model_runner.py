# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD edge data-plane compute host (moved verbatim from LwdEdgeWorker).

The edge worker only dispatches EMBED/UNEMBED batches; the compute —
embedding forward + Hash expert lookup + UP send, DOWN receive + lm_head
+ rank replay, the MM cache and the tid2eid tables — lives here. The
native per-step runner pipeline stays dormant (no _prepare_inputs /
_model_forward / sample_tokens): the edge hijack semantics are
unchanged, this only restores the native worker↔runner division.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import torch
from vllm.lwd_logger import lwd_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.v1.core.sched.output import LwdEmbedBatch, LwdUnembedBatch
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
    ConnectionId,
    LwdChannelType,
    LwdTransferRequest,
    connection_from,
)
from vllm_ascend.worker.lwd_base_model.lwd_base_model_runner import LwdBaseModelRunner
from vllm_ascend.worker.lwd_edge_model.lwd_edge_mm_cache import LwdEdgeMmCache
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import pack_hash_payload
from vllm_ascend.worker.lwd_hash.lwd_hash_tables import load_tid2eid_tables

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheSpec

    from vllm_ascend.worker.lwd_base_model.lwd_worker_base import LwdWorkerBase


def select_token_batch(logits: torch.Tensor, ranks: list[int]) -> list[int]:
    """Batched rank→token lookup: one sort, one D2H sync for all rows.

    logits [R,V] rows sorted together (NPU sorts rows in parallel);
    sorted_idx[i, ranks[i]] is the token id at rank ranks[i] for row i.
    Stable sorting preserves ascending token-ID order for equal logits.
    """
    _, sorted_idx = torch.sort(logits, dim=-1, descending=True, stable=True)
    row = torch.arange(len(ranks), device=logits.device)
    return sorted_idx[row, torch.tensor(ranks, device=logits.device)].tolist()


# ---------------------------------------------------------------------------
# Dual-instance localization probes (observe-only; env-tunable, 0 disables).
# The edge EngineCore is a SINGLE-THREADED loop serving both instances
# time-shared on one NPU card, so one instance's batch can starve the
# other's. The probes below attribute every slow batch and every loop
# window to (batch type, instance) to make that starvation — or its
# absence — visible in plain logs.
# ---------------------------------------------------------------------------
# One batch holding the loop past this logs an ERROR naming type/instance.
_LWD_LOOP_HOLD_WARN_S = float(os.environ.get("LWD_LOOP_HOLD_WARN_S", "5.0"))
# Loop-share accounting window: every _LWD_LOOP_SHARE_INTERVAL_S of loop
# activity, log how many seconds each (type, instance) consumed plus the
# unaccounted remainder (scheduler/idle between batches).
_LWD_LOOP_SHARE_INTERVAL_S = float(os.environ.get("LWD_LOOP_SHARE_INTERVAL_S", "30.0"))
# Slow single-batch threshold for the phase-breakdown ERROR lines.
_LWD_SLOW_BATCH_MS = float(os.environ.get("LWD_SLOW_BATCH_MS", "1000"))
# DOWN receive cadence watchdog: gap past this between consecutive UNEMBEDs
# of one connection logs an ERROR (that instance's decode is starving).
_LWD_DOWN_GAP_WARN_S = float(os.environ.get("LWD_DOWN_GAP_WARN_S", "5.0"))
# Stall heartbeat: a daemon thread that beats every N seconds and reports
# (ERROR) any batch still holding the loop past _LWD_LOOP_HOLD_WARN_S WHILE
# it holds it. Two facts the after-the-fact probes cannot give:
#   1. heartbeats present during a long hold => the process is alive and the
#      DOWN frame is genuinely absent (a logical stall on the data plane);
#      heartbeats absent + a "heartbeat gap" line on wake => the whole
#      process/host was frozen or CPU-starved (OS/driver level, not LWD).
#   2. whether the recv future is still UNBOUND (no frame arrived yet) vs
#      BOUND (frame landed, stuck later in the chain).
_LWD_STALL_HEARTBEAT_S = 10.0


class LwdEdgeModelRunner(LwdBaseModelRunner):
    """Embed + unembed compute for the LWD edge (no native step pipeline)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        worker: LwdWorkerBase | None = None,
    ) -> None:
        super().__init__(vllm_config, device, worker)
        self.mm_cache = LwdEdgeMmCache()
        self.tid2eid_tables: list[torch.Tensor] = []
        self._lwd_up_send_inflight: dict[ConnectionId, deque[tuple[int, LwdTransferFuture]]] = {}
        self._lwd_up_send_watermarks: dict[ConnectionId, int] = {}
        # Loop-occupancy probes: last batch (type, connection, seqno,
        # entry ts) and the per-window share by (type, instance edge_id).
        self._lwd_loop_batch: tuple[str, ConnectionId, int, float] | None = None
        self._lwd_loop_share: dict[tuple[str, int], float] = {}
        self._lwd_loop_window_t0 = 0.0
        self._lwd_loop_window_total = 0.0
        # DOWN receive cadence watchdog: last UNEMBED arrival per connection.
        self._lwd_unembed_last_arrival: dict[ConnectionId, float] = {}
        # Stall heartbeat context: set only while a DOWN recv wait is in
        # flight (connection, seqno, wait-start ts, future); the observer
        # thread below reads it between beats.
        self._lwd_down_wait: tuple[ConnectionId, int, float, LwdTransferFuture] | None = None
        self._lwd_last_heartbeat = time.monotonic()
        threading.Thread(
            target=self._lwd_stall_heartbeat,
            name="lwd-edge-stall-heartbeat",
            daemon=True,
        ).start()

    def _lwd_stall_heartbeat(self) -> None:
        """Observe-only: witness long loop holds WHILE they hold.

        The after-the-fact probes (HELD/share/slow-batch) only fire once the
        blocking batch finally releases, so a frozen hour looks identical to
        an idle hour. This thread beats every _LWD_STALL_HEARTBEAT_S and
        reports any in-flight DOWN wait, plus detects its own scheduling
        gaps: a "heartbeat gap" line means the thread itself could not run —
        the whole process/host was frozen or CPU-starved, not LWD logic.
        """
        while True:
            time.sleep(_LWD_STALL_HEARTBEAT_S)
            now = time.monotonic()
            gap = now - self._lwd_last_heartbeat
            self._lwd_last_heartbeat = now
            if gap > _LWD_STALL_HEARTBEAT_S * 3:
                lwd_logger.error(
                    "[Lwd][edge-loop] heartbeat gap %.1fs -- the heartbeat "
                    "thread itself could not run; the whole process/host was "
                    "frozen or CPU-starved in between (not LWD logic)",
                    gap,
                )
            wait = self._lwd_down_wait
            if wait is not None:
                conn, seqno, t0, fut = wait
                elapsed = now - t0
                if elapsed >= _LWD_LOOP_HOLD_WARN_S:
                    try:
                        frame_state = (
                            "BOUND (frame arrived; stuck later in the chain)"
                            if fut._done_event is not None
                            else "ABSENT (no DOWN frame reached this recv yet)"
                        )
                    except Exception:  # noqa: BLE001
                        frame_state = "UNKNOWN"
                    queue_state = ""
                    try:
                        # Decisive discriminator for the deferred-hole freeze:
                        # if this recv sits in the queue's _held (its seqno is
                        # ahead of _next_seqno), the recv was never posted to
                        # UCX and the wait can only end when the missing lower
                        # seqno arrives or is skipped -- i.e. the peer burned/
                        # lost a seqno (exception between its seqno allocation
                        # and the wire), not a slow frame.
                        from vllm_ascend.distributed.lwd_data_plane.lwd_data_plane import (
                            lwd_get_data_plane,
                        )

                        _q = lwd_get_data_plane()._queues.get((conn, LwdChannelType.DOWN, "recv"))
                        if _q is not None:
                            queue_state = (
                                f" recv_queue_next={_q._next_seqno}"
                                f" held={sorted(_q._held)}"
                                f" this_recv={'HELD-DEFERRED (never posted to UCX)'
                                if seqno in _q._held else 'posted'}"
                            )
                    except Exception:  # noqa: BLE001
                        pass
                    lwd_logger.error(
                        "[Lwd][edge-loop] still waiting for DOWN seqno=%s "
                        "instance=edge%d for %.1fs: frame=%s%s (process alive, "
                        "so the cloud sent nothing this connection)",
                        seqno,
                        conn.edge_id,
                        elapsed,
                        frame_state,
                        queue_state,
                    )
                    continue
            batch = self._lwd_loop_batch
            if batch is not None:
                kind, conn, seqno, t0 = batch
                elapsed = now - t0
                # Cap at 10 minutes: beyond that the loop is simply idle
                # (benchmark drained) and endless lines would spam the log;
                # a frozen host shows up as heartbeat-gap lines instead.
                if _LWD_LOOP_HOLD_WARN_S <= elapsed < 600.0:
                    lwd_logger.error(
                        "[Lwd][edge-loop] no new batch for %.1fs; last is %s "
                        "seqno=%s instance=edge%d (scheduler gate/no-notify "
                        "idle, or stuck inside that batch)",
                        elapsed,
                        kind,
                        seqno,
                        conn.edge_id,
                    )

    def _lwd_note_loop_batch(self, kind: str, connection: ConnectionId, seqno: int) -> None:
        """Stamp loop occupancy at batch entry; attribute the PREVIOUS
        batch's full duration to (type, instance) and report slow holders
        plus a periodic per-instance share of the single-threaded loop.

        The share is the direct answer to "is instance A's batch starving
        instance B": a window like `UNEMBED/e0=22s EMBED/e1=3s
        other=5s` says instance 0's DOWN-wait batches consumed most of
        the loop while instance 1's UP prefill supply queued behind.
        """
        now = time.monotonic()
        prev = self._lwd_loop_batch
        self._lwd_loop_batch = (kind, connection, seqno, now)
        if prev is not None:
            p_kind, p_conn, p_seqno, p_t0 = prev
            held = now - p_t0
            key = (p_kind, p_conn.edge_id)
            self._lwd_loop_share[key] = self._lwd_loop_share.get(key, 0.0) + held
            self._lwd_loop_window_total += held
            if held >= _LWD_LOOP_HOLD_WARN_S:
                lwd_logger.error(
                    "[Lwd][edge-loop] %s connection=(edge=%d, cloud=%d, dp=%d) "
                    "seqno=%s HELD the loop for %.1fs -- every other "
                    "instance's EMBED/UNEMBED (and its UP sends) waited "
                    "behind it",
                    p_kind,
                    p_conn.edge_id,
                    p_conn.cloud_id,
                    p_conn.dp_idx,
                    p_seqno,
                    held,
                )
        if self._lwd_loop_window_t0 == 0.0:
            self._lwd_loop_window_t0 = now
            return
        window = now - self._lwd_loop_window_t0
        if window >= _LWD_LOOP_SHARE_INTERVAL_S and self._lwd_loop_share:
            other = max(window - self._lwd_loop_window_total, 0.0)
            share_str = " ".join(
                f"{kind}/e{edge}={secs:.1f}s"
                for (kind, edge), secs in sorted(self._lwd_loop_share.items())
            )
            lwd_logger.error(
                "[Lwd][edge-loop] share over %.0fs loop window: %s "
                "other/idle=%.1fs (single-threaded loop, one NPU card, "
                "both instances time-shared)",
                window,
                share_str,
                other,
            )
            self._lwd_loop_share = {}
            self._lwd_loop_window_total = 0.0
            self._lwd_loop_window_t0 = now

    def load_model(self) -> None:
        """Load the edge model and read the CPU Hash routing tables for
        DeepSeek V4 from the weight files."""
        super().load_model()
        # [layer-trace] Parity instrumentation (env-gated, zero overhead
        # when off): the edge install point; cloud and centralized share
        # the same code (see lwd_debug.lwd_layer_trace).
        from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
            install_lwd_layer_trace,
        )

        install_lwd_layer_trace(self.get_model())
        config = self.model_config.hf_config
        if getattr(config, "model_type", None) != "deepseek_v4":
            return
        num_hash_layers = config.num_hash_layers
        if not 0 <= num_hash_layers <= config.num_hidden_layers:
            raise ValueError(f"Invalid DeepSeek V4 num_hash_layers: {num_hash_layers}")
        if num_hash_layers == 0:
            lwd_logger.info("[lwd-edge] tid2eid loading skipped: num_hash_layers=0")
            return
        lwd_logger.info("[lwd-edge] loading tid2eid tables for %d Hash MoE layers", num_hash_layers)
        # Reuse vLLM's checkpoint resolution (revision, cache and shard filtering).
        loader = DefaultModelLoader(self.vllm_config.load_config)
        _, weight_files, use_safetensors = loader._prepare_weights(
            self.model_config.model,
            subfolder=None,
            revision=self.model_config.revision,
            fall_back_to_pt=False,
            allow_patterns_overrides=None,
        )
        if not use_safetensors:
            raise ValueError("LWD DeepSeek V4 tid2eid loading requires safetensors weights")
        self.tid2eid_tables = load_tid2eid_tables(
            weight_files,
            num_hash_layers,
            config.vocab_size,
            config.num_experts_per_tok,
            config.n_routed_experts,
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """The LWD edge runs no attention/transformer, so it needs no KV cache."""
        return {}

    def _lwd_embed_ids_h2d(self, flat_token_ids: list[int]) -> torch.Tensor:
        """Async H2D for token ids: a pinned ring (4 slots, same order as
        the dispatch queue depth) + non_blocking, replacing the pageable
        synchronous copy of torch.tensor(list, device=...) — the latter
        stalls the main stream waiting for the device to drain (including
        the pairing wait on the previous chunk's UP send), leaking
        cross-side lockstep into the edge worker's main thread. Same-stream
        ordering guarantees the embed kernel reads this chunk's values
        (H2D enqueues ahead of the embed)."""
        return self._lwd_staging(
            "edge_ids",
            dtype=torch.int64,
            min_capacity=8192,
        ).stage_to_device(flat_token_ids, len(flat_token_ids))

    def _lwd_submit_up_send(self, request: LwdTransferRequest) -> None:
        future = self.worker.comm_service.submit_send(request)
        pending = self._lwd_up_send_inflight.setdefault(request.connection, deque())
        pending.append((request.seqno, future))

    def _lwd_pop_completed_up_sends(self) -> None:
        """Advance each connection watermark across its successful FIFO prefix."""
        for connection, pending in self._lwd_up_send_inflight.items():
            while pending and pending[0][1].done():
                seqno, future = pending[0]
                future.result()
                pending.popleft()
                self._lwd_up_send_watermarks[connection] = seqno

    def _lwd_watermarked_output(self, connection: ConnectionId | int | None = None) -> ModelRunnerOutput:
        self._lwd_pop_completed_up_sends()
        if isinstance(connection, int):
            connection = ConnectionId(0, 0, connection)
        if connection is None:
            watermarks = {known: self._lwd_up_send_watermarks.get(known, -1) for known in self._lwd_up_send_inflight}
        else:
            watermarks = {connection: self._lwd_up_send_watermarks.get(connection, -1)}
        return ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            lwd_up_send_watermarks=watermarks,
        )

    def _lwd_gather_mm(
        self,
        batch_meta: LwdEmbedBatch,
        scheduler_output: SchedulerOutput,
        model: torch.nn.Module,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        is_refill = getattr(batch_meta, "is_refill", False)
        new_reqs = scheduler_output.scheduled_new_reqs
        snapshot = self.mm_cache.snapshot_features([req_data.req_id for req_data in new_reqs]) if is_refill else {}
        try:
            for req_data in new_reqs:
                feats = [f for f in req_data.mm_features if f.modality != "prompt_embeds"]
                if feats:
                    self.mm_cache.register(req_data.req_id, feats)
            return self.mm_cache.gather_chunk(
                batch_meta,
                model,
                self.device,
                self.pin_memory,
            )
        finally:
            if is_refill:
                self.mm_cache.restore_features(snapshot)

    @torch.inference_mode()
    def execute_lwd_embed(
        self,
        seqno: int,
        batch_meta: LwdEmbedBatch,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        model = self.get_model()
        connection = connection_from(batch_meta)
        self._lwd_note_loop_batch("EMBED", connection, seqno)
        if not batch_meta.token_ids:
            return self._lwd_watermarked_output(connection)

        # Flatten all requests' prompt tokens into one batch (order = req_ids).
        flat_token_ids = [tid for token_ids in batch_meta.token_ids for tid in token_ids]
        lwd_logger.debug("[Lwd][edge-worker] embed token_ids=%s", flat_token_ids)
        token_ids_tensor = self._lwd_embed_ids_h2d(flat_token_ids)
        _t0 = time.monotonic()
        # Multimodal merge: when any request in the batch carries mm data,
        # collect vision-tower embeds for this chunk window + build the
        # is_multimodal mask and take the model's native embed_input_ids
        # merge branch; text-only batches take the original direct path.
        mm_embeds, is_mm = self._lwd_gather_mm(batch_meta, scheduler_output, model)
        if mm_embeds:
            embeds = model.embed_input_ids(
                token_ids_tensor,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm,
            )
        else:
            embeds = model.embed_input_ids(token_ids_tensor)  # (total_N, H)
        _t_fwd = time.monotonic()
        payload = embeds
        if self.tid2eid_tables:
            # The original prompt IDs stay on the edge; transmit only lookups.
            cpu_ids = torch.tensor(flat_token_ids, dtype=torch.int64)
            expert_ids = torch.stack([table[cpu_ids] for table in self.tid2eid_tables], dim=1)
            payload = pack_hash_payload(embeds, expert_ids)
            lwd_logger.info(
                "[lwd-edge] UP Hash MoE seqno=%d expert_shape=%s payload_numel=%d",
                seqno,
                tuple(expert_ids.shape),
                payload.numel(),
            )

        # mrope positions ride the control plane with the RangeNotify
        # (wire_version>=2); the data plane sends only the main embeds
        # frame. The aux frame mechanism stays in the channel layer for
        # deepstack and other future payloads; mrope no longer uses it.
        request = LwdTransferRequest(
            channel=LwdChannelType.UP,
            op="send",
            num_elements=payload.numel(),
            dp_idx=connection.dp_idx,
            edge_id=connection.edge_id,
            cloud_id=connection.cloud_id,
            tensor=payload,
            seqno=seqno,
        )
        self._lwd_submit_up_send(request)
        # Slow-batch breakdown: a healthy EMBED (embed forward + submit) is
        # sub-second; anything past the threshold names which phase ate the
        # time and for WHICH instance — forward = the card's embed compute
        # (card saturation), submit_send = the data-plane snapshot/submit
        # (backpressure/lockstep).
        _embed_total_ms = (time.monotonic() - _t0) * 1000
        if _embed_total_ms > _LWD_SLOW_BATCH_MS:
            lwd_logger.error(
                "[Lwd][edge-worker] EMBED slow batch: seqno=%s total=%.1fms "
                "forward=%.2f submit_send=%.2f tokens=%d instance=edge%d",
                seqno,
                _embed_total_ms,
                (_t_fwd - _t0) * 1000,
                (time.monotonic() - _t_fwd) * 1000,
                len(flat_token_ids),
                connection.edge_id,
            )
        # [Lwd][perf] TTFT probes: forward = the embed forward;
        # submit_send = snapshot clone + broadcast submit (including the
        # bridge's handle.wait) — if the UP world broadcast is waiting on
        # the cloud's enqueue, this segment grows visibly (direct evidence
        # of chunk-level lockstep)
        lwd_logger.info(
            "[Lwd][perf] embed seqno=%s forward=%.2f submit_send=%.2f total=%.2fms tokens=%d",
            seqno,
            (_t_fwd - _t0) * 1000,
            (time.monotonic() - _t_fwd) * 1000,
            (time.monotonic() - _t0) * 1000,
            len(flat_token_ids),
        )
        return self._lwd_watermarked_output(connection)

    @torch.inference_mode()
    def execute_lwd_unembed(self, seqno: int, batch_meta: LwdUnembedBatch) -> ModelRunnerOutput:
        model = self.get_model()
        connection = connection_from(batch_meta)
        self._lwd_note_loop_batch("UNEMBED", connection, seqno)
        # DOWN receive cadence watchdog: gaps name the instance whose decode
        # is starving (its cloud sent no DOWN frame in between).
        _now = time.monotonic()
        _prev = self._lwd_unembed_last_arrival.get(connection)
        self._lwd_unembed_last_arrival[connection] = _now
        if _prev is not None and _now - _prev >= _LWD_DOWN_GAP_WARN_S:
            lwd_logger.error(
                "[Lwd][down-gap-recv] DOWN receive gap %.1fs on instance=edge%d "
                "(cloud=%d, dp=%d) before seqno=%s -- no DOWN frame reached "
                "this instance in between",
                _now - _prev,
                connection.edge_id,
                connection.cloud_id,
                connection.dp_idx,
                seqno,
            )
        if not batch_meta.req_ids:
            return self._lwd_watermarked_output()

        # Finish-only notifies (hidden_num_elements == 0, down_seqno == -1):
        # the cloud closes requests without any DOWN data (connection
        # failure cleanup). Submitting the recv anyway posts a tag that
        # never arrives and, under the abort path, crashes the whole edge
        # ("seqno -1 was skipped (aborted)") -- taking BOTH instances down.
        # No data means no receive, no lm_head: deliver zero tokens and let
        # the scheduler's row disposition finish the requests.
        if batch_meta.recv_num_elements == 0 or seqno < 0:
            lwd_logger.warning(
                "[Lwd][edge-worker] UNEMBED finish-only seqno=%s reqs=%s "
                "recv_num_elements=%d: skipping DOWN receive",
                seqno,
                list(batch_meta.req_ids),
                batch_meta.recv_num_elements,
            )
            output = self._lwd_watermarked_output()
            output.req_ids = batch_meta.req_ids
            output.req_id_to_index = {req_id: index for index, req_id in enumerate(batch_meta.req_ids)}
            output.sampled_token_ids = [[] for _ in batch_meta.req_ids]
            return output

        _t0 = time.monotonic()
        recv_future = self.worker.comm_service.submit_recv(
            LwdTransferRequest(
                channel=LwdChannelType.DOWN,
                op="recv",
                num_elements=batch_meta.recv_num_elements,  # int = rows_total * hidden_size
                dp_idx=connection.dp_idx,
                edge_id=connection.edge_id,
                cloud_id=connection.cloud_id,
                seqno=seqno,
            )
        )
        _t_post = time.monotonic()
        # Stall heartbeat context: while this wait is in flight the observer
        # thread can report seqno/elapsed/frame-state, and the outcome line
        # below tells whether a long wait ended by the frame finally arriving
        # or by an error -- the after-the-fact HELD line cannot say which.
        self._lwd_down_wait = (connection, seqno, _t0, recv_future)
        try:
            # wait_on_current_stream: device-ordered wait; the host never blocks
            recv_future.wait_on_current_stream()
            result = recv_future.result()
        except BaseException:
            self._lwd_down_wait = None
            if time.monotonic() - _t_post >= _LWD_LOOP_HOLD_WARN_S:
                lwd_logger.error(
                    "[Lwd][down-wait] seqno=%s instance=edge%d DOWN wait "
                    "FAILED after %.1fs (exception, not a late frame)",
                    seqno,
                    connection.edge_id,
                    time.monotonic() - _t_post,
                )
            raise
        self._lwd_down_wait = None
        if time.monotonic() - _t_post >= _LWD_LOOP_HOLD_WARN_S:
            lwd_logger.error(
                "[Lwd][down-wait] seqno=%s instance=edge%d DOWN wait ended "
                "after %.1fs -- the frame finally arrived (no exception)",
                seqno,
                connection.edge_id,
                time.monotonic() - _t_post,
            )
        # The receive tensor was allocated on the channel stream but is
        # consumed by lm_head on the compute stream. Keep its allocation
        # alive until that stream's work completes, including after the
        # future and its local result leave scope.
        result.tensor.record_stream(torch.npu.current_stream())
        _t_tensor = time.monotonic()
        hidden_size = self.model_config.get_hidden_size()
        hidden_states = result.tensor.view(-1, hidden_size)  # (rows_total, H)
        # lm_head must not be called directly (ParallelLMHead.forward
        # forces the sampler path); compute_logits is the standard
        # interface present on both wrapper and monolithic models.
        logits = model.compute_logits(hidden_states)  # (rows_total, V)
        _t_lm = time.monotonic()
        # L5 parity check: the first row's top-20 compared entrywise
        # against the cloud sampler's [layer-trace] top-20 — any rank
        # swap is a direct view of replay drift.
        try:
            from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
                lwd_layer_trace_enabled,
            )

            if lwd_layer_trace_enabled() and logits.dim() == 2:
                row = logits[0].detach().float()
                vals, tids = torch.topk(row, min(20, row.numel()))
                lwd_logger.info(
                    "[layer-trace] edge lm_head logits shape=%s row0 l2=%.4f top20=%s",
                    tuple(logits.shape),
                    row.norm().item(),
                    list(zip(tids.tolist(), [round(v, 3) for v in vals.tolist()])),
                )
        except Exception:  # noqa: BLE001
            pass

        flat_ids = select_token_batch(logits, [r for ranks in batch_meta.top_id_ths for r in ranks])
        sampled_token_ids = []
        off = 0
        for accept, ranks in zip(batch_meta.num_accept_tokens, batch_meta.top_id_ths):
            sampled_token_ids.append(flat_ids[off : off + accept])
            off += len(ranks)
        _t_sel = time.monotonic()
        # Slow-batch breakdown: wait_tensor is the DOWN-wait phase -- the
        # number that quantifies how long THIS instance's decode batch
        # blocked the single-threaded loop waiting for its cloud's frame.
        # post_recv/lm_head/select give the compute phases for contrast.
        _unembed_total_ms = (_t_sel - _t0) * 1000
        if _unembed_total_ms > _LWD_SLOW_BATCH_MS:
            lwd_logger.error(
                "[Lwd][edge-worker] UNEMBED slow batch: seqno=%s total=%.1fms "
                "post_recv=%.2f wait_tensor=%.2f lm_head=%.2f select=%.2f "
                "reqs=%d instance=edge%d",
                seqno,
                _unembed_total_ms,
                (_t_post - _t0) * 1000,
                (_t_tensor - _t_post) * 1000,
                (_t_lm - _t_tensor) * 1000,
                (_t_sel - _t_lm) * 1000,
                len(batch_meta.req_ids),
                connection.edge_id,
            )
        # [Lwd][perf] Temporary probes attributing single-request latency:
        # post_recv = posting the irecv; wait_tensor = waiting for the
        # tensor to land (network); lm_head = the vocabulary forward
        # (weight bandwidth); select = argsort name lookup (.item sync);
        # total = the whole edge tail
        lwd_logger.info(
            "[Lwd][perf] unembed seqno=%s post_recv=%.2f wait_tensor=%.2f "
            "lm_head=%.2f select=%.2f total=%.2fms reqs=%d",
            seqno,
            (_t_post - _t0) * 1000,
            (_t_tensor - _t_post) * 1000,
            (_t_lm - _t_tensor) * 1000,
            (_t_sel - _t_lm) * 1000,
            (_t_sel - _t0) * 1000,
            len(batch_meta.req_ids),
        )
        output = self._lwd_watermarked_output()
        output.req_ids = batch_meta.req_ids
        output.req_id_to_index = {req_id: index for index, req_id in enumerate(batch_meta.req_ids)}
        output.sampled_token_ids = sampled_token_ids
        return output
