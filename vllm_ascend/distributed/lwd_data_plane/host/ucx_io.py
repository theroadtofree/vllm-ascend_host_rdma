# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD UCX data-plane NPU assembly layer.

Generic host-host transport (the vLLM repo's ``vllm.v1.lwd_comm.data_comm``
UHC1 channel family) plus NPU device orchestration (D2H/H2D) and future/callback
wiring, on top of ``LwdTransferFuture``.

Layering and responsibilities:
  * transport: LwdUcxChannel -- seqno is the tag, zero frame header, zero
    software CRC; the receive block pool is built in. Transport-layer
    supervise/reconnect is kept, and ``on_peer_restart`` IS wired (embedded
    mode): the host-channel builder registers the seqno re-baseline and the
    cloud's early-recv ledger drop, so a peer process restart recovers without
    restarting the deployment (sidecar/daemon mode keeps the
    whole-deployment-restart semantics — the daemon protocol has no restart
    event).
  * send: sender thread + library-pool staging -- ``ch.acquire_staging()`` takes
    a pool block (pool_staging: the channel's pool blocks are not attached to
    the wildcard, they serve purely as D2H staging), one D2H copy writes
    directly into it, then ``ch.send(tag, blk.addr)``; the completion callback
    releases the staging back. Pool depth = in-flight cap = backpressure.
  * recv: ``ch.recv(tag, cb)`` callback (library IO thread) submits an async
    H2D + descramble and records an NPU event, then ``future._bind``; with
    stage_segments it further D2H-copies per segment into each request's pinned
    buffer and records staged_event.
  * aux: packed contiguously (payload then aux) into ONE message; the receiver
    splits it back into two H2D copies sized by the buffers the control plane
    already knows.
  * skip: aborted seqnos get a pre-armed discard recv; peer-restart resets
    the baseline lazily (the first arriving seqno of the new stream wins).

Deployment: both peers must use the same transport; UCX 1.22 + pinned TLS. The
RMA-put large-frame path stays off (LwdUcxConfig.put_thresh default 0). Block
capacity is per direction and comes from the topology
(``deployment.data_plane_host.up_buffer_size`` / ``down_buffer_size``):
identical on both peers and verified per link at handshake.
"""

from __future__ import annotations

import contextlib
import ctypes
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from vllm.logger import logger
from vllm.v1.lwd_comm.data_comm.lwd_probe import LwdProbeSet, probe_enabled
from vllm.v1.lwd_comm.data_comm.ucx_channel import (
    LwdHostDoneEvent,
    LwdUcxConfig,
    LwdUcxListener,
)
from vllm.v1.lwd_comm.data_comm.ucx_port import LwdUcxTransportPort, open_transport

from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType, LwdLinkError

# Phase-0 baseline probes (see vllm .../data_comm/lwd_probe.py): sender-side
# occupancy (staging acquire wait / synchronous D2H / post); the recv-side
# IO occupancy lives in ucx_channel itself. Same LWD_PROBE=0 kill switch.
_PROBE_ON = probe_enabled()

# Blocks per channel: absorbs the early frames and provides send backpressure.
BLOCKS_PER_CHANNEL = 4
# Host-landing copy pool: worker threads per UP-recv IO. The frame rate
# caps the pool's work, so 4 absorbs bursts without stealing the IO thread
# (the baseline probes pin the copy cost per frame; raise only if
# landing_copy p95 tracks the pool, not the memory).
_LANDING_POOL_THREADS = 4
# Early frames that arrived before their receive was pre-posted. The old
# branch allowed 4096; leaving this at LwdUcxConfig's default caps the store
# at BLOCKS_PER_CHANNEL and silently drops the oldest frame on overflow.
_EARLY_MAX = 4096
_HS_TIMEOUT_S = 600.0
# Poll tick of the submit-time reconnect wait (hot path cost when healthy:
# one attribute read; only a broken link ever enters the loop).
_RECONNECT_POLL_S = 0.05


# Reserved tag for frames without a seqno (production request streams all
# carry seqnos; only defends the queue's non-sequenced path — at most one
# such frame in flight at a time).
_NO_SEQNO_TAG = 1 << 60


@dataclass(frozen=True)
class LwdNpuUcxChannelIoConfig:
    """Static settings of one (channel, op) NPU IO object."""

    op: Literal["send", "recv"]
    up_block_mb: int
    down_block_mb: int
    host: str = ""
    port: int = 0
    serve: bool = False
    token: int = 0
    listener: LwdUcxListener | None = None
    transport_mode: Literal["embedded", "daemon"] = "embedded"
    traffic_class: int | None = None
    sl: int | None = None
    submit_wait_ready_s: float = 5.0
    """Bounded submit-time wait for the link to leave a reconnect window
    (the shared code constant ``LWD_SUBMIT_WAIT_READY_S`` flows in from the
    channel wiring; identical on both peers by construction); beyond it the
    submit proceeds and fails fast like any broken-link op."""

    landing: Literal["device", "host"] = "device"
    """UP receive landing mode (cloud UP-recv only; "device" everywhere
    else). "device" = current path: H2D into a device slot, then D2H
    staging into the per-request pinned buffers. "host" = the frame lands
    straight in the pinned buffers through the copy pool (zero PCIe round
    trip; requires in_process deployment, and stage_segments on every
    receive; scramble is rejected on this path until it moves)."""


class LwdNpuUcxChannelIo:
    """One (channel, op) LWD IO: LwdUcxChannel + sender thread + recv H2D.

    Roles: the edge connects (host, port); the cloud serves (serve=True,
    host, port) — i.e. the RECEIVER of a (channel, op) link serves.
    """

    def __init__(
        self,
        channel: LwdChannelType,
        stream: torch.npu.Stream,
        config: LwdNpuUcxChannelIoConfig,
    ) -> None:
        self._channel = channel
        self._op = config.op
        self._stream = stream
        self.tag = f"{channel.value}/{config.op}"
        # Phase-0 probe; token discriminates connections that share a channel
        # name. Series pre-created so the sender thread never constructs.
        self._probe = LwdProbeSet(f"{self.tag} token={config.token:#x}")
        for _probe_series in ("send_acq", "send_d2h", "send_post"):
            self._probe.series(_probe_series)
        self._stopped = False
        # Host landing (Phase-2): only the cloud's UP receive qualifies.
        # The edge's DOWN receive must keep the device path (its consumer
        # wants a device tensor, no stage_segments), send IOs have no
        # receive path, and the daemon transport hands shm views whose
        # recycle goes through mark_recycle (not op.release_data) -- a
        # host-landing daemon port is a later extension, not a fallback.
        self._landing_host = (
            config.landing == "host"
            and config.op == "recv"
            and channel is LwdChannelType.UP
        )
        if config.landing == "host" and not self._landing_host:
            raise RuntimeError(
                f"{self.tag}: landing=host is only valid on the cloud UP "
                f"receive IO (got op={config.op!r} channel={channel.value}); "
                "edge DOWN receives keep the device path"
            )
        if self._landing_host and config.transport_mode != "embedded":
            raise RuntimeError(
                f"{self.tag}: landing=host requires in_process deployment "
                "(the daemon port's shm recycle path is a later extension)"
            )
        if not config.token:
            raise RuntimeError(
                f"{self.tag}: UCX handshake token is 0; ucx_channels must "
                "pass a non-zero token derived from (dp_idx, channel, op)"
            )
        # Per-direction block capacity (see the file header); identical on
        # both peers and verified per link at handshake.
        _is_up = channel is LwdChannelType.UP
        _block_mb = int(config.up_block_mb if _is_up else config.down_block_mb)
        _block_env = "data_plane_host.up_buffer_size" if _is_up else "data_plane_host.down_buffer_size"
        if _block_mb <= 0:
            raise RuntimeError(f"{self.tag}: {_block_env} must be positive, got {_block_mb}MiB")
        self._block_bytes = _block_mb << 20
        self._block_env = _block_env
        cfg = LwdUcxConfig(
            block_mb=self._block_bytes >> 20,
            n_blocks=BLOCKS_PER_CHANNEL,
            early_max=_EARLY_MAX,
            hs_timeout_s=_HS_TIMEOUT_S,
            token=config.token,
            # Send-only link: pool blocks are not attached to the
            # wildcard; they serve purely as D2H staging (cycled via
            # acquire_staging/release_staging).
            pool_staging=(config.op == "send"),
            traffic_class=config.traffic_class,
            sl=config.sl,
            # Host landing rides on hold-recv: the library delivers the
            # receive block itself (no bytes copy-out) and the copy pool
            # reads it, releasing the block back when done.
            hold_recv=self._landing_host,
        )
        # Transport executor: embedded = in-process direct LwdUcxChannel
        # (default); daemon = dedicated transport daemon (ucx_daemon +
        # shm ring). The factory assembles by mode; this layer only faces
        # the LwdUcxTransportPort protocol.
        self._ch: LwdUcxTransportPort = open_transport(
            cfg,
            mode=config.transport_mode,
            serve=config.serve,
            host=config.host,
            port=config.port,
            listener=config.listener,
        )
        self._submit_wait_ready_s = config.submit_wait_ready_s
        # Host-landing copy pool: the IO thread only enqueues; worker
        # threads memmove the frame into the per-request pinned segments.
        # The queue is the landing permit: its backlog plus the transport
        # block pool bound the in-flight frames (the device-slot gate is
        # gone in this mode).
        self._copy_tasks: queue.Queue[tuple | None] | None = None
        self._copy_threads: list[threading.Thread] = []
        if self._landing_host:
            self._probe.series("landing_copy")
            self._copy_tasks = queue.Queue()
            for _i in range(_LANDING_POOL_THREADS):
                th = threading.Thread(
                    target=self._copy_worker,
                    name=f"lwd-ucx-cpy-{self.tag}-{_i}",
                    daemon=True,
                )
                th.start()
                self._copy_threads.append(th)
        # Peer-restart recovery (embedded): the port IS the channel in
        # embedded mode, so the session-fingerprint restart event reaches
        # the queue/ledger resets registered by lwd_host_channels. The
        # daemon protocol has no cross-process restart event yet, so the
        # registration below no-ops there (sidecar keeps the
        # whole-deployment-restart semantics).
        # Send direction: task queue + sender thread (staging blocks come
        # from the library pool via ch.acquire_staging; this layer never
        # allocates its own memory/MR).
        self._sq: queue.Queue[tuple | None] = queue.Queue()
        self._send_thread: threading.Thread | None = None
        if config.op == "send":
            self._send_thread = threading.Thread(target=self._sender_loop, name=f"lwd-ucx-send-{self.tag}", daemon=True)
            self._send_thread.start()
        # skip arming (only meaningful on the recv side)
        self._skip_armed: set[int] = set()
        self._skip_lock = threading.Lock()
        # Dedicated copy stream for segmented D2H (built at recv-side
        # construction to avoid creating streams on the lib IO thread)
        self._copy_stream: torch.npu.Stream | None = torch.npu.Stream() if config.op == "recv" else None
        # First-frame delimitation traces (error level, once per IO each)
        self._tr_submit_send = True
        self._tr_send_done = True
        self._tr_d2h = True
        self._tr_submit_recv = True
        self._tr_recv_done = True

    # ------------------------------------------------------------------ #
    # Queue call surface                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _nbytes(tensor: Any) -> int:
        # symbolic-meta-ok: host-side frame size in bytes, not a meta shape
        return tensor.numel() * tensor.element_size()

    @classmethod
    def _frame_nbytes(cls, tensor: Any, aux_tensor: Any) -> int:
        aux_bytes = cls._nbytes(aux_tensor) if aux_tensor is not None else 0
        return cls._nbytes(tensor) + aux_bytes

    def _raise_if_frame_too_large(self, seqno: int | None, tensor: Any, aux_tensor: Any) -> None:
        """Fail at submit time, before queueing a frame the peer cannot hold."""
        total = self._frame_nbytes(tensor, aux_tensor)
        if total <= self._block_bytes:
            return
        raise RuntimeError(
            f"[lwd-ucx] {self.tag} seqno={seqno}: frame "
            f"{total / 2**20:.1f}MiB ({total} B) > "
            f"{self._block_env}={self._block_bytes >> 20}MiB; "
            f"increase {self._block_env} on both edge and cloud "
            "(or reduce max_num_batched_tokens/chunk size)"
        )

    def _wait_reconnect_window(self) -> None:
        """Bounded submit-time wait for the link to leave a broken/reconnect
        window. ``wait_ready`` only reflects the FIRST link, so poll the
        transport's live ``ready`` state (set at adopt, dropped while
        broken); past the budget proceed and fail fast like any broken-link
        op. The daemon port has no ``ready``: no wait, matching its no
        cross-process recovery semantics."""
        waited = False
        deadline = time.monotonic() + self._submit_wait_ready_s
        while not getattr(self._ch, "ready", True):
            waited = True
            if time.monotonic() >= deadline:
                return
            time.sleep(_RECONNECT_POLL_S)
        if waited:
            logger.info("[lwd-ucx] %s submit waited out a reconnect window", self.tag)

    def submit_send(
        self,
        seqno: int | None,
        tensor: Any,
        aux_tensor: Any,
        req: Any,
        future: Any,
    ) -> None:
        # Fast fail on the edge: a frame larger than the negotiated block
        # can never reach the cloud. Queueing it would leave the request
        # awaiting forever because _do_send only finalizes the future.
        self._raise_if_frame_too_large(seqno, tensor, aux_tensor)
        # Absorb the reconnect window (peer restart / network break): a
        # send landing inside the gap would fail its op fail-fast.
        self._wait_reconnect_window()
        # The caller enqueues the snapshot clone asynchronously on the
        # producer stream, while the sender thread D2Hs on this channel
        # stream. Order the D2H after that producer work or it can read
        # half-written bytes under back-to-back submits.
        producer = torch.npu.current_stream()
        self._stream.wait_stream(producer)
        if self._tr_submit_send:
            self._tr_submit_send = False
            logger.error(
                "[lwd-ucx-trace] %s SUBMIT_SEND seqno=%s payload=%dB aux=%dB sq=%d",
                self.tag,
                seqno,
                self._nbytes(tensor),
                (self._nbytes(aux_tensor) if aux_tensor is not None else 0),
                self._sq.qsize(),
            )
        self._sq.put((seqno, tensor, aux_tensor, future))

    def submit_recv(
        self,
        seqno: int | None,
        req: Any,
        buffer: Any,
        aux_buffer: Any,
        future: Any,
        *,
        post_h2d: Any = None,
    ) -> None:
        tag = seqno if seqno is not None else _NO_SEQNO_TAG
        if self._tr_submit_recv:
            self._tr_submit_recv = False
            logger.error(
                "[lwd-ucx-trace] %s SUBMIT_RECV seqno=%s tag=%d numel=%s aux_numel=%s stage=%s",
                self.tag,
                seqno,
                tag,
                # host landing passes no device buffer: report the planned
                # numel from the request instead
                buffer.numel() if buffer is not None else req.num_elements,  # symbolic-meta-ok: log-only size
                aux_buffer.numel() if aux_buffer is not None else 0,  # symbolic-meta-ok: log-only size
                req.stage_segments is not None,
            )
        self._ch.recv(tag, cb=self._recv_cb(buffer, aux_buffer, future, req, post_h2d))

    def add_skipped(self, seqno: int) -> None:
        """Silently drop this seqno's frame on the recv side: a pre-armed
        discard recv (dedup)."""
        if seqno is None or self._op != "recv":
            return
        with self._skip_lock:
            if seqno in self._skip_armed:
                return
            self._skip_armed.add(seqno)
        # daemon mode goes through recv_discard (occupies no recv block,
        # produces no completion); embedded falls back to the generic path
        # (a cb that discards).
        discard = getattr(self._ch, "recv_discard", None)
        if discard is not None:
            discard(seqno)
        else:
            self._ch.recv(seqno, cb=lambda op: None)

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ch.wait_ready(timeout=timeout)

    def force_teardown(self, reason: str) -> None:
        """Linked teardown (control-plane restart verdict): tear the link
        down at once — in-flight ops end with ERROR (the containment
        catches them per request) and the connect side rebuilds. The
        daemon port's fail detaches the session instead, so this stays an
        embedded-mode operation."""
        self._ch.fail(reason)

    def on_peer_restart(self, cb: Callable[[], None]) -> None:
        """Register the peer-restart hook (fires on the transport IO thread
        when the handshake session changes at link adopt). Embedded mode
        only: the port IS the channel there; the daemon protocol has no
        cross-process restart event yet, so registration no-ops there."""
        register = getattr(self._ch, "on_peer_restart", None)
        if register is None:
            return
        register(cb)

    def shutdown(self, timeout: float = 2.0) -> None:
        self._stopped = True
        if self._send_thread is not None:
            self._sq.put(None)
            self._send_thread.join(timeout=timeout)
        for _th in self._copy_threads:
            # landing workers exit within one poll tick of _stopped
            _th.join(timeout=timeout)
        self._ch.close(timeout=timeout)
        with self._skip_lock:
            self._skip_armed.clear()

    # ------------------------------------------------------------------ #
    # Send: pool block (acquire_staging) -> direct D2H -> ch.send        #
    # ------------------------------------------------------------------ #
    def _sender_loop(self) -> None:
        last_probe = 0.0
        while True:
            task = self._sq.get()
            if task is None:
                break
            seqno, tensor, aux_tensor, future = task
            try:
                self._do_send(seqno, tensor, aux_tensor, future)
            except Exception as exc:  # noqa: BLE001
                logger.exception("[lwd-ucx-trace] %s SEND_FAILED seqno=%s err=%r", self.tag, seqno, exc)
                with contextlib.suppress(Exception):
                    future._finalize(error=exc)
            if _PROBE_ON:
                # flush cadence advances only while frames flow: a quiet
                # interval reports nothing, and the counters of the next busy
                # interval cover everything since the previous flush.
                _now = time.monotonic()
                if _now - last_probe >= 10.0:
                    for _probe_line in self._probe.take_lines():
                        logger.error("[lwd-probe] %s", _probe_line)
                    last_probe = _now

    def _do_send(self, seqno: int | None, tensor: Any, aux_tensor: Any, future: Any) -> None:
        payload_bytes = self._nbytes(tensor)
        aux_bytes = self._nbytes(aux_tensor) if aux_tensor is not None else 0
        total = payload_bytes + aux_bytes
        if total > self._block_bytes:
            # Defensive: submit_send already rejects this; keep the sender
            # path consistent if a frame is enqueued by another caller.
            self._raise_if_frame_too_large(seqno, tensor, aux_tensor)
            return
        # Library-pool staging block (pool depth = in-flight cap, i.e.
        # backpressure; blocks until one frees — the completion callback
        # releases it back; registration/rcache handled by the library).
        if _PROBE_ON:
            _t0 = time.perf_counter()
        stg = self._ch.acquire_staging()
        if _PROBE_ON:
            _t1 = time.perf_counter()
            self._probe.series("send_acq").add(_t1 - _t0)
        sent = False
        try:
            tag = seqno if seqno is not None else _NO_SEQNO_TAG
            if _PROBE_ON:
                _t2 = time.perf_counter()
            t0 = time.monotonic()
            with torch.inference_mode(), torch.npu.stream(self._stream):
                mv = memoryview(stg.buf)
                dst = torch.frombuffer(mv[:payload_bytes], dtype=tensor.dtype)
                dst.copy_(tensor.reshape(dst.shape))
                if aux_bytes:
                    adst = torch.frombuffer(mv[payload_bytes:total], dtype=aux_tensor.dtype)
                    adst.copy_(aux_tensor.reshape(adst.shape))
            if _PROBE_ON:
                # synchronous D2H into the staging block: this is the local
                # machine's real PCIe D2H rate — the number the landing
                # redesign compares host memcpy against.
                self._probe.series("send_d2h").add(time.perf_counter() - _t2)
            if self._tr_d2h:
                self._tr_d2h = False
                logger.error(
                    "[lwd-ucx-trace] %s SEND_D2H seqno=%s dur=%.1fms "
                    "(copy returned means host-readable; no such line = "
                    "stuck in D2H or staging acquire)",
                    self.tag,
                    seqno,
                    (time.monotonic() - t0) * 1000,
                )
            event = LwdHostDoneEvent()
            # bind BEFORE send: the handle must be in place before the
            # completion callback (IO thread) sets the event.
            future._bind(handles=[], done_event=event)
            if _PROBE_ON:
                _t3 = time.perf_counter()
            self._ch.send(tag, stg.addr, total, ref=stg, cb=self._send_done(event, stg, future))
            if _PROBE_ON:
                self._probe.series("send_post").add(time.perf_counter() - _t3)
                self._probe.count("send_frames")
                self._probe.count("send_bytes", total)
            sent = True
        finally:
            if not sent:
                self._ch.release_staging(stg)

    def _send_done(self, event: LwdHostDoneEvent, stg: Any, future: Any) -> Callable[[Any], None]:
        def cb(op) -> None:  # noqa: ANN001 - LwdUcxOp (lib IO thread)
            if self._tr_send_done:
                self._tr_send_done = False
                logger.error("[lwd-ucx-trace] %s SEND_DONE tag=%d size=%s err=%r", self.tag, op.tag, op.size, op.error)
            # op completion = data handed over: return the staging block
            # to the library pool.
            self._ch.release_staging(stg)
            event.set()
            if op.error is not None:
                with contextlib.suppress(Exception):
                    # Transport op errors are link-class: the consumers
                    # classify LwdLinkError as transient (peer restart /
                    # network outage) instead of a permanent failure.
                    future._finalize(error=LwdLinkError(f"{self.tag} send failed: {op.error!r}"))

        return cb

    # ------------------------------------------------------------------ #
    # Recv: lib IO thread callback -> H2D + descramble + staging -> bind  #
    # ------------------------------------------------------------------ #
    def _copy_stream_or_create(self) -> torch.npu.Stream:
        if self._copy_stream is None:
            self._copy_stream = torch.npu.Stream()
        return self._copy_stream

    def _recv_cb(
        self,
        buffer: Any,
        aux_buffer: Any,
        future: Any,
        req: Any,
        post_h2d: Any = None,
    ) -> Callable[[Any], None]:
        stream = self._stream
        tag_str = self.tag

        def cb(op) -> None:  # noqa: ANN001 - LwdUcxOp (lib IO thread)
            if self._tr_recv_done:
                self._tr_recv_done = False
                logger.error(
                    "[lwd-ucx-trace] %s RECV_DONE tag=%d len=%s err=%r",
                    tag_str,
                    op.tag,
                    len(op.data) if op.data is not None else None,
                    op.error,
                )
            if op.error is not None:
                with contextlib.suppress(Exception):
                    # Transport op errors are link-class: the consumers
                    # classify LwdLinkError as transient (peer restart /
                    # network outage) instead of a permanent failure.
                    future._finalize(error=LwdLinkError(f"{tag_str} recv failed: {op.error!r}"))
                return
            data = op.data
            if self._landing_host:
                self._landing_dispatch(op, data, req, future)
                return
            try:
                payload_len = self._nbytes(buffer)
                aux_len = self._nbytes(aux_buffer) if aux_buffer is not None else 0
                if payload_len + aux_len != len(data):
                    err = ConnectionError(
                        f"payload size mismatch on {tag_str}: got {len(data)}B expect {payload_len}+{aux_len}"
                    )
                    future._finalize(error=err)
                    self._ch.fail(err)
                    return
                with torch.inference_mode(), torch.npu.stream(stream):
                    mv = memoryview(data)
                    buffer.copy_(torch.frombuffer(mv[:payload_len], dtype=buffer.dtype), non_blocking=True)
                    if aux_buffer is not None:
                        aux_buffer.copy_(torch.frombuffer(mv[payload_len:], dtype=aux_buffer.dtype), non_blocking=True)
                    # The HCCL queue descrambles after _bridge_and_record;
                    # UCX must run the same logic after H2D and before
                    # recording done_event.
                    if post_h2d is not None:
                        post_h2d(buffer, req)
                    done_event = torch.npu.Event()
                    done_event.record()
                staged_event = None
                if req.stage_segments is not None:
                    staged_event = self._stage_to_buffers(buffer, req, done_event)
                future._bind(
                    handles=[],
                    done_event=done_event,
                    tensor=buffer,
                    keepalive=(data,),
                    aux_tensor=aux_buffer,
                    staged_event=staged_event,
                )
                # daemon mode: data is a shm-block view — register the H2D
                # completion event so the port reaper can return the recv
                # block (embedded's LwdUcxOp has no such method; skip).
                recycle = getattr(op, "mark_recycle", None)
                if recycle is not None:
                    recycle(done_event)
            except Exception as exc:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    future._finalize(error=exc)

        return cb

    # ------------------------------------------------------------------ #
    # Host landing: IO thread enqueues, copy pool memmoves into pinned      #
    # ------------------------------------------------------------------ #
    @property
    def landing_host(self) -> bool:
        """Whether this IO lands receives directly into pinned host buffers
        (cloud UP-recv, Phase-2 mode). The queue reads this to skip its
        device-buffer allocation."""
        return self._landing_host

    def _landing_dispatch(
        self, op: Any, data: Any, req: Any, future: Any
    ) -> None:
        """IO-thread half of host landing: validate the frame, resolve its
        host address, enqueue for the copy pool, and return immediately.

        Frame layout (see _do_send): payload (bf16, main hash body + expert
        tail packed) then aux (int64 mrope positions). The cloud data plane
        has NO consumer for the aux tensor -- mrope reaches the runner via
        the control plane (``_lwd_prime_mrope_positions``) -- so aux is
        validated for the length check and deliberately not copied; a future
        consumer must land it inside the copy pool, never on the IO thread.
        """
        payload_bytes = int(req.num_elements) * 2  # bf16 main+tail
        aux_bytes = int(req.aux_num_elements) * 8  # int64 aux
        if payload_bytes + aux_bytes != len(data):
            err = ConnectionError(
                f"payload size mismatch on {self.tag}: got {len(data)}B "
                f"expect {payload_bytes}+{aux_bytes}"
            )
            future._finalize(error=err)
            self._ch.fail(err)
            op.release_data()
            return
        segments = req.stage_segments
        if not segments:
            err = RuntimeError(
                f"host landing on {self.tag} requires req.stage_segments "
                "(per-request pinned buffers must be planned before submit)"
            )
            future._finalize(error=err)
            op.release_data()
            return
        for dst, _src_off, _numel in segments:
            if dst.element_size() != 2:
                err = RuntimeError(
                    f"host landing segment dtype is {dst.dtype} on {self.tag}; "
                    "the frame is packed bf16"
                )
                future._finalize(error=err)
                op.release_data()
                return
        if op._blk is not None:
            # held delivery: the receive block's registered memory
            src = op._blk.addr
        else:
            # early-parked frame: standalone bytes owned by the early store;
            # the future's keepalive holds them until finalize
            src = ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p).value
        tasks = self._copy_tasks
        assert tasks is not None
        tasks.put((src, data, segments, future, op))

    def _copy_worker(self) -> None:
        """Landing worker: memmove each segment straight into its pinned
        slice (no torch, no GIL-heavy ops), bind a host completion event,
        then release the held block back to the library."""
        last_probe = 0.0
        tasks = self._copy_tasks
        assert tasks is not None
        while True:
            try:
                task = tasks.get(timeout=0.05)
            except queue.Empty:
                task = None
            if task is None:
                if self._stopped:
                    return
                continue
            src, keep, segments, future, op = task
            try:
                if _PROBE_ON:
                    _t0 = time.perf_counter()
                for dst, src_off, numel in segments:
                    # segments are element-indexed into the bf16 frame
                    ctypes.memmove(dst.data_ptr(), src + src_off * 2, numel * 2)
                if _PROBE_ON:
                    self._probe.series("landing_copy").add(time.perf_counter() - _t0)
                    self._probe.count("landing_frames")
                    self._probe.count("landing_bytes", sum(n * 2 for _, _, n in segments))
                event = LwdHostDoneEvent()
                # bind BEFORE set: the handle must be in place before the
                # gate polls done(). keepalive only anchors the frame view
                # for GC; the pinned destinations are the durable copy.
                future._bind(
                    handles=[],
                    done_event=event,
                    tensor=None,
                    keepalive=(keep,),
                    aux_tensor=None,
                    staged_event=None,
                )
                event.set()
            except Exception as exc:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    future._finalize(
                        error=RuntimeError(f"{self.tag} host landing copy failed: {exc!r}")
                    )
            finally:
                # release in every outcome: a leaked held block drains the
                # transport pool and stalls the sender (logged stuck by the
                # library's IO loop after 5s)
                op.release_data()
                if _PROBE_ON:
                    _now = time.monotonic()
                    if _now - last_probe >= 10.0:
                        for _probe_line in self._probe.take_lines():
                            logger.error("[lwd-probe] %s", _probe_line)
                        last_probe = _now

    def _stage_to_buffers(self, buffer: Any, req: Any, done_event: torch.npu.Event) -> torch.npu.Event:
        """D2H-copy the main frame per stage_segment into each request's
        pinned buffer.

        Same semantics as the HCCL queue's staging path: the copy stream
        first waits done_event (descramble included), segments stay FIFO
        on one stream, staged_event is recorded after the last segment.
        """
        copy_stream = self._copy_stream_or_create()
        copy_stream.wait_event(done_event)
        with torch.npu.stream(copy_stream):
            for dst, src_off, n in req.stage_segments:
                dst.copy_(buffer[src_off : src_off + n], non_blocking=True)
            staged_event = torch.npu.Event()
            staged_event.record()
        return staged_event
