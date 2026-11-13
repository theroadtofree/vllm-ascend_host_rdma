# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdTransferFuture: completion handle for one data plane transfer.

Ported (simplified) from the demo branch's ``lwd_comm/future.py``:
pure-CPU ``done()`` via ``event.query()``, blocking ``wait()`` for the
debug/timeout path, in-place ``_bind`` for deferred (out-of-order
sequenced) futures, and exactly-once ``_finalize`` releasing the send
keepalive.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from typing import Any

import torch

from vllm.lwd_logger import lwd_logger

from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
    LwdTransferRequest,
    LwdTransferResult,
    LwdTransferStatus,
)

# Debug stall watchdog: how long wait_on_current_stream may sit waiting for
# the sender/IO thread to bind the completion event before dumping every
# thread's stack at ERROR level. A healthy bind is milliseconds; the default
# 5s matches the ucx_channel recv-stuck watchdog so both sides of a stall
# fire together. 0 disables the dump (plain wait, old behaviour).
_FUTURE_BIND_TIMEOUT_S = float(os.environ.get("LWD_FUTURE_BIND_TIMEOUT_S", "5.0"))


def _dump_all_threads(reason: str) -> None:
    """ERROR-level snapshot of every live thread's stack (stall forensics).

    The wait's victim is usually the EngineCore main thread; the *cause*
    lives in other threads (UCX IO, sender, comm threads), so the whole
    process is captured, not just the waiter.
    """
    lines = [reason]
    names = {t.ident: t.name for t in threading.enumerate()}
    for tid, frame in sys._current_frames().items():
        lines.append(f"---- thread {names.get(tid, '?')} (tid={tid}) ----")
        lines.extend(traceback.format_stack(frame))
    lwd_logger.error("\n".join(lines))


class LwdTransferFuture:
    """Tracks one posted (or held) channel op.

    Completion means the cross-node P2P op has finished on the device,
    witnessed by ``done_event`` recorded on the channel stream right
    behind the bridged HCCL op.
    """

    def __init__(
        self,
        request: LwdTransferRequest,
        handles: list[Any],
        done_event: torch.npu.Event | None,
        tensor: torch.Tensor | None = None,
        keepalive: Any = None,
        aux_tensor: torch.Tensor | None = None,
        staged_event: torch.npu.Event | None = None,
    ) -> None:
        self._request = request
        self._handles = handles
        self._done_event = done_event
        self._tensor = tensor
        self._aux_tensor = aux_tensor
        self._keepalive = keepalive
        self._staged_event = staged_event
        self._status = LwdTransferStatus.PENDING
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._done_cond = threading.Condition(self._lock)
        self._finalized = False

    # ------------------------------------------------------------------ #
    # Deferred (out-of-order sequenced) support                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def deferred(cls, request: LwdTransferRequest) -> LwdTransferFuture:
        """A future for a held request; bound in place when its seqno
        turn arrives."""
        return cls(request, [], None)

    def _bind(
        self,
        *,
        handles: list[Any],
        done_event: torch.npu.Event | None,
        tensor: torch.Tensor | None = None,
        keepalive: Any = None,
        aux_tensor: torch.Tensor | None = None,
        staged_event: torch.npu.Event | None = None,
    ) -> None:
        with self._done_cond:
            self._handles = handles
            self._done_event = done_event
            self._tensor = tensor
            self._aux_tensor = aux_tensor
            self._keepalive = keepalive
            self._staged_event = staged_event
            self._done_cond.notify_all()

    # ------------------------------------------------------------------ #
    # Status                                                              #
    # ------------------------------------------------------------------ #

    @property
    def request(self) -> LwdTransferRequest:
        return self._request

    def done(self) -> bool:
        """Pure CPU query; never blocks, never touches the device."""
        with self._lock:
            if self._status is not LwdTransferStatus.PENDING:
                return True
            event = self._done_event
        if event is None:
            return False
        try:
            if event.query():
                with self._lock:
                    if self._status is LwdTransferStatus.PENDING:
                        self._status = LwdTransferStatus.OK
                        self._done_cond.notify_all()
                return True
        except Exception as exc:  # device error surfaces as ERROR
            with self._lock:
                if self._status is LwdTransferStatus.PENDING:
                    self._status = LwdTransferStatus.ERROR
                    self._error = exc
                    self._done_cond.notify_all()
            return True
        return False

    def wait(self, timeout: float | None = None) -> LwdTransferResult:
        """Block until completion (readiness-gate path)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._done_cond:
            while self._status is LwdTransferStatus.PENDING:
                if self._done_event is None:
                    # Deferred future not yet bound; wait for _bind.
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError(
                            f"lwd-comm wait timed out (unbound) on "
                            f"{self._request.channel.value} seqno="
                            f"{self._request.seqno}"
                        )
                    self._done_cond.wait(timeout=remaining or 0.001)
                    continue
                if self._done_event.query():
                    self._status = LwdTransferStatus.OK
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f"lwd-comm wait timed out on {self._request.channel.value} seqno={self._request.seqno}"
                    )
                self._done_cond.wait(timeout=min(0.001, remaining or 0.001))
            if self._status is LwdTransferStatus.ERROR:
                raise RuntimeError(
                    f"lwd-comm op failed on {self._request.channel.value} seqno={self._request.seqno}"
                ) from self._error
        return LwdTransferResult(
            status=self._status,
            tensor=self._tensor,
            aux_tensor=self._aux_tensor,
        )

    def result(self) -> LwdTransferResult:
        # TODO(lwd): confirm the behavior for a held future that is not bound
        # yet: it returns tensor=None, and wait_on_current_stream() is a no-op
        # before _bind.
        with self._lock:
            status, tensor, aux_tensor, error = (
                self._status,
                self._tensor,
                self._aux_tensor,
                self._error,
            )
        if status is LwdTransferStatus.ERROR:
            raise RuntimeError(
                f"lwd-comm op failed on {self._request.channel.value} seqno={self._request.seqno}"
            ) from error
        return LwdTransferResult(status=status, tensor=tensor, aux_tensor=aux_tensor)

    # ------------------------------------------------------------------ #
    # Host staging                                                        #
    # ------------------------------------------------------------------ #

    def staged_done(self) -> bool:
        """Query D2H completion without blocking the communication thread."""
        event = self._staged_event
        if event is None:
            return True
        try:
            return bool(event.query())
        except Exception as exc:
            with self._lock:
                if self._status is not LwdTransferStatus.ERROR:
                    self._status = LwdTransferStatus.ERROR
                    self._error = exc
                    self._done_cond.notify_all()
            return True

    def release_device_tensor(self) -> None:
        """Release a staged receive's device slot, not a normal recv result."""
        if self._staged_event is None:
            return
        with self._done_cond:
            self._tensor = None

    # ------------------------------------------------------------------ #
    # Device-side ordering for consumers                                  #
    # ------------------------------------------------------------------ #

    def wait_on_current_stream(self) -> None:
        """Order the *current* stream after the channel-stream completion
        event (pure device-side; CPU returns immediately).

        UCX paths bind the event asynchronously (IO thread / sender
        thread): first wait until the future is bound — reading
        ``result()`` before the bind would surface ``tensor=None`` and
        crash the consumer.  Host-side completion events (UCX sends)
        carry no device ordering semantics; ordering after them is a
        no-op (their data is host-staged, already readable when set).

        Debug stall watchdog: this wait is a suspected engine-loop
        pinning point under concurrent large prompts (one connection's
        DOWN frame not arriving freezes the single-threaded EngineCore
        scheduler here, and the OTHER connection's UP sends stall with
        it — matching the 11-13 incident where both clouds reported
        stuck within the same second). A healthy bind takes milliseconds;
        anything past LWD_FUTURE_BIND_TIMEOUT_S dumps every thread's
        stack at ERROR level (rate-limited per future) so the wait's
        cause is visible, then keeps waiting (fail-fast would take the
        engine down; the watchdog in ucx_channel owns that decision).
        """
        deadline = time.monotonic() + _FUTURE_BIND_TIMEOUT_S
        dumped = False
        with self._done_cond:
            while self._done_event is None and self._status is LwdTransferStatus.PENDING:
                self._done_cond.wait(timeout=0.01)
                if not dumped and time.monotonic() > deadline:
                    dumped = True
                    _dump_all_threads(
                        f"[lwd-comm] future bind stalled: {self._request.channel.value} "
                        f"seqno={self._request.seqno} connection=(edge={self._request.edge_id}, "
                        f"cloud={self._request.cloud_id}, dp={self._request.dp_idx}) waited "
                        f"{time.monotonic() - (deadline - _FUTURE_BIND_TIMEOUT_S):.1f}s "
                        f"for the sender/IO thread to bind the completion event -- "
                        f"if the engine main thread is here, the whole scheduler loop "
                        f"is pinned on this single op"
                    )
            if self._status is LwdTransferStatus.ERROR:
                raise RuntimeError(
                    f"lwd-comm op failed on {self._request.channel.value} seqno={self._request.seqno}"
                ) from self._error
        event = self._done_event
        if event is not None and not getattr(event, "is_host_event", False):
            torch.npu.current_stream().wait_event(event)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def _finalize(self, error: BaseException | None = None) -> None:
        """Exactly-once terminal bookkeeping: release the send keepalive.
        With ``error`` the future completes in ERROR state immediately
        (abort/skip path) so waiters fail promptly instead of timing out."""
        with self._done_cond:
            if self._finalized:
                return
            self._finalized = True
            self._keepalive = None
            self._handles = []
            if error is not None:
                self._status = LwdTransferStatus.ERROR
                self._error = error
                self._done_cond.notify_all()
                return
            if self._status is LwdTransferStatus.PENDING:
                self._status = (
                    LwdTransferStatus.OK
                    if (self._done_event is not None and self._done_event.query())
                    else self._status
                )
