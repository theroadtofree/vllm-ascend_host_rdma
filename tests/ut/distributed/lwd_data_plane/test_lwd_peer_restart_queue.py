# SPDX-License-Identifier: Apache-2.0
"""Peer-restart recovery of the transfer FIFOs (lazy baseline + reset).

A restarted peer's seqno stream continues at an unknown value, so a queue
must take the first arriving frame as its baseline instead of assuming 0
(pinning to 0 held every frame of a continuing counter forever), and the
plane's reset must clear the old stream's skip holes without failing the
connection that is coming back.
"""

from __future__ import annotations

import pytest

from vllm_ascend.distributed.lwd_data_plane import lwd_data_plane as plane_module
from vllm_ascend.distributed.lwd_data_plane.lwd_data_plane import LwdDataPlane
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_queue import LwdTransferQueue
from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
    ConnectionId,
    LwdChannelType,
    LwdTransferRequest,
)

UP = LwdChannelType.UP
CONNECTION = ConnectionId(0, 0, 0)


class _RecordingQueue(LwdTransferQueue):
    """Records executed seqnos; no NPU work, futures never complete."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.executed: list[int] = []

    def _execute(self, req, predecessor, into=None) -> LwdTransferFuture:  # noqa: ANN001
        self.executed.append(req.seqno)
        if into is None:
            return LwdTransferFuture(req, [], None)
        return into


def _recv(seqno: int) -> LwdTransferRequest:
    return LwdTransferRequest(channel=UP, op="recv", num_elements=8, seqno=seqno)


@pytest.fixture()
def plane(monkeypatch: pytest.MonkeyPatch) -> LwdDataPlane:
    monkeypatch.setattr(plane_module, "_queue_class", lambda: _RecordingQueue)
    return LwdDataPlane()


def test_first_arriving_seqno_is_the_baseline() -> None:
    """Regression: a fresh queue pinned its baseline to 0, so a restarted
    peer's continuing counter held every frame forever."""
    queue = _RecordingQueue(CONNECTION, UP, "recv")
    queue.submit(_recv(7))
    assert queue.executed == [7]


def test_reset_fails_stale_holds_and_re_baselines() -> None:
    """A restart fails the old stream's held frames promptly (their waiters
    must not hang) and the next frame re-baselines the FIFO."""
    queue = _RecordingQueue(CONNECTION, UP, "recv")
    queue.submit(_recv(5))
    held = queue.submit(_recv(7))
    assert queue.executed == [5]
    queue.reset_peer_baseline()
    with pytest.raises(RuntimeError, match="lwd-comm op failed"):
        held.result()
    queue.submit(_recv(2))
    assert queue.executed == [5, 2]


def test_reset_clears_the_planes_pending_skips_of_the_old_stream(plane: LwdDataPlane) -> None:
    """Skip holes recorded before the FIFO exists belong to the old stream;
    after a restart they must not replay onto (and wedge) the new one."""
    plane.skip_seqno(UP, seqno=0, op="recv", connection_id=CONNECTION)
    plane.reset_channel_seqno(CONNECTION, UP, "recv")
    plane.submit_recv(_recv(0))
    queue = plane._queues[(CONNECTION, UP, "recv")]
    assert isinstance(queue, _RecordingQueue)
    assert queue.executed == [0]


def test_reset_keeps_the_coming_back_connection_submittable(plane: LwdDataPlane) -> None:
    """A restart is not an abort: submissions on the connection must keep
    working after the reset (unlike abort_connection's failed marking)."""
    plane.submit_recv(_recv(4))
    plane.reset_channel_seqno(CONNECTION, UP, "recv")
    plane.submit_recv(_recv(0))  # must NOT raise "connection has failed"


def test_reset_without_a_queue_logs_miss(plane: LwdDataPlane, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every built IO registers its own reset, so a miss is a wiring bug that
    must surface in the log instead of passing silently."""
    errors: list[str] = []
    monkeypatch.setattr(plane_module.logger, "error", lambda msg, *args: errors.append(msg % args if args else msg))
    plane.reset_channel_seqno(ConnectionId(3, 3, 0), UP, "recv")
    assert any("reset_channel_seqno MISS" in message for message in errors)
