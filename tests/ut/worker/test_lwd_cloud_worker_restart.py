# SPDX-License-Identifier: Apache-2.0
"""Early-recv ledger drop on edge restart (peer-restart recovery).

A restarted edge's from-zero hints must not be judged 'already delivered'
by the surviving idempotence watermark: the drop clears only the restarted
connection's bookkeeping and, unlike the failure path, does not mark the
connection failed — it is coming back.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from vllm_ascend.distributed.lwd_data_plane.lwd_types import ConnectionId, LwdChannelType
from vllm_ascend.worker.lwd_cloud_model import lwd_cloud_worker
from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_worker import LwdCloudWorker

UP = LwdChannelType.UP
CONNECTION_A = ConnectionId(0, 0, 0)
CONNECTION_B = ConnectionId(1, 0, 0)


def _worker() -> LwdCloudWorker:
    worker = object.__new__(LwdCloudWorker)
    worker._early_recv_lock = threading.Lock()
    worker._early_recv_futures = {
        (CONNECTION_A, UP, 1): MagicMock(),
        (CONNECTION_B, UP, 10): MagicMock(),
    }
    worker._early_recv_pending = {
        (CONNECTION_A, UP, 1): {"req": "a"},
        (CONNECTION_B, UP, 11): {"req": "b"},
    }
    worker._early_recv_reserved = {(CONNECTION_A, UP, 2), (CONNECTION_B, UP, 12)}
    worker._early_recv_last_delivered = {CONNECTION_A: 4, CONNECTION_B: 9}
    worker._early_recv_failed_connections = set()
    worker._early_recv_current_key = (CONNECTION_B, UP, 13)
    worker._lwd_stager = MagicMock()
    return worker


def test_drop_clears_only_the_restarted_connection() -> None:
    worker = _worker()
    worker.drop_connection_early_recv(CONNECTION_B)
    # Bookkeeping of the restarted connection is gone, the peer's survives.
    assert set(worker._early_recv_futures) == {(CONNECTION_A, UP, 1)}
    assert set(worker._early_recv_pending) == {(CONNECTION_A, UP, 1)}
    assert worker._early_recv_reserved == {(CONNECTION_A, UP, 2)}
    assert worker._early_recv_last_delivered == {CONNECTION_A: 4}
    assert worker._early_recv_current_key is None
    # The pre-posted receive slot of the dropped future is released.
    worker._early_recv_futures[(CONNECTION_A, UP, 1)].release_device_tensor.assert_not_called()
    # The stale-watermark hang this drop exists for: the new edge starts at
    # seqno 0, judged against a cleared watermark instead of the old 9.
    worker._lwd_stager.abandon_connection.assert_called_once_with(CONNECTION_B)
    # A restart is coming back, not a failure: no failed marking.
    assert worker._early_recv_failed_connections == set()


def test_dropped_futures_release_their_receive_slots() -> None:
    worker = _worker()
    dropped = worker._early_recv_futures[(CONNECTION_B, UP, 10)]
    worker.drop_connection_early_recv(CONNECTION_B)
    dropped.release_device_tensor.assert_called_once()


def test_module_hook_without_an_endpoint_worker_logs_and_continues(monkeypatch) -> None:  # noqa: ANN001
    """The hook fires on the transport IO thread: a missing worker must be
    visible, and must not take the IO thread down."""
    monkeypatch.setattr(lwd_cloud_worker, "_ENDPOINT_WORKER", None)
    errors: list[str] = []
    monkeypatch.setattr(
        lwd_cloud_worker.lwd_logger, "error", lambda msg, *args: errors.append(msg % args if args else msg)
    )
    lwd_cloud_worker.drop_endpoint_early_recv(CONNECTION_B)  # must not raise
    assert any("early-recv drop MISS" in message for message in errors)
