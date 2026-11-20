# SPDX-License-Identifier: Apache-2.0
"""Link-error classification of the cloud early-recv path.

A transport link failure (LwdLinkError, peer restart or network outage)
must stay transient: the ledger drops and the engine retires the
connection's requests WITHOUT any failed marking, so the connection is
recoverable when the peer returns. Application errors keep the permanent
failure path.
"""

from __future__ import annotations

from typing import Any

from vllm.config.lwd.topology import LwdConnectionId

from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdLinkError
from vllm_ascend.worker.lwd_cloud_model import lwd_cloud_worker
from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_worker import LwdCloudWorker

CONNECTION = LwdConnectionId(1, 0, 0)


def _runtime_error_from(cause: BaseException) -> RuntimeError:
    try:
        raise RuntimeError("lwd-comm op failed") from cause
    except RuntimeError as exc:
        return exc


def test_link_error_cause_unwraps_the_future_wrapper() -> None:
    wrapped = _runtime_error_from(LwdLinkError("transport died"))
    assert isinstance(lwd_cloud_worker._link_error_cause(wrapped), LwdLinkError)
    assert isinstance(lwd_cloud_worker._link_error_cause(LwdLinkError("raw")), LwdLinkError)
    # Application errors must NOT classify as link errors.
    assert lwd_cloud_worker._link_error_cause(_runtime_error_from(ValueError("bad geometry"))) is None
    assert lwd_cloud_worker._link_error_cause(ValueError("raw")) is None


def _worker(monkeypatch: Any, drops: list, fails: list, enqueued: list) -> LwdCloudWorker:
    worker = object.__new__(LwdCloudWorker)
    worker._early_recv_failed_connections = set()
    worker._irecv_done_mqs = {CONNECTION: _FakeMq(enqueued)}
    worker._irecv_done_mq = None
    monkeypatch.setattr(worker, "drop_connection_early_recv", lambda connection: drops.append(connection))
    monkeypatch.setattr(
        worker,
        "_lwd_fail_early_recv_connection",
        lambda connection, seqno, exc: fails.append((connection, seqno)),
    )
    return worker


class _FakeMq:
    def __init__(self, enqueued: list) -> None:
        self._enqueued = enqueued

    def enqueue(self, item: Any) -> None:
        self._enqueued.append(item)


def test_link_class_early_recv_error_stays_transient(monkeypatch) -> None:  # noqa: ANN001
    drops: list = []
    fails: list = []
    enqueued: list = []
    worker = _worker(monkeypatch, drops, fails, enqueued)
    worker._lwd_route_early_recv_error(CONNECTION, 7, _runtime_error_from(LwdLinkError("link down")))
    # Transient: ledger dropped, link_error reported, nothing failed.
    assert drops == [CONNECTION]
    assert fails == []
    assert enqueued == [(CONNECTION, "link_error", 7, repr(_runtime_error_from(LwdLinkError("link down"))))]
    assert worker._early_recv_failed_connections == set()


def test_application_class_early_recv_error_keeps_the_failure_path(monkeypatch) -> None:  # noqa: ANN001
    drops: list = []
    fails: list = []
    enqueued: list = []
    worker = _worker(monkeypatch, drops, fails, enqueued)
    worker._lwd_route_early_recv_error(CONNECTION, 7, ValueError("protocol violation"))
    assert drops == []
    assert fails == [(CONNECTION, 7)]
    assert enqueued == []
