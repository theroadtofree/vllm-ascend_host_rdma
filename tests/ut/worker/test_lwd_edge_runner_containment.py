# SPDX-License-Identifier: Apache-2.0
"""Edge runner containment: a failed UP send settles its batch and raises
the contained error instead of killing the engine."""

from __future__ import annotations

from collections import deque

import pytest
from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.lwd_exceptions import LwdTransferFailedError

from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
    LwdChannelType,
    LwdTransferRequest,
)
from vllm_ascend.worker.lwd_edge_model.lwd_edge_model_runner import (
    LwdEdgeModelRunner,
)

CONNECTION = LwdConnectionId(0, 0, 0)


def _request(seqno: int) -> LwdTransferRequest:
    return LwdTransferRequest(channel=LwdChannelType.UP, op="send", num_elements=4, seqno=seqno)


def _failed_future(seqno: int) -> LwdTransferFuture:
    future = LwdTransferFuture(_request(seqno), [], None)
    future._finalize(error=RuntimeError(f"transport died on seqno {seqno}"))
    return future


def _runner(inflight: dict) -> LwdEdgeModelRunner:
    runner = object.__new__(LwdEdgeModelRunner)
    runner._lwd_up_send_inflight = inflight
    runner._lwd_up_send_watermarks = {}
    return runner


def test_failed_up_send_settles_the_watermark_and_raises_contained() -> None:
    """The failed batch settles (watermark advances so later sends
    release) and surfaces as LwdTransferFailedError, not a bare engine
    exception."""
    runner = _runner({CONNECTION: deque([(5, _failed_future(5))])})
    with pytest.raises(LwdTransferFailedError) as raised:
        runner._lwd_pop_completed_up_sends()
    assert raised.value.connection == CONNECTION
    assert raised.value.seqnos == [5]
    assert runner._lwd_up_send_watermarks[CONNECTION] == 5
    assert not runner._lwd_up_send_inflight[CONNECTION]


def test_successful_pops_still_advance_the_watermark() -> None:
    future = LwdTransferFuture(_request(3), [], None)
    runner = _runner({CONNECTION: deque([(3, future)])})
    # Not done: the prefix does not advance.
    runner._lwd_pop_completed_up_sends()
    assert runner._lwd_up_send_watermarks == {}
    future._finalize()
    runner._lwd_pop_completed_up_sends()
    assert runner._lwd_up_send_watermarks[CONNECTION] == 3
