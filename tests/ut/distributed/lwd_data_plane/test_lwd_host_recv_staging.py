# SPDX-License-Identifier: Apache-2.0
"""Host-plane direct staging: one host copy, straight from the received block.

On the host data plane the UP frame already arrives in host memory and the
per-request staging buffers live there too, so a staged receive allocates no
landing buffer at all: each segment is copied from the block into its request
buffer exactly once, descrambled in place on the destination, and completion is
reported through host events.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_ascend.distributed.lwd_data_plane.host import ucx_io
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import (
    LwdTransferFuture,
)
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

HIDDEN = 8
ROWS = 3
NUMEL = ROWS * HIDDEN


class _NpuTrap:
    """Fail loudly if the host path reaches for the device at all."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"direct staging touched torch.npu.{name}")


@pytest.fixture(autouse=True)
def npu_trap(monkeypatch) -> None:
    monkeypatch.setattr(ucx_io.torch, "npu", _NpuTrap(), raising=False)


def _recv_io() -> Any:
    """A recv IO with just the state the callback touches."""
    io = object.__new__(ucx_io.LwdNpuUcxChannelIo)
    io.tag = "lwd_up/recv"
    io._tr_recv_done = False
    io._stream = None
    io._copy_stream = None
    io._ch = SimpleNamespace(fail=lambda err: None)
    return io


def _future(stage_segments: list[tuple[torch.Tensor, int, int]] | None) -> LwdTransferFuture:
    request = SimpleNamespace(
        channel=LwdChannelType.UP,
        op="recv",
        seqno=7,
        num_elements=NUMEL,
        stage_segments=stage_segments,
    )
    return LwdTransferFuture(request, [], None)


def _values() -> torch.Tensor:
    """bf16 rows whose ramp makes segment boundaries visible."""
    return torch.arange(NUMEL, dtype=torch.int16).to(torch.bfloat16)


def _payload() -> bytes:
    # bf16 has no numpy dtype: ship the raw bit pattern.
    return _values().view(torch.int16).numpy().tobytes()


def _op(data: bytes | None = None) -> SimpleNamespace:
    payload = _payload() if data is None else data
    return SimpleNamespace(tag=7, data=payload, error=None, size=len(payload))


def test_staged_recv_copies_each_segment_once() -> None:
    first = torch.empty(HIDDEN, dtype=torch.bfloat16)
    tail = torch.empty(2, dtype=torch.bfloat16)
    segments = [(first, 0, HIDDEN), (tail, NUMEL - 2, 2)]
    future = _future(segments)
    request = SimpleNamespace(num_elements=NUMEL, stage_segments=segments)
    cb = _recv_io()._recv_cb(None, None, future, request)

    cb(_op())

    assert future.done()
    assert future.staged_done()
    result = future.result()
    # No landing buffer: the segments are the only copy of the frame.
    assert result.tensor is None
    assert result.aux_tensor is None
    values = _values()
    assert torch.equal(first, values[:HIDDEN])
    assert torch.equal(tail, values[NUMEL - 2 :])


def test_staged_recv_descrambles_each_segment_in_place() -> None:
    dst = torch.empty(NUMEL, dtype=torch.bfloat16)
    segments = [(dst, 0, NUMEL)]
    future = _future(segments)

    def post_h2d(tensor: torch.Tensor, _req: Any) -> None:
        tensor.add_(1)

    cb = _recv_io()._recv_cb(None, None, future, SimpleNamespace(num_elements=NUMEL, stage_segments=segments), post_h2d)

    cb(_op())

    # The segment is descrambled after its copy, so the staged data is plain.
    assert torch.equal(dst, _values() + 1)


def _failure_cause(future: LwdTransferFuture) -> str:
    """result() wraps the transport error; the cause carries the detail."""
    with pytest.raises(RuntimeError) as excinfo:
        future.result()
    cause = excinfo.value.__cause__
    assert cause is not None
    return str(cause)


def test_staged_recv_rejects_a_landing_buffer() -> None:
    """The contract is "omit recv_buffer", not "pass a host buffer"."""
    dst = torch.empty(NUMEL, dtype=torch.bfloat16)
    segments = [(dst, 0, NUMEL)]
    future = _future(segments)
    buffer = torch.empty(NUMEL, dtype=torch.bfloat16)
    cb = _recv_io()._recv_cb(buffer, None, future, SimpleNamespace(num_elements=NUMEL, stage_segments=segments))

    cb(_op())

    assert future.done()
    assert "omit recv_buffer" in _failure_cause(future)
    assert not torch.equal(dst, _values())


def test_staged_recv_fails_fast_on_a_short_frame() -> None:
    dst = torch.empty(NUMEL, dtype=torch.bfloat16)
    segments = [(dst, 0, NUMEL)]
    future = _future(segments)
    cb = _recv_io()._recv_cb(None, None, future, SimpleNamespace(num_elements=NUMEL, stage_segments=segments))

    cb(_op(_payload()[:-2]))

    assert future.done()
    assert "payload size mismatch" in _failure_cause(future)
    # Nothing was staged from a truncated frame.
    assert not torch.equal(dst, _values())


class _Event:
    def __init__(self) -> None:
        self.recorded = False

    def record(self) -> None:
        self.recorded = True

    def query(self) -> bool:
        return self.recorded


class _FakeNpuTensor:
    """Just enough of an NPU tensor for the device-path assertions."""

    def __init__(self, numel: int, dtype: torch.dtype = torch.bfloat16) -> None:
        self.dtype = dtype
        self.device = SimpleNamespace(type="npu")
        self._numel = numel
        self.copies: list[tuple[torch.Tensor, bool]] = []

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return 2

    def copy_(self, src: torch.Tensor, non_blocking: bool = False) -> _FakeNpuTensor:
        self.copies.append((src.clone(), non_blocking))
        return self


def test_non_staged_recv_keeps_the_device_path(monkeypatch) -> None:
    """A recv without stage segments still goes through the NPU stream."""
    stream_calls: list[str] = []

    class _StreamCtx:
        def __enter__(self) -> None:
            stream_calls.append("enter")

        def __exit__(self, *exc: Any) -> None:
            stream_calls.append("exit")

    monkeypatch.setattr(ucx_io.torch, "npu", SimpleNamespace(stream=lambda _s: _StreamCtx(), Event=_Event))
    buffer = _FakeNpuTensor(NUMEL)
    future = _future(None)
    cb = _recv_io()._recv_cb(buffer, None, future, SimpleNamespace(num_elements=NUMEL, stage_segments=None))

    cb(_op())

    assert stream_calls == ["enter", "exit"]
    assert len(buffer.copies) == 1
    assert future.done()
