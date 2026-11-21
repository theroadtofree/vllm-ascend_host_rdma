# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU tests for the cloud worker's staged early-receive lifecycle."""

from __future__ import annotations

import ast
import enum
import importlib.util
import logging
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, path: Path, modules: dict[str, types.ModuleType] | None = None):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with mock.patch.dict(sys.modules, modules or {}):
        spec.loader.exec_module(module)
    return module


lwd_types = load_module(
    "lwd_types_for_cloud_early_recv_test",
    ROOT / "vllm_ascend/distributed/lwd_data_plane/lwd_types.py",
)
envs = load_module("envs_for_cloud_early_recv_test", ROOT / "vllm_ascend/envs.py")
parents = {
    name: types.ModuleType(name)
    for name in (
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.lwd_data_plane",
    )
}
pre_recv = load_module(
    "lwd_pre_recv_for_cloud_early_recv_test",
    ROOT / "vllm_ascend/worker/lwd_cloud_model/lwd_pre_recv.py",
    {
        **parents,
        "vllm_ascend.envs": envs,
        "vllm_ascend.distributed.lwd_data_plane.lwd_types": lwd_types,
    },
)
fake_vllm = types.ModuleType("vllm")
fake_logger_module = types.ModuleType("vllm.lwd_logger")
fake_logger_module.lwd_logger = logging.getLogger("lwd-cloud-early-recv-test")
host_stage = load_module(
    "lwd_host_stage_for_cloud_early_recv_test",
    ROOT / "vllm_ascend/worker/lwd_cloud_model/lwd_host_stage.py",
    {"vllm": fake_vllm, "vllm.lwd_logger": fake_logger_module},
)


def load_worker_methods():
    path = ROOT / "vllm_ascend/worker/lwd_cloud_model/lwd_cloud_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_init_lwd_early_recv_state",
        "_lwd_init_lwd_host_stager",
        "_lwd_build_up_recv_request",
        "_lwd_early_recv_loop",
        "_early_recv_error_seqno",
        "_lwd_fail_early_recv_connection",
        "_lwd_notify_early_recv_error",
        "start_early_irecv",
        "_lwd_submit_early_recv",
        "_early_recv_pump",
        "_early_recv_quota",
        "_pick_pending_locked",
        "_report_irecv_completions",
        "_lwd_up_post_recvs",
        "_lwd_cloud_flush_finished",
        "lwd_attach_runtime_connection_mqs",
    }
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    namespace = {
        "Base": object,
        "threading": threading,
        "time": time,
        "torch": NS(npu=NS(set_device=lambda device: None)),
        "LwdChannelType": lwd_types.LwdChannelType,
        "LwdTransferRequest": lwd_types.LwdTransferRequest,
        "connection_from": lwd_types.connection_from,
        "LwdHostStageSlotBusy": host_stage.LwdHostStageSlotBusy,
        "LwdHostStager": host_stage.LwdHostStager,
        "LwdRecvHintFields": pre_recv.LwdRecvHintFields,
        "hash_layer_count": lambda hf_config: hf_config.num_hash_layers,
        "hash_payload_numel": lambda rows, hidden, layers, top_k: rows * (hidden + 4 * layers * top_k),
        "lwd_get_data_plane": lambda: None,
        "lwd_logger": fake_logger_module.lwd_logger,
        "_LWD_EARLY_RECV_DEQUEUE_TIMEOUT_S": 0.0005,
        "_LWD_EARLY_RECV_ERROR_SLEEP_S": 0.01,
        "_LWD_EARLY_RECV_LOOP_SLEEP_S": 0.002,
        "_LWD_EARLY_RECV_MAX_INFLIGHT": 2,
        "_LWD_HOST_STAGE_SLOTS": 2,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["LwdCloudWorker"], namespace


Worker, worker_globals = load_worker_methods()
DEFAULT_CONNECTION = lwd_types.ConnectionId(0, 0, 0)
OTHER_CONNECTION = lwd_types.ConnectionId(1, 0, 0)


class FakeBuffer:
    def __init__(self, numel: int) -> None:
        self._numel = numel

    def numel(self) -> int:
        return self._numel

    def __getitem__(self, key: slice) -> FakeBuffer:
        start = key.start or 0
        stop = self._numel if key.stop is None else key.stop
        return FakeBuffer(stop - start)


class FakeFuture:
    def __init__(self) -> None:
        self.is_done = True
        self.is_staged = True
        self.error: RuntimeError | None = None
        self.staged_error: RuntimeError | None = None
        self.events: list[str] = []

    def done(self) -> bool:
        self.events.append("done")
        return self.is_done

    def result(self) -> None:
        self.events.append("result")
        if self.error is not None:
            raise self.error

    def staged_done(self) -> bool:
        self.events.append("staged_done")
        if self.staged_error is not None:
            self.error = self.staged_error
            return True
        return self.is_staged

    def release_device_tensor(self) -> None:
        self.events.append("release")


class FakeDataPlane:
    def __init__(self) -> None:
        self.requests: list[object] = []
        self.futures: list[FakeFuture] = []
        self.error: RuntimeError | None = None
        self.errors: dict[object, RuntimeError] = {}
        self.aborted_connections: list[object] = []

    def submit_recv(self, request: object) -> FakeFuture:
        error = self.errors.get(lwd_types.connection_from(request), self.error)
        if error is not None:
            raise error
        future = FakeFuture()
        self.requests.append(request)
        self.futures.append(future)
        return future

    def abort_connection(self, connection: object, error: RuntimeError) -> None:
        self.aborted_connections.append(connection)


class FakeDoneMq:
    def __init__(self) -> None:
        self.items: list[object] = []
        self.on_enqueue = None

    def enqueue(self, item: object) -> None:
        self.items.append(item)
        if self.on_enqueue is not None:
            self.on_enqueue()


class FakeBatchType(enum.Enum):
    LWD_EMBED = "lwd_embed"
    LWD_UNEMBED = "lwd_unembed"


def make_hint(
    seqno: int,
    req_ids: list[str],
    rows: list[int],
    offsets: list[int],
    prompt_lens: list[int],
    refills: list[bool] | None = None,
    prefix_nonces: list[int] | None = None,
    connection: lwd_types.ConnectionId = DEFAULT_CONNECTION,
) -> dict:
    return pre_recv.make_recv_hint(
        seqno,
        sum(rows),
        0,
        connection_id=connection,
        req_ids=req_ids,
        rows=rows,
        prompt_offsets=offsets,
        prompt_lens=prompt_lens,
        refills=refills,
        prefix_nonces=prefix_nonces,
    )


def start_early_irecv(
    worker,
    hint: dict,
    connection: lwd_types.ConnectionId = DEFAULT_CONNECTION,
) -> None:
    Worker.start_early_irecv(worker, hint, connection_id=connection)


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch):
    packages = {
        name: types.ModuleType(name)
        for name in (
            "vllm_ascend",
            "vllm_ascend.worker",
            "vllm_ascend.worker.lwd_cloud_model",
            "vllm",
            "vllm.v1",
            "vllm.v1.core",
            "vllm.v1.core.sched",
        )
    }
    output = types.ModuleType("vllm.v1.core.sched.output")
    output.LwdBatchType = FakeBatchType
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.output", output)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv",
        pre_recv,
    )
    data_plane = FakeDataPlane()
    monkeypatch.setitem(worker_globals, "lwd_get_data_plane", lambda: data_plane)
    instance = Worker()
    instance._init_lwd_early_recv_state()
    instance.device = "cpu"
    instance.rank = 0
    instance._lwd_dp_idx = 0
    instance._lwd_up_hidden_size = 4
    instance._lwd_up_num_hash_layers = 0
    instance._lwd_up_num_experts_per_tok = 0
    instance._early_recv_max_inflight = 2
    instance._lwd_stager = host_stage.LwdHostStager(
        device="cpu",
        slots=2,
        empty_factory=FakeBuffer,
        device_factory=FakeBuffer,
    )
    instance._lwd_early_recv_connections = (DEFAULT_CONNECTION,)
    instance._irecv_done_mqs = {DEFAULT_CONNECTION: FakeDoneMq()}
    instance._data_plane = data_plane
    instance.model_runner = NS(lwd_release_finished=lambda req_ids: None)
    return instance


@pytest.fixture
def two_connection_worker(worker):
    worker._lwd_early_recv_connections = (DEFAULT_CONNECTION, OTHER_CONNECTION)
    worker._irecv_done_mqs[OTHER_CONNECTION] = FakeDoneMq()
    return worker


def test_host_stager_init_uses_scheduler_token_capacity(worker) -> None:
    worker._lwd_up_num_hash_layers = 2
    worker._lwd_up_num_experts_per_tok = 3
    worker.vllm_config = NS(scheduler_config=NS(max_num_batched_tokens=7))

    worker._lwd_init_lwd_host_stager()
    assert worker._early_recv_max_inflight == 2
    assert worker._lwd_stager.slots == 2
    assert worker._lwd_stager.max_numel == 7 * (4 + 4 * 2 * 3)


def test_completion_reports_cumulative_rows_and_releases_device(worker) -> None:
    start_early_irecv(worker, make_hint(0, ["a", "b"], [2, 2], [0, 0], [5, 4]))
    worker._report_irecv_completions()
    start_early_irecv(worker, make_hint(1, ["a", "b"], [3, 2], [2, 2], [5, 4]))
    worker._report_irecv_completions()

    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [
        ((DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 0), [("a", 2), ("b", 2)]),
        ((DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 1), [("a", 5), ("b", 4)]),
    ]
    assert worker._data_plane.futures[0].events == ["done", "result", "staged_done", "result", "release"]
    assert worker._early_recv_last_delivered[DEFAULT_CONNECTION] == 1
    assert not worker._early_recv_futures


def test_refill_done_reports_nonce_after_staging_before_watermark(worker) -> None:
    start_early_irecv(worker, make_hint(0, ["a"], [6], [4], [10], prefix_nonces=[7]))
    worker._report_irecv_completions()
    start_early_irecv(worker, make_hint(1, ["a"], [2], [2], [10], refills=[True], prefix_nonces=[7]))
    future = worker._data_plane.futures[1]
    future.is_staged = False

    worker._report_irecv_completions()
    assert len(worker._irecv_done_mqs[DEFAULT_CONNECTION].items) == 1
    assert worker._lwd_stager.take_refill_done(DEFAULT_CONNECTION) == []

    future.is_staged = True
    worker._report_irecv_completions()
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [
        ((DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 0), [("a", 10)]),
        ("refill_done", "a", 7),
        ((DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 1), [("a", 4)]),
    ]


def test_aborted_refill_reports_only_watermark(worker) -> None:
    start_early_irecv(worker, make_hint(0, ["a"], [6], [4], [10]))
    worker._report_irecv_completions()
    start_early_irecv(worker, make_hint(1, ["a"], [2], [2], [10], refills=[True]))
    scheduler_output = NS(
        lwd_batch=NS(
            batch_type=FakeBatchType.LWD_EMBED,
            batch_meta=NS(aborted_req_ids=["a"]),
        )
    )
    worker._lwd_up_post_recvs(scheduler_output)

    worker._report_irecv_completions()
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [
        ((DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 0), [("a", 10)]),
        ((DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 1), []),
    ]


def test_unstaged_completion_is_not_published(worker) -> None:
    start_early_irecv(worker, make_hint(0, ["a"], [2], [0], [4]))
    future = worker._data_plane.futures[0]
    future.is_staged = False

    worker._report_irecv_completions()
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == []
    assert worker._lwd_stager.recv_rows("a") == 0
    assert worker._early_recv_last_delivered.get(DEFAULT_CONNECTION, -1) == -1
    assert (DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 0) in worker._early_recv_futures

    future.is_staged = True
    worker._report_irecv_completions()
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [
        ((DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 0), [("a", 2)])
    ]


def test_receive_error_is_not_published(worker) -> None:
    start_early_irecv(worker, make_hint(0, ["a"], [2], [0], [4]))
    future = worker._data_plane.futures[0]
    future.error = RuntimeError("recv failed")

    worker._report_irecv_completions()
    error = repr(RuntimeError("recv failed"))
    assert future.events == ["done", "result", "release"]
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [(DEFAULT_CONNECTION, "error", 0, error)]
    assert worker._lwd_stager.recv_rows("a") == 0
    assert worker._lwd_stager.busy_slots() == []


def test_staged_event_error_is_not_published_and_cleans_slot(worker) -> None:
    start_early_irecv(worker, make_hint(0, ["a"], [2], [0], [4]))
    future = worker._data_plane.futures[0]
    error = RuntimeError("D2H failed")
    future.staged_error = error

    worker._report_irecv_completions()
    assert future.events == ["done", "result", "staged_done", "result", "release"]
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [(DEFAULT_CONNECTION, "error", 0, repr(error))]
    assert worker._lwd_stager.recv_rows("a") == 0
    assert worker._lwd_stager.busy_slots() == []


def test_duplicate_hints_do_not_submit_or_reappear_after_delivery(worker) -> None:
    hint = make_hint(0, ["a"], [2], [0], [4])
    start_early_irecv(worker, hint)
    start_early_irecv(worker, hint)
    assert len(worker._data_plane.requests) == 1

    worker._report_irecv_completions()
    start_early_irecv(worker, hint)
    assert len(worker._data_plane.requests) == 1


def test_busy_slot_keeps_hint_queued_until_delivery(worker) -> None:
    worker._lwd_stager = host_stage.LwdHostStager(
        device="cpu",
        slots=1,
        empty_factory=FakeBuffer,
        device_factory=FakeBuffer,
    )
    start_early_irecv(worker, make_hint(0, ["a"], [2], [0], [4]))
    start_early_irecv(worker, make_hint(1, ["b"], [1], [0], [2]))
    assert len(worker._data_plane.requests) == 1
    assert (DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 1) in worker._early_recv_pending
    assert worker._lwd_stager.staged_count == 1

    worker._report_irecv_completions()
    assert len(worker._data_plane.requests) == 2
    assert not worker._early_recv_pending


def test_submit_failure_abandons_device_allocation(worker) -> None:
    error = RuntimeError("submit failed")
    worker._data_plane.error = error
    start_early_irecv(worker, make_hint(0, ["a"], [2], [0], [4]))
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [(DEFAULT_CONNECTION, "error", 0, repr(error))]
    assert worker._lwd_stager.busy_slots() == []
    assert not worker._early_recv_pending
    assert not worker._early_recv_futures


def test_loop_reports_submit_error_and_stops_reading_hints(worker) -> None:
    hints = iter([make_hint(4, ["a"], [2], [0], [4]), make_hint(5, ["b"], [1], [0], [2])])
    dequeues = []

    def dequeue(timeout: float) -> dict:
        dequeues.append(timeout)
        return next(hints)

    error = RuntimeError("submit failed")
    worker._data_plane.error = error
    worker._irecv_hint_mqs = {DEFAULT_CONNECTION: NS(dequeue=dequeue)}
    worker._lwd_early_recv_loop()

    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [(DEFAULT_CONNECTION, "error", 4, repr(error))]
    assert len(dequeues) == 1
    assert worker._lwd_stager.busy_slots() == []
    assert not worker._early_recv_pending


def test_loop_reports_staged_error_and_cleans_slot(worker) -> None:
    start_early_irecv(worker, make_hint(6, ["a"], [2], [0], [4]))
    error = RuntimeError("D2H failed")
    worker._data_plane.futures[0].staged_error = error
    worker._irecv_done_mqs[DEFAULT_CONNECTION].on_enqueue = worker._lwd_early_recv_stop.set

    worker._lwd_early_recv_loop()
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [(DEFAULT_CONNECTION, "error", 6, repr(error))]
    assert worker._lwd_stager.recv_rows("a") == 0
    assert worker._lwd_stager.busy_slots() == []


def test_abort_only_batch_and_finish_release_host_descriptors(worker) -> None:
    start_early_irecv(worker, make_hint(0, ["a", "b"], [2, 1], [0, 0], [4, 3]))
    worker._report_irecv_completions()
    scheduler_output = NS(
        lwd_batch=NS(
            batch_type=FakeBatchType.LWD_EMBED,
            seqno=-1,
            batch_meta=NS(req_ids=[], token_ids=[], aborted_req_ids=["a"]),
        )
    )
    worker._lwd_up_post_recvs(scheduler_output)
    assert worker._lwd_stager.take_req("a").aborted
    assert worker._lwd_stager.request_count == 2

    released = []
    worker.model_runner = NS(lwd_release_finished=released.extend)
    worker._lwd_cloud_flush_finished(["a", "b"])
    assert released == ["a", "b"]
    assert worker._lwd_stager.request_count == 0
    assert worker._lwd_stager.host_bytes == 0
    assert worker._lwd_stager.staged_count == 0


def test_finish_before_publish_drains_descriptor_but_reports_watermark(worker) -> None:
    start_early_irecv(worker, make_hint(0, ["a"], [2], [0], [4]))
    worker._lwd_cloud_flush_finished(["a"])
    assert worker._lwd_stager.host_bytes == 0
    assert worker._lwd_stager.staged_count == 1

    worker._report_irecv_completions()
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [
        ((DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 0), [])
    ]
    assert worker._lwd_stager.host_bytes == 0
    assert worker._lwd_stager.staged_count == 0
    assert worker._early_recv_last_delivered[DEFAULT_CONNECTION] == 0


def test_two_connections_have_independent_lifecycles(two_connection_worker) -> None:
    worker = two_connection_worker
    for connection, req_id in ((DEFAULT_CONNECTION, "a"), (OTHER_CONNECTION, "b")):
        start_early_irecv(
            worker,
            make_hint(0, [req_id], [2], [0], [4], connection=connection),
            connection,
        )
    for connection, req_id in ((DEFAULT_CONNECTION, "a"), (OTHER_CONNECTION, "b")):
        start_early_irecv(
            worker,
            make_hint(1, [req_id], [2], [2], [4], connection=connection),
            connection,
        )

    assert len(worker._early_recv_futures) == 2
    assert len(worker._early_recv_pending) == 2
    worker._report_irecv_completions()

    for connection, req_id in ((DEFAULT_CONNECTION, "a"), (OTHER_CONNECTION, "b")):
        assert worker._irecv_done_mqs[connection].items == [
            ((connection, lwd_types.LwdChannelType.UP, 0), [(req_id, 2)])
        ]
        worker._lwd_up_post_recvs(
            NS(
                lwd_batch=NS(
                    batch_type=FakeBatchType.LWD_EMBED,
                    batch_meta=NS(connection_id=connection, aborted_req_ids=[req_id]),
                )
            )
        )

    worker._report_irecv_completions()
    for connection, req_id in ((DEFAULT_CONNECTION, "a"), (OTHER_CONNECTION, "b")):
        assert worker._irecv_done_mqs[connection].items == [
            ((connection, lwd_types.LwdChannelType.UP, 0), [(req_id, 2)]),
            ((connection, lwd_types.LwdChannelType.UP, 1), []),
        ]
        assert worker._early_recv_last_delivered[connection] == 1
    assert not worker._early_recv_futures
    assert not worker._early_recv_pending


@pytest.mark.parametrize("failure_kind", ["submit", "future", "staged_event"])
def test_connection_failure_does_not_stop_other(two_connection_worker, failure_kind: str) -> None:
    worker = two_connection_worker
    error = RuntimeError(f"{failure_kind} failed")
    if failure_kind == "submit":
        worker._data_plane.errors[DEFAULT_CONNECTION] = error

    start_early_irecv(worker, make_hint(0, ["a"], [2], [0], [4]))
    start_early_irecv(
        worker,
        make_hint(0, ["b"], [2], [0], [4], connection=OTHER_CONNECTION),
        OTHER_CONNECTION,
    )
    if failure_kind != "submit":
        future = worker._data_plane.futures[0]
        if failure_kind == "future":
            future.error = error
        else:
            future.staged_error = error

    worker._report_irecv_completions()
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [(DEFAULT_CONNECTION, "error", 0, repr(error))]
    assert worker._irecv_done_mqs[OTHER_CONNECTION].items == [
        ((OTHER_CONNECTION, lwd_types.LwdChannelType.UP, 0), [("b", 2)])
    ]
    assert worker._data_plane.aborted_connections == [DEFAULT_CONNECTION]

    start_early_irecv(
        worker,
        make_hint(1, ["b"], [2], [2], [4], connection=OTHER_CONNECTION),
        OTHER_CONNECTION,
    )
    worker._report_irecv_completions()
    assert worker._irecv_done_mqs[OTHER_CONNECTION].items[-1] == (
        (OTHER_CONNECTION, lwd_types.LwdChannelType.UP, 1),
        [("b", 4)],
    )
    assert worker._early_recv_failed_connections == {DEFAULT_CONNECTION}
    assert worker._early_recv_last_delivered == {OTHER_CONNECTION: 1}


def test_max_inflight_two_round_robins_connections(two_connection_worker) -> None:
    worker = two_connection_worker
    for connection in (DEFAULT_CONNECTION, OTHER_CONNECTION):
        start_early_irecv(
            worker,
            make_hint(0, [str(connection.edge_id)], [2], [0], [4], connection=connection),
            connection,
        )
    for connection in (DEFAULT_CONNECTION, OTHER_CONNECTION):
        for seqno in (1, 2):
            start_early_irecv(
                worker,
                make_hint(
                    seqno,
                    [str(connection.edge_id)],
                    [1],
                    [seqno],
                    [4],
                    connection=connection,
                ),
                connection,
            )

    assert len(worker._early_recv_futures) == 2
    assert len(worker._early_recv_pending) == 4
    worker._report_irecv_completions()
    submitted = [(lwd_types.connection_from(request), request.seqno) for request in worker._data_plane.requests]
    assert submitted == [
        (DEFAULT_CONNECTION, 0),
        (OTHER_CONNECTION, 0),
        (DEFAULT_CONNECTION, 1),
        (OTHER_CONNECTION, 1),
    ]

    worker._report_irecv_completions()
    submitted = [(lwd_types.connection_from(request), request.seqno) for request in worker._data_plane.requests]
    assert submitted[4:] == [(DEFAULT_CONNECTION, 2), (OTHER_CONNECTION, 2)]
    assert len(worker._early_recv_futures) == 2
    assert not worker._early_recv_pending


def test_idle_connection_capacity_is_reused(two_connection_worker) -> None:
    worker = two_connection_worker
    for seqno in (0, 1):
        start_early_irecv(
            worker,
            make_hint(seqno, ["a"], [1], [seqno], [4]),
        )

    assert [(lwd_types.connection_from(request), request.seqno) for request in worker._data_plane.requests] == [
        (DEFAULT_CONNECTION, 0),
        (DEFAULT_CONNECTION, 1),
    ]
    assert len(worker._early_recv_futures) == 2
    assert not worker._early_recv_pending


def test_watermark_gap_isolated_and_recovers_without_starvation(
    two_connection_worker,
) -> None:
    worker = two_connection_worker
    start_early_irecv(worker, make_hint(0, ["a"], [2], [0], [4]))
    start_early_irecv(
        worker,
        make_hint(0, ["b"], [2], [0], [4], connection=OTHER_CONNECTION),
        OTHER_CONNECTION,
    )
    blocked_future = worker._data_plane.futures[0]
    blocked_future.is_done = False
    for connection, req_id in ((DEFAULT_CONNECTION, "a"), (OTHER_CONNECTION, "b")):
        start_early_irecv(
            worker,
            make_hint(1, [req_id], [2], [2], [4], connection=connection),
            connection,
        )

    worker._report_irecv_completions()
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == []
    assert worker._irecv_done_mqs[OTHER_CONNECTION].items == [
        ((OTHER_CONNECTION, lwd_types.LwdChannelType.UP, 0), [("b", 2)])
    ]
    assert (OTHER_CONNECTION, lwd_types.LwdChannelType.UP, 1) in worker._early_recv_futures

    worker._report_irecv_completions()
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == []
    assert worker._irecv_done_mqs[OTHER_CONNECTION].items[-1] == (
        (OTHER_CONNECTION, lwd_types.LwdChannelType.UP, 1),
        [("b", 4)],
    )
    assert (DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 1) in worker._early_recv_futures
    assert DEFAULT_CONNECTION not in worker._early_recv_last_delivered

    blocked_future.is_done = True
    worker._report_irecv_completions()
    assert worker._irecv_done_mqs[DEFAULT_CONNECTION].items == [
        ((DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 0), [("a", 2)]),
        ((DEFAULT_CONNECTION, lwd_types.LwdChannelType.UP, 1), [("a", 4)]),
    ]
    assert worker._early_recv_last_delivered == {
        DEFAULT_CONNECTION: 1,
        OTHER_CONNECTION: 1,
    }
    assert not worker._early_recv_futures
    assert not worker._early_recv_pending


def test_runtime_attach_connection_mqs_uses_zero_arg_report_factory(worker, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime MQ attachment must use the same zero-arg factory as startup."""
    worker._lwd_early_recv_connections = (OTHER_CONNECTION,)
    worker._irecv_hint_mqs = {}
    worker._irecv_done_mqs = {}
    hint_mq = object()
    report_mq = NS(export_handle=lambda: "done-handle")
    report_calls = []

    def create_report_mq():
        report_calls.append(())
        return report_mq

    monkeypatch.setattr(pre_recv, "create_report_mq", create_report_mq)
    monkeypatch.setattr(pre_recv, "_decode_handle", lambda raw: raw)
    message_queue = NS(create_from_handle=lambda handle, rank: hint_mq)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.shm_broadcast",
        NS(MessageQueue=message_queue),
    )

    handle = worker.lwd_attach_runtime_connection_mqs(OTHER_CONNECTION, "hint-handle")

    assert handle == "done-handle"
    assert worker._irecv_hint_mqs[OTHER_CONNECTION] is hint_mq
    assert worker._irecv_done_mqs[OTHER_CONNECTION] is report_mq
    assert report_calls == [()]
    assert worker.lwd_attach_runtime_connection_mqs(OTHER_CONNECTION, "hint-handle") == "done-handle"
    assert report_calls == [()]
