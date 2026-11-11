# SPDX-License-Identifier: Apache-2.0
"""Guard tests of the host data-plane bootstrap, multi-instance aware.

These build real ``LwdTopology`` objects but drive
``lwd_init_host_channels`` with fake listeners and IOs: no sockets, no
libucp, no NPU. They pin the registry layout of each scene -- in
cloud_share / lwd_cluster several edges connect to one cloud DP, which
must bind exactly ONE shared listener instead of one per connection.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from vllm.config.lwd.topology import LwdDP, LwdDPConnection, LwdTopology

from vllm_ascend.distributed.lwd_data_plane.host import lwd_host_channels
from vllm_ascend.distributed.lwd_data_plane.host.lwd_host_channels import _link_token
from vllm_ascend.distributed.lwd_data_plane.host.ucx_io import LwdNpuUcxChannelIoConfig
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

UP = LwdChannelType.UP
DOWN = LwdChannelType.DOWN


class _FakeListener:
    """Stands in for LwdUcxListener: records the bind and the accept start."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.start_calls = 0

    def start(self) -> None:
        self.start_calls += 1


class _FakeIo:
    """Stands in for LwdNpuUcxChannelIo: records the handed-in config."""

    def __init__(self, channel: LwdChannelType, stream: object, config: LwdNpuUcxChannelIoConfig) -> None:
        self.channel = channel
        self.config = config

    def wait_ready(self, timeout: float | None = None) -> bool:
        return True


@pytest.fixture(autouse=True)
def _reset_host_channel_state() -> Iterator[None]:
    yield
    lwd_host_channels._UCX_IOS.clear()
    lwd_host_channels._UCX_LISTENERS.clear()
    lwd_host_channels._UCX_CHANNELS_INITIALIZED = False


def _cloud_share_topology(*, sidecar: bool = False) -> LwdTopology:
    """Two edges sharing one cloud DP -- the scene that needs listener sharing."""
    return LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "cloud_share",
                "edges_num": 2,
                "clouds_num": 1,
                "data_plane_comm": "host",
                "data_plane_host": {"deployment": "sidecar"} if sidecar else None,
            },
            "feature_ctrl": {},
            "edges": [
                {"id": 0, "dp": [{"dp_idx": 0, "addr": "10.0.0.1"}], "coordination": {"enabled": True}},
                {"id": 1, "dp": [{"dp_idx": 0, "addr": "10.0.0.2"}], "coordination": {"enabled": True}},
            ],
            "clouds": [
                {
                    "id": 0,
                    "dp": [{"dp_idx": 0, "addr": "10.1.0.1", "ctrl_port": 7000, "data_port": 5560}],
                    "coordination": {"enabled": True},
                }
            ],
            "instance_links": [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 0}],
        }
    )


def _single_instance_topology(*, memory_budget_mb: int | None = None) -> LwdTopology:
    return LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "single_instance",
                "edges_num": 1,
                "clouds_num": 1,
                "data_plane_comm": "host",
                "data_plane_host": {"memory_budget_mb": memory_budget_mb} if memory_budget_mb is not None else None,
            },
            "feature_ctrl": {},
            "edges": [{"id": 0, "dp": [{"dp_idx": 0, "addr": "10.0.0.1"}]}],
            "clouds": [{"id": 0, "dp": [{"dp_idx": 0, "addr": "10.1.0.1", "ctrl_port": 7000, "data_port": 5560}]}],
            "instance_links": [{"edge": 0, "cloud": 0}],
        }
    )


def _lwd_cluster_topology() -> LwdTopology:
    """One edge fully meshed with two clouds -- the edge-side multi-instance form."""
    return LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "lwd_cluster",
                "edges_num": 1,
                "clouds_num": 2,
                "data_plane_comm": "host",
            },
            "feature_ctrl": {},
            "edges": [{"id": 0, "dp": [{"dp_idx": 0, "addr": "10.0.0.1"}], "coordination": {"enabled": True}}],
            "clouds": [
                {
                    "id": 0,
                    "dp": [{"dp_idx": 0, "addr": "10.1.0.1", "ctrl_port": 7000, "data_port": 5560}],
                    "coordination": {"enabled": True},
                },
                {
                    "id": 1,
                    "dp": [{"dp_idx": 0, "addr": "10.1.0.2", "ctrl_port": 7000, "data_port": 5560}],
                    "coordination": {"enabled": True},
                },
            ],
            "instance_links": [{"edge": 0, "cloud": 0}, {"edge": 0, "cloud": 1}],
        }
    )


def _run_init(
    monkeypatch: pytest.MonkeyPatch,
    topology: LwdTopology | SimpleNamespace,
    *,
    role: str,
    instance_id: int,
    dp_rank: int = 0,
    tp_rank: int = 0,
) -> list[_FakeListener]:
    """Run the bootstrap with fake listeners, IOs and identity; return the binds."""
    made_listeners: list[_FakeListener] = []

    def fake_listener(host: str, port: int) -> _FakeListener:
        listener = _FakeListener(host, port)
        made_listeners.append(listener)
        return listener

    vllm_config = SimpleNamespace(
        lwd_config=SimpleNamespace(topology=topology, role=role, instance_id=instance_id),
        parallel_config=SimpleNamespace(data_parallel_rank=dp_rank),
    )
    monkeypatch.setattr(lwd_host_channels, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(lwd_host_channels, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(lwd_host_channels, "_resolve_up_block_mb", lambda: 32)
    monkeypatch.setattr(lwd_host_channels, "LwdUcxListener", fake_listener)
    monkeypatch.setattr(lwd_host_channels, "LwdNpuUcxChannelIo", _FakeIo)
    monkeypatch.setattr(
        lwd_host_channels.torch,
        "npu",
        SimpleNamespace(Stream=lambda priority=-1: object()),
        raising=False,
    )
    lwd_host_channels.lwd_init_host_channels()
    return made_listeners


def test_cloud_share_binds_one_listener_for_all_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two edges, one cloud DP: every connection reuses the same bound listener."""
    listeners = _run_init(monkeypatch, _cloud_share_topology(), role="cloud", instance_id=0)
    assert len(listeners) == 1, "one cloud DP must bind one listener, not one per connection"
    (listener,) = listeners
    assert (listener.host, listener.port) == ("10.1.0.1", 5560)
    assert listener.start_calls == 1
    assert set(lwd_host_channels._UCX_LISTENERS) == {(0, 0)}
    assert lwd_host_channels._UCX_LISTENERS[(0, 0)] is listener
    # Every edge connection owns its UP/DOWN pair, all served by the ONE listener.
    ios = lwd_host_channels._UCX_IOS
    assert set(ios) == {
        (0, 0, 0, UP, "recv"),
        (0, 0, 0, DOWN, "send"),
        (1, 0, 0, UP, "recv"),
        (1, 0, 0, DOWN, "send"),
    }
    for io in ios.values():
        assert io.config.listener is listener
    # Tokens stay per-connection (edge, cloud, dp, channel): four distinct ones.
    assert ios[(1, 0, 0, UP, "recv")].config.token == _link_token(1, 0, 0, UP)
    assert len({io.config.token for io in ios.values()}) == 4


def test_single_instance_cloud_keeps_one_listener_and_two_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listeners = _run_init(monkeypatch, _single_instance_topology(), role="cloud", instance_id=0)
    assert len(listeners) == 1
    assert set(lwd_host_channels._UCX_IOS) == {(0, 0, 0, UP, "recv"), (0, 0, 0, DOWN, "send")}


def test_edge_role_connects_out_without_listeners(monkeypatch: pytest.MonkeyPatch) -> None:
    """lwd_cluster edge: one link pair per cloud, each at that cloud's addr; never binds."""
    _run_init(monkeypatch, _lwd_cluster_topology(), role="edge", instance_id=0)
    assert lwd_host_channels._UCX_LISTENERS == {}
    ios = lwd_host_channels._UCX_IOS
    assert set(ios) == {
        (0, 0, 0, UP, "send"),
        (0, 0, 0, DOWN, "recv"),
        (0, 1, 0, UP, "send"),
        (0, 1, 0, DOWN, "recv"),
    }
    for io in ios.values():
        assert io.config.listener is None
        assert io.config.serve is False
    # The edge resolves each cloud instance's own address, not a single one.
    assert ios[(0, 0, 0, UP, "send")].config.host == "10.1.0.1"
    assert ios[(0, 1, 0, UP, "send")].config.host == "10.1.0.2"
    assert {io.config.port for io in ios.values()} == {5560}


def test_sidecar_cloud_attaches_without_an_engine_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sidecar deployment: the transport daemon owns the listener, never the engine."""
    _run_init(monkeypatch, _cloud_share_topology(sidecar=True), role="cloud", instance_id=0)
    assert lwd_host_channels._UCX_LISTENERS == {}
    assert len(lwd_host_channels._UCX_IOS) == 4
    for io in lwd_host_channels._UCX_IOS.values():
        assert io.config.listener is None
        assert io.config.transport_mode == "daemon"


def test_non_endpoint_rank_holds_no_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the endpoint process (dp_rank == dp_idx and TP rank 0) owns IO objects."""
    _run_init(monkeypatch, _single_instance_topology(), role="cloud", instance_id=0, tp_rank=1)
    assert lwd_host_channels._UCX_IOS == {}
    assert lwd_host_channels._UCX_LISTENERS == {}


def test_memory_budget_from_the_topology_fails_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-rank budget is topology config: an over-budget need fails before any link."""
    topology = _single_instance_topology(memory_budget_mb=100)
    with pytest.raises(RuntimeError, match="memory budget exceeded"):
        _run_init(monkeypatch, topology, role="cloud", instance_id=0)
    assert lwd_host_channels._UCX_IOS == {}
    assert lwd_host_channels._UCX_LISTENERS == {}


def test_missing_cloud_data_port_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defense in depth: a connection whose cloud DP lacks data_port must not build."""
    conn = LwdDPConnection(
        0,
        0,
        0,
        LwdDP(dp_idx=0, addr="10.0.0.1"),
        LwdDP(dp_idx=0, addr="10.1.0.1", ctrl_port=7000),  # data_port omitted
    )
    topology = SimpleNamespace(
        deployment=SimpleNamespace(data_plane_host=None),
        qos=None,
        connections=lambda role, instance_id: (conn,),
    )
    with pytest.raises(RuntimeError, match="data_port"):
        _run_init(monkeypatch, topology, role="cloud", instance_id=0)
