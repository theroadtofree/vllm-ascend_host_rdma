# SPDX-License-Identifier: Apache-2.0
"""Host channel bootstrap: one listener per cloud DP, shared by its edges.

The listener binds in its constructor, so a second edge connected to the
same cloud DP must reuse that DP's listener instead of binding data_port
again. Edges and sidecar engines build no engine-side listener at all.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.config.lwd import LwdConfig
from vllm.config.lwd.topology import (
    LwdConnectionId,
    LwdDP,
    LwdDPConnection,
    LwdTopology,
)

from vllm_ascend.distributed.lwd_data_plane.host import lwd_host_channels
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

CLOUD_ADDR = "10.1.0.1"
CLOUD_DATA_PORT = 7000

UP = LwdChannelType.UP
DOWN = LwdChannelType.DOWN


class _FakeListener:
    """Stands in for LwdUcxListener: records the bind, the accept start
    and the token route table (serve_via's register call)."""

    created: list[tuple[str, int]] = []

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.started = False
        self.start_calls = 0
        self._routes: dict[int, object] = {}
        _FakeListener.created.append((host, port))

    def register(self, token: int, channel: object, *, name: str = "") -> None:
        self._routes[token] = channel

    def start(self) -> None:
        self.started = True


class _FakeIo:
    """Stands in for LwdNpuUcxChannelIo: records the handed-in config, the
    registered peer-restart hooks and the forced teardowns; when serving
    on a shared listener, registers the token — what the real
    LwdUcxChannel.serve_via does at construction."""

    def __init__(
        self,
        channel: LwdChannelType,
        stream: object,
        config: Any,
    ) -> None:
        self.channel = channel
        self.config = config
        self.peer_restart_cbs: list = []
        self.forced_teardowns: list = []
        if config.serve and config.listener is not None:
            config.listener.register(config.token, self)

    def on_peer_restart(self, cb: object) -> None:
        self.peer_restart_cbs.append(cb)

    def force_teardown(self, reason: str) -> None:
        self.forced_teardowns.append(reason)

    def wait_ready(self, timeout: float) -> bool:
        return True

    def shutdown(self, timeout: float | None = None) -> None:
        self.forced_teardowns.append("shutdown")


# The scale-line tests patch the fakes by their older names.
_Io = _FakeIo
_Listener = _FakeListener


@pytest.fixture(autouse=True)
def _reset_host_channel_state() -> Iterator[None]:
    from vllm.v1.lwd_control.lwd_comm import lwd_data_plane_hooks

    saved_hooks = list(lwd_data_plane_hooks._hooks)
    yield
    lwd_data_plane_hooks._hooks[:] = saved_hooks
    lwd_host_channels._UCX_IOS.clear()
    lwd_host_channels._UCX_LISTENERS.clear()
    lwd_host_channels._UCX_CHANNELS_INITIALIZED = False
    lwd_host_channels._CACHED_PLAN = None
    lwd_host_channels._ALLOCATED_MB = 0


def _cloud_share_topology(*, sidecar: bool = False) -> LwdTopology:
    """Two edges sharing one cloud DP -- the scene that needs listener
    sharing. The coordination contract of this core carries the cloud's
    HTTP endpoint on its own DP block (coord_port/coord_url); the edges'
    coordination needs only the enabled block and a key file."""
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
            "feature_ctrl": {"enable_early_recv": False},
            "edges": [
                {
                    "id": 0,
                    "dp": [{"dp_idx": 0, "addr": "10.0.0.1"}],
                    "coordination": {
                        "enabled": True,
                        "tenant_key_file": "/etc/lwd/tenant-0.key",
                    },
                },
                {
                    "id": 1,
                    "dp": [{"dp_idx": 0, "addr": "10.0.0.2"}],
                    "coordination": {
                        "enabled": True,
                        "tenant_key_file": "/etc/lwd/tenant-1.key",
                    },
                },
            ],
            "clouds": [
                {
                    "id": 0,
                    "dp": [
                        {
                            "dp_idx": 0,
                            "addr": "10.1.0.1",
                            "ctrl_port": 7000,
                            "data_port": 5560,
                            "coord_port": 8100,
                            "coord_url": "http://10.1.0.1:8100",
                        }
                    ],
                    "coordination": {"enabled": True, "listen_host": "0.0.0.0"},
                }
            ],
            "instance_links": [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 0}],
        }
    )


def _instance(identity: int, prefix: str, *, cloud: bool) -> dict[str, Any]:
    dp: dict[str, Any] = {"dp_idx": 0, "addr": f"{prefix}.{identity + 1}"}
    if cloud:
        dp["ctrl_port"] = 5550 + identity
        dp["data_port"] = CLOUD_DATA_PORT + identity
        dp["coord_port"] = 8100 + identity
        dp["coord_url"] = f"http://{prefix}.{identity + 1}:{8100 + identity}"
    # The scene validator needs an enabled coordination block on every
    # instance; an edge also needs a key file. The cloud's HTTP endpoint
    # lives on its own DP block (coord_port/coord_url) on this schema.
    dp_coordination: dict[str, Any] = {"enabled": True}
    if not cloud:
        dp_coordination["tenant_key_file"] = f"/etc/lwd/tenant-{identity}.key"
    return {"id": identity, "dp": [dp], "coordination": dp_coordination}


def _host_config(role: str, *, deployment: str = "in_process") -> LwdConfig:
    """cloud_share (2 edges, 1 cloud) and lwd_cluster (1 edge, 2 clouds)."""
    edges = 2 if role == "cloud" else 1
    clouds = 1 if role == "cloud" else 2
    topology = LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "cloud_share" if role == "cloud" else "lwd_cluster",
                "edges_num": edges,
                "clouds_num": clouds,
                "data_plane_comm": "host",
                "data_plane_host": {
                    "down_buffer_size": 32,
                    "deployment": deployment,
                },
            },
            "feature_ctrl": {"enable_early_recv": False},
            "edges": [_instance(identity, "10.0.0", cloud=False) for identity in range(edges)],
            "clouds": [_instance(identity, "10.1.0", cloud=True) for identity in range(clouds)],
            "instance_links": [{"edge": edge, "cloud": cloud} for edge in range(edges) for cloud in range(clouds)],
        }
    )
    return LwdConfig.from_topology("host_topology.yaml", role, 0, topology)


def _single_instance_config(role: str, *, dp_count: int = 2) -> LwdConfig:
    """single_instance host layout: one edge, one cloud, several DPs."""

    def instance(identity: int, prefix: str, *, cloud: bool) -> dict[str, Any]:
        dps: list[dict[str, Any]] = []
        for dp_idx in range(dp_count):
            dp: dict[str, Any] = {
                "dp_idx": dp_idx,
                "addr": f"{prefix}.{identity + 1}",
            }
            if cloud:
                dp["ctrl_port"] = 5550 + dp_idx
                dp["data_port"] = CLOUD_DATA_PORT + dp_idx
            dps.append(dp)
        return {"id": identity, "dp": dps}

    topology = LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "single_instance",
                "edges_num": 1,
                "clouds_num": 1,
                "data_plane_comm": "host",
                "data_plane_host": {"down_buffer_size": 32},
            },
            "feature_ctrl": {"enable_early_recv": False},
            "edges": [instance(0, "10.0.0", cloud=False)],
            "clouds": [instance(0, "10.1.0", cloud=True)],
            "instance_links": [{"edge": 0, "cloud": 0}],
        }
    )
    return LwdConfig.from_topology("host_topology.yaml", role, 0, topology)


def _edge_share_config() -> LwdConfig:
    """edge_share: one physical edge engine serves two logical edges, each
    connected to its own cloud DP over a one-to-one instance link."""
    topology = LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "edge_share",
                "edges_num": 2,
                "clouds_num": 2,
                "data_plane_comm": "host",
                "data_plane_host": {"down_buffer_size": 32},
            },
            "feature_ctrl": {"enable_early_recv": False},
            "edges": [_instance(identity, "10.0.0", cloud=False) for identity in range(2)],
            "clouds": [_instance(identity, "10.1.0", cloud=True) for identity in range(2)],
            "instance_links": [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 1}],
        }
    )
    return LwdConfig.from_topology("host_topology.yaml", "edge", 0, topology)


@pytest.fixture(autouse=True)
def _host_bootstrap(monkeypatch) -> None:
    """Isolate the module registries and stub every device-side dependency."""
    _Listener.created = []
    monkeypatch.setattr(lwd_host_channels, "_UCX_IOS", {})
    monkeypatch.setattr(lwd_host_channels, "_UCX_LISTENERS", {})
    monkeypatch.setattr(lwd_host_channels, "_UCX_CHANNELS_INITIALIZED", False)
    monkeypatch.setattr(lwd_host_channels, "LwdUcxListener", _Listener)
    monkeypatch.setattr(lwd_host_channels, "LwdNpuUcxChannelIo", _Io)
    monkeypatch.setattr(
        lwd_host_channels.torch,
        "npu",
        SimpleNamespace(Stream=lambda **_: object()),
        raising=False,
    )
    monkeypatch.setattr(lwd_host_channels, "_resolve_up_block_mb", lambda: 8)
    monkeypatch.setattr(
        lwd_host_channels,
        "_qos_data_plane",
        lambda: SimpleNamespace(traffic_class=None, sl=None),
    )


def _init(monkeypatch, config: LwdConfig, *, dp_identity: tuple[int, int] = (0, 0)) -> None:
    monkeypatch.setattr(
        lwd_host_channels,
        "get_current_vllm_config",
        lambda: SimpleNamespace(lwd_config=config),
    )
    monkeypatch.setattr(lwd_host_channels, "_my_dp_identity", lambda: dp_identity)
    lwd_host_channels.lwd_init_host_channels()


def _edge_share_topology() -> LwdTopology:
    """Two 1:1 edge/cloud pairs -- the multi-frontend shared-engine scene."""
    return LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "edge_share",
                "edges_num": 2,
                "clouds_num": 2,
                "data_plane_comm": "host",
            },
            "feature_ctrl": {"enable_early_recv": False},
            "edges": [
                {"id": 0, "dp": [{"dp_idx": 0, "addr": "10.0.0.1"}]},
                {"id": 1, "dp": [{"dp_idx": 0, "addr": "10.0.0.1"}]},
            ],
            "clouds": [
                {
                    "id": 0,
                    "dp": [
                        {
                            "dp_idx": 0,
                            "addr": "10.1.0.1",
                            "ctrl_port": 7000,
                            "data_port": 5560,
                        }
                    ],
                },
                {
                    "id": 1,
                    "dp": [
                        {
                            "dp_idx": 0,
                            "addr": "10.1.0.2",
                            "ctrl_port": 7000,
                            "data_port": 5560,
                        }
                    ],
                },
            ],
            "instance_links": [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 1}],
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
    connections: tuple[LwdDPConnection, ...] | None = None,
) -> list[_FakeListener]:
    """Run the bootstrap with fake listeners, IOs and identity; return the
    binds. Self-contained: it patches everything it needs, so it also works
    for the runtime-addition tests that boot with a restricted connection
    subset (a yaml that planned fewer pairs than the file grown later)."""
    made_listeners: list[_FakeListener] = []

    def fake_listener(host: str, port: int) -> _FakeListener:
        listener = _FakeListener(host, port)
        made_listeners.append(listener)
        return listener

    if connections is None:
        connections = topology.connections(role, instance_id)
    vllm_config = SimpleNamespace(
        lwd_config=SimpleNamespace(
            topology=topology,
            role=role,
            instance_id=instance_id,
            connections=connections,
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=dp_rank),
    )
    monkeypatch.setattr(
        lwd_host_channels, "get_current_vllm_config", lambda: vllm_config
    )
    monkeypatch.setattr(
        lwd_host_channels, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
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


def test_two_edges_of_one_cloud_dp_share_one_listener(monkeypatch) -> None:
    """Regression: the second edge bound data_port again and broke 2E1C."""
    _init(monkeypatch, _host_config("cloud"))

    assert _Listener.created == [(CLOUD_ADDR, CLOUD_DATA_PORT)]
    listener = lwd_host_channels._UCX_LISTENERS[(0, 0)]
    assert listener.started
    ios = lwd_host_channels._UCX_IOS
    assert len(ios) == 4
    for edge_id in (0, 1):
        up = ios[(edge_id, 0, 0, LwdChannelType.UP, "recv")]
        down = ios[(edge_id, 0, 0, LwdChannelType.DOWN, "send")]
        assert up.config.listener is listener
        assert down.config.listener is listener


def test_edge_builds_no_listener(monkeypatch) -> None:
    """Edges connect out to every cloud DP and never listen."""
    _init(monkeypatch, _host_config("edge"))

    assert _Listener.created == []
    assert lwd_host_channels._UCX_LISTENERS == {}
    ios = lwd_host_channels._UCX_IOS
    assert len(ios) == 4
    assert all(config.listener is None for config in (io.config for io in ios.values()))


def test_edge_share_shared_engine_owns_every_logical_edge(monkeypatch) -> None:
    """The one physical edge engine serves both frontends, so it must hold the
    IO of every logical edge; a per-instance view would drop the non-primary
    edge's links and fail its first UP submit with "no host send IO"."""
    _init(monkeypatch, _edge_share_config())

    assert _Listener.created == []
    assert lwd_host_channels._UCX_LISTENERS == {}
    ios = lwd_host_channels._UCX_IOS
    assert set(ios) == {
        (0, 0, 0, LwdChannelType.UP, "send"),
        (0, 0, 0, LwdChannelType.DOWN, "recv"),
        (1, 1, 0, LwdChannelType.UP, "send"),
        (1, 1, 0, LwdChannelType.DOWN, "recv"),
    }
    # Each logical edge connects out to its OWN cloud, not only the primary's.
    assert ios[(0, 0, 0, LwdChannelType.UP, "send")].config.host == "10.1.0.1"
    assert ios[(1, 1, 0, LwdChannelType.UP, "send")].config.host == "10.1.0.2"


def test_edge_owns_every_cloud_dp_link(monkeypatch) -> None:
    """One edge engine serves both cloud DPs.

    Dropping the non-primary DP's connection (dp_idx != the edge's own, always
    0) leaves that DP without IO, and its first UP submit fails with
    "no host send IO".
    """
    _init(monkeypatch, _single_instance_config("edge"))

    assert _Listener.created == []
    assert set(lwd_host_channels._UCX_IOS) == {
        (0, 0, 0, LwdChannelType.UP, "send"),
        (0, 0, 0, LwdChannelType.DOWN, "recv"),
        (0, 0, 1, LwdChannelType.UP, "send"),
        (0, 0, 1, LwdChannelType.DOWN, "recv"),
    }


def test_cloud_dp_owns_only_its_own_link(monkeypatch) -> None:
    """Each cloud DP engine binds its own listener and holds its own link."""
    _init(monkeypatch, _single_instance_config("cloud"), dp_identity=(1, 0))

    assert _Listener.created == [(CLOUD_ADDR, CLOUD_DATA_PORT + 1)]
    assert set(lwd_host_channels._UCX_IOS) == {
        (0, 0, 1, LwdChannelType.UP, "recv"),
        (0, 0, 1, LwdChannelType.DOWN, "send"),
    }


def test_peer_restart_registration_wires_resets_and_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    """One peer restart: every IO re-baselines its own FIFO; the cloud's UP
    recv IO (the ledger's owner) additionally drops its connection's
    early-recv ledger, so one restart fires the drop exactly once."""
    from vllm_ascend.distributed.lwd_data_plane import lwd_data_plane as plane_module
    from vllm_ascend.worker.lwd_cloud_model import lwd_cloud_worker

    resets: list[tuple[object, object, str]] = []
    ledger_drops: list[object] = []

    class _FakePlane:
        def reset_channel_seqno(self, connection, channel, op) -> None:  # noqa: ANN001
            resets.append((connection, channel, op))

    monkeypatch.setattr(plane_module, "lwd_get_data_plane", lambda: _FakePlane())
    monkeypatch.setattr(lwd_cloud_worker, "drop_endpoint_early_recv", ledger_drops.append)
    _run_init(monkeypatch, _cloud_share_topology(), role="cloud", instance_id=0)
    ios = lwd_host_channels._UCX_IOS
    assert all(len(io.peer_restart_cbs) == 1 for io in ios.values())
    ios[(0, 0, 0, UP, "recv")].peer_restart_cbs[0]()
    assert resets == [(LwdConnectionId(0, 0, 0), UP, "recv")]
    assert ledger_drops == [LwdConnectionId(0, 0, 0)]
    ios[(0, 0, 0, DOWN, "send")].peer_restart_cbs[0]()
    assert resets == [
        (LwdConnectionId(0, 0, 0), UP, "recv"),
        (LwdConnectionId(0, 0, 0), DOWN, "send"),
    ]
    assert ledger_drops == [LwdConnectionId(0, 0, 0)]  # exactly once per restart


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


def test_sidecar_cloud_leaves_the_listener_to_the_daemon(monkeypatch) -> None:
    """The transport daemon owns the listener; the engine only attaches."""
    _init(monkeypatch, _host_config("cloud", deployment="sidecar"))

    assert _Listener.created == []
    assert lwd_host_channels._UCX_LISTENERS == {}
    ios = lwd_host_channels._UCX_IOS
    assert {io.config.transport_mode for io in ios.values()} == {"daemon"}


def test_dp_identity_survives_the_independent_dp_reset(monkeypatch) -> None:
    """A non-MoE DP engine resets `data_parallel_rank` to 0.

    The host endpoint must key on `data_parallel_index` (the launched DP rank);
    otherwise every cloud DP claims dp0 and binds the same listener port.
    """
    monkeypatch.setattr(
        lwd_host_channels,
        "get_current_vllm_config",
        lambda: SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=1, data_parallel_rank=0)),
    )
    monkeypatch.setattr(lwd_host_channels, "get_tensor_model_parallel_rank", lambda: 0)

    assert lwd_host_channels._my_dp_identity() == (1, 0)


def test_dp_identity_falls_back_to_the_rank_field(monkeypatch) -> None:
    """Configs without the index field keep their previous meaning."""
    monkeypatch.setattr(
        lwd_host_channels,
        "get_current_vllm_config",
        lambda: SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=2)),
    )
    monkeypatch.setattr(lwd_host_channels, "get_tensor_model_parallel_rank", lambda: 3)

    assert lwd_host_channels._my_dp_identity() == (2, 3)


def test_the_cloud_dp_requires_a_data_port(monkeypatch) -> None:
    topology = _cloud_share_topology()
    for cloud in topology.clouds:
        for dp in cloud.dp:
            object.__setattr__(dp, "data_port", None)
    with pytest.raises(RuntimeError, match="data_port"):
        _run_init(monkeypatch, topology, role="cloud", instance_id=0)


def test_linked_teardown_hook_fails_only_the_restarted_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    """The liveness verdict tears down every IO of one (cloud, dp) and
    nothing else: the OTHER cloud's links keep serving."""
    from vllm.v1.lwd_control.lwd_comm import lwd_restart_hooks

    hooks: list = []
    monkeypatch.setattr(
        lwd_restart_hooks,
        "register_cloud_restart_hook",
        hooks.append,
    )
    topology = _edge_share_topology()
    connections = tuple(connection for edge in topology.edges for connection in topology.connections("edge", edge.id))
    _run_init(monkeypatch, topology, role="edge", instance_id=0, connections=connections)
    (teardown,) = hooks
    teardown(1, 0, "heartbeat silent 10s")
    ios = lwd_host_channels._UCX_IOS
    # Cloud 1's pair tore down with the verdict; cloud 0's is untouched.
    assert ios[(1, 1, 0, UP, "send")].forced_teardowns == ["heartbeat silent 10s"]
    assert ios[(1, 1, 0, DOWN, "recv")].forced_teardowns == ["heartbeat silent 10s"]
    assert ios[(0, 0, 0, UP, "send")].forced_teardowns == []
    assert ios[(0, 0, 0, DOWN, "recv")].forced_teardowns == []
    # A verdict without local IOs is a silent no-op (another edge's cloud).
    teardown(9, 9, "unrelated")


# --------------------------------------------------------------------- #
# Runtime link addition (elastic scale-out, phase C)                     #
# --------------------------------------------------------------------- #


def _runtime_add_connection() -> LwdDPConnection:
    """Pair 1's connection of the edge_share fixture: absent from the
    startup plan, added at runtime."""
    topology = _edge_share_topology()
    (connection,) = topology.connections("edge", 1)
    return connection


def _run_edge_init_with_pair_zero_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boot the shared edge engine with only pair 0 wired, as if the yaml
    that started it planned a single instance."""
    topology = _edge_share_topology()
    (pair_zero,) = topology.connections("edge", 0)
    _run_init(
        monkeypatch,
        topology,
        role="edge",
        instance_id=0,
        connections=(pair_zero,),
    )


def test_runtime_add_builds_both_links_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime half of the bootstrap: one connection -> two IOs (UP
    send / DOWN recv), both carrying the connection's token and restart
    closures; the ledger grows by exactly one pair. Idempotent on
    replay."""
    _run_edge_init_with_pair_zero_only(monkeypatch)
    connection = _runtime_add_connection()
    assert lwd_host_channels.lwd_add_runtime_connection(connection) is True
    ios = lwd_host_channels._UCX_IOS
    for channel, op in ((UP, "send"), (DOWN, "recv")):
        key = (1, 1, 0, channel, op)
        assert key in ios, key
        assert ios[key].config.token == _link_token(1, 1, 0, channel)
        assert ios[key].peer_restart_cbs, "restart closures registered"
    # init pair (4*(32+32)) + runtime pair.
    assert lwd_host_channels._ALLOCATED_MB == 2 * 4 * (32 + 32)
    # A replay is an idempotent True: no third pair of IOs.
    assert lwd_host_channels.lwd_add_runtime_connection(connection) is True
    assert len(ios) == 4


def test_runtime_add_refuses_when_the_budget_is_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shortfall refuses THE CONNECTION (fail-fast per instance), leaves
    the registry and the ledger untouched -- the engine keeps serving on
    its startup links."""
    _run_edge_init_with_pair_zero_only(monkeypatch)
    monkeypatch.setattr(lwd_host_channels, "_HOST_MEM_BUDGET_MB", 4 * (32 + 32) + 1)
    assert lwd_host_channels.lwd_add_runtime_connection(_runtime_add_connection()) is False
    assert len(lwd_host_channels._UCX_IOS) == 2
    assert lwd_host_channels._ALLOCATED_MB == 4 * (32 + 32)


def test_runtime_add_refuses_an_uninitialized_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert lwd_host_channels.lwd_add_runtime_connection(_runtime_add_connection()) is False
    assert lwd_host_channels._UCX_IOS == {}


def test_runtime_add_builds_a_cloud_pair_on_the_shared_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C-3: the cloud builds its serve pair on the shared listener at
    runtime — UP recv / DOWN send, both registered via serve_via (the
    token lands in the listener's locked route table), so the already
    retrying edge's next UHC1 handshake completes."""
    made_listeners = _run_init(monkeypatch, _cloud_share_topology(), role="cloud", instance_id=0)
    (listener,) = made_listeners
    edge_two_conn = LwdDPConnection(
        LwdConnectionId(1, 0, 0),
        LwdDP(dp_idx=0, addr="10.0.0.2"),
        LwdDP(dp_idx=0, addr="10.1.0.1", ctrl_port=7000, data_port=5560),
    )
    assert lwd_host_channels.lwd_add_runtime_connection(edge_two_conn) is True
    ios = lwd_host_channels._UCX_IOS
    # The cloud's serve pair: UP recv / DOWN send on the shared listener.
    assert (1, 0, 0, UP, "recv") in ios, "UP recv registered"
    assert (1, 0, 0, DOWN, "send") in ios, "DOWN send registered"
    for channel in (UP, DOWN):
        io = ios[(1, 0, 0, channel, "recv" if channel is UP else "send")]
        assert io.config.serve, f"{channel} serves"
        assert io.config.token == _link_token(1, 0, 0, channel)
    # The token registered on the shared listener via serve_via.
    for channel in (UP, DOWN):
        token = _link_token(1, 0, 0, channel)
        assert token in listener._routes, f"token {hex(token)} on listener"
        assert listener._routes[token] is ios[(1, 0, 0, channel, "recv" if channel is UP else "send")]
    # Idempotent replay.
    assert lwd_host_channels.lwd_add_runtime_connection(edge_two_conn) is True
    assert len(ios) == 4  # 2 startup + 2 runtime
    # Budget: startup 1 pair + runtime 1 pair = 2 * 4 * (32+32) MiB.
    assert lwd_host_channels._ALLOCATED_MB == 2 * 4 * (32 + 32)


def test_runtime_add_refuses_a_cloud_without_a_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cloud whose listener vanished (sidecar daemon owns it, or the
    registry was cleared) refuses the addition — the edge's UP frames
    keep bouncing and the next increment retries."""
    _run_init(
        monkeypatch,
        _cloud_share_topology(),
        role="cloud",
        instance_id=0,
    )
    saved_listener = lwd_host_channels._UCX_LISTENERS.pop((0, 0))
    try:
        edge_two_conn = LwdDPConnection(
            LwdConnectionId(1, 0, 0),
            LwdDP(dp_idx=0, addr="10.0.0.2"),
            LwdDP(dp_idx=0, addr="10.1.0.1", ctrl_port=7000, data_port=5560),
        )
        assert lwd_host_channels.lwd_add_runtime_connection(edge_two_conn) is False
        assert (1, 0, 0, UP, "recv") not in lwd_host_channels._UCX_IOS
    finally:
        lwd_host_channels._UCX_LISTENERS[(0, 0)] = saved_listener


def test_runtime_add_rolls_back_a_partial_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure mid-build leaves no half-registered pair behind: the
    already-built UP IO is shut down and unregistered."""

    class _FailingOnDown(_FakeIo):
        def __init__(self, channel, stream, config):
            if channel is DOWN:
                raise RuntimeError("boom on the DOWN link")
            super().__init__(channel, stream, config)

    _run_edge_init_with_pair_zero_only(monkeypatch)
    monkeypatch.setattr(lwd_host_channels, "LwdNpuUcxChannelIo", _FailingOnDown)
    assert lwd_host_channels.lwd_add_runtime_connection(_runtime_add_connection()) is False
    assert (1, 1, 0, UP, "send") not in lwd_host_channels._UCX_IOS
    assert len(lwd_host_channels._UCX_IOS) == 2
    assert lwd_host_channels._ALLOCATED_MB == 4 * (32 + 32)


def test_runtime_add_refuses_a_connection_without_data_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_edge_init_with_pair_zero_only(monkeypatch)
    connection = LwdDPConnection(
        LwdConnectionId(1, 1, 0),
        LwdDP(dp_idx=0, addr="10.0.0.1"),
        LwdDP(dp_idx=0, addr="10.1.0.3", ctrl_port=7002),
    )
    assert lwd_host_channels.lwd_add_runtime_connection(connection) is False
