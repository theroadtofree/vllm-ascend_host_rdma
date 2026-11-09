# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UCX link bootstrap and registry for the LWD data plane.

Migrated from the host_rdma branch's ``lwd_wire_host.py`` (commits
f309213fd / f9b2f11e5 / 9f11d96e9), adapted to this repo's topology
file instead of the old registry/dispatcher split-world config.

``data_plane_comm: host`` has NO global comm domain: the edge and the
cloud are separate torch worlds, so link endpoints derive from identity
alone — ``lwd_config.role`` + ``instance_id`` + the per-connection
``LwdDPConnection`` from ``topology.connections()`` (which carries
addr/data_port/dp_idx directly; topology ranks are absent in host mode).
No global ranks, no LwdParallelGroupsSpec, no world collectives.

Link model (identical to the old branch):
  * per DP connection TWO UHC1 links (UP/DOWN), each an independent
    UcxChannel with its own UCP ep / IO thread / block pool;
  * the CLOUD DP binds ONE shared ``UcxListener`` on its ``data_port``
    and BOTH links attach via ``serve_via`` — the handshake token
    (blake2b over "edge_id:cloud_id:dp_idx:channel", like the old
    ``_ucx_token``) routes each incoming TCP connection to its channel;
  * the EDGE never listens: its UP/send and DOWN/recv IOs connect out
    (UHC1 is full-duplex; who binds the handshake socket is independent
    of the data direction);
  * endpoint process within an instance: dp_rank == dp_idx and TP rank 0
    (the old branch's "my_rank == 0" with cloud dp=1 is the dp_idx=0
    special case of this rule);
  * warmup: ONLY the edge waits for its links (``wait_ready``, budget
    = HS_TIMEOUT_S + 5) — the cloud starts before edges, so it just
    reports the listener bound and adopts each edge asynchronously as
    it connects; NO cross-world barrier exists to join.

Block sizing: DOWN uses VLLM_ASCEND_LWD_UCX_DOWN_BLOCK_MB; UP resolves
through ``ucx_block_policy.resolve`` (model hidden_size x
max_num_batched_tokens auto-sizing, explicit env override wins) — both
peers derive the same value from the same config.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.v1.data_comm.ucx_channel import UcxListener

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType
from vllm_ascend.distributed.lwd_data_plane.host import ucx_block_policy
from vllm_ascend.distributed.lwd_data_plane.host.ucx_io import NpuUcxChannelIo

logger = init_logger(__name__)

# (edge_id, cloud_id, dp_idx, channel, op) -> NpuUcxChannelIo
_UCX_IOS: dict[tuple[int, int, int, LwdChannelType, str], NpuUcxChannelIo] = {}
# (cloud_id, dp_idx) -> the cloud DP's shared listener (handshake entry
# only; data flows over RoCE). No stop() exists — the accept thread is a
# daemon and dies with the process, matching the old branch.
_UCX_LISTENERS: dict[tuple[int, int], Any] = {}
_UCX_CHANNELS_INITIALIZED = False


# Cached at channel-init time (config is only set inside the init
# context — get_current_vllm_config() RAISES at request time after vLLM
# clears the current config, so request-path readers must use the cache).
_CACHED_COMM: str | None = None


def lwd_data_plane_comm() -> str:
    """Transport comm selected by the topology file ('device'|'host').
    The YAML is the single source of truth — there is no env override.

    The first call MUST happen during channel init (worker init_device),
    where the current config is guaranteed; it caches the value for the
    request path, where the global config is no longer set."""
    global _CACHED_COMM
    if _CACHED_COMM is not None:
        return _CACHED_COMM
    lwd_config = get_current_vllm_config().lwd_config
    if lwd_config is None:
        comm = "device"
    else:
        comm = getattr(
            lwd_config.topology.deployment, "data_plane_comm", "device"
        )
    _CACHED_COMM = comm
    return comm


def lwd_data_plane_is_host() -> bool:
    return lwd_data_plane_comm() == "host"


def _link_token(
    edge_id: int, cloud_id: int, dp_idx: int, channel: LwdChannelType
) -> int:
    """Channel token: identical on both peers of a link, mismatch-proof.
    Same scheme as the old branch's ``_ucx_token`` (edge:cloud:engine:
    channel, blake2b-8) with dp_idx standing in for engine_index."""
    raw = f"{edge_id}:{cloud_id}:{dp_idx}:{channel.value}:1"
    token = int.from_bytes(
        hashlib.blake2b(raw.encode("utf-8"), digest_size=8).digest(), "little"
    )
    return token or 1


def _my_dp_identity() -> tuple[int, int]:
    """(dp_rank, tp_rank) of this process within its OWN instance world —
    in host comm these are instance-local (the edge and cloud worlds are
    independent), never global LWD ranks."""
    vllm_config = get_current_vllm_config()
    dp_rank = getattr(vllm_config.parallel_config, "data_parallel_rank", 0)
    try:
        tp_rank = get_tensor_model_parallel_rank()
    except Exception:
        tp_rank = 0
    return int(dp_rank or 0), int(tp_rank)


def _hf_attr(cfg: Any, name: str) -> Any:
    """Attribute of an hf_config (or its nested text_config), or None.
    Dict configs are supported like the old branch's helper."""
    if isinstance(cfg, dict):
        return cfg.get(name)
    return getattr(cfg, name, None)


def _resolve_up_block_mb() -> int:
    """Both peers derive the same UP block from the same model config.

    Nested-config fallback mirrors the old branch's
    ``_resolve_ucx_block_plan``: multimodal/moe models (e.g. qwen3_5)
    keep the real values inside ``hf_config.text_config``, so every
    sizing attribute falls back to it when the top level is absent.
    """
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        raise RuntimeError(
            "[LWD] host UP block auto-sizing failed: model_config.hf_config "
            "is unavailable"
        )
    text_config = _hf_attr(hf_config, "text_config")

    def attr(name: str) -> Any:
        value = _hf_attr(hf_config, name)
        if value is None and text_config is not None:
            value = _hf_attr(text_config, name)
        return value

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_batched_tokens = getattr(
        scheduler_config, "max_num_batched_tokens", None
    )
    if max_num_batched_tokens is None:
        raise RuntimeError(
            "[LWD] host UP block auto-sizing failed: "
            "scheduler_config.max_num_batched_tokens is unavailable"
        )
    try:
        plan = ucx_block_policy.resolve(
            model_type=attr("model_type") or "",
            hidden_size=attr("hidden_size"),
            num_hash_layers=attr("num_hash_layers") or 0,
            num_experts_per_tok=attr("num_experts_per_tok") or 0,
            max_num_batched_tokens=max_num_batched_tokens,
            explicit_up_block_mb=ascend_envs.ucx_up_block_mb_explicit(),
        )
    except ucx_block_policy.UcxBlockPolicyError as exc:
        raise RuntimeError(
            f"[LWD] host UP block auto-sizing failed: {exc}"
        ) from exc
    # Same record format as the old branch's _log_ucx_block_plan (log
    # analysis tooling parses it).
    for warning in plan.warnings:
        logger.warning("[Lwd][ucx-block] %s", warning)
    logger.info(
        "[Lwd][ucx-block] model_type=%s max_num_batched_tokens=%d "
        "per_token=%d raw=%.1fMiB auto=%dMiB source=%s up_block_mb=%d",
        plan.model_type or "<unknown>",
        plan.max_num_batched_tokens,
        plan.per_token_elements,
        plan.raw_mib,
        plan.auto_up_block_mb,
        plan.source,
        plan.up_block_mb,
    )
    return plan.up_block_mb


def _host_memory_need_mb(
    channels: int, *, up_block_mb: int | None = None
) -> int:
    """Host data-plane memory estimate: one pool per channel (the send
    direction reuses its channel's pool as D2H staging — pools =
    channels). ``channels=2`` (UP/DOWN per connection) =>
    N_BLOCKS x (UP_MB + DOWN_MB) per connection pair. Same formula as
    the old branch's ``_ucx_memory_need_mb`` and the preflight tool.
    """
    n_blocks = int(ascend_envs.VLLM_ASCEND_LWD_UCX_N_BLOCKS)
    up_mb = (
        int(up_block_mb)
        if up_block_mb is not None
        else int(ascend_envs.VLLM_ASCEND_LWD_UCX_UP_BLOCK_MB)
    )
    down_mb = int(ascend_envs.VLLM_ASCEND_LWD_UCX_DOWN_BLOCK_MB)
    pairs = channels // 2
    return pairs * n_blocks * (up_mb + down_mb)


def lwd_init_host_channels() -> None:
    """Create this process's UCX links from its role identity. Idempotent.

    Called by ``lwd_channels.lwd_init_channels`` when the topology
    selects ``data_plane_comm: host``. Identity-driven: an edge instance
    connects out to every linked cloud DP; a cloud instance binds one
    listener per its own DP and adopts edges asynchronously. Only the
    endpoint process (dp_rank == dp_idx, TP rank 0) of each connection
    creates IO objects — the same single-endpoint-per-link rule as the
    old branch's ``my_rank == 0``.
    """
    global _UCX_CHANNELS_INITIALIZED
    if _UCX_CHANNELS_INITIALIZED:
        return
    lwd_config = get_current_vllm_config().lwd_config
    assert lwd_config is not None, "the data plane requires lwd_config"
    topology = lwd_config.topology
    role = lwd_config.role
    instance_id = lwd_config.instance_id
    connections = topology.connections(role, instance_id)
    if not connections:
        raise RuntimeError(
            f"[LWD] no DP connections for role={role} instance={instance_id}"
        )
    dp_rank, tp_rank = _my_dp_identity()
    up_block_mb = _resolve_up_block_mb()

    # Host memory budget gate (old branch's _ucx_memory_need_mb): every
    # link owns one pool of N_BLOCKS blocks (send directions reuse their
    # channel's pool as D2H staging, no extra arena), so need =
    # links/2 * N_BLOCKS * (UP_MB + DOWN_MB). RMA-put mode (PUT_THRESH>0)
    # adds a control pool on top — budget headroom when enabling it.
    endpoint_conns = [
        conn
        for conn in connections
        if conn.dp_idx == dp_rank and tp_rank == 0
    ]
    links = 2 * len(endpoint_conns)
    need_mb = _host_memory_need_mb(links, up_block_mb=up_block_mb)
    limit_mb = int(ascend_envs.VLLM_ASCEND_LWD_DP_MEM_BUDGET_MB)
    if need_mb > limit_mb:
        raise RuntimeError(
            "host data plane memory budget exceeded: "
            f"need={need_mb}MiB limit={limit_mb}MiB per rank "
            f"(links={links}, pools={links}, N_BLOCKS="
            f"{ascend_envs.VLLM_ASCEND_LWD_UCX_N_BLOCKS}, "
            f"UP_MB={up_block_mb}, DOWN_MB="
            f"{ascend_envs.VLLM_ASCEND_LWD_UCX_DOWN_BLOCK_MB}); raise "
            "VLLM_ASCEND_LWD_DP_MEM_BUDGET_MB or lower N_BLOCKS"
        )
    logger.info(
        "[LWD] host memory plan links=%d pools=%d need=%dMiB limit=%dMiB",
        links, links, need_mb, limit_mb,
    )

    for conn in endpoint_conns:
        if conn.cloud.data_port is None:
            raise RuntimeError(
                f"[LWD] host data plane requires 'data_port' on the cloud DP "
                f"(cloud={conn.cloud_id}, dp_idx={conn.dp_idx}); set it in "
                "the topology file (edges never listen; ctrl_port is "
                "control-plane only)"
            )
        addr = conn.cloud.addr
        port = conn.cloud.data_port  # used exactly as configured
        # Cloud DP: one shared listener per DP connection; both links
        # (UP recv / DOWN send) attach to it via serve_via, token-routed.
        listener = None
        if role == "cloud":
            listener = UcxListener(addr, port)
            _UCX_LISTENERS[(conn.cloud_id, conn.dp_idx)] = listener
        for channel in (LwdChannelType.UP, LwdChannelType.DOWN):
            # Edge: UP sender / DOWN receiver, both connect out (the edge
            # never listens — UHC1 is full-duplex). Cloud: UP receiver /
            # DOWN sender, both on the shared listener.
            if role == "edge":
                op = "send" if channel is LwdChannelType.UP else "recv"
                serve = False
            else:
                op = "recv" if channel is LwdChannelType.UP else "send"
                serve = True
            _UCX_IOS[(conn.edge_id, conn.cloud_id, conn.dp_idx, channel, op)] = (
                NpuUcxChannelIo(
                    channel, op, torch.npu.Stream(priority=-1),
                    host=addr, port=port, serve=serve,
                    token=_link_token(conn.edge_id, conn.cloud_id,
                                      conn.dp_idx, channel),
                    listener=listener, up_block_mb=up_block_mb,
                )
            )
    # Listeners bind at construction; accept loops start only after every
    # channel has registered its token (UcxListener.register rejects
    # duplicates, so registration order does not matter).
    for listener in _UCX_LISTENERS.values():
        listener.start()
    _UCX_CHANNELS_INITIALIZED = True
    logger.info(
        "[LWD] host data-plane links created for role=%s instance=%d "
        "dp_rank=%d tp_rank=%d (ios=%s)",
        role,
        instance_id,
        dp_rank,
        tp_rank,
        sorted(
            f"e{e}c{c}/{d}/{ch.value}/{op}"
            for e, c, d, ch, op in _UCX_IOS
        ),
    )
    if role == "edge":
        # Only the EDGE waits for its links (old-branch semantics: the
        # cloud starts before edges — its listener is bound and each edge
        # is adopted asynchronously as it connects; blocking here would
        # deadlock a cloud-first deployment within the warmup timeout).
        lwd_warm_up_host_channels()
    else:
        logger.info(
            "[LWD] host cloud listener ready; waiting for %d edge link(s) "
            "asynchronously",
            len(_UCX_IOS),
        )


def _io(
    edge_id: int, cloud_id: int, dp_idx: int, channel: LwdChannelType, op: str
) -> NpuUcxChannelIo:
    key = (edge_id, cloud_id, dp_idx, channel, op)
    try:
        return _UCX_IOS[key]
    except KeyError:
        raise RuntimeError(
            f"[LWD] this process holds no UCX {op} IO of {channel.value} on "
            f"(edge={edge_id}, cloud={cloud_id}, dp_idx={dp_idx}); only the "
            f"endpoint process (dp_rank==dp_idx, tp_rank==0) of the "
            "connection does"
        ) from None


def lwd_warm_up_host_channels(timeout: float | None = None) -> None:
    """Wait for every LOCAL link's readiness — no cross-world barrier
    exists to join in host comm (old-branch semantics: the cloud reports
    its listener bound and adopts edges asynchronously; the edge waits
    for every cloud link, bounded by HS_TIMEOUT_S + 5)."""
    if timeout is None:
        timeout = max(
            5.0,
            float(int(ascend_envs.VLLM_ASCEND_LWD_UCX_HS_TIMEOUT_S)) + 5.0,
        )
    if not _UCX_IOS:
        logger.info(
            "[LWD] host data-plane: no local links (non-endpoint process)"
        )
        return
    import time as _time

    t0 = _time.perf_counter()
    for (edge_id, cloud_id, dp_idx, channel, op), io in _UCX_IOS.items():
        if not io.wait_ready(timeout=timeout):
            raise RuntimeError(
                f"[Lwd][handshake] host link not ready: "
                f"edge={edge_id} cloud={cloud_id} dp_idx={dp_idx} "
                f"channel={channel.value} op={op} within {timeout:.0f}s"
            )
    logger.info(
        "[Lwd][handshake] OK data_plane=host links=%d %.1fms "
        "(edge-side wait_ready only, no cross-world barrier)",
        len(_UCX_IOS),
        (_time.perf_counter() - t0) * 1000.0,
    )


def lwd_host_channels_initialized() -> bool:
    return _UCX_CHANNELS_INITIALIZED


def lwd_get_host_io(
    channel: LwdChannelType,
    dp_idx: int = 0,
    op: str = "send",
    edge_id: int | None = None,
    cloud_id: int | None = None,
) -> NpuUcxChannelIo:
    """The queue factory's lookup. Single-connection callers omit the
    instance ids; the (edge, cloud) pair disambiguates multi-instance
    topologies (token identity is per (edge, cloud, dp, channel))."""
    if edge_id is None or cloud_id is None:
        if len({(e, c) for e, c, _d, _ch, _op in _UCX_IOS}) != 1:
            raise RuntimeError(
                "[LWD] ambiguous UCX link lookup: multiple (edge, cloud) "
                "connections exist; pass edge_id/cloud_id"
            )
        (edge_id, cloud_id), = {
            (e, c) for e, c, _d, _ch, _op in _UCX_IOS
        }
    return _io(edge_id, cloud_id, dp_idx, channel, op)


def lwd_destroy_host_channels(timeout: float = 2.0) -> None:
    global _UCX_CHANNELS_INITIALIZED
    for key, io in list(_UCX_IOS.items()):
        try:
            io.shutdown(timeout=timeout)
        except Exception:
            logger.warning("[LWD] failed to shut down the UCX IO for %s", key)
    _UCX_IOS.clear()
    for key, listener in list(_UCX_LISTENERS.items()):
        try:
            listener.close()
        except Exception:
            logger.warning("[LWD] failed to close host listener %s", key)
    _UCX_LISTENERS.clear()
    global _CACHED_COMM
    _CACHED_COMM = None
    _UCX_CHANNELS_INITIALIZED = False
