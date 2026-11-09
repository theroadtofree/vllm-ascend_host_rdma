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
  * warmup: each side only waits its OWN links' readiness
    (``wait_ready``, budget = HS_TIMEOUT_S + 5) — NO cross-world
    barrier exists to join; the cloud just reports the listener bound
    and adopts edges asynchronously (old branch semantics).

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
from vllm_ascend.distributed.lwd_data_plane.ucx import ucx_block_policy
from vllm_ascend.distributed.lwd_data_plane.ucx.ucx_io import NpuUcxChannelIo

logger = init_logger(__name__)

# (edge_id, cloud_id, dp_idx, channel, op) -> NpuUcxChannelIo
_UCX_IOS: dict[tuple[int, int, int, LwdChannelType, str], NpuUcxChannelIo] = {}
# (cloud_id, dp_idx) -> the cloud DP's shared listener (handshake entry
# only; data flows over RoCE). No stop() exists — the accept thread is a
# daemon and dies with the process, matching the old branch.
_UCX_LISTENERS: dict[tuple[int, int], Any] = {}
_UCX_CHANNELS_INITIALIZED = False


def lwd_data_plane_comm() -> str:
    """Transport comm selected by the topology file ('device'|'host').
    The YAML is the single source of truth — there is no env override."""
    lwd_config = get_current_vllm_config().lwd_config
    if lwd_config is None:
        return "device"
    return getattr(
        lwd_config.topology.deployment, "data_plane_comm", "device"
    )


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


def _resolve_up_block_mb() -> int:
    """Both peers derive the same UP block from the same model config."""
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    try:
        plan = ucx_block_policy.resolve(
            model_type=model_config.hf_config.model_type,
            hidden_size=getattr(model_config.hf_config, "hidden_size", None),
            max_num_batched_tokens=(
                vllm_config.scheduler_config.max_num_batched_tokens
            ),
            explicit_up_block_mb=ascend_envs.ucx_up_block_mb_explicit(),
        )
    except ucx_block_policy.UcxBlockPolicyError as exc:
        raise RuntimeError(
            f"[LWD] UCX UP block auto-sizing failed: {exc}"
        ) from exc
    for warning in plan.warnings:
        logger.warning("[LWD] UCX block policy: %s", warning)
    logger.info(
        "[LWD] UCX UP block resolved: %dMiB (%s)", plan.up_block_mb, plan.source
    )
    return plan.up_block_mb


def lwd_init_ucx_channels() -> None:
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

    for conn in connections:
        if conn.dp_idx != dp_rank or tp_rank != 0:
            continue  # not this process's connection
        if conn.cloud.data_port is None:
            raise RuntimeError(
                f"[LWD] UCX data plane requires 'data_port' on the cloud DP "
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
        "[LWD] UCX data-plane links created for role=%s instance=%d "
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
    lwd_warm_up_ucx_channels()


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


def lwd_warm_up_ucx_channels(timeout: float | None = None) -> None:
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
            "[LWD] UCX data-plane: no local links (non-endpoint process)"
        )
        return
    for key, io in _UCX_IOS.items():
        if not io.wait_ready(timeout=timeout):
            logger.error(
                "[LWD] UCX link %s NOT ready after %.0fs",
                key,
                timeout,
            )
            raise RuntimeError(
                f"[LWD] UCX link {key} handshake timed out after "
                f"{timeout:.0f}s"
            )
    logger.info(
        "[LWD] UCX data-plane warmup done (%d link(s), role-side "
        "wait_ready only, no cross-world barrier)",
        len(_UCX_IOS),
    )


def lwd_ucx_channels_initialized() -> bool:
    return _UCX_CHANNELS_INITIALIZED


def lwd_get_ucx_io(
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


def lwd_destroy_ucx_channels(timeout: float = 2.0) -> None:
    global _UCX_CHANNELS_INITIALIZED
    for key, io in list(_UCX_IOS.items()):
        try:
            io.shutdown(timeout=timeout)
        except Exception:
            logger.warning("[LWD] failed to shut down the UCX IO for %s", key)
    _UCX_IOS.clear()
    # Listeners have no stop(): their accept threads are daemons and die
    # with the process; dropping the references is all the cleanup there is.
    _UCX_LISTENERS.clear()
    _UCX_CHANNELS_INITIALIZED = False
