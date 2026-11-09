# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UCX link bootstrap and registry for the LWD data plane.

Ported (re-worked) from the host_rdma branch's ``lwd_wire_host.py``: the
old branch bootstrapped UCX links over a TCP registry in a split world;
this repo shares ONE torch world between edge and cloud (see
``lwd_channels.lwd_init_channels``), and the topology file already
carries every DP's unicast ``addr`` and its ``data_port`` (the base of
its UCX data-plane links), so link endpoints are derived deterministically
from it — no registry, no address exchange, no extra collectives.

Per DP connection TWO independent UHC1 links exist, one per direction
(UP/DOWN UcxChannels, each with its own UCP ep / IO thread / block
pool), aligned with the old host_rdma branch's listener model: the
CLOUD DP of the connection binds ONE ``UcxListener`` on its
``data_port`` and BOTH links attach to it via ``serve_via`` — the
handshake token (derived from (dp_idx, channel)) routes each incoming
TCP connection to the right channel. The EDGE side never listens: both
its IOs (UP send, DOWN recv) connect out to the cloud port — UHC1 is
full-duplex, so who binds the handshake socket is independent of the
data direction.

``data_port`` is therefore required on CLOUD DPs only (the topology's
``ctrl_port`` belongs to the control plane and is never used here); no
env-var port base exists.

Block sizing: DOWN uses VLLM_ASCEND_LWD_UCX_DOWN_BLOCK_MB; UP resolves
through ``ucx_block_policy.resolve`` (model hidden_size x
max_num_batched_tokens auto-sizing, explicit env override wins) — both
peers derive the same value from the same config.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed.lwd import LwdChannelEndpoints, LwdParallelGroupsSpec
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.v1.data_comm.ucx_channel import UcxListener

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType
from vllm_ascend.distributed.lwd_data_plane.ucx import ucx_block_policy
from vllm_ascend.distributed.lwd_data_plane.ucx.ucx_io import NpuUcxChannelIo

logger = init_logger(__name__)

# (dp_idx, channel, op) -> NpuUcxChannelIo, endpoint ranks only
_UCX_IOS: dict[tuple[int, LwdChannelType, str], NpuUcxChannelIo] = {}
# dp_idx -> the cloud DP's shared listener (handshake entry only; data
# flows over RoCE). No stop() exists — the accept thread is a daemon and
# dies with the process, matching the old branch.
_UCX_LISTENERS: dict[int, Any] = {}
_UCX_CHANNELS_INITIALIZED = False

# Handshake token base: nonzero, derived from (dp_idx, channel, op).
_TOKEN_BASE = 0x4C57_0000


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


def _link_token(dp_idx: int, channel: LwdChannelType) -> int:
    """Token identifies the LINK, so it must be identical on both ends:
    derived from (dp_idx, channel) only, never from the op."""
    return _TOKEN_BASE + ((dp_idx & 0xFFFF) << 8) + (
        0x01 if channel is LwdChannelType.DOWN else 0x00
    )


def _link_port(data_port: int) -> int:
    """The cloud DP's data_port is the listener port itself — one port
    per DP connection, no offset: both links share one UcxListener."""
    return data_port


def _cloud_dp_endpoint(
    endpoints: LwdChannelEndpoints, channel: LwdChannelType
) -> tuple[str, int]:
    """The cloud DP's (addr, data_port) of this connection — the cloud is
    the only listener (both links attach to one UcxListener on this
    port). ``data_port`` is mandatory on cloud DPs for UCX;
    ``ctrl_port`` belongs to the control plane and is never consulted
    here."""
    topology = get_current_vllm_config().lwd_config.topology
    cloud_rank = endpoints.cloud_rank
    for instance in topology.clouds:
        for dp in instance.dp:
            if dp.ranks and dp.ranks[0] == cloud_rank:
                if dp.data_port is None:
                    raise RuntimeError(
                        f"[LWD] UCX data plane requires 'data_port' on the "
                        f"cloud DP (dp_idx={dp.dp_idx}, addr={dp.addr}); "
                        f"set it in the topology file (edges never listen; "
                        "ctrl_port is control-plane only)"
                    )
                return dp.addr, dp.data_port
    raise RuntimeError(
        f"[LWD] cloud rank {cloud_rank} of the {channel.value} link on "
        f"dp_idx {endpoints.dp_idx} is not the first rank of any cloud DP"
    )


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
    """Create the UCX links of every DP connection. Idempotent.

    Called by ``lwd_channels.lwd_init_channels`` instead of the HCCL
    group path when the topology selects ``data_plane_comm: host``.
    Only the endpoint ranks of each connection create IO objects; the
    world barrier at the end keeps init collective-symmetric.
    """
    global _UCX_CHANNELS_INITIALIZED
    if _UCX_CHANNELS_INITIALIZED:
        return
    lwd_config = get_current_vllm_config().lwd_config
    assert lwd_config is not None, "the data plane requires lwd_config"
    spec = LwdParallelGroupsSpec.from_topology(lwd_config.topology)
    my_rank = dist.get_rank()
    up_block_mb = _resolve_up_block_mb()

    for endpoints in spec.channel_endpoints:
        # Cloud DP: one shared listener per DP connection; both links
        # (UP recv / DOWN send) attach to it via serve_via, token-routed.
        addr, data_port = _cloud_dp_endpoint(
            endpoints, LwdChannelType.UP
        )
        port = _link_port(data_port)
        listener = None
        if my_rank == endpoints.cloud_rank:
            listener = UcxListener(addr, port)
            _UCX_LISTENERS[endpoints.dp_idx] = listener
        for channel in (LwdChannelType.UP, LwdChannelType.DOWN):
            # Edge: UP sender / DOWN receiver, both connect out (the edge
            # never listens — UHC1 is full-duplex). Cloud: UP receiver /
            # DOWN sender, both on the shared listener.
            if my_rank == endpoints.edge_rank:
                op = "send" if channel is LwdChannelType.UP else "recv"
                serve = False
            elif my_rank == endpoints.cloud_rank:
                op = "recv" if channel is LwdChannelType.UP else "send"
                serve = True
            else:
                continue
            _UCX_IOS[(endpoints.dp_idx, channel, op)] = NpuUcxChannelIo(
                channel, op, torch.npu.Stream(),
                host=addr, port=port, serve=serve,
                token=_link_token(endpoints.dp_idx, channel),
                listener=listener, up_block_mb=up_block_mb,
            )
    # Listeners bind at construction; accept loops start only after every
    # channel has registered its token (UcxListener.register rejects
    # duplicates, so registration order does not matter).
    for listener in _UCX_LISTENERS.values():
        listener.start()
    _UCX_CHANNELS_INITIALIZED = True
    logger.info(
        "[LWD] UCX data-plane links created for %s (my_rank=%d, ios=%s)",
        spec.channel_endpoints,
        my_rank,
        sorted(f"{dp}/{ch.value}/{op}" for dp, ch, op in _UCX_IOS),
    )
    lwd_warm_up_ucx_channels()


def _io(dp_idx: int, channel: LwdChannelType, op: str) -> NpuUcxChannelIo:
    try:
        return _UCX_IOS[(dp_idx, channel, op)]
    except KeyError:
        raise RuntimeError(
            f"[LWD] rank {dist.get_rank()} holds no UCX {op} IO of "
            f"{channel.value} on dp_idx {dp_idx}; only the endpoint ranks "
            f"of that DP connection do"
        ) from None


def lwd_warm_up_ucx_channels(timeout: float | None = None) -> None:
    """Wait for every local link's handshake to complete (bounded by the
    transport's hs_timeout), then barrier the world."""
    if timeout is None:
        timeout = float(
            max(1, int(ascend_envs.VLLM_ASCEND_LWD_UCX_HS_TIMEOUT_S))
        )
    for key, io in _UCX_IOS.items():
        if not io.wait_ready(timeout=timeout):
            logger.warning(
                "[LWD] UCX link %s not ready after %.0fs", key, timeout
            )
    get_world_group().barrier()
    if _UCX_IOS:
        logger.info(
            "[LWD] UCX data-plane warmup done (%d link(s))", len(_UCX_IOS)
        )


def lwd_ucx_channels_initialized() -> bool:
    return _UCX_CHANNELS_INITIALIZED


def lwd_get_ucx_io(
    channel: LwdChannelType, dp_idx: int = 0, op: str = "send"
) -> NpuUcxChannelIo:
    return _io(dp_idx, channel, op)


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
