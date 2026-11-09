# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UCX link bootstrap and registry for the LWD data plane.

Ported (re-worked) from the host_rdma branch's ``lwd_wire_host.py``: the
old branch bootstrapped UCX links over a TCP registry in a split world;
this repo shares ONE torch world between edge and cloud (see
``lwd_channels.lwd_init_channels``), and the topology file already
carries every DP's unicast ``addr`` (and optional ``ctrl_port``), so link
endpoints are derived deterministically from it — no registry, no
address exchange, no extra collectives.

Per DP connection and channel TWO independent UHC1 links exist, one per
direction: the receiver of a (channel, op) link serves on a deterministic
port and the sender connects to the server's topology address. Link
identity is verified at handshake by a token derived from
(dp_idx, channel, op) on both peers.

Port allocation (both peers derive identically): the serving DP's
``ctrl_port`` when configured, else VLLM_ASCEND_LWD_UCX_PORT_BASE
(default 29530); then ``base + dp_idx * 4 + (0|2 for UP|DOWN) + (0|1 for
recv|send)`` — dp_idx is bounded well below 2**16 by the topology schema.

Block sizing: DOWN uses VLLM_ASCEND_LWD_UCX_DOWN_BLOCK_MB; UP resolves
through ``ucx_block_policy.resolve`` (model hidden_size x
max_num_batched_tokens auto-sizing, explicit env override wins) — both
peers derive the same value from the same config.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed.lwd import LwdChannelEndpoints, LwdParallelGroupsSpec
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType
from vllm_ascend.distributed.lwd_data_plane.ucx import ucx_block_policy
from vllm_ascend.distributed.lwd_data_plane.ucx.ucx_io import NpuUcxChannelIo

logger = init_logger(__name__)

# (dp_idx, channel, op) -> NpuUcxChannelIo, endpoint ranks only
_UCX_IOS: dict[tuple[int, LwdChannelType, str], NpuUcxChannelIo] = {}
_UCX_CHANNELS_INITIALIZED = False

# Handshake token base: nonzero, derived from (dp_idx, channel, op).
_TOKEN_BASE = 0x4C57_0000

_DEFAULT_PORT_BASE = 5560


def lwd_data_plane_family() -> str:
    """Transport family selected by the topology file ('hccl'|'ucx')."""
    lwd_config = get_current_vllm_config().lwd_config
    if lwd_config is None:
        return "hccl"
    return getattr(lwd_config.topology, "data_plane_family", "hccl")


def lwd_data_plane_is_ucx() -> bool:
    return lwd_data_plane_family() == "ucx"


def _link_token(dp_idx: int, channel: LwdChannelType, op: str) -> int:
    return _TOKEN_BASE + ((dp_idx & 0xFFFF) << 8) + (
        (0x02 if channel is LwdChannelType.DOWN else 0x00)
        | (0x01 if op == "recv" else 0x00)
    )


def _link_port(
    base: int, dp_idx: int, channel: LwdChannelType, op: str
) -> int:
    return (
        base
        + (dp_idx & 0xFFFF) * 4
        + (0 if channel is LwdChannelType.UP else 2)
        + (0 if op == "recv" else 1)
    )


def _serving_dp_addr(
    endpoints: LwdChannelEndpoints, channel: LwdChannelType, serve_rank: int
) -> tuple[str, int]:
    """The serving side's (addr, port_base) from the topology: the DP whose
    first rank is the server. ``ctrl_port`` wins when configured, else the
    env default base."""
    topology = get_current_vllm_config().lwd_config.topology
    for instance in (*topology.clouds, *topology.edges):
        for dp in instance.dp:
            if dp.ranks and dp.ranks[0] == serve_rank:
                base = dp.ctrl_port or int(
                    ascend_envs.VLLM_ASCEND_LWD_UCX_PORT_BASE
                )
                return dp.addr, base
    raise RuntimeError(
        f"[LWD] serving rank {serve_rank} of the {channel.value} link on "
        f"dp_idx {endpoints.dp_idx} is not the first rank of any topology DP"
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
    group path when the topology selects ``data_plane_family: ucx``.
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
        for channel in (LwdChannelType.UP, LwdChannelType.DOWN):
            # The receiver of each direction's link serves.
            recv_rank = (
                endpoints.cloud_rank if channel is LwdChannelType.UP
                else endpoints.edge_rank
            )
            send_rank = (
                endpoints.edge_rank if channel is LwdChannelType.UP
                else endpoints.cloud_rank
            )
            for op in ("recv", "send"):
                key = (endpoints.dp_idx, channel, op)
                server_rank = recv_rank if op == "recv" else send_rank
                addr, base = _serving_dp_addr(endpoints, channel, server_rank)
                port = _link_port(base, endpoints.dp_idx, channel, op)
                if my_rank == server_rank:
                    _UCX_IOS[key] = NpuUcxChannelIo(
                        channel, op, torch.npu.Stream(),
                        host=addr, port=port, serve=True,
                        token=_link_token(*key), up_block_mb=up_block_mb,
                    )
                elif my_rank in (recv_rank, send_rank):
                    _UCX_IOS[key] = NpuUcxChannelIo(
                        channel, op, torch.npu.Stream(),
                        host=addr, port=port, serve=False,
                        token=_link_token(*key), up_block_mb=up_block_mb,
                    )
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
    _UCX_CHANNELS_INITIALIZED = False
