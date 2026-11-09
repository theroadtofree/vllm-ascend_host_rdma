# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UCX (host RoCE) backend of the LWD data plane.

Transport core lives in the vllm repo (``vllm/v1/data_comm``, pure
Python ctypes over libucp); this package assembles it for NPU:
``ucx_io`` (link IO + D2H/H2D orchestration), ``ucx_channels``
(deterministic topology-derived bootstrap + registry) and
``lwd_ucx_queue`` (the LwdTransferQueue implementation the data-plane
factory selects when the topology sets ``data_plane_comm: host``).
"""

from vllm_ascend.distributed.lwd_data_plane.ucx.ucx_channels import (
    lwd_data_plane_comm,
    lwd_data_plane_is_host,
    lwd_destroy_ucx_channels,
    lwd_get_ucx_io,
    lwd_init_ucx_channels,
    lwd_ucx_channels_initialized,
)
from vllm_ascend.distributed.lwd_data_plane.ucx.lwd_ucx_queue import (
    LwdUcxTransferQueue,
)

__all__ = [
    "LwdUcxTransferQueue",
    "lwd_data_plane_comm",
    "lwd_data_plane_is_host",
    "lwd_destroy_ucx_channels",
    "lwd_get_ucx_io",
    "lwd_init_ucx_channels",
    "lwd_ucx_channels_initialized",
]
