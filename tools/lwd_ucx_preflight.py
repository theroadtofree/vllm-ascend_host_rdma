#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""1e1c UCX/HRDP 数据面预检(纯标准库,不需要 NPU/torch)。

用法::

    python tools/lwd_ucx_preflight.py --role cloud
    python tools/lwd_ucx_preflight.py --role cloud \
        --model /models/DeepSeek-V4 --max-num-batched-tokens 20480
    python tools/lwd_ucx_preflight.py --role edge --cloud-host 10.0.0.2
    # 两端起来后,再探测端口:
    python tools/lwd_ucx_preflight.py --role edge --cloud-host 10.0.0.2 \
        --probe-ports

退出码 0=全部 PASS(允许 WARN),1=存在 FAIL。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path

DEFAULT_DATA_PORT = 5560
DEFAULT_UP_BLOCK_MB = 128
DEFAULT_DOWN_BLOCK_MB = 32
DEFAULT_N_BLOCKS = 4
DEFAULT_BUDGET_MB = 4096

_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm_ascend"
    / "distributed"
    / "lwd_data_plane"
    / "ucx"
    / "ucx_block_policy.py"
)
_policy_spec = importlib.util.spec_from_file_location(
    "lwd_ucx_block_policy", _POLICY_PATH
)
if _policy_spec is None or _policy_spec.loader is None:
    raise RuntimeError(f"cannot load UCX block policy: {_POLICY_PATH}")
_block_policy = importlib.util.module_from_spec(_policy_spec)
sys.modules[_policy_spec.name] = _block_policy
_policy_spec.loader.exec_module(_block_policy)
DEFAULT_UP_BLOCK_MB = _block_policy.DEFAULT_UP_BLOCK_MB


def memory_need_mb(
    up_block_mb: int = DEFAULT_UP_BLOCK_MB,
    down_block_mb: int = DEFAULT_DOWN_BLOCK_MB,
    n_blocks: int = DEFAULT_N_BLOCKS,
    channels: int = 2,
) -> int:
    """1e1c 每端 host 内存:每通道一个池(发送向复用本通道池,无 arena)。

    ``channels=2`` => N × (UP_MB + DOWN_MB)(默认 640 MiB);put 模式另加
    控制池。
    """
    if up_block_mb <= 0 or down_block_mb <= 0 or n_blocks <= 0:
        raise ValueError("block_mb and n_blocks must be positive")
    pairs = channels // 2
    return pairs * n_blocks * (up_block_mb + down_block_mb)


def data_plane_port(data_port: int) -> int:
    """UCX 链路端口:直接使用 topology 中云 DP 的 data_port(每个云 DP
    一个共享 listener,UP/DOWN 两链路经 token 路由共用);越界抛
    ValueError。"""
    if data_port < 1 or data_port > 65535:
        raise ValueError(
            f"data_port={data_port} invalid: need 1..65535 "
            "(one listener per DP, used as configured)"
        )
    return data_port


def _check(ok: bool, name: str, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return ok


def _warn(name: str, detail: str = "") -> None:
    print(f"[WARN] {name}" + (f": {detail}" if detail else ""))


def probe_ports(role: str, host: str, data_port: int, timeout: float = 2.0) -> bool:
    """cloud=bind 单端口;edge=connect 单端口。返回是否成功。"""
    port = data_plane_port(data_port)
    if role == "cloud":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            print(f"[PASS] port bind 0.0.0.0:{port}")
            return True
        except OSError as exc:
            print(f"[FAIL] port bind 0.0.0.0:{port}: {exc}")
            return False
        finally:
            sock.close()
    if not host:
        print("[FAIL] port connect: --cloud-host is empty")
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            print(f"[PASS] port connect {host}:{port}")
            return True
    except OSError as exc:
        print(f"[FAIL] port connect {host}:{port}: {exc}")
        return False


def load_model_metadata(model_path: str | Path) -> dict:
    """Read local ``config.json`` metadata used for UP block sizing."""
    path = Path(model_path)
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        raise ValueError(f"model config not readable: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse model config {path}: {exc}") from exc
    text_config = config.get("text_config") or {}
    return {
        "model_type": str(
            config.get("model_type")
            or text_config.get("model_type")
            or ""
        ),
        "hidden_size": config.get(
            "hidden_size", text_config.get("hidden_size")
        ),
        "num_hash_layers": config.get(
            "num_hash_layers", text_config.get("num_hash_layers", 0)
        ),
        "num_experts_per_tok": config.get(
            "num_experts_per_tok",
            text_config.get("num_experts_per_tok", 0),
        ),
    }


def _policy_up_block_mb(
    *,
    model_path: str,
    max_num_batched_tokens: int | None,
    explicit_up_block_mb: int | None,
) -> tuple[int, str, list[str]]:
    """Resolve (up_block_mb, source, warnings) for preflight."""
    if explicit_up_block_mb is not None and (
        not model_path or max_num_batched_tokens is None
    ):
        return explicit_up_block_mb, "explicit", []
    if max_num_batched_tokens is None and model_path:
        raise ValueError(
            "--max-num-batched-tokens is required with --model"
        )
    if not model_path:
        if explicit_up_block_mb is not None:
            return explicit_up_block_mb, "explicit", []
        return (
            DEFAULT_UP_BLOCK_MB,
            "fallback",
            [
                "no --model/--up-block-mb; fallback "
                f"UP_BLOCK_MB={DEFAULT_UP_BLOCK_MB}"
            ],
        )
    metadata = load_model_metadata(model_path)
    hidden_size = metadata["hidden_size"]
    plan = _block_policy.resolve(
        model_type=metadata["model_type"],
        hidden_size=int(hidden_size) if hidden_size is not None else None,
        num_hash_layers=int(metadata["num_hash_layers"] or 0),
        num_experts_per_tok=int(metadata["num_experts_per_tok"] or 0),
        max_num_batched_tokens=int(max_num_batched_tokens),
        explicit_up_block_mb=explicit_up_block_mb,
    )
    return plan.up_block_mb, plan.source, list(plan.warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="1e1c LWD UCX preflight")
    parser.add_argument("--role", choices=("edge", "cloud"), required=True)
    parser.add_argument("--cloud-host", default="")
    parser.add_argument(
        "--data-port",
        type=int,
        default=DEFAULT_DATA_PORT,
        help="the cloud DP's topology data_port (shared listener)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="local model directory or config.json for auto UP block",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="scheduler max_num_batched_tokens; required with --model",
    )
    _env_up_block = os.getenv("VLLM_ASCEND_LWD_UCX_UP_BLOCK_MB")
    parser.add_argument(
        "--up-block-mb",
        type=int,
        default=(
            int(_env_up_block)
            if _env_up_block not in (None, "")
            else None
        ),
        help="explicit UP block; omit to derive from --model",
    )
    parser.add_argument(
        "--edges",
        type=int,
        default=1,
        help="number of edges sharing this cloud engine (channels=2*edges)",
    )
    parser.add_argument(
        "--down-block-mb",
        type=int,
        default=int(
            os.getenv(
                "VLLM_ASCEND_LWD_UCX_DOWN_BLOCK_MB", DEFAULT_DOWN_BLOCK_MB
            )
        ),
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=int(os.getenv("VLLM_ASCEND_LWD_UCX_N_BLOCKS", DEFAULT_N_BLOCKS)),
    )
    parser.add_argument(
        "--budget-mb",
        type=int,
        default=int(
            os.getenv("VLLM_ASCEND_LWD_DP_MEM_BUDGET_MB", DEFAULT_BUDGET_MB)
        ),
    )
    parser.add_argument(
        "--family",
        default=os.getenv("VLLM_ASCEND_LWD_DATA_PLANE_FAMILY", "hccl"),
    )
    parser.add_argument(
        "--probe-ports",
        action="store_true",
        help="cloud binds / edge connects to the single data port",
    )
    args = parser.parse_args(argv)

    ok = True
    family = str(args.family).strip().lower()
    ok &= _check(
        family == "ucx",
        "data_plane_family",
        f"{family!r} (expected 'ucx'; default hccl keeps HCCL)",
    )

    try:
        up_block_mb, block_source, block_warnings = _policy_up_block_mb(
            model_path=args.model,
            max_num_batched_tokens=args.max_num_batched_tokens,
            explicit_up_block_mb=args.up_block_mb,
        )
        for warning in block_warnings:
            _warn("ucx block", warning)
        ok &= _check(
            True,
            "ucx block",
            f"up={up_block_mb}MiB source={block_source}",
        )
        need = memory_need_mb(
            up_block_mb,
            args.down_block_mb,
            args.n_blocks,
            channels=2 * max(1, args.edges),
        )
        ok &= _check(
            need <= args.budget_mb,
            "memory budget",
            f"need={need}MiB limit={args.budget_mb}MiB "
            f"({args.n_blocks}*({up_block_mb}"
            f"+{args.down_block_mb})*edges={args.edges})",
        )
    except ValueError as exc:
        ok &= _check(False, "ucx block/memory args", str(exc))
        up_block_mb = DEFAULT_UP_BLOCK_MB

    try:
        port = data_plane_port(args.data_port)
        ok &= _check(
            True,
            "data plane port",
            f"{port} (cloud bind, edge connect; single port)",
        )
    except ValueError as exc:
        ok &= _check(False, "data plane port", str(exc))
        port = -1

    if args.role == "edge":
        ok &= _check(
            bool(args.cloud_host),
            "cloud host",
            args.cloud_host or "set --cloud-host / data_plane_host",
        )

    ucx_tls = os.getenv("UCX_TLS", "")
    if ucx_tls:
        _check(True, "UCX_TLS", ucx_tls)
    else:
        _warn("UCX_TLS", "unset; pin rc_verbs/rc_mlx5 + tcp,self explicitly")
    if os.getenv("UCX_NET_DEVICES"):
        _check(True, "UCX_NET_DEVICES", os.getenv("UCX_NET_DEVICES", ""))
    else:
        _warn("UCX_NET_DEVICES", "unset; pin the RDMA NIC explicitly")

    if args.probe_ports:
        ok &= probe_ports(args.role, args.cloud_host, args.data_port)

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
