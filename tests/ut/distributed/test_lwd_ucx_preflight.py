# SPDX-License-Identifier: Apache-2.0
"""1e1c UCX preflight 纯函数/退出码测试(不需要 NPU/torch)。"""

import importlib.util
import json
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "lwd_ucx_preflight.py"
_spec = importlib.util.spec_from_file_location("lwd_ucx_preflight", _TOOL)
assert _spec is not None and _spec.loader is not None
preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preflight)


def test_memory_need_default_1e1c():
    assert preflight.memory_need_mb(128, 32, 4, channels=2) == 640


def test_memory_need_rejects_bad_args():
    with pytest.raises(ValueError):
        preflight.memory_need_mb(0, 32, 4, channels=2)


def test_data_plane_port():
    assert preflight.data_plane_port(5560) == 5560
    assert preflight.data_plane_port(65535) == 65535
    with pytest.raises(ValueError):
        preflight.data_plane_port(0)
    with pytest.raises(ValueError):
        preflight.data_plane_port(65536)


def test_main_passes_with_ucx_and_budget(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LWD_DATA_PLANE_FAMILY", "ucx")
    monkeypatch.setenv("VLLM_ASCEND_LWD_DP_MEM_BUDGET_MB", "4096")
    assert preflight.main(["--role", "cloud"]) == 0


def test_main_fails_when_budget_too_small(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LWD_DATA_PLANE_FAMILY", "ucx")
    monkeypatch.setenv("VLLM_ASCEND_LWD_DP_MEM_BUDGET_MB", "512")
    assert preflight.main(["--role", "cloud"]) == 1


def test_main_fails_on_hccl(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LWD_DATA_PLANE_FAMILY", "hccl")
    assert preflight.main(["--role", "cloud"]) == 1


def _write_model_config(tmp_path, **overrides):
    config = {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hash_layers": 3,
        "num_experts_per_tok": 6,
    }
    config.update(overrides)
    (tmp_path / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return str(tmp_path)


def test_local_model_auto_block(tmp_path):
    model = _write_model_config(tmp_path)
    up, source, warnings = preflight._policy_up_block_mb(
        model_path=model,
        max_num_batched_tokens=20_480,
        explicit_up_block_mb=None,
    )
    assert up == 192
    assert source == "auto"
    assert warnings == []


def test_unknown_model_falls_back_in_preflight(tmp_path):
    model = _write_model_config(
        tmp_path,
        model_type="unknown_model",
        hidden_size=7168,
        num_hash_layers=0,
        num_experts_per_tok=0,
    )
    up, source, warnings = preflight._policy_up_block_mb(
        model_path=model,
        max_num_batched_tokens=20_480,
        explicit_up_block_mb=None,
    )
    assert up == 128
    assert source == "fallback"
    assert warnings


def test_model_requires_max_num_batched_tokens(tmp_path):
    model = _write_model_config(tmp_path)
    with pytest.raises(ValueError, match="max-num-batched-tokens"):
        preflight._policy_up_block_mb(
            model_path=model,
            max_num_batched_tokens=None,
            explicit_up_block_mb=None,
        )


def test_preflight_main_with_local_model(tmp_path, monkeypatch):
    model = _write_model_config(tmp_path)
    monkeypatch.setenv("VLLM_ASCEND_LWD_DATA_PLANE_FAMILY", "ucx")
    monkeypatch.setenv("VLLM_ASCEND_LWD_DP_MEM_BUDGET_MB", "4096")
    assert preflight.main(
        [
            "--role",
            "cloud",
            "--model",
            model,
            "--max-num-batched-tokens",
            "20480",
        ]
    ) == 0
