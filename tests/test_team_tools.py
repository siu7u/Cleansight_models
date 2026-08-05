"""组员工具（tools/team_*.py）逻辑测试。

覆盖：
  - team_train：--list 输出、模型解析、-S 覆盖解析、未知模型报错
  - team_dataset：check_required_datasets 在临时目录下的缺失判定
  - team_env：check_env 在无 torch 环境下不崩溃并标记必需项缺失
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Args:
    def __init__(self, model="yolo", group=None):
        self.model = model
        self.group = group


def test_team_train_list_contains_all_models():
    from tools.team_train import list_models

    text = list_models()
    for name in ("yolo", "gru", "mstcn", "mstcn2", "transformer", "feature_fusion",
                 "yolo11n", "yolo11s", "yolo11m"):
        assert name in text
    assert "--group" in text


def test_team_train_resolve_yolo_weights_and_group():
    from tools.team_train import resolve_model

    info = resolve_model(_Args(model="yolo11s", group="group1_large"))
    assert info["weights"] == "yolo11s.pt"
    assert info["group"] == "group1_large"
    assert info["config"].endswith("yolo-clean-large.yaml")

    # 默认 yolo 不带显式 weights，但 group 可覆盖
    info2 = resolve_model(_Args(model="yolo", group="group2_small"))
    assert info2["group"] == "group2_small"


def test_team_train_unknown_model_fails():
    from tools.team_train import resolve_model

    with pytest.raises(SystemExit, match="未知模型"):
        resolve_model(_Args(model="yolov5"))


def test_team_train_parse_overrides():
    from tools.team_train import _parse_overrides

    overrides = _parse_overrides(["train.epochs=200", "model.imgsz=960", "train.cos_lr=true"])
    assert ("train.epochs", 200) in overrides
    assert ("model.imgsz", 960) in overrides
    assert ("train.cos_lr", True) in overrides

    with pytest.raises(SystemExit, match="KEY=VALUE"):
        _parse_overrides(["no-equals"])


def test_team_dataset_check_missing_in_tmp(tmp_path, monkeypatch):
    from tools.team_dataset import REQUIRED_FILES, check_required_datasets

    # 临时目录里什么都没有 → 全部缺失
    monkeypatch.setattr("tools.team_dataset.ROOT", tmp_path)
    missing = check_required_datasets(["yolo", "actionmixed"])
    assert set(missing) == {"yolo", "actionmixed"}

    # 伪造一个 yolo data.yaml → yolo 就绪
    (tmp_path / "datasets/cleansight-yolo/group1_large").mkdir(parents=True)
    (tmp_path / "datasets/cleansight-yolo/group2_small").mkdir(parents=True)
    for rel in REQUIRED_FILES["yolo"]:
        (tmp_path / rel).write_text("train: images/train\n", encoding="utf-8")
    missing = check_required_datasets(["yolo", "actionmixed"])
    assert missing == ["actionmixed"]


def test_team_env_check_without_torch(tmp_path, monkeypatch):
    """无 torch 环境下 check_env 仍可运行并标记 torch 缺失。"""

    from tools.team_env import CHECKS, check_env

    # 强制所有可选导入失败（模拟无依赖环境）
    def fake_importable(name):
        return name is None  # 只有 python 检查通过

    monkeypatch.setattr("tools.team_env._importable", fake_importable)
    results = check_env()
    by_name = {pkg: (ok, required) for pkg, ok, _, required in results}
    assert by_name["python"][0] is True
    assert by_name["torch"][0] is False
    assert by_name["torch"][1] is True  # 必需
    assert by_name["ultralytics"][1] is False  # 可选
    # CHECKS 与结果一一对应
    assert len(results) == len(CHECKS)
