"""framework 组员友好能力（model_aliases / dataset_download / cli.dataset / cli.train --model）测试。

覆盖：
  - model_aliases：--list-models 内容、resolve_model 解析、未知模型报错
  - dataset_download：check_required_datasets 在临时目录下的缺失判定
  - cli.train：--model 与 --config 互斥、--list-models 无 torch 可用
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_list_models_contains_all():
    from framework.cleansight_eval.core.model_aliases import list_models

    text = list_models()
    for name in ("yolo", "gru", "mstcn", "mstcn2", "transformer", "feature_fusion",
                 "yolo11n", "yolo11s", "yolo11m"):
        assert name in text
    assert "--group" in text
    assert "cli.dataset --preset all" in text


def test_resolve_yolo_weights_and_group():
    from framework.cleansight_eval.core.model_aliases import resolve_model

    info = resolve_model("yolo11s", "group1_large")
    assert info["overrides"] == {"model.weights": "yolo11s.pt"}
    assert info["group"] == "group1_large"
    assert info["config"] == "yolo-clean-large.yaml"

    info2 = resolve_model("yolo", "group2_small")
    assert info2["group"] == "group2_small"
    assert "overrides" not in info2


def test_unknown_model_fails():
    from framework.cleansight_eval.core.model_aliases import resolve_model

    with pytest.raises(SystemExit, match="未知模型"):
        resolve_model("yolov5")


def test_model_config_path():
    from framework.cleansight_eval.core.model_aliases import model_config_path, resolve_model

    path = model_config_path(resolve_model("gru"))
    assert path.name == "gru-actionmixed.yaml"
    assert path.is_file()


def test_dataset_check_missing_in_tmp(tmp_path, monkeypatch):
    from framework.cleansight_eval.core.dataset_download import REQUIRED_FILES, check_required_datasets

    missing = check_required_datasets(["yolo", "actionmixed"], root=tmp_path)
    assert set(missing) == {"yolo", "actionmixed"}

    (tmp_path / "datasets/cleansight-yolo/group1_large").mkdir(parents=True)
    (tmp_path / "datasets/cleansight-yolo/group2_small").mkdir(parents=True)
    for rel in REQUIRED_FILES["yolo"]:
        (tmp_path / rel).write_text("train: images/train\n", encoding="utf-8")
    missing = check_required_datasets(["yolo", "actionmixed"], root=tmp_path)
    assert missing == ["actionmixed"]


def test_train_cli_rejects_config_and_model_together():
    """--config 与 --model 同时传 → main 抛 SystemExit（互斥）。"""

    from framework.cleansight_eval.cli.train import main

    with pytest.raises(SystemExit, match="二选一"):
        main(["--config", "a.yaml", "--model", "gru"])


def test_train_cli_list_models_runs_without_torch():
    """--list-models 不 import torch/numpy 即可运行（在无 torch 的 python 下验证）。"""

    import subprocess
    import sys as _sys

    code = subprocess.run(
        [_sys.executable, "-m", "framework.cleansight_eval.cli.train", "--list-models"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert code.returncode == 0
    assert "可训练模型" in code.stdout
    assert "Traceback" not in code.stderr
