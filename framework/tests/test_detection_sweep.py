"""检测优化实验编排（sweep）逻辑测试。

注入假 adapter（仿 test_detection_smoke.py），验证：
  - dry-run 不触发训练、不 import ultralytics
  - run_experiment 产出整体 + 逐类指标
  - classify 阈值语义（在 analysis 单测覆盖）
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from framework.cleansight_eval.cli import sweep as cli_sweep
from framework.cleansight_eval.detection import sweep


class _FakeAdapter:
    """假 YOLO adapter：train 返回假 best.pt 路径，val 返回同形指标 dict。"""

    def __init__(self, best_path):
        self.best_path = best_path
        self.train_calls = 0
        self.val_calls = 0
        self.train_kwargs = []

    def train(self, **kwargs):
        self.train_calls += 1
        self.train_kwargs.append(kwargs)
        best = self.best_path
        best.write_bytes(b"fake-weight")
        return best, 2_600_000, {0: "hand", 1: "scope_control_body"}, 2

    def val(self, **kwargs):
        self.val_calls += 1
        return {
            "map50": 0.5123,
            "map50_95": 0.1800,
            "precision": 0.5900,
            "recall": 0.5000,
            "per_class": {
                "hand": {"precision": 0.8, "recall": 0.7, "map50": 0.75},
                "scope_control_body": {"precision": 0.4, "recall": 0.3, "map50": 0.35},
            },
        }


@pytest.fixture
def fake_dataset(monkeypatch, tmp_path):
    """把 DATASET_BASE 指到临时目录并造 data.yaml。"""
    group_dir = tmp_path / "datasets" / "cleansight-yolo" / "group1_large"
    group_dir.mkdir(parents=True)
    (group_dir / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames:\n  0: hand\n  1: scope_control_body\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sweep, "DATASET_BASE", tmp_path / "datasets" / "cleansight-yolo")
    monkeypatch.setattr(sweep, "RUNS_BASE", tmp_path / "runs")
    return group_dir


def test_get_augment_params_shapes():
    default = sweep.get_augment_params("default", 640)
    strong = sweep.get_augment_params("strong", 640)
    copy_paste = sweep.get_augment_params("copy_paste", 640)
    assert default["mosaic"] == 1.0 and default["mixup"] == 0.0
    assert strong["mixup"] > 0
    assert copy_paste["copy_paste"] > 0


def test_adjust_batch_for_imgsz():
    assert sweep.adjust_batch_for_imgsz(16, 640, 640) == 16
    assert sweep.adjust_batch_for_imgsz(16, 640, 1280) == 4
    assert sweep.adjust_batch_for_imgsz(16, 640, 2560) == 2  # 下限保护


def test_dry_run_does_not_touch_adapter(fake_dataset, monkeypatch):
    calls = {"train": 0}

    def fake_train(**kwargs):
        calls["train"] += 1
        raise AssertionError("dry-run 不应触发训练")

    monkeypatch.setattr(sweep, "get_adapter", lambda mt: SimpleNamespace(train=fake_train))

    result = sweep.run_experiment("group1_large", "large_baseline",
                                  dict(sweep.PRESETS["large_baseline"]), dry_run=True)

    assert result["dry_run"] is True
    assert calls["train"] == 0


def test_run_experiment_produces_metrics(fake_dataset, monkeypatch, tmp_path):
    best = tmp_path / "best.pt"
    adapter = _FakeAdapter(best)
    monkeypatch.setattr(sweep, "get_adapter", lambda mt: adapter)
    cfg = dict(sweep.PRESETS["large_s"])
    cfg["device"] = "cpu"  # 显式设备，跳过 auto 的设备探测（避免依赖 torch/numpy）

    result = sweep.run_experiment("group1_large", "large_s", cfg, dry_run=False)

    assert adapter.train_calls == 1
    assert adapter.val_calls == 1
    assert result["val"]["map50"] == 0.5123
    assert result["val"]["per_class"]["hand"]["precision"] == 0.8
    assert "error" not in result


def test_run_grid_builds_combinations(fake_dataset, monkeypatch, tmp_path):
    results = []

    def fake_experiment(group, preset_name, cfg, dry_run=False, smoke=False, device=None):
        results.append(preset_name)
        return {"name": preset_name, "dry_run": True, "cfg": cfg}

    monkeypatch.setattr(sweep, "run_experiment", fake_experiment)

    sweep.run_grid("group1_large", ["models", "resolutions"], dry_run=True)

    # 3 模型 × 3 分辨率 = 9
    assert len(results) == 9
    assert "mn-r640" in results and "mm-r1280" in results


def test_smoke_mode_caps_epochs_and_subsamples(fake_dataset, monkeypatch, tmp_path):
    best = tmp_path / "best.pt"
    adapter = _FakeAdapter(best)
    monkeypatch.setattr(sweep, "get_adapter", lambda mt: adapter)
    cfg = dict(sweep.PRESETS["large_m_960"])  # 原 epochs=200 / patience=40
    cfg["device"] = "cpu"

    result = sweep.run_experiment("group1_large", "large_m_960", cfg,
                                  dry_run=False, smoke=True)

    assert result["smoke"] is True
    assert adapter.train_calls == 1
    train_kwargs = adapter.train_kwargs[0]["train_cfg"]
    assert train_kwargs["epochs"] == sweep.SMOKE_EPOCHS
    assert train_kwargs["patience"] == sweep.SMOKE_PATIENCE
    assert train_kwargs["fraction"] == sweep.SMOKE_FRACTION
    # 探针仍产出同口径 val 指标
    assert result["val"]["map50"] == 0.5123


def test_smoke_dry_run_marks_plan_without_training(fake_dataset, monkeypatch):
    calls = {"train": 0}

    def fake_train(**kwargs):
        calls["train"] += 1
        raise AssertionError("dry-run 不应触发训练")

    monkeypatch.setattr(sweep, "get_adapter", lambda mt: SimpleNamespace(train=fake_train))

    result = sweep.run_experiment("group1_large", "large_s",
                                  dict(sweep.PRESETS["large_s"]),
                                  dry_run=True, smoke=True)

    assert result["dry_run"] is True
    assert result["smoke"] is True
    assert calls["train"] == 0


def test_run_experiment_resolves_local_weights_to_repo_root(fake_dataset, monkeypatch, tmp_path):
    # 仓库根目录存在同名权重时，裸权重名提前解析为绝对路径（chdir 到分组目录后
    # 相对路径会失效并触发 ultralytics 从 GitHub 下载）。
    weight_dir = tmp_path / "repo"
    weight_dir.mkdir()
    (weight_dir / "yolo11n.pt").write_bytes(b"fake-weight")
    monkeypatch.setattr(sweep, "REPO_ROOT", weight_dir)
    best = tmp_path / "best.pt"
    adapter = _FakeAdapter(best)
    monkeypatch.setattr(sweep, "get_adapter", lambda mt: adapter)
    cfg = dict(sweep.PRESETS["large_baseline"])
    cfg["device"] = "cpu"

    sweep.run_experiment("group1_large", "large_baseline", cfg, dry_run=False)

    weights = adapter.train_kwargs[0]["weights"]
    assert Path(weights) == weight_dir / "yolo11n.pt"


def test_run_experiment_explicit_device_wins(fake_dataset, monkeypatch, tmp_path):
    best = tmp_path / "best.pt"
    adapter = _FakeAdapter(best)
    monkeypatch.setattr(sweep, "get_adapter", lambda mt: adapter)

    sweep.run_experiment("group1_large", "large_s",
                         dict(sweep.PRESETS["large_s"]), dry_run=False,
                         device="1")

    assert adapter.train_kwargs[0]["device"] == "1"


def test_cli_forwards_device_to_experiments(monkeypatch):
    calls = []

    def fake_run_experiment(group, preset_name, cfg, dry_run=False, smoke=False, device=None):
        calls.append((group, preset_name, device))
        return {"name": preset_name, "dry_run": True, "cfg": cfg}

    monkeypatch.setattr(cli_sweep, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(cli_sweep, "run_grid", lambda *a, **k: [])

    cli_sweep.main(["--group", "group1_large", "--preset", "large_s",
                    "--device", "1", "--dry-run"])

    assert calls[0][2] == "1"
