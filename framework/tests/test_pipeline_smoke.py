"""端到端冒烟：train → eval → matrix，使用合成的迷你 ActionMixed 数据集。

机械验证 CLI 全链路可跑通、产物结构正确（含 loader：ActionMixed 目录 → bbox→40维
特征 → windowed_causal 加窗）。数值对齐验收需在有真实数据的机器上执行。
"""

import json

import numpy as np
import yaml

from cleansight_eval.cli import eval as eval_cli
from cleansight_eval.cli import matrix as matrix_cli
from cleansight_eval.cli import train as train_cli
from cleansight_eval.core.checkpoint import load_training_checkpoint
from cleansight_eval.core.envelope import MetricState

_ACTIONS = ["idle", "air_injection", "flush", "long_brush_insert", "long_brush_withdraw", "short_brush_cleaning"]


def _make_actionmixed(root, seed=0):
    """造迷你 ActionMixed：labels/<split>/<vid>.mp4.txt + frames/<split>/<vid>-<f>.txt。"""
    rng = np.random.default_rng(seed)
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "data.yaml").write_text(
        "nc: 6\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(_ACTIONS))
    )
    # split -> {video: n_sampled_frames}
    layout = {"train": {"v0": 60, "v1": 50}, "test": {"v2": 40}}
    for split, vids in layout.items():
        (root / "labels" / split).mkdir(parents=True)
        (root / "frames" / split).mkdir(parents=True)
        for vid, T in vids.items():
            frame_ids = list(range(1, T * 4, 4))  # stride-4 采样帧号
            actions = rng.integers(0, len(_ACTIONS), size=T)
            (root / "labels" / split / f"{vid}.mp4.txt").write_text(
                "\n".join(f"{fid} {a}" for fid, a in zip(frame_ids, actions)) + "\n"
            )
            for fid in frame_ids:
                # 随机 0-3 个 bbox（8 类），部分帧留空（无检测）
                n = rng.integers(0, 4)
                lines = [
                    f"{rng.integers(0,8)} {rng.random():.4f} {rng.random():.4f} {rng.random()*0.3:.4f} {rng.random()*0.3:.4f}"
                    for _ in range(n)
                ]
                (root / "frames" / split / f"{vid}.mp4-{fid:06d}.txt").write_text("\n".join(lines) + "\n")


def _write_config(path, data_root):
    cfg = {
        "pipeline": "sliding_window_temporal",
        "model": {"type": "gru", "input_dim": 40, "num_classes": 6, "hidden": 16, "num_layers": 1},
        "data": {
            "name": "synthetic-actionmixed",
            "root": str(data_root),
            "action_mapping": "labels/data.yaml",
            "labels_dir": "labels",
            "frames_dir": "frames",
            "split_train": "train",
            "split_eval": "test",
        },
        "feature_schema": {"dim": 40, "version": "actionmixed-bbox-8cls-v1"},
        "train": {"epochs": 1, "lr": 0.01, "batch_size": 8, "window": 8},
    }
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True))


def test_end_to_end(tmp_path):
    data_root = tmp_path / "cleansight-ActionMixed"
    _make_actionmixed(data_root)
    cfg_path = tmp_path / "gru.yaml"
    _write_config(cfg_path, data_root)
    runs_dir = tmp_path / "runs"

    # train
    ckpt = train_cli.main(["--config", str(cfg_path), "--runs-dir", str(runs_dir)])
    assert (tmp_path / "runs").exists()
    assert ckpt.endswith(".pt")
    run_dir = next(runs_dir.iterdir())
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "status.json").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "succeeded"
    assert status["best_metric"]["name"] == "val_acc"
    payload, _meta = load_training_checkpoint(run_dir / "checkpoints" / "last.pt")
    assert payload["epoch"] == 1
    assert "optimizer_state" in payload

    # eval → 一份信封（训练与评估同属一条流水线，输入构造与输出语义一致）
    envelopes = eval_cli.main(["--config", str(cfg_path), "--ckpt", ckpt])
    assert len(envelopes) == 1
    data = json.loads(open(envelopes[0]).read())
    assert data["schema_version"] == 2
    assert data["pipeline"] == "sliding_window_temporal"
    summary = data["metrics"]["summary"]
    assert summary["acc"]["state"] in (
        MetricState.COMPUTED.value,
        MetricState.MISSING.value,
    )
    # framework 已复用 benchmark 的逐视频一对一片段匹配口径。
    assert summary["tp@0.5"]["state"] == MetricState.COMPUTED.value
    assert summary["precision@0.5"]["state"] == MetricState.COMPUTED.value
    assert summary["recall@0.5"]["state"] == MetricState.COMPUTED.value
    assert "temporal_iou@0.5" in summary
    assert "tp@0.1" not in summary
    assert not any(key.startswith("frame.f1:") for key in summary)
    assert data["artifacts"]["predictions"]["recomputable"] is True
    temporal_details = data["metrics"]["details"]["temporal"]
    assert temporal_details["metric_spec"]["video_boundaries_preserved"] is True
    assert "0.10" in temporal_details["segment"]["details_at_iou"]
    assert temporal_details["frame"]["per_class"]
    # 滑窗流水线：测实时延迟、记录窗口与冷启动语义
    assert data["performance"]["latency_mean_ms"]["state"] == MetricState.COMPUTED.value
    assert data["inference"]["window"] == 8
    assert "cold_start" in data["inference"]
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    checkpoint_report = run_dir / "checkpoints" / "best.eval.md"
    version_report = run_dir / "checkpoints" / "EVALUATION_REPORT.md"
    assert ckpt == str(ckpt_path)
    assert checkpoint_report.exists()
    assert version_report.exists()
    report_text = checkpoint_report.read_text(encoding="utf-8")
    assert "Checkpoint 评估报告：best.pt" in report_text
    assert "人工维护区" in report_text
    version_text = version_report.read_text(encoding="utf-8")
    assert "版本化评估报告" in version_text
    assert "checkpoint 专属报告：`best.eval.md`" in version_text

    # matrix
    matrix_json = matrix_cli.main(["--runs", str(runs_dir)])
    matrix = json.loads(open(matrix_json).read())
    assert len(matrix["rows"]) == 1
    assert "perf.latency_mean_ms" in matrix["metric_columns"]
    md = open(matrix_json.replace(".json", ".md")).read()
    assert "N/A" in md  # 图例中说明 N/A 语义


def test_resume_from_last_checkpoint(tmp_path):
    data_root = tmp_path / "cleansight-ActionMixed"
    _make_actionmixed(data_root)
    cfg_path = tmp_path / "gru.yaml"
    _write_config(cfg_path, data_root)
    runs_dir = tmp_path / "runs"

    first_ckpt = train_cli.main(["--config", str(cfg_path), "--runs-dir", str(runs_dir)])
    first_run = next(runs_dir.iterdir())
    resume_path = first_run / "checkpoints" / "last.pt"
    assert first_ckpt.endswith("best.pt")

    second_ckpt = train_cli.main([
        "--config", str(cfg_path),
        "--runs-dir", str(runs_dir),
        "--resume", str(resume_path),
        "-S", "train.epochs=2",
    ])
    second_run = sorted(runs_dir.iterdir())[-1]
    payload, _meta = load_training_checkpoint(second_run / "checkpoints" / "last.pt")
    assert second_ckpt.endswith("best.pt")
    assert payload["epoch"] == 2
    status = json.loads((second_run / "status.json").read_text())
    assert status["state"] == "succeeded"
