"""checkpoint 级评估报告落盘规则。"""

import os
from pathlib import Path

from cleansight_eval.core.envelope import EvalEnvelope, MetricValue
from cleansight_eval.core.report import write_checkpoint_reports


def _env(ckpt, timestamp):
    return EvalEnvelope(
        model_type="gru",
        model_id="gru-1k",
        pipeline="sliding_window_temporal",
        checkpoint=str(ckpt),
        dataset="synthetic",
        feature_schema={"dim": 40, "version": "test"},
        metrics={"acc": MetricValue.computed(0.5, spec="acc/v1")},
        performance={"latency_mean_ms": MetricValue.computed(1.2, spec="latency/model-forward/v1")},
        inference_semantics={"mode": "windowed_causal"},
        integrity={"ok": True, "issues": []},
        num_params=123,
        timestamp=timestamp,
    )


def _yolo_env(ckpt, timestamp):
    env = _env(ckpt, timestamp)
    env.pipeline = "detection"
    env.model_type = "yolo"
    env.model_id = "yolo-2.6M"
    env.num_params = 2600000
    env.metrics = {
        "mAP@0.5": MetricValue.computed(0.93, spec="map/coco-0.5/v1"),
        "mAP@0.5:0.95": MetricValue.computed(0.58, spec="map/coco-0.5:0.95/v1"),
    }
    env.metric_details = {
        "per_class": {
            "hand": {"precision": 0.9, "recall": 0.8},
            "scope": {
                "precision": {"state": "missing", "reason": "验证集无样本"},
                "recall": {"state": "missing", "reason": "验证集无样本"},
            },
        },
        "per_class_specs": {"precision": "precision/v1", "recall": "recall/v1"},
    }
    env.artifacts = {
        "predictions": {
            "path": "artifacts/predictions.json",
            "sha256": "a" * 64,
            "schema_version": 1,
        }
    }
    return env


def test_checkpoint_report_and_parent_version_report_append(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"")

    report, version = write_checkpoint_reports(_env(ckpt, "t1"), tmp_path / "e1.envelope.json")
    assert report == tmp_path / "best.eval.md"
    assert version == tmp_path / "EVALUATION_REPORT.md"
    report_text = report.read_text(encoding="utf-8")
    assert "Checkpoint 评估报告：best.pt" in report_text
    assert "checkpoint：[`" + str(ckpt) + "`](<best.pt>)" in report_text
    assert "evaluation result：[`" + str(tmp_path / "e1.envelope.json") + "`](<e1.envelope.json>)" in report_text
    first = version.read_text(encoding="utf-8")
    assert first.count("checkpoint 专属报告：[best.eval.md](<best.eval.md>)") == 1

    write_checkpoint_reports(_env(ckpt, "t2"), tmp_path / "e2.envelope.json")
    second = version.read_text(encoding="utf-8")
    assert second.count("checkpoint 专属报告：[best.eval.md](<best.eval.md>)") == 2
    assert "t1 · best.pt" in second
    assert "t2 · best.pt" in second


def test_version_report_groups_temporal_and_yolo(tmp_path):
    temporal = tmp_path / "temporal.pt"
    yolo = tmp_path / "yolo.pt"
    temporal.write_bytes(b"")
    yolo.write_bytes(b"")

    _, version = write_checkpoint_reports(_env(temporal, "t1"), tmp_path / "t1.json")
    write_checkpoint_reports(_yolo_env(yolo, "t2"), tmp_path / "t2.json")

    text = version.read_text(encoding="utf-8")
    assert text.count("## 时序模型") == 1
    assert text.count("## YOLO 探测模型") == 1
    assert "### t1 · temporal.pt" in text
    assert "### t2 · yolo.pt" in text


def test_yolo_report_renders_map95_per_class_and_artifact_table(tmp_path):
    ckpt = tmp_path / "yolo.pt"
    ckpt.write_bytes(b"")

    report, _ = write_checkpoint_reports(_yolo_env(ckpt, "t1"), tmp_path / "result.json")
    text = report.read_text(encoding="utf-8")

    assert "| mAP@0.5:0.95 | 0.58" in text
    assert "## 逐类指标" in text
    assert "| hand | 0.9 | 0.8 |" in text
    assert "| scope | MISSING — 验证集无样本 | MISSING — 验证集无样本 |" in text
    assert "| predictions | [`artifacts/predictions.json`](<artifacts/predictions.json>) | 1 |" in text
    assert "需结合 testset 真值" in text
    assert "{'path':" not in text


def test_report_links_are_relative_to_checkpoint_directory(tmp_path):
    run_dir = tmp_path / "run with spaces"
    checkpoints = run_dir / "checkpoints"
    evals = run_dir / "evals"
    artifacts = run_dir / "artifacts"
    checkpoints.mkdir(parents=True)
    evals.mkdir()
    artifacts.mkdir()
    ckpt = checkpoints / "best.pt"
    ckpt.write_bytes(b"")
    result_path = evals / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    (artifacts / "predictions.json").write_text("{}", encoding="utf-8")

    env = _yolo_env(ckpt, "t1")
    env.artifacts["predictions"]["path"] = "artifacts/predictions.json"
    report, version = write_checkpoint_reports(env, result_path)

    text = report.read_text(encoding="utf-8")
    assert f"[`{ckpt}`](<best.pt>)" in text
    assert f"[`{result_path}`](<../evals/result.json>)" in text
    assert "[`artifacts/predictions.json`](<../artifacts/predictions.json>)" in text
    assert "checkpoint 专属报告：[best.eval.md](<best.eval.md>)" in version.read_text(encoding="utf-8")


def test_config_link_uses_repo_root_when_running_from_framework(tmp_path, monkeypatch):
    framework_dir = Path(__file__).resolve().parents[1]
    config = framework_dir / "experiments" / "yolo-clean-large.yaml"
    monkeypatch.chdir(framework_dir)
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"")
    env = _env(ckpt, "t1")
    env.run = {"config": "framework/experiments/yolo-clean-large.yaml"}

    report, _ = write_checkpoint_reports(env, tmp_path / "result.json")

    relative_config = Path(os.path.relpath(config, start=tmp_path)).as_posix()
    text = report.read_text(encoding="utf-8")
    assert f"[`framework/experiments/yolo-clean-large.yaml`](<{relative_config}>)" in text
