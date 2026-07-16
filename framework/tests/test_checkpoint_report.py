"""checkpoint 级评估报告落盘规则。"""

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
    return env


def test_checkpoint_report_and_parent_version_report_append(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"")

    report, version = write_checkpoint_reports(_env(ckpt, "t1"), tmp_path / "e1.envelope.json")
    assert report == tmp_path / "best.eval.md"
    assert version == tmp_path / "EVALUATION_REPORT.md"
    assert "Checkpoint 评估报告：best.pt" in report.read_text(encoding="utf-8")
    first = version.read_text(encoding="utf-8")
    assert first.count("checkpoint 专属报告：`best.eval.md`") == 1

    write_checkpoint_reports(_env(ckpt, "t2"), tmp_path / "e2.envelope.json")
    second = version.read_text(encoding="utf-8")
    assert second.count("checkpoint 专属报告：`best.eval.md`") == 2
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
