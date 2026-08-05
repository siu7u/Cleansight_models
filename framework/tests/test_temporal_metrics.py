"""benchmark 时序指标口径可独立测试。"""

from framework.cleansight_eval.core.metrics import edit_score
from benchmark.core.result import MetricState
from benchmark.evaluators.temporal import (
    SPEC_ACC,
    compute_temporal_metrics,
    compute_temporal_metrics_by_item,
)


def test_perfect_prediction():
    gt = ["Idle"] * 30 + ["Long"] * 30 + ["Idle"] * 30
    metrics = compute_temporal_metrics(gt, gt)
    assert metrics["acc"].state is MetricState.COMPUTED
    assert metrics["acc"].value == 100.0
    assert metrics["acc"].spec == SPEC_ACC
    # 完美预测：edit 与 F1 均为满分
    assert metrics["edit"].value == 100.0
    assert metrics["f1@0.5"].value == 100.0


def test_edit_and_fscore_basic():
    gt = ["Idle", "Idle", "Long", "Long"]
    pred = ["Idle", "Long", "Long", "Long"]
    # 逐帧 3/4 正确
    m = compute_temporal_metrics(pred, gt)
    assert m["acc"].value == 75.0
    # segmental：两个序列在主 IoU 阈值下至少匹配一个片段。
    assert m["tp@0.5"].value >= 1


def test_length_mismatch_is_missing():
    m = compute_temporal_metrics(["Idle"], ["Idle", "Long"])
    assert m["acc"].state is MetricState.MISSING
    assert m["edit"].state is MetricState.MISSING


def test_edit_score_symmetric_labels():
    seq = ["A", "B", "A"]
    assert edit_score(seq, seq) == 1.0


def test_benchmark_metrics_preserve_video_boundaries_and_counts():
    metrics = compute_temporal_metrics_by_item(
        {"a": ["Long", "Long"], "b": ["Long", "Long"]},
        {"a": ["Long", "Long"], "b": ["Long", "Long"]},
        ["Idle", "Long"],
    )
    assert metrics["tp@0.5"].value == 2
    assert metrics["fp@0.5"].value == 0
    assert metrics["fn@0.5"].value == 0
    assert metrics["precision@0.5"].value == 100.0
    assert metrics["recall@0.5"].value == 100.0
    assert metrics["temporal_iou@0.5"].value == 100.0
