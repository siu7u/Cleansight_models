"""时序指标口径可独立测试（需求 §12.3）。"""

from cleansight_eval.core.envelope import MetricState
from cleansight_eval.temporal.metrics import (
    SPEC_ACC,
    compute_temporal_metrics,
    edit_score,
    f_score,
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
    # segmental：两个序列各 1 个非背景段，IoU 高 → tp>=1
    tp, fp, fn = f_score(pred, gt, 0.1)
    assert tp >= 1


def test_length_mismatch_is_missing():
    m = compute_temporal_metrics(["Idle"], ["Idle", "Long"])
    assert m["acc"].state is MetricState.MISSING
    assert m["edit"].state is MetricState.MISSING


def test_edit_score_symmetric_labels():
    seq = ["A", "B", "A"]
    assert edit_score(seq, seq) == 100.0
