"""检测指标组装的单元测试（免 ultralytics）。

核心对齐点（§8.2 / §10 / §13.11）：
- 整体 mAP/P/R 与逐类"有样本"的指标 → COMPUTED，且声明口径 spec；
- 逐类"检出为 0"仍是 COMPUTED 0（不是 MISSING、不是 N/A）；
- 逐类"验证集无样本" → MISSING（不是 0、不是 N/A）；
- 全程不含任何 PASS/FAIL / 达标判断字段。
"""

from cleansight_eval.core.envelope import MetricState
from cleansight_eval.tasks.detection.metrics import build_detection_metrics


def _fake_val():
    # data.yaml 声明 3 类；验证集里 hand 有样本且检出，scope_mid 有样本但全漏检(0)，
    # scope_ctrl 无样本（不在 per_class 中）。
    return {
        "map50": 0.612345,
        "map50_95": 0.401111,
        "precision": 0.5,
        "recall": 0.45,
        "names": {0: "hand", 1: "scope_control_body", 2: "scope_mid_section"},
        "per_class": {
            "hand": {"precision": 0.8, "recall": 0.7, "map50": 0.75},
            "scope_mid_section": {"precision": 0.0, "recall": 0.0, "map50": 0.0},
        },
    }


def test_overall_metrics_computed_with_spec():
    m = build_detection_metrics(_fake_val())
    for key in ("mAP@0.5", "mAP@0.5:0.95", "precision", "recall"):
        assert m[key].state is MetricState.COMPUTED
        assert m[key].spec, f"{key} 已计算必须声明口径 spec"
    assert m["mAP@0.5"].value == 0.6123


def test_detected_zero_is_computed_not_missing():
    m = build_detection_metrics(_fake_val())
    # scope_mid_section 有样本但全漏检 → COMPUTED 0，绝不能是 MISSING/N/A
    assert m["recall:scope_mid_section"].state is MetricState.COMPUTED
    assert m["recall:scope_mid_section"].value == 0.0


def test_no_sample_class_is_missing_not_zero():
    m = build_detection_metrics(_fake_val())
    # scope_control_body 验证集无样本 → MISSING（区分于检出为 0）
    assert m["recall:scope_control_body"].state is MetricState.MISSING
    assert m["precision:scope_control_body"].state is MetricState.MISSING
    assert m["recall:scope_control_body"].value is None


def test_no_passfail_fields():
    m = build_detection_metrics(_fake_val())
    # 指标里不得出现任何达标/判决语义的键
    forbidden = {"passed", "pass", "fail", "accepted", "verdict", "threshold"}
    assert not (forbidden & set(m.keys()))
    for mv in m.values():
        assert mv.state in (MetricState.COMPUTED, MetricState.MISSING, MetricState.NOT_APPLICABLE)
