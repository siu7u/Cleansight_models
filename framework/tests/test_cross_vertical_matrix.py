"""跨流水线单一矩阵守卫（简化后最关键的不变量，需求 §9）。

时序与检测是三条独立流水线，但它们的信封必须能汇入**同一份**异构矩阵：不同流水线有
不同指标列、三态严格区分、不生成综合分数。这是外部模型管理仓库消费的唯一产出，简化绝不
能把它拆成两份。
"""

from benchmark.core.result import EvaluationResult, MetricState, MetricValue
from benchmark.core.matrix import build_matrix, collect_envelopes, render_markdown


def _temporal_env() -> EvaluationResult:
    return EvaluationResult(
        model_type="gru",
        model_id="gru-128h",
        pipeline="sliding_window_temporal",
        checkpoint="runs/gru/checkpoints/x.pt",
        dataset="endo-v1",
        feature_schema={"dim": 40, "version": "actionmixed-bbox-8cls-v1"},
        metrics={
            "accuracy": MetricValue.computed(0.91, spec="acc/frame-wise/v1"),
            "edit": MetricValue.computed(0.83, spec="edit/levenstein-norm/v1"),
        },
        performance={
            "latency_mean_ms": MetricValue.computed(1.2, spec="latency/single_tick_ms/v1"),
        },
        num_params=123456,
    )


def _detection_env() -> EvaluationResult:
    return EvaluationResult(
        model_type="yolo",
        model_id="yolo-group1",
        pipeline="detection",
        checkpoint="runs/yolo/checkpoints/best.pt",
        dataset="group1_large",
        feature_schema={"modality": "image", "imgsz": 640},
        metrics={
            "mAP@0.5": MetricValue.computed(0.61, spec="map/coco-0.5/v1"),
            "recall:scope_ctrl": MetricValue.missing(reason="验证集无该类样本"),
        },
        performance={
            "latency_mean_ms": MetricValue.not_applicable("单帧检测不测实时延迟"),
        },
        num_params=2600000,
    )


def test_two_verticals_fold_into_one_heterogeneous_matrix(tmp_path):
    runs = tmp_path / "runs"
    _temporal_env().write(runs / "gru" / "evals" / "a.envelope.json")
    _detection_env().write(runs / "yolo" / "evals" / "b.envelope.json")

    envs = collect_envelopes(runs)
    assert len(envs) == 2  # 同一次扫描收纳两域

    matrix = build_matrix(envs)
    cols = set(matrix["metric_columns"])
    # 异构列：时序与检测的指标在同一张表里并存
    assert {"accuracy", "edit", "mAP@0.5", "recall:scope_ctrl"} <= cols
    assert {"perf.latency_mean_ms"} <= cols

    rows = {r["model_type"]: r for r in matrix["rows"]}
    gru, yolo = rows["gru"], rows["yolo"]

    # 时序行没有检测列 → 空白（既非 N/A 也非 MISSING）
    assert "mAP@0.5" not in gru["cells"]
    # 检测行没有时序列 → 空白
    assert "accuracy" not in yolo["cells"]

    # 三态严格区分：COMPUTED / MISSING / N/A 各归其位
    assert gru["cells"]["accuracy"]["state"] == MetricState.COMPUTED.value
    assert yolo["cells"]["recall:scope_ctrl"]["state"] == MetricState.MISSING.value
    assert yolo["cells"]["perf.latency_mean_ms"]["state"] == MetricState.NOT_APPLICABLE.value
    assert gru["cells"]["perf.latency_mean_ms"]["state"] == MetricState.COMPUTED.value


def test_matrix_has_no_combined_score_column(tmp_path):
    runs = tmp_path / "runs"
    _temporal_env().write(runs / "a.envelope.json")
    _detection_env().write(runs / "b.envelope.json")
    matrix = build_matrix(collect_envelopes(runs))
    # 不对异构指标生成统一综合分数
    forbidden = {"score", "overall", "combined", "total"}
    assert not (forbidden & {c.lower() for c in matrix["metric_columns"]})
    md = render_markdown(matrix)
    assert "N/A" in md and "MISSING" in md
