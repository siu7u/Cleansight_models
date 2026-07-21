"""时序预测评估器：消费逐视频事实，计算指标并构造可复算产物。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from benchmark.core.artifacts import build_temporal_prediction_artifact
from benchmark.core.metrics import DEFAULT_INTERVAL_IOU_THRESHOLDS, temporal_metrics
from benchmark.core.result import EvaluationResult, MetricValue


# v4 将段级匹配统一为全局 IoU 贪心一对一；帧级和 Edit 未变，继续沿用 v3。
SPEC_ACC = "accuracy/frame-wise-micro-across-items/percent/v3; source=benchmark.core.metrics"
SPEC_EDIT = "edit/levenshtein-item-macro-mean/percent/v3; source=benchmark.core.metrics"
SPEC_F1 = "segmental_f1/counts-micro-across-items-label-aware-one-to-one-global-greedy-iou/percent/v4; source=benchmark.core.metrics"
SPEC_PRECISION = "segmental_precision/counts-micro-across-items-label-aware-one-to-one-global-greedy-iou/percent/v4; source=benchmark.core.metrics"
SPEC_RECALL = "segmental_recall/counts-micro-across-items-label-aware-one-to-one-global-greedy-iou/percent/v4; source=benchmark.core.metrics"
SPEC_COUNTS = "segmental_counts/micro-across-items-label-aware-one-to-one-global-greedy-iou/v4; source=benchmark.core.metrics"
SPEC_TEMPORAL_IOU = "temporal_iou/matched-segment-global-greedy-micro-pool-mean/percent/v4; source=benchmark.core.metrics"
SPEC_FRAME_CLASS = "classification/frame-micro-pool-per-class/percent/v3; source=benchmark.core.metrics"
SPEC_MODEL_FORWARD = "latency/model-forward-single-window/ms/v2; excludes=data,postprocess,io,production"


def _value(output: Any, name: str, default=None):
    """同时支持 dataclass 属性和普通 mapping，避免 benchmark 反向依赖 framework。"""

    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


def _percent_metric(value, spec: str, reason: str = "指标没有可计算样本") -> MetricValue:
    """把 benchmark 的 0..1 比率转换为对外兼容的 0..100 指标。"""

    if value is None:
        return MetricValue.missing(reason, spec=spec)
    return MetricValue.computed(round(float(value) * 100.0, 2), spec=spec)


def compute_temporal_metrics_by_item(
    pred_by_item: Mapping[str, Sequence[str]],
    truth_by_item: Mapping[str, Sequence[str]],
    labels: Sequence[str],
    *,
    start_frame: int = 0,
    return_details: bool = False,
):
    """按视频边界计算指标，并显式声明各指标的跨视频聚合方式。"""

    try:
        raw = temporal_metrics(
            pred_by_item,
            truth_by_item,
            labels=list(labels),
            start_frame=start_frame,
            thresholds=DEFAULT_INTERVAL_IOU_THRESHOLDS,
            ignore_index=-1,
        )
    except ValueError as exc:
        reason = f"benchmark metrics 输入无效: {exc}"
        missing = {
            "acc": MetricValue.missing(reason, spec=SPEC_ACC),
            "edit": MetricValue.missing(reason, spec=SPEC_EDIT),
            **{
                f"f1@{threshold}": MetricValue.missing(reason, spec=SPEC_F1)
                for threshold in DEFAULT_INTERVAL_IOU_THRESHOLDS
            },
        }
        return (missing, {"error": reason}) if return_details else missing

    frame = raw["frame"]
    segment = raw["segment"]
    metrics: dict[str, MetricValue] = {
        "acc": _percent_metric(frame.get("accuracy"), SPEC_ACC),
        "edit": _percent_metric(segment.get("edit"), SPEC_EDIT),
        "frame.macro_f1": _percent_metric(frame.get("macro_f1"), SPEC_FRAME_CLASS),
        "frame.macro_iou": _percent_metric(frame.get("macro_iou"), SPEC_FRAME_CLASS),
        "frame.micro_f1": _percent_metric(frame.get("micro_f1"), SPEC_FRAME_CLASS),
    }
    for threshold in DEFAULT_INTERVAL_IOU_THRESHOLDS:
        key = f"{threshold:.2f}"
        detail = segment["details_at_iou"][key]
        suffix = str(threshold)
        metrics[f"f1@{suffix}"] = _percent_metric(detail.get("f1"), SPEC_F1)
        if threshold == 0.5:
            metrics[f"tp@{suffix}"] = MetricValue.computed(int(detail["tp"]), spec=SPEC_COUNTS)
            metrics[f"fp@{suffix}"] = MetricValue.computed(int(detail["fp"]), spec=SPEC_COUNTS)
            metrics[f"fn@{suffix}"] = MetricValue.computed(int(detail["fn"]), spec=SPEC_COUNTS)
            metrics[f"precision@{suffix}"] = _percent_metric(detail.get("precision"), SPEC_PRECISION)
            metrics[f"recall@{suffix}"] = _percent_metric(detail.get("recall"), SPEC_RECALL)
            metrics[f"temporal_iou@{suffix}"] = _percent_metric(
                detail.get("mean_matched_iou"),
                SPEC_TEMPORAL_IOU,
                reason="该 IoU 阈值下没有匹配片段",
            )
    return (metrics, raw) if return_details else metrics


def compute_temporal_metrics(pred_labels: list[str], gt_labels: list[str]) -> dict[str, MetricValue]:
    """单序列兼容入口；正式评估应传逐 item 映射。"""

    labels = sorted(set(pred_labels) | set(gt_labels))
    return compute_temporal_metrics_by_item(
        {"item-0": pred_labels},
        {"item-0": gt_labels},
        labels,
    )


def summarize_model_forward_timing(timing: Mapping[str, Any]) -> dict[str, MetricValue]:
    """汇总模型单窗前向样本；名称明确排除数据、后处理、I/O 和生产延迟。"""

    samples = sorted(float(value) for value in timing.get("samples_ms", []))
    if not samples:
        reason = "未采集到模型前向耗时样本"
        return {
            "model_forward_mean_ms": MetricValue.missing(reason, spec=SPEC_MODEL_FORWARD),
            "model_forward_median_ms": MetricValue.missing(reason, spec=SPEC_MODEL_FORWARD),
            "model_forward_p95_ms": MetricValue.missing(reason, spec=SPEC_MODEL_FORWARD),
        }
    mean_ms = sum(samples) / len(samples)
    median_ms = samples[len(samples) // 2]
    p95_ms = samples[min(len(samples) - 1, int(0.95 * len(samples)))]
    context = timing.get("context") or {}
    spec = (
        f"{SPEC_MODEL_FORWARD}; device={timing.get('device')}; window={context.get('window')}; "
        f"warmup={timing.get('warmup')}; runs={timing.get('runs')}"
    )
    return {
        "model_forward_mean_ms": MetricValue.computed(round(mean_ms, 4), spec=spec),
        "model_forward_median_ms": MetricValue.computed(round(median_ms, 4), spec=spec),
        "model_forward_p95_ms": MetricValue.computed(round(p95_ms, 4), spec=spec),
    }


def not_applicable_model_forward(reason: str) -> dict[str, MetricValue]:
    """声明当前推理模式未测模型前向耗时，不用 0 冒充测量结果。"""

    return {
        "model_forward_mean_ms": MetricValue.not_applicable(reason, spec=SPEC_MODEL_FORWARD),
        "model_forward_median_ms": MetricValue.not_applicable(reason, spec=SPEC_MODEL_FORWARD),
        "model_forward_p95_ms": MetricValue.not_applicable(reason, spec=SPEC_MODEL_FORWARD),
    }


def _prediction_artifact(output: Any) -> dict:
    """把标签名序列编码为 benchmark 唯一时序 prediction artifact。"""

    labels = list(_value(output, "labels", []))
    name_to_id = {name: index for index, name in enumerate(labels)}
    predictions = _value(output, "predictions", {})
    targets = _value(output, "targets", {})
    pred_ids = {name: [name_to_id[value] for value in values] for name, values in predictions.items()}
    truth_ids = {name: [name_to_id[value] for value in values] for name, values in targets.items()}
    metadata = dict(_value(output, "metadata", {}) or {})
    semantics = dict(_value(output, "inference_semantics", {}) or {})
    return build_temporal_prediction_artifact(
        pred_by_item=pred_ids,
        truth_by_item=truth_ids,
        index_to_action={index: name for name, index in name_to_id.items()},
        window=metadata.get("window"),
        inference_mode=str(semantics.get("mode") or "unknown"),
        prediction_start_frame=0,
    )


def evaluate(output: Any, options: Mapping[str, Any] | None = None) -> EvaluationResult:
    """消费时序 PredictionOutput，生成不含业务判决的 EvaluationResult。"""

    del options
    predictions = _value(output, "predictions", {})
    targets = _value(output, "targets", {})
    labels = _value(output, "labels", [])
    metrics, details = compute_temporal_metrics_by_item(
        predictions,
        targets,
        labels,
        return_details=True,
    )
    pipeline = str(_value(output, "pipeline"))
    timing = _value(output, "timing", {}) or {}
    performance = (
        summarize_model_forward_timing(timing)
        if pipeline == "sliding_window_temporal"
        else not_applicable_model_forward("离线全序列不测模型单窗前向耗时")
    )
    result = EvaluationResult(
        model_type=str(_value(output, "model_type")),
        model_id=str(_value(output, "model_id")),
        pipeline=pipeline,
        checkpoint=str(_value(output, "checkpoint")),
        dataset=str(_value(output, "dataset")),
        feature_schema=dict(_value(output, "feature_schema", {}) or {}),
        metrics=metrics,
        metric_details={"temporal": details},
        performance=performance,
        inference_semantics=dict(_value(output, "inference_semantics", {}) or {}),
        num_params=_value(output, "num_params"),
    )
    result.pending_artifacts["predictions"] = _prediction_artifact(output)
    return result
