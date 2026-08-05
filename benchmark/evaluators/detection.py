"""检测预测评估器：翻译引擎原生指标并构造逐图预测产物。"""

from __future__ import annotations

from typing import Any, Mapping

from benchmark.core.artifacts import build_detection_prediction_artifact
from benchmark.core.result import EvaluationResult, MetricValue


SPEC_MAP50 = "map/coco-0.5/v1"
SPEC_MAP50_95 = "map/coco-0.5:0.95/v1"
SPEC_PRECISION = "precision/detection-iou0.5/v1"
SPEC_RECALL = "recall/detection-iou0.5/v1"
SPEC_MODEL_FORWARD = "latency/model-forward/not-measured/v2; excludes=production"
_NO_SAMPLE = "验证集无该类样本，无法评估"


def _value(output: Any, name: str, default=None):
    """同时支持 dataclass 属性和普通 mapping。"""

    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


def build_detection_metrics(val: Mapping[str, Any]) -> dict[str, MetricValue]:
    """把检测引擎的普通 dict 翻译成三态指标，不执行 PASS/FAIL。"""

    metrics: dict[str, MetricValue] = {
        "mAP@0.5": MetricValue.computed(round(float(val["map50"]), 4), spec=SPEC_MAP50),
        "mAP@0.5:0.95": MetricValue.computed(round(float(val["map50_95"]), 4), spec=SPEC_MAP50_95),
        "precision": MetricValue.computed(round(float(val["precision"]), 4), spec=SPEC_PRECISION),
        "recall": MetricValue.computed(round(float(val["recall"]), 4), spec=SPEC_RECALL),
    }
    per_class = val.get("per_class", {})
    for cid, name in sorted(val.get("names", {}).items(), key=lambda item: int(item[0])):
        del cid
        if name in per_class:
            item = per_class[name]
            metrics[f"precision:{name}"] = MetricValue.computed(
                round(float(item["precision"]), 4), spec=SPEC_PRECISION
            )
            metrics[f"recall:{name}"] = MetricValue.computed(
                round(float(item["recall"]), 4), spec=SPEC_RECALL
            )
        else:
            metrics[f"precision:{name}"] = MetricValue.missing(_NO_SAMPLE, spec=SPEC_PRECISION)
            metrics[f"recall:{name}"] = MetricValue.missing(_NO_SAMPLE, spec=SPEC_RECALL)
    return metrics


def _not_measured_performance() -> dict[str, MetricValue]:
    """检测未测前向耗时时只保留一个公共均值槽位，避免重复三份 N/A。"""

    reason = "检测离线评估未测模型前向耗时；生产延迟由后端测量"
    return {
        "model_forward_mean_ms": MetricValue.not_applicable(reason, spec=SPEC_MODEL_FORWARD),
    }


def _split_metric_details(metrics: dict[str, MetricValue]) -> tuple[dict[str, MetricValue], dict]:
    """把逐类 P/R 移出主指标，并用共享 spec 避免为每个类别重复口径文本。"""

    summary: dict[str, MetricValue] = {}
    per_class: dict[str, dict[str, Any]] = {}
    for name, metric in metrics.items():
        prefix = next(
            (candidate for candidate in ("precision:", "recall:") if name.startswith(candidate)),
            None,
        )
        if prefix is None:
            summary[name] = metric
            continue
        metric_name = prefix[:-1]
        class_name = name[len(prefix):]
        if metric.state.value == "computed":
            detail: Any = metric.value
        else:
            detail = {"state": metric.state.value}
            if metric.reason:
                detail["reason"] = metric.reason
        per_class.setdefault(class_name, {})[metric_name] = detail
    return summary, {
        "per_class": per_class,
        "per_class_specs": {"precision": SPEC_PRECISION, "recall": SPEC_RECALL},
    }


def evaluate(output: Any, options: Mapping[str, Any] | None = None) -> EvaluationResult:
    """消费检测 PredictionOutput，生成指标、有效参数和逐图预测产物。"""

    options = dict(options or {})
    metadata = dict(_value(output, "metadata", {}) or {})
    metrics, metric_details = _split_metric_details(
        build_detection_metrics(_value(output, "native_metrics", {}) or {})
    )
    metric_details["effective_parameters"] = metadata.get("effective_parameters", {})
    result = EvaluationResult(
        model_type=str(_value(output, "model_type")),
        model_id=str(_value(output, "model_id")),
        pipeline=str(_value(output, "pipeline")),
        checkpoint=str(_value(output, "checkpoint")),
        dataset=str(_value(output, "dataset")),
        feature_schema=dict(_value(output, "feature_schema", {}) or {}),
        metrics=metrics,
        metric_details=metric_details,
        performance=_not_measured_performance(),
        inference_semantics=dict(_value(output, "inference_semantics", {}) or {}),
        num_params=_value(output, "num_params"),
    )
    predictions = _value(output, "predictions", {}) or {}
    if predictions and options.get("save_predictions", True):
        result.pending_artifacts["predictions"] = build_detection_prediction_artifact(
            items=predictions,
            labels=_value(output, "labels", {}) or {},
            split=str(metadata.get("split") or "unknown"),
            prediction_format=str(metadata.get("prediction_format") or "class_confidence_xywhn"),
        )
    else:
        errors = list(_value(output, "errors", []) or [])
        if errors:
            result.artifacts["predictions"] = {"state": "missing", "reason": errors[0]}
    return result
