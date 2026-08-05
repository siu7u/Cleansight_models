"""ROI 分类预测评估器：把多标签 P/R/F1 原生指标翻译成三态指标。"""

from __future__ import annotations

from typing import Any, Mapping

from benchmark.core.result import EvaluationResult, MetricValue

SPEC_PRECISION = "precision/multi-label-micro/v1; source=framework.cleansight_eval.classification"
SPEC_RECALL = "recall/multi-label-micro/v1; source=framework.cleansight_eval.classification"
SPEC_F1 = "f1/multi-label-micro/v1; source=framework.cleansight_eval.classification"
SPEC_EXACT_MATCH = "exact-match/multi-label/v1; source=framework.cleansight_eval.classification"
SPEC_MODEL_FORWARD = "latency/model-forward/not-measured/v2; excludes=production"


def _value(output: Any, name: str, default=None):
    """同时支持 dataclass 属性和普通 mapping。"""

    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


def evaluate(output: Any, options: Mapping[str, Any] | None = None) -> EvaluationResult:
    """消费 roi_classification PredictionOutput，生成三态指标结果。"""

    options = dict(options or {})
    native = dict(_value(output, "native_metrics", {}) or {})
    micro = dict(native.get("micro", {}) or {})
    labels = dict(native.get("labels", {}) or {})
    per_class = dict(native.get("per_class", {}) or {})

    metrics: dict[str, MetricValue] = {
        "precision": MetricValue.computed(
            round(float(micro.get("precision", 0.0)), 4), spec=SPEC_PRECISION
        ),
        "recall": MetricValue.computed(
            round(float(micro.get("recall", 0.0)), 4), spec=SPEC_RECALL
        ),
        "f1": MetricValue.computed(
            round(float(micro.get("f1", 0.0)), 4), spec=SPEC_F1
        ),
        "exact_match": MetricValue.computed(
            round(float(native.get("exact_match", 0.0)), 4), spec=SPEC_EXACT_MATCH
        ),
    }

    metric_details: dict[str, Any] = {"per_class": {}}
    for cid, name in sorted(labels.items(), key=lambda item: int(item[0])):
        if str(name) in per_class:
            item = per_class[str(name)]
            metric_details["per_class"][str(name)] = {
                "precision": round(float(item.get("precision", 0)), 4),
                "recall": round(float(item.get("recall", 0)), 4),
                "f1": round(float(item.get("f1", 0)), 4),
                "support": int(item.get("support", 0)),
            }
        else:
            metric_details["per_class"][str(name)] = {
                "precision": {"state": "missing"},
                "recall": {"state": "missing"},
            }

    result = EvaluationResult(
        model_type=str(_value(output, "model_type")),
        model_id=str(_value(output, "model_id")),
        pipeline=str(_value(output, "pipeline")),
        checkpoint=str(_value(output, "checkpoint")),
        dataset=str(_value(output, "dataset")),
        feature_schema=dict(_value(output, "feature_schema", {}) or {}),
        metrics=metrics,
        metric_details=metric_details,
        performance={
            "model_forward_mean_ms": MetricValue.not_applicable(
                "ROI 分类离线评估未测前向耗时；生产延迟由后端测量", spec=SPEC_MODEL_FORWARD
            ),
        },
        inference_semantics=dict(_value(output, "inference_semantics", {}) or {}),
        num_params=_value(output, "num_params"),
    )
    return result
