"""评估预测 artifact 的公共 schema 与复算工具。"""

from __future__ import annotations

from typing import Mapping, Sequence

from benchmark.core.metrics import Label, temporal_metrics


TEMPORAL_PREDICTION_SCHEMA_VERSION = 1


def _labels_to_names(values: Sequence[int], index_to_action: Mapping[int, str]) -> list[str]:
    """把类别编号转换为可读标签名；未知编号保留为字符串。"""

    return [index_to_action.get(int(value), str(int(value))) for value in values]


def build_temporal_prediction_artifact(
    *,
    pred_by_item: Mapping[str, Sequence[int]],
    truth_by_item: Mapping[str, Sequence[int]],
    index_to_action: Mapping[int, str],
    window: int,
    inference_mode: str,
) -> dict:
    """构造可复算指标的逐视频预测 artifact，显式保留视频边界。"""

    if set(pred_by_item) != set(truth_by_item):
        raise ValueError("pred_by_item 与 truth_by_item 的 item 必须一致")
    items = {}
    for name in sorted(pred_by_item):
        predictions = [int(value) for value in pred_by_item[name]]
        truths = [int(value) for value in truth_by_item[name]]
        if len(predictions) != len(truths):
            raise ValueError(f"{name}: 预测/真值长度不同")
        items[name] = {
            "prediction_start_frame": window - 1,
            "num_predictions": len(predictions),
            "predicted_label_ids": predictions,
            "truth_label_ids": truths,
            "predicted_labels": _labels_to_names(predictions, index_to_action),
            "truth_labels": _labels_to_names(truths, index_to_action),
        }

    return {
        "schema_version": TEMPORAL_PREDICTION_SCHEMA_VERSION,
        "task_type": "temporal",
        "prediction_format": "frame_labels",
        "inference": {
            "mode": inference_mode,
            "window": window,
            "alignment": "prediction[t] corresponds to source frame t + window - 1",
        },
        "labels": [
            {"id": int(index), "name": index_to_action[index]}
            for index in sorted(index_to_action)
        ],
        "items": items,
    }


def temporal_metrics_from_prediction_artifact(
    artifact: Mapping,
    *,
    thresholds: Sequence[float] = (0.1, 0.25, 0.5),
    ignore_index: Label = -1,
) -> dict:
    """从逐视频预测 artifact 复算时序指标，验证 envelope 汇总可追溯。"""

    if artifact.get("schema_version") != TEMPORAL_PREDICTION_SCHEMA_VERSION:
        raise ValueError(f"不支持 temporal prediction artifact schema_version={artifact.get('schema_version')!r}")
    if artifact.get("task_type") != "temporal":
        raise ValueError("prediction artifact task_type 必须是 temporal")
    labels = [int(item["id"]) for item in artifact.get("labels", [])]
    items = artifact.get("items")
    if not isinstance(items, Mapping) or not items:
        raise ValueError("prediction artifact 缺少 items")

    pred_by_item: dict[str, list[int]] = {}
    truth_by_item: dict[str, list[int]] = {}
    for name, payload in items.items():
        predictions = [int(value) for value in payload.get("predicted_label_ids", [])]
        truths = [int(value) for value in payload.get("truth_label_ids", [])]
        if len(predictions) != len(truths):
            raise ValueError(f"{name}: 预测/真值长度不同")
        pred_by_item[str(name)] = predictions
        truth_by_item[str(name)] = truths

    return temporal_metrics(
        pred_by_item,
        truth_by_item,
        labels=labels,
        thresholds=thresholds,
        ignore_index=ignore_index,
    )
