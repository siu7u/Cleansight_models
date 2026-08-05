"""时序与检测预测 artifact 的唯一 schema、校验和复算工具。"""

from __future__ import annotations

from numbers import Real
from typing import Any, Mapping, Sequence

from framework.cleansight_eval.core.metrics import Label, temporal_metrics


TEMPORAL_PREDICTION_SCHEMA_VERSION = 1
DETECTION_PREDICTION_SCHEMA_VERSION = 1


def _labels_to_names(values: Sequence[int], index_to_action: Mapping[int, str]) -> list[str]:
    """把类别编号转换为可读标签名；未知编号保留为字符串。"""

    return [index_to_action.get(int(value), str(int(value))) for value in values]


def build_temporal_prediction_artifact(
    *,
    pred_by_item: Mapping[str, Sequence[int]],
    truth_by_item: Mapping[str, Sequence[int]],
    index_to_action: Mapping[int, str],
    window: int | None,
    inference_mode: str,
    prediction_start_frame: int | None = None,
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
            "prediction_start_frame": (
                prediction_start_frame
                if prediction_start_frame is not None
                else (window - 1 if window is not None else 0)
            ),
            "num_predictions": len(predictions),
            "predicted_label_ids": predictions,
            "truth_label_ids": truths,
            "predicted_labels": _labels_to_names(predictions, index_to_action),
            "truth_labels": _labels_to_names(truths, index_to_action),
        }

    artifact = {
        "schema_version": TEMPORAL_PREDICTION_SCHEMA_VERSION,
        "task_type": "temporal",
        "prediction_format": "frame_labels",
        "inference": {
            "mode": inference_mode,
            "window": window,
            "alignment": "predicted_label_ids 与 truth_label_ids 在每个 item 内逐项对齐",
        },
        "labels": [
            {"id": int(index), "name": index_to_action[index]}
            for index in sorted(index_to_action)
        ],
        "items": items,
    }
    validate_temporal_prediction_artifact(artifact)
    return artifact


def validate_temporal_prediction_artifact(artifact: Mapping[str, Any]) -> None:
    """校验逐视频时序 artifact 的 schema、标签和逐帧对齐。"""

    if artifact.get("schema_version") != TEMPORAL_PREDICTION_SCHEMA_VERSION:
        raise ValueError(f"不支持 temporal prediction artifact schema_version={artifact.get('schema_version')!r}")
    if artifact.get("task_type") != "temporal":
        raise ValueError("prediction artifact task_type 必须是 temporal")
    labels = [int(item["id"]) for item in artifact.get("labels", [])]
    items = artifact.get("items")
    if not isinstance(items, Mapping) or not items:
        raise ValueError("prediction artifact 缺少 items")

    known_labels = set(labels)
    for name, payload in items.items():
        predictions = [int(value) for value in payload.get("predicted_label_ids", [])]
        truths = [int(value) for value in payload.get("truth_label_ids", [])]
        if len(predictions) != len(truths):
            raise ValueError(f"{name}: 预测/真值长度不同")
        if payload.get("num_predictions") != len(predictions):
            raise ValueError(f"{name}: num_predictions 与预测长度不一致")
        unknown = (set(predictions) | set(truths)) - known_labels
        if unknown:
            raise ValueError(f"{name}: 出现未登记类别 id: {sorted(unknown)}")


def build_detection_prediction_artifact(
    *,
    items: Mapping[str, Mapping[str, Any]],
    labels: Mapping[int | str, str],
    split: str,
    prediction_format: str = "class_confidence_xywhn",
) -> dict[str, Any]:
    """构造逐图检测 artifact；每个框为 class_id、confidence 和归一化 xywh。"""

    artifact = {
        "schema_version": DETECTION_PREDICTION_SCHEMA_VERSION,
        "task_type": "detection",
        "prediction_format": prediction_format,
        "split": split,
        "labels": {str(key): str(value) for key, value in labels.items()},
        "items": {str(name): dict(payload) for name, payload in sorted(items.items())},
    }
    validate_detection_prediction_artifact(artifact)
    return artifact


def validate_detection_prediction_artifact(artifact: Mapping[str, Any]) -> None:
    """校验逐图检测 artifact；真值由固定 testset 提供，因此这里只验证预测结构。"""

    if artifact.get("schema_version") != DETECTION_PREDICTION_SCHEMA_VERSION:
        raise ValueError(f"不支持 detection prediction artifact schema_version={artifact.get('schema_version')!r}")
    if artifact.get("task_type") != "detection":
        raise ValueError("prediction artifact task_type 必须是 detection")
    if artifact.get("prediction_format") != "class_confidence_xywhn":
        raise ValueError(f"不支持 detection prediction_format={artifact.get('prediction_format')!r}")
    if not artifact.get("split"):
        raise ValueError("detection prediction artifact 缺少 split")
    labels = artifact.get("labels")
    if not isinstance(labels, Mapping) or not labels:
        raise ValueError("detection prediction artifact 缺少 labels")
    known_labels = {str(key) for key in labels}
    items = artifact.get("items")
    if not isinstance(items, Mapping) or not items:
        raise ValueError("detection prediction artifact 缺少 items")

    for name, payload in items.items():
        predictions = payload.get("predictions") if isinstance(payload, Mapping) else None
        if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
            raise ValueError(f"{name}: predictions 必须是列表")
        for index, box in enumerate(predictions):
            if not isinstance(box, Mapping):
                raise ValueError(f"{name}[{index}]: box 必须是映射")
            class_id = str(box.get("class_id"))
            if class_id not in known_labels:
                raise ValueError(f"{name}[{index}]: 未登记类别 id={class_id}")
            confidence = box.get("confidence")
            if not isinstance(confidence, Real) or not 0.0 <= float(confidence) <= 1.0:
                raise ValueError(f"{name}[{index}]: confidence 必须在 0..1")
            xywhn = box.get("xywhn")
            if not isinstance(xywhn, Sequence) or isinstance(xywhn, (str, bytes)) or len(xywhn) != 4:
                raise ValueError(f"{name}[{index}]: xywhn 必须含 4 个数")
            if any(not isinstance(value, Real) for value in xywhn):
                raise ValueError(f"{name}[{index}]: xywhn 必须为数值")


def validate_prediction_artifact(artifact: Mapping[str, Any]) -> None:
    """按 task_type 分派到唯一的 artifact schema 校验器。"""

    task_type = artifact.get("task_type")
    if task_type == "temporal":
        validate_temporal_prediction_artifact(artifact)
    elif task_type == "detection":
        validate_detection_prediction_artifact(artifact)
    else:
        raise ValueError(f"未知 prediction artifact task_type={task_type!r}")


def temporal_metrics_from_prediction_artifact(
    artifact: Mapping,
    *,
    thresholds: Sequence[float] = (0.1, 0.25, 0.5),
    ignore_index: Label = -1,
) -> dict:
    """从逐视频预测 artifact 复算时序指标，验证结果汇总可追溯。"""

    validate_temporal_prediction_artifact(artifact)
    labels = [int(item["id"]) for item in artifact.get("labels", [])]
    items = artifact["items"]

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


def prediction_artifact_recomputable(artifact: Mapping[str, Any]) -> bool | None:
    """校验 artifact，并返回能否仅凭 artifact 复算指标；检测需外部真值，返回 None。"""

    validate_prediction_artifact(artifact)
    if artifact.get("task_type") == "temporal":
        temporal_metrics_from_prediction_artifact(artifact)
        return True
    return None
