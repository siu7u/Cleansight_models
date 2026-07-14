"""时序与端到端时间线共用的 F1、IoU、Edit 和帧级指标。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Hashable, Iterable, Mapping, Sequence


Label = Hashable


@dataclass(frozen=True)
class Interval:
    """一个带类别的半开区间 `[start, end)`，单位可为帧或秒。"""

    label: Label
    start: float
    end: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.start)) or not math.isfinite(float(self.end)):
            raise ValueError("区间起止必须是有限数值")
        if self.end <= self.start:
            raise ValueError(f"区间必须满足 end > start: {self.start}, {self.end}")


@dataclass(frozen=True)
class MatchCounts:
    """一次 label-aware 一对一匹配的计数和误差样本。"""

    tp: int
    fp: int
    fn: int
    matched_ious: tuple[float, ...] = ()
    start_errors: tuple[float, ...] = ()
    end_errors: tuple[float, ...] = ()

    def as_metrics(self) -> dict:
        """以 0..1 比率返回 precision/recall/F1 与匹配质量。"""

        both_empty = self.tp == self.fp == self.fn == 0
        precision = 1.0 if both_empty else _safe_ratio(self.tp, self.tp + self.fp, 0.0)
        recall = 1.0 if both_empty else _safe_ratio(self.tp, self.tp + self.fn, 0.0)
        f1 = 1.0 if both_empty else _f1(precision, recall)
        boundary_errors = self.start_errors + self.end_errors
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": mean(self.matched_ious) if self.matched_ious else None,
            "boundary_mae": mean(boundary_errors) if boundary_errors else None,
            "start_mae": mean(self.start_errors) if self.start_errors else None,
            "end_mae": mean(self.end_errors) if self.end_errors else None,
        }


def _safe_ratio(numerator: int | float, denominator: int | float, default=None):
    return numerator / denominator if denominator else default


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def interval_iou(first: Interval, second: Interval) -> float:
    """计算两个半开区间的 IoU；类别判断由匹配器负责。"""

    intersection = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    union = max(first.end, second.end) - min(first.start, second.start)
    return intersection / union if union > 0 else 0.0


def segments_from_labels(
    labels: Iterable[Label], ignore_labels: Iterable[Label] | None = None
) -> list[Interval]:
    """把逐帧标签折叠成半开片段，正确保留单帧和最后一个片段。"""

    values = list(labels)
    if not values:
        return []
    ignored = set(ignore_labels or ())
    segments: list[Interval] = []
    start = 0
    current = values[0]
    for index in range(1, len(values) + 1):
        changed = index == len(values) or values[index] != current
        if not changed:
            continue
        if current not in ignored:
            segments.append(Interval(current, float(start), float(index)))
        if index < len(values):
            start = index
            current = values[index]
    return segments


def match_intervals(
    predictions: Sequence[Interval], truths: Sequence[Interval], iou_threshold: float
) -> MatchCounts:
    """按类别和最大 IoU 做一对一匹配；重复预测只能产生一个 TP。"""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold 必须位于 0..1")
    used_truths: set[int] = set()
    matched_ious: list[float] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    tp = 0
    for prediction in predictions:
        candidates = [
            (interval_iou(prediction, truth), index, truth)
            for index, truth in enumerate(truths)
            if index not in used_truths and prediction.label == truth.label
        ]
        if not candidates:
            continue
        best_iou, best_index, best_truth = max(candidates, key=lambda item: item[0])
        if best_iou < iou_threshold:
            continue
        used_truths.add(best_index)
        tp += 1
        matched_ious.append(best_iou)
        start_errors.append(abs(prediction.start - best_truth.start))
        end_errors.append(abs(prediction.end - best_truth.end))
    return MatchCounts(
        tp=tp,
        fp=len(predictions) - tp,
        fn=len(truths) - tp,
        matched_ious=tuple(matched_ious),
        start_errors=tuple(start_errors),
        end_errors=tuple(end_errors),
    )


def classification_metrics(
    predictions: Sequence[Label], truths: Sequence[Label], labels: Sequence[Label]
) -> dict:
    """返回混淆矩阵、逐类 P/R/F1/IoU 与 macro/micro 指标。"""

    pred_values = list(predictions)
    truth_values = list(truths)
    label_values = list(labels)
    if len(pred_values) != len(truth_values):
        raise ValueError("predictions 与 truths 长度不同")
    if len(set(label_values)) != len(label_values):
        raise ValueError("labels 不得重复")
    label_to_index = {label: index for index, label in enumerate(label_values)}
    confusion = [[0 for _ in label_values] for _ in label_values]
    for prediction, truth in zip(pred_values, truth_values):
        if truth not in label_to_index:
            raise ValueError(f"truth 出现未知标签: {truth}")
        if prediction not in label_to_index:
            raise ValueError(f"prediction 出现未知标签: {prediction}")
        confusion[label_to_index[truth]][label_to_index[prediction]] += 1

    per_class: dict[str, dict] = {}
    defined_f1: list[float] = []
    defined_iou: list[float] = []
    total_tp = total_fp = total_fn = 0
    for index, label in enumerate(label_values):
        tp = confusion[index][index]
        fp = sum(row[index] for row in confusion) - tp
        fn = sum(confusion[index]) - tp
        support = sum(confusion[index])
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = None if precision is None or recall is None else _f1(precision, recall)
        iou = _safe_ratio(tp, tp + fp + fn)
        if f1 is not None:
            defined_f1.append(f1)
        if iou is not None:
            defined_iou.append(iou)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_class[str(label)] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }

    correct = sum(confusion[index][index] for index in range(len(label_values)))
    micro_precision = _safe_ratio(total_tp, total_tp + total_fp, 0.0)
    micro_recall = _safe_ratio(total_tp, total_tp + total_fn, 0.0)
    return {
        "num_frames": len(truth_values),
        "accuracy": _safe_ratio(correct, len(truth_values)),
        "macro_f1": mean(defined_f1) if defined_f1 else None,
        "macro_iou": mean(defined_iou) if defined_iou else None,
        "micro_f1": _f1(micro_precision, micro_recall) if truth_values else None,
        "per_class": per_class,
        "confusion_matrix_rows_truth_cols_prediction": confusion,
    }


def edit_score(predictions: Sequence[Label], truths: Sequence[Label]) -> float:
    """对折叠后的标签序列计算归一化 Levenshtein 相似度 0..1。"""

    pred_labels = [segment.label for segment in segments_from_labels(predictions)]
    truth_labels = [segment.label for segment in segments_from_labels(truths)]
    rows, columns = len(pred_labels), len(truth_labels)
    if rows == columns == 0:
        return 1.0
    distance = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        distance[row][0] = row
    for column in range(columns + 1):
        distance[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            substitution = 0 if pred_labels[row - 1] == truth_labels[column - 1] else 1
            distance[row][column] = min(
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
                distance[row - 1][column - 1] + substitution,
            )
    return 1.0 - distance[-1][-1] / max(rows, columns)


def _combine_matches(matches: Sequence[MatchCounts]) -> MatchCounts:
    return MatchCounts(
        tp=sum(item.tp for item in matches),
        fp=sum(item.fp for item in matches),
        fn=sum(item.fn for item in matches),
        matched_ious=tuple(value for item in matches for value in item.matched_ious),
        start_errors=tuple(value for item in matches for value in item.start_errors),
        end_errors=tuple(value for item in matches for value in item.end_errors),
    )


def temporal_metrics(
    pred_by_item: Mapping[str, Sequence[Label]],
    truth_by_item: Mapping[str, Sequence[Label]],
    labels: Sequence[Label],
    start_frame: int = 0,
    thresholds: Sequence[float] = (0.1, 0.25, 0.5),
    ignore_index: Label = -1,
) -> dict:
    """在同一裁剪范围按视频分别分段，再汇总帧级与片段级指标。"""

    if start_frame < 0:
        raise ValueError("start_frame 不得为负数")
    if set(pred_by_item) != set(truth_by_item):
        missing_pred = sorted(set(truth_by_item) - set(pred_by_item))
        missing_truth = sorted(set(pred_by_item) - set(truth_by_item))
        raise ValueError(f"预测/真值 item 不一致: missing_pred={missing_pred}, missing_truth={missing_truth}")

    all_predictions: list[Label] = []
    all_truths: list[Label] = []
    edits: list[float] = []
    matches_by_threshold: dict[float, list[MatchCounts]] = {float(value): [] for value in thresholds}

    for item_id in sorted(pred_by_item):
        predictions = list(pred_by_item[item_id])
        truths = list(truth_by_item[item_id])
        if len(predictions) != len(truths):
            raise ValueError(f"{item_id}: 预测/真值长度不同 {len(predictions)} != {len(truths)}")
        paired = [
            (prediction, truth)
            for prediction, truth in zip(predictions[start_frame:], truths[start_frame:])
            if prediction != ignore_index and truth != ignore_index
        ]
        item_predictions = [pair[0] for pair in paired]
        item_truths = [pair[1] for pair in paired]
        all_predictions.extend(item_predictions)
        all_truths.extend(item_truths)
        edits.append(edit_score(item_predictions, item_truths))
        pred_segments = segments_from_labels(item_predictions)
        truth_segments = segments_from_labels(item_truths)
        for threshold in matches_by_threshold:
            matches_by_threshold[threshold].append(
                match_intervals(pred_segments, truth_segments, threshold)
            )

    details = {
        _threshold_key(threshold): _combine_matches(items).as_metrics()
        for threshold, items in matches_by_threshold.items()
    }
    return {
        "metric_spec": {
            "ratio_range": [0.0, 1.0],
            "interval": "[start, end)",
            "matching": "label-aware one-to-one maximum IoU",
            "video_boundaries_preserved": True,
            "start_frame": start_frame,
        },
        "frame": classification_metrics(all_predictions, all_truths, labels),
        "segment": {
            "num_items": len(pred_by_item),
            "edit": mean(edits) if edits else None,
            "f1_at_iou": {key: value["f1"] for key, value in details.items()},
            "details_at_iou": details,
        },
    }


def _actions_to_intervals(actions: Sequence[Mapping]) -> list[Interval]:
    intervals = []
    for action in actions:
        label = action.get("name")
        if label is None:
            raise ValueError("时间线动作缺少 name")
        intervals.append(
            Interval(label=label, start=float(action["start_sec"]), end=float(action["end_sec"]))
        )
    return intervals


def timeline_metrics(
    predictions: Sequence[Mapping],
    truths: Sequence[Mapping],
    thresholds: Sequence[float] = (0.1, 0.25, 0.5),
) -> dict:
    """用与时序片段相同的一对一 IoU 匹配评估端到端动作时间线。"""

    pred_intervals = _actions_to_intervals(predictions)
    truth_intervals = _actions_to_intervals(truths)
    details = {
        _threshold_key(threshold): match_intervals(
            pred_intervals, truth_intervals, float(threshold)
        ).as_metrics()
        for threshold in thresholds
    }
    labels = sorted({str(item.label) for item in pred_intervals + truth_intervals})
    per_class = {}
    for label in labels:
        pred_for_label = [item for item in pred_intervals if str(item.label) == label]
        truth_for_label = [item for item in truth_intervals if str(item.label) == label]
        per_class[label] = {
            key: match_intervals(pred_for_label, truth_for_label, float(key)).as_metrics()
            for key in details
        }
    return {
        "metric_spec": {
            "ratio_range": [0.0, 1.0],
            "interval": "[start_sec, end_sec)",
            "matching": "label-aware one-to-one maximum IoU",
            "boundary_error_unit": "seconds",
        },
        "num_prediction_segments": len(pred_intervals),
        "num_truth_segments": len(truth_intervals),
        "f1_at_iou": {key: value["f1"] for key, value in details.items()},
        "details_at_iou": details,
        "per_class": per_class,
    }
