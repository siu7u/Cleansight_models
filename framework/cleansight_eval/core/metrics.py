"""时序与端到端时间线共用的 F1、IoU、Edit 和帧级指标。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any, Hashable, Iterable, Mapping, Sequence


Label = Hashable
DEFAULT_INTERVAL_IOU_THRESHOLDS = (0.1, 0.25, 0.5)
INTERVAL_MATCHING_METHOD = "label-aware one-to-one global-greedy maximum IoU"
INTERVAL_METRIC_VERSION = "interval-matching-v2"

# ---------------------------------------------------------------------------
# 指标口径注册表：**全仓库唯一的指标定义处**
# ---------------------------------------------------------------------------
# 同一批指标会出现在四个面上，过去各处各自拼名字/口径字符串，容易出现"训练侧
# val_f1_0.5、评测侧 f1@0.5、单位一个是百分数一个是 0..1 比率"这类隐性分歧。
# 现在四个面全部引用本表：
#   1. 训练期 validation（``temporal/training_validation.py``）→ ``training_key``；
#   2. 正式评测结果（``benchmark/evaluators/temporal.py``）→ 注册表键 = summary 键 + ``spec``；
#   3. 训练 history/曲线（``core/history.py`` + 两条流水线）→ ``training_key`` 列；
#   4. 矩阵汇总与报告（``tools/*``、``benchmark/core/report.py``）→ summary 键。
#
# 字段含义：
#   spec          口径版本字符串，直接落进 evaluation.json 的 MetricValue.spec；
#   unit          该指标对外报出的单位（``percent`` = 0..100，已四舍五入到 2 位）；
#   detail_path   ``metrics.details.temporal`` 里同口径原始值的路径元组（比率 0..1 或计数）；
#   training_key  训练期 history 列名（无则说明该指标不参与选点/不写 history）。
#
# 口径变更（改了聚合方式或单位）必须递增 spec 版本号并同步更新 docs/EVAL.md。

_SOURCE = "framework.cleansight_eval.core.metrics"
FRAME_CLASS_SPEC = f"classification/frame-micro-pool-per-class/percent/v3; source={_SOURCE}"
ACC_SPEC = f"accuracy/frame-wise-micro-across-items/percent/v3; source={_SOURCE}"
EDIT_SPEC = f"edit/levenshtein-item-macro-mean/percent/v3; source={_SOURCE}"
_SEGMENT_SPEC = "counts-micro-across-items-label-aware-one-to-one-global-greedy-iou"
# 计数指标（tp/fp/fn@0.5）的历史口径字符串没有 `counts-` 前缀，与 F1/P/R 不同——这是既有
# evaluation.json 里已落盘的口径标识，注册表原样保留，不做"顺手统一"（改字符串等于改口径）。
_SEGMENT_COUNTS_SPEC = "micro-across-items-label-aware-one-to-one-global-greedy-iou"
SEGMENTAL_F1_SPEC = f"segmental_f1/{_SEGMENT_SPEC}/percent/v4; source={_SOURCE}"
SEGMENTAL_PRECISION_SPEC = f"segmental_precision/{_SEGMENT_SPEC}/percent/v4; source={_SOURCE}"
SEGMENTAL_RECALL_SPEC = f"segmental_recall/{_SEGMENT_SPEC}/percent/v4; source={_SOURCE}"
SEGMENTAL_COUNTS_SPEC = f"segmental_counts/{_SEGMENT_COUNTS_SPEC}/v4; source={_SOURCE}"
TEMPORAL_IOU_SPEC = f"temporal_iou/matched-segment-global-greedy-micro-pool-mean/percent/v4; source={_SOURCE}"

TEMPORAL_METRIC_SPECS: dict[str, dict[str, object]] = {
    "acc": {
        "spec": ACC_SPEC,
        "unit": "percent",
        "detail_path": ("frame", "accuracy"),
        "training_key": "val_acc",
    },
    "edit": {
        "spec": EDIT_SPEC,
        "unit": "percent",
        "detail_path": ("segment", "edit"),
        "training_key": "val_edit",
    },
    "frame.macro_f1": {"spec": FRAME_CLASS_SPEC, "unit": "percent", "detail_path": ("frame", "macro_f1")},
    "frame.macro_iou": {"spec": FRAME_CLASS_SPEC, "unit": "percent", "detail_path": ("frame", "macro_iou")},
    "frame.micro_f1": {"spec": FRAME_CLASS_SPEC, "unit": "percent", "detail_path": ("frame", "micro_f1")},
    "precision@0.5": {"spec": SEGMENTAL_PRECISION_SPEC, "unit": "percent", "detail_path": ("segment", "details_at_iou", "0.50", "precision")},
    "recall@0.5": {"spec": SEGMENTAL_RECALL_SPEC, "unit": "percent", "detail_path": ("segment", "details_at_iou", "0.50", "recall")},
    "temporal_iou@0.5": {"spec": TEMPORAL_IOU_SPEC, "unit": "percent", "detail_path": ("segment", "details_at_iou", "0.50", "mean_matched_iou")},
    "tp@0.5": {"spec": SEGMENTAL_COUNTS_SPEC, "unit": "count", "detail_path": ("segment", "details_at_iou", "0.50", "tp")},
    "fp@0.5": {"spec": SEGMENTAL_COUNTS_SPEC, "unit": "count", "detail_path": ("segment", "details_at_iou", "0.50", "fp")},
    "fn@0.5": {"spec": SEGMENTAL_COUNTS_SPEC, "unit": "count", "detail_path": ("segment", "details_at_iou", "0.50", "fn")},
}
# 分段 F1 逐阈值展开（summary 键 `f1@0.1/0.25/0.5`，details 键 `0.10/0.25/0.50`）。
for _threshold in DEFAULT_INTERVAL_IOU_THRESHOLDS:
    TEMPORAL_METRIC_SPECS[f"f1@{_threshold:g}"] = {
        "spec": SEGMENTAL_F1_SPEC,
        "unit": "percent",
        "detail_path": ("segment", "f1_at_iou", f"{float(_threshold):.2f}"),
        "training_key": f"val_f1_{_threshold:g}",
    }
del _threshold


def metric_spec(name: str) -> str:
    """返回指标的 spec 口径字符串；未注册的名字立即报错，避免各处自造口径。"""

    if name not in TEMPORAL_METRIC_SPECS:
        raise KeyError(f"未注册的时序指标: {name!r}；已注册: {sorted(TEMPORAL_METRIC_SPECS)}")
    return TEMPORAL_METRIC_SPECS[name]["spec"]


def training_metric_keys() -> tuple[str, ...]:
    """训练期 history 的验证指标列（同时也是 ``train.best_metric`` 的可选值）。"""

    return tuple(
        entry["training_key"]
        for entry in TEMPORAL_METRIC_SPECS.values()
        if entry.get("training_key")
    )


def training_metric_name(training_key: str) -> str:
    """训练侧列名 → 评测侧 summary 键名（如 ``val_f1_0.5`` → ``f1@0.5``）。"""

    for name, entry in TEMPORAL_METRIC_SPECS.items():
        if entry.get("training_key") == training_key:
            return name
    raise KeyError(f"未注册的训练期指标: {training_key!r}；已注册: {training_metric_keys()}")


def detail_value(raw: Mapping, detail_path: Sequence[str]):
    """按注册表的 ``detail_path`` 元组从 ``temporal_metrics`` 结果取原始值。

    路径用元组而非点分字符串：段级 IoU 阈值键本身就是 ``"0.50"`` 这种带小数点的形式，
    点分字符串会被拆成 ``"0"`` / ``"50"`` 两个键。
    """

    node: object = raw
    for part in detail_path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
        if node is None:
            return None
    return node


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
class IntervalMatch:
    """一对已匹配区间及其索引；误差为 prediction - truth，保留方向。"""

    prediction_index: int
    truth_index: int
    iou: float
    start_error: float
    end_error: float


@dataclass(frozen=True)
class MatchCounts:
    """一次 label-aware 一对一匹配的计数和误差样本。"""

    tp: int
    fp: int
    fn: int
    matched_ious: tuple[float, ...] = ()
    start_errors: tuple[float, ...] = ()
    end_errors: tuple[float, ...] = ()
    matches: tuple[IntervalMatch, ...] = ()

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
    """按类别做全局贪心最大 IoU 一对一匹配；重复预测只能产生一个 TP。"""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold 必须位于 0..1")
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            if prediction.label != truth.label:
                continue
            iou = interval_iou(prediction, truth)
            if iou >= iou_threshold:
                candidates.append((iou, prediction_index, truth_index, prediction, truth))
    # 先选择全体候选中的最大 IoU，降低预测遍历顺序影响；索引用于同 IoU 时稳定排序。
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matched_ious: list[float] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    matches: list[IntervalMatch] = []
    for iou, prediction_index, truth_index, prediction, truth in candidates:
        if prediction_index in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(prediction_index)
        used_truths.add(truth_index)
        start_error = prediction.start - truth.start
        end_error = prediction.end - truth.end
        matches.append(
            IntervalMatch(
                prediction_index=prediction_index,
                truth_index=truth_index,
                iou=iou,
                start_error=start_error,
                end_error=end_error,
            )
        )
        matched_ious.append(iou)
        start_errors.append(abs(start_error))
        end_errors.append(abs(end_error))
    tp = len(matches)
    return MatchCounts(
        tp=tp,
        fp=len(predictions) - tp,
        fn=len(truths) - tp,
        matched_ious=tuple(matched_ious),
        start_errors=tuple(start_errors),
        end_errors=tuple(end_errors),
        matches=tuple(matches),
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
    thresholds: Sequence[float] = DEFAULT_INTERVAL_IOU_THRESHOLDS,
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
            "version": INTERVAL_METRIC_VERSION,
            "ratio_range": [0.0, 1.0],
            "interval": "[start, end)",
            "matching": INTERVAL_MATCHING_METHOD,
            "video_boundaries_preserved": True,
            "start_frame": start_frame,
            "aggregation": {
                "frame": "micro over frames pooled across items",
                "edit": "macro mean over items",
                "segment_counts_precision_recall_f1": "micro from TP/FP/FN summed across items",
                "matched_temporal_iou": "mean over all matched segments pooled across items",
            },
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


def match_timeline(
    predictions: Sequence[Mapping],
    truths: Sequence[Mapping],
    iou_threshold: float,
) -> MatchCounts:
    """把秒级动作时间线转换为区间，并复用统一的一对一匹配算法。"""

    return match_intervals(
        _actions_to_intervals(predictions),
        _actions_to_intervals(truths),
        iou_threshold,
    )


def timeline_metrics(
    predictions: Sequence[Mapping],
    truths: Sequence[Mapping],
    thresholds: Sequence[float] = DEFAULT_INTERVAL_IOU_THRESHOLDS,
) -> dict:
    """用与时序片段相同的一对一 IoU 匹配评估端到端动作时间线。"""

    pred_intervals = _actions_to_intervals(predictions)
    truth_intervals = _actions_to_intervals(truths)
    details = {
        _threshold_key(threshold): match_timeline(
            predictions, truths, float(threshold)
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
            "version": INTERVAL_METRIC_VERSION,
            "ratio_range": [0.0, 1.0],
            "interval": "[start_sec, end_sec)",
            "matching": INTERVAL_MATCHING_METHOD,
            "boundary_error_unit": "seconds",
        },
        "num_prediction_segments": len(pred_intervals),
        "num_truth_segments": len(truth_intervals),
        "f1_at_iou": {key: value["f1"] for key, value in details.items()},
        "details_at_iou": details,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# 多标签分类（ROI）指标：口径注册表 + 唯一实现
# ---------------------------------------------------------------------------
# 与上面的时序注册表同构：名字、spec、单位、训练侧别名只有这一个定义处；训练期 validation
# （``classification/pipeline.py::_fit``）与正式评测（``benchmark/evaluators/classification.py``）
# 调用同一批纯函数，因此同名指标必然同口径。
#
# 本任务的指标刻意放在 core 内核而不是 ``framework.cleansight_eval.classification`` 下：
# benchmark evaluator 只允许依赖 framework core 的纯指标原语
# （见 ``tests/test_evaluator_boundaries.py``），实现与口径必须一起落在允许的模块里。
#
# 与时序注册表有意保留的差别（不做"顺手统一"）：
# - 单位是 **0..1 比率**（对外保留 4 位小数）；时序侧是 0..100 百分数（2 位）。
# - 判正阈值 0.5 属口径的一部分，登记为 :data:`DECISION_THRESHOLD`：训练与评测都经
#   :func:`decide` 生成 0/1 预测，改阈值只改一处。
# 计算函数在函数内延迟 import numpy，保持内核在无 numpy 环境也可导入。

CLASSIFICATION_SOURCE = "framework.cleansight_eval.classification"
CLASSIFICATION_PRECISION_SPEC = f"precision/multi-label-micro/v1; source={CLASSIFICATION_SOURCE}"
CLASSIFICATION_RECALL_SPEC = f"recall/multi-label-micro/v1; source={CLASSIFICATION_SOURCE}"
CLASSIFICATION_F1_SPEC = f"f1/multi-label-micro/v1; source={CLASSIFICATION_SOURCE}"
CLASSIFICATION_EXACT_MATCH_SPEC = f"exact-match/multi-label/v1; source={CLASSIFICATION_SOURCE}"

# 判正阈值（sigmoid 概率 > 该值记为正）。训练期验证与正式评测共用，改这里即改全链路。
DECISION_THRESHOLD = 0.5
# 对外报出的比率小数位（评测结果与训练 history 一致）。
CLASSIFICATION_DECIMALS = 4

# 指标注册表：键 = 评测 ``metrics.summary`` 里的键名。
#   spec         口径版本字符串，直接落进 evaluation.json 的 MetricValue.spec；
#   unit         对外单位（本任务恒为 0..1 比率）；
#   decimals     外报小数位；
#   value_path   ``multilabel_metrics`` 结果里的取值路径元组；
#   training_key 训练期 history 键（无则说明该指标不参与训练侧记录）。
CLASSIFICATION_METRIC_SPECS: dict[str, dict[str, Any]] = {
    "precision": {
        "spec": CLASSIFICATION_PRECISION_SPEC,
        "unit": "ratio",
        "decimals": CLASSIFICATION_DECIMALS,
        "value_path": ("micro", "precision"),
        "training_key": "val_precision",
    },
    "recall": {
        "spec": CLASSIFICATION_RECALL_SPEC,
        "unit": "ratio",
        "decimals": CLASSIFICATION_DECIMALS,
        "value_path": ("micro", "recall"),
        "training_key": "val_recall",
    },
    "f1": {
        "spec": CLASSIFICATION_F1_SPEC,
        "unit": "ratio",
        "decimals": CLASSIFICATION_DECIMALS,
        "value_path": ("micro", "f1"),
        "training_key": "val_f1",
    },
    "exact_match": {
        "spec": CLASSIFICATION_EXACT_MATCH_SPEC,
        "unit": "ratio",
        "decimals": CLASSIFICATION_DECIMALS,
        "value_path": ("exact_match",),
        "training_key": "val_exact_match",
    },
}


def classification_metric_spec(name: str) -> str:
    """返回分类指标的 spec 口径字符串；未注册立即报错，避免各处自造口径。"""

    if name not in CLASSIFICATION_METRIC_SPECS:
        raise KeyError(
            f"未注册的分类指标: {name!r}；已注册: {sorted(CLASSIFICATION_METRIC_SPECS)}"
        )
    return CLASSIFICATION_METRIC_SPECS[name]["spec"]


def classification_training_keys() -> tuple[str, ...]:
    """训练期 history 的分类指标键（顺序与注册表一致）。"""

    return tuple(
        entry["training_key"]
        for entry in CLASSIFICATION_METRIC_SPECS.values()
        if entry.get("training_key")
    )


def classification_training_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    """按注册表的 ``value_path`` 从 :func:`multilabel_metrics` 结果取出训练侧指标值。

    训练循环与测试都用它把"算出来的指标"映射成 history 键，避免在流水线里再抄一份
    「键 → 取值位置」对照表。
    """

    values: dict[str, float] = {}
    for entry in CLASSIFICATION_METRIC_SPECS.values():
        training_key = entry.get("training_key")
        if not training_key:
            continue
        values[training_key] = detail_value(metrics, entry["value_path"])
    return values


def decide(scores, threshold: float = DECISION_THRESHOLD):
    """概率 → 0/1 判正矩阵（唯一阈值入口）。

    输入 ``[N, C]`` 的 sigmoid 概率（numpy 数组或可转数组的序列），返回同形状 float32 0/1 数组。
    训练期验证与正式评测都走这里，因此"阈值 0.5"不会在两处各写一遍。
    """

    import numpy as np

    return (np.asarray(scores, dtype=np.float32) > float(threshold)).astype(np.float32)


def confusion_counts(binary_preds, labels) -> dict:
    """逐类与整体的 tp/fp/fn 计数，供逐 batch 累加。

    输入 ``[N, C]`` 的 0/1 预测与 0/1 真值。返回普通 dict（python int / list），
    可用 :func:`merge_counts` 跨 batch 相加，最后交给 :func:`metrics_from_counts` 出指标。
    """

    import numpy as np

    preds = np.asarray(binary_preds, dtype=np.float32) > 0.5
    truth = np.asarray(labels, dtype=np.float32) > 0.5
    if preds.shape != truth.shape:
        raise ValueError(f"预测与真值形状不一致: {preds.shape} vs {truth.shape}")
    per_class_tp, per_class_fp, per_class_fn = [], [], []
    for index in range(truth.shape[1]):
        pred_col, truth_col = preds[:, index], truth[:, index]
        per_class_tp.append(int((pred_col & truth_col).sum()))
        per_class_fp.append(int((pred_col & ~truth_col).sum()))
        per_class_fn.append(int((~pred_col & truth_col).sum()))
    return {
        "per_class_tp": per_class_tp,
        "per_class_fp": per_class_fp,
        "per_class_fn": per_class_fn,
        "tp": int((preds & truth).sum()),
        "fp": int((preds & ~truth).sum()),
        "fn": int((~preds & truth).sum()),
        "samples": int(truth.shape[0]),
        "exact_matches": int((preds == truth).all(axis=1).sum()),
    }


def merge_counts(parts: Sequence[Mapping[str, Any]]) -> dict:
    """把多个 batch 的计数相加（逐类列表按位相加）。"""

    merged: dict[str, Any] = {
        "per_class_tp": [], "per_class_fp": [], "per_class_fn": [],
        "tp": 0, "fp": 0, "fn": 0, "samples": 0, "exact_matches": 0,
    }
    for part in parts:
        for key in ("per_class_tp", "per_class_fp", "per_class_fn"):
            values = list(part.get(key, []))
            if not merged[key]:
                merged[key] = [0] * len(values)
            if len(values) != len(merged[key]):
                raise ValueError("逐类计数长度不一致，无法合并")
            merged[key] = [a + b for a, b in zip(merged[key], values)]
        for key in ("tp", "fp", "fn", "samples", "exact_matches"):
            merged[key] += int(part.get(key, 0))
    return merged


def metrics_from_counts(counts: Mapping[str, Any], class_names: Sequence[str]) -> dict:
    """计数 → micro P/R/F1 + 样本级 exact_match + 逐类 P/R/F1/support。

    非空输入下与历史实现数值完全一致：``precision = tp / max(tp + fp, 1)``、
    ``f1 = 2pr / max(p + r, 1e-8)``、``support = tp + fn``，逐项四舍五入到 :data:`CLASSIFICATION_DECIMALS` 位。
    逐类计数长度必须与 ``class_names`` 一致：一个 batch 都没累积到就调用会立即报错
    （不静默返回一组 0 指标把接线错误盖掉）；``samples == 0`` 时 ``exact_match`` 为 0.0。
    """

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / max(denominator, 1)

    def f1_of(precision: float, recall: float) -> float:
        return 2 * precision * recall / max(precision + recall, 1e-8)

    names = list(class_names)
    per_class_tp = list(counts.get("per_class_tp", []))
    per_class_fp = list(counts.get("per_class_fp", []))
    per_class_fn = list(counts.get("per_class_fn", []))
    if len(per_class_tp) != len(names):
        raise ValueError("逐类计数与类别数不一致")

    per_class = {}
    for index, name in enumerate(names):
        precision = ratio(per_class_tp[index], per_class_tp[index] + per_class_fp[index])
        recall = ratio(per_class_tp[index], per_class_tp[index] + per_class_fn[index])
        per_class[name] = {
            "precision": round(float(precision), CLASSIFICATION_DECIMALS),
            "recall": round(float(recall), CLASSIFICATION_DECIMALS),
            "f1": round(float(f1_of(precision, recall)), CLASSIFICATION_DECIMALS),
            "support": int(per_class_tp[index] + per_class_fn[index]),
        }

    micro_precision = ratio(counts.get("tp", 0), counts.get("tp", 0) + counts.get("fp", 0))
    micro_recall = ratio(counts.get("tp", 0), counts.get("tp", 0) + counts.get("fn", 0))
    samples = int(counts.get("samples", 0))
    exact_match = (counts.get("exact_matches", 0) / samples) if samples else 0.0
    return {
        "per_class": per_class,
        "micro": {
            "precision": round(float(micro_precision), CLASSIFICATION_DECIMALS),
            "recall": round(float(micro_recall), CLASSIFICATION_DECIMALS),
            "f1": round(float(f1_of(micro_precision, micro_recall)), CLASSIFICATION_DECIMALS),
        },
        "exact_match": round(float(exact_match), CLASSIFICATION_DECIMALS),
        "labels": {index: name for index, name in enumerate(names)},
    }


def multilabel_metrics(binary_preds, labels, class_names: Sequence[str]) -> dict:
    """一次算完多标签分类指标（计数 → 指标），供评测路径直接调用。"""

    return metrics_from_counts(confusion_counts(binary_preds, labels), class_names)
