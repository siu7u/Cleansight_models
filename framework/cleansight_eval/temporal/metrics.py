"""时序指标与延迟测量（两条时序流水线共用）。

从 ``temporal-*/util.py`` 迁移的口径一致实现：edit 距离、segmental F1、逐帧
accuracy，以及因果平滑决策 ``causal_decision``。每个指标声明口径版本 ``spec``
（需求 §8.2），已计算的指标以 ``MetricValue`` 三态信封返回。

延迟测量（``measure_single_tick`` / ``not_applicable_perf``）也放这里：滑窗流水线测单
tick 延迟，全序列流水线标 N/A 而非造假。口径与原实现保持一致，未做数值改动，便于与旧
benchmark 对齐验收。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ..core.envelope import MetricValue

# 过渡接入：benchmark 仍位于仓库根目录。兼容从仓库根目录运行
# ``python -m framework...`` 和进入 framework 后运行 ``python -m cleansight_eval...``。
try:
    from benchmark.core.metrics import temporal_metrics as _benchmark_temporal_metrics
except ModuleNotFoundError:  # pragma: no cover - 仅 framework 作为 cwd 时触发
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from benchmark.core.metrics import temporal_metrics as _benchmark_temporal_metrics

# 口径版本：任何影响数值的口径变化都应递增版本号。
SPEC_ACC = "accuracy/frame-wise/percent/v2; source=benchmark.core.metrics"
SPEC_EDIT = "edit/levenshtein-item-mean/percent/v2; source=benchmark.core.metrics"
SPEC_F1 = "segmental_f1/label-aware-one-to-one-iou/percent/v2; source=benchmark.core.metrics"
SPEC_PRECISION = "segmental_precision/label-aware-one-to-one-iou/percent/v2; source=benchmark.core.metrics"
SPEC_RECALL = "segmental_recall/label-aware-one-to-one-iou/percent/v2; source=benchmark.core.metrics"
SPEC_COUNTS = "segmental_counts/label-aware-one-to-one-iou/v2; source=benchmark.core.metrics"
SPEC_TEMPORAL_IOU = "temporal_iou/matched-segment-mean/percent/v2; source=benchmark.core.metrics"
SPEC_FRAME_CLASS = "classification/per-class/percent/v2; source=benchmark.core.metrics"
SPEC_LATENCY = "latency/single_tick_ms/v1"

BG_CLASS = ["background"]


def get_labels_start_end_time(frame_wise_labels, bg_class=BG_CLASS):
    labels, starts, ends = [], [], []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row, n_col = len(p), len(y)
    D = np.zeros([m_row + 1, n_col + 1], float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i
    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    if norm:
        return (1 - D[-1, -1] / max(m_row, n_col)) * 100
    return D[-1, -1]


def edit_score(recognized, ground_truth, norm=True, bg_class=BG_CLASS):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=BG_CLASS):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp, fp = 0, 0
    hits = np.zeros(len(y_label))
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        idx = np.array(IoU).argmax()
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def causal_decision(last, pending, stable, count, num_classes: int | None = None):
    """因果平滑：转移先验 + 最小持续时长，迁移自 util.causal_decision。

    仅在 3 类（Idle/Long/Short）时应用带类别语义的转移先验；其他类别数时退化为
    仅最小持续时长平滑，避免对未知类别硬编码先验。
    """

    prob = torch.softmax(last, dim=-1).cpu().numpy()
    C = len(prob)

    transition_prior = np.zeros((C, C))
    if C == 3:
        idle_id, long_id, short_id = 0, 1, 2
        transition_prior[idle_id, idle_id] = 2.0
        transition_prior[long_id, long_id] = 2.0
        transition_prior[short_id, short_id] = 1.5
        transition_prior[long_id, short_id] = -1.0
        transition_prior[short_id, long_id] = -1.0

    scores = np.zeros(C)
    for j in range(C):
        scores[j] = np.log(prob[j] + 1e-8) + transition_prior[stable, j]
    candidate = int(np.argmax(scores))

    MIN_DURATION = 25
    if candidate == pending:
        count += 1
    else:
        pending = candidate
        count = 1
    if count >= MIN_DURATION:
        stable = pending if pending is not None else 0
    return pending, stable, count


def _percent_metric(value, spec: str, reason: str = "指标没有可计算样本") -> MetricValue:
    """把 benchmark 的 0..1 比率转成 framework 历史兼容的 0..100 三态指标。"""

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
    """调用 benchmark 公共实现，按视频边界汇总时序和逐帧指标。

    benchmark 原始比率为 0..1；framework 对外继续用 0..100，避免已有 history、报告和
    best checkpoint 口径静默变化。新增片段 TP/FP/FN、P/R、matched temporal IoU，及逐类
    帧级 P/R/F1/IoU。输入 item 必须分别对应一段独立视频，禁止预先拼接。
    """

    try:
        raw = _benchmark_temporal_metrics(
            pred_by_item,
            truth_by_item,
            labels=list(labels),
            start_frame=start_frame,
            thresholds=(0.1, 0.25, 0.5),
            ignore_index=-1,
        )
    except ValueError as exc:
        reason = f"benchmark metrics 输入无效: {exc}"
        missing = {
            "acc": MetricValue.missing(reason, spec=SPEC_ACC),
            "edit": MetricValue.missing(reason, spec=SPEC_EDIT),
            **{f"f1@{threshold}": MetricValue.missing(reason, spec=SPEC_F1) for threshold in (0.1, 0.25, 0.5)},
        }
        return (missing, {"error": reason}) if return_details else missing

    frame = raw["frame"]
    segment = raw["segment"]
    out: dict[str, MetricValue] = {
        "acc": _percent_metric(frame.get("accuracy"), SPEC_ACC),
        "edit": _percent_metric(segment.get("edit"), SPEC_EDIT),
        "frame.macro_f1": _percent_metric(frame.get("macro_f1"), SPEC_FRAME_CLASS),
        "frame.macro_iou": _percent_metric(frame.get("macro_iou"), SPEC_FRAME_CLASS),
        "frame.micro_f1": _percent_metric(frame.get("micro_f1"), SPEC_FRAME_CLASS),
    }

    for threshold in (0.1, 0.25, 0.5):
        key = f"{threshold:.2f}"
        detail = segment["details_at_iou"][key]
        suffix = str(threshold)
        out[f"f1@{suffix}"] = _percent_metric(detail.get("f1"), SPEC_F1)

        # summary 仅保留主阈值 0.5 的诊断量；其他阈值的 counts/P/R/IoU 和逐类
        # 指标完整保存在 metrics.details.temporal，避免报告与矩阵无限横向膨胀。
        if threshold == 0.5:
            out[f"tp@{suffix}"] = MetricValue.computed(int(detail["tp"]), spec=SPEC_COUNTS)
            out[f"fp@{suffix}"] = MetricValue.computed(int(detail["fp"]), spec=SPEC_COUNTS)
            out[f"fn@{suffix}"] = MetricValue.computed(int(detail["fn"]), spec=SPEC_COUNTS)
            out[f"precision@{suffix}"] = _percent_metric(detail.get("precision"), SPEC_PRECISION)
            out[f"recall@{suffix}"] = _percent_metric(detail.get("recall"), SPEC_RECALL)
            out[f"temporal_iou@{suffix}"] = _percent_metric(
                detail.get("mean_matched_iou"),
                SPEC_TEMPORAL_IOU,
                reason="该 IoU 阈值下没有匹配片段",
            )
    return (out, raw) if return_details else out


def compute_temporal_metrics(pred_labels: list[str], gt_labels: list[str]) -> dict[str, MetricValue]:
    """单序列兼容入口；流水线评估应优先使用 ``compute_temporal_metrics_by_item``。"""

    labels = sorted(set(pred_labels) | set(gt_labels))
    return compute_temporal_metrics_by_item(
        {"item-0": pred_labels},
        {"item-0": gt_labels},
        labels,
    )


def measure_single_tick(
    model, window: int, input_dim: int, device, warmup: int = 20, runs: int = 200
) -> dict[str, MetricValue]:
    """测量单窗口 ``[1, window, input_dim]`` 前向延迟（取末帧，模拟滑窗流式一 tick）。"""

    model.eval()
    x = torch.randn(1, window, input_dim, device=device)

    def _tick():
        return model(x)[0, -1]  # 末帧 logits，滑窗流式的一步

    with torch.no_grad():
        for _ in range(warmup):
            _tick()
        if device.type == "cuda":
            torch.cuda.synchronize()

        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _tick()
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    mean_ms = sum(samples) / len(samples)
    median_ms = samples[len(samples) // 2]
    p95_ms = samples[min(len(samples) - 1, int(0.95 * len(samples)))]
    spec = f"{SPEC_LATENCY}; device={device}; window={window}; warmup={warmup}; runs={runs}"
    return {
        "latency_mean_ms": MetricValue.computed(round(mean_ms, 4), spec=spec),
        "latency_median_ms": MetricValue.computed(round(median_ms, 4), spec=spec),
        "latency_p95_ms": MetricValue.computed(round(p95_ms, 4), spec=spec),
    }


def not_applicable_perf(reason: str = "该流水线不测量实时延迟") -> dict[str, MetricValue]:
    return {
        "latency_mean_ms": MetricValue.not_applicable(reason, spec=SPEC_LATENCY),
        "latency_median_ms": MetricValue.not_applicable(reason, spec=SPEC_LATENCY),
        "latency_p95_ms": MetricValue.not_applicable(reason, spec=SPEC_LATENCY),
    }
