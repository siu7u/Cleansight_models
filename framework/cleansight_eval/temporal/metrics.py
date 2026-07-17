"""时序训练兼容工具；正式评估指标由 benchmark.evaluators.temporal 提供。

从 ``temporal-*/util.py`` 迁移的口径一致实现：edit 距离、segmental F1、逐帧
accuracy，以及因果平滑决策 ``causal_decision``。每个指标声明口径版本 ``spec``
（需求 §8.2），已计算的指标以 ``MetricValue`` 三态结果返回。

延迟测量（``measure_single_tick`` / ``not_applicable_perf``）也放这里：滑窗流水线测单
tick 延迟，全序列流水线标 N/A 而非造假。口径与原实现保持一致，未做数值改动，便于与旧
benchmark 对齐验收。
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.execution import sample_callable_latency
from .util import causal_decision  # 历史兼容导出；实现属于推理后处理，不属于指标

try:
    from benchmark.core.result import MetricValue
    from benchmark.evaluators.temporal import (
        SPEC_ACC,
        SPEC_COUNTS,
        SPEC_EDIT,
        SPEC_F1,
        SPEC_FRAME_CLASS,
        SPEC_MODEL_FORWARD,
        SPEC_PRECISION,
        SPEC_RECALL,
        SPEC_TEMPORAL_IOU,
        compute_temporal_metrics,
        compute_temporal_metrics_by_item,
        not_applicable_model_forward as not_applicable_perf,
        summarize_model_forward_timing as summarize_single_tick_timing,
    )
except ModuleNotFoundError:  # pragma: no cover - 仅 framework 作为 cwd 时触发
    import sys
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from benchmark.core.result import MetricValue
    from benchmark.evaluators.temporal import (
        SPEC_ACC,
        SPEC_COUNTS,
        SPEC_EDIT,
        SPEC_F1,
        SPEC_FRAME_CLASS,
        SPEC_MODEL_FORWARD,
        SPEC_PRECISION,
        SPEC_RECALL,
        SPEC_TEMPORAL_IOU,
        compute_temporal_metrics,
        compute_temporal_metrics_by_item,
        not_applicable_model_forward as not_applicable_perf,
        summarize_model_forward_timing as summarize_single_tick_timing,
    )

SPEC_LATENCY = SPEC_MODEL_FORWARD  # 历史兼容名称

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


def measure_single_tick(
    model, window: int, input_dim: int, device, warmup: int = 20, runs: int = 200
) -> dict[str, MetricValue]:
    """兼容入口：采集原始单 tick 样本后，按既有口径汇总。"""

    model.eval()
    x = torch.randn(1, window, input_dim, device=device)

    def _tick():
        return model(x)[0, -1]  # 末帧 logits，滑窗流式的一步

    timing = sample_callable_latency(
        _tick,
        device,
        warmup=warmup,
        runs=runs,
        scope="model_forward_single_window",
        context={"window": window, "input_dim": input_dim, "input_shape": [1, window, input_dim]},
    )
    return summarize_single_tick_timing(timing)
