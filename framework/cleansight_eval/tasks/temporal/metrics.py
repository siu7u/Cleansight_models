"""时序任务指标（任务层）。

从 ``temporal-*/util.py`` 迁移的口径一致实现：edit 距离、segmental F1、逐帧
accuracy，以及因果平滑决策 ``causal_decision``。每个指标声明口径版本 ``spec``
（需求 §8.2），已计算的指标以 ``MetricValue`` 三态信封返回。

口径与原实现保持一致，未做数值改动，便于与旧 benchmark 对齐验收。
"""

from __future__ import annotations

import numpy as np
import torch

from ...core.envelope import MetricValue

# 口径版本：任何影响数值的口径变化都应递增版本号。
SPEC_ACC = "acc/frame-wise/v1"
SPEC_EDIT = "edit/levenstein-norm/v1"
SPEC_F1 = "segmental_f1/iou/v1"

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


def compute_temporal_metrics(pred_labels: list[str], gt_labels: list[str]) -> dict[str, MetricValue]:
    """计算逐帧 accuracy、edit、segmental F1@{0.1,0.25,0.5}，返回三态信封。"""

    n = len(pred_labels)
    if n == 0 or len(gt_labels) != n:
        reason = "预测与真值无法对齐或为空"
        return {
            "acc": MetricValue.missing(reason, spec=SPEC_ACC),
            "edit": MetricValue.missing(reason, spec=SPEC_EDIT),
            **{f"f1@{o}": MetricValue.missing(reason, spec=SPEC_F1) for o in (0.1, 0.25, 0.5)},
        }

    correct = sum(p == g for p, g in zip(pred_labels, gt_labels))
    acc = round(100.0 * correct / n, 2)
    edit = round(edit_score(pred_labels, gt_labels), 2)

    out = {
        "acc": MetricValue.computed(acc, spec=SPEC_ACC),
        "edit": MetricValue.computed(edit, spec=SPEC_EDIT),
    }
    for overlap in (0.1, 0.25, 0.5):
        tp, fp, fn = f_score(pred_labels, gt_labels, overlap)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = round(2.0 * precision * recall / (precision + recall + 1e-8) * 100, 2)
        out[f"f1@{overlap}"] = MetricValue.computed(f1, spec=SPEC_F1)
    return out
