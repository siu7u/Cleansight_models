import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

def load_mappings(file: str):
    mapping = {}
    for line in Path(file).read_text().splitlines():
        idx, action = line.split()
        mapping[action] = int(idx)
    return mapping

def load_features(root: str, name: str):
    path = Path(root) / f"{name}.npy"
    features = np.load(path)
    # 原数据为了对齐 MS-TCN2，形状 (F, T)
    # 这里取转置
    return features.T

def load_truths(root: str, name: str, mappings: dict):
    path = Path(root) / f"{name}.txt"

    with open(path) as f:
        labels = [line.strip() for line in f]

    return [mappings[lbl] for lbl in labels]

def compute_class_weights(dataloader):
    counter = Counter()
    for _, y in dataloader:
        counter.update(y.cpu().numpy())

    total = sum(counter.values())
    weights = {
        cls: total / count
        for cls, count in counter.items()
    }

    # 归一化
    max_w = max(weights.values())
    weights = {k: v / max_w for k, v in weights.items()}

    return weights

def get_current_timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def causal_decision(last, pending, stable, count):

    prob = torch.softmax(last, dim=-1).cpu().numpy()

    idle_id, long_id, short_id = 0, 1, 2

    C = len(prob)

    transition_prior = np.zeros((C, C))

    # 自转移偏好
    transition_prior[idle_id, idle_id] = 2.0
    transition_prior[long_id, long_id] = 2.0
    transition_prior[short_id, short_id] = 1.5

    # 动作间切换惩罚
    transition_prior[long_id, short_id] = -1.0
    transition_prior[short_id, long_id] = -1.0

    # ===== 转移得分 =====
    scores = np.zeros(C)
    for j in range(C):
        emit = np.log(prob[j] + 1e-8)
        trans = transition_prior[stable, j]
        scores[j] = emit + trans

    candidate = np.argmax(scores)

    # ===== Min Duration：和“待确认”比较 =====
    MIN_DURATION = 25

    if candidate == pending:
        count += 1
    else:
        pending = candidate
        count = 1

    # ===== 稳定后才确认 =====
    if count >= MIN_DURATION:
        stable = pending if pending is not None else idle_id

    return pending, stable, count

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
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
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)



def plot_temporal_results(
        recognitions: list[str], ground_truths: list[str],
        metadata: dict, output_path
    ):
    # 1. 将动作标签映射为数字 ID
    actions = sorted(list(set(ground_truths + recognitions)))
    action2id = {a: i for i, a in enumerate(actions)}
    
    gt_ids = [action2id[a] for a in ground_truths]
    pred_ids = [action2id[a] for a in recognitions]

    # 2. 绘图
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # 绘制 Ground Truth (上方)
    ax.broken_barh([(i, 1) for i in range(len(gt_ids))], (1.2, 0.8), 
                   facecolors=[plt.cm.tab20(i % 20) for i in gt_ids])
    
    # 绘制 Prediction (下方)
    ax.broken_barh([(i, 1) for i in range(len(pred_ids))], (0.2, 0.8), 
                   facecolors=[plt.cm.tab20(i % 20) for i in pred_ids])

    metrics_text = (
        f"timestamp: {metadata['timestamp']}\n"
        f"model: {metadata['model']}\n"
        f"path: {metadata['path']}\n"
        f"num_params: {metadata['num_params']}\n"
        f"Acc: {metadata['acc']:.2f}\n"
        f"Edit: {metadata['edit']:.2f}\n"
        f"F1@0.1: {metadata['f1'][0.1]:.2f}\n"
        f"F1@0.25: {metadata['f1'][0.25]:.2f}\n"
        f"F1@0.5: {metadata['f1'][0.5]:.2f}\n"
    )

    ax.annotate(
        metrics_text,
        xy=(1.02, 1.0),
        xycoords='axes fraction',
        fontsize=10,
        va='top', ha='left',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # 3. 美化
    TOTAL_FRAMES = len(gt_ids)
    FPS = 30
    MAX_SEC = TOTAL_FRAMES / FPS

    # 每 10 秒一个刻度
    xticks = np.arange(0, int(np.ceil(MAX_SEC / 10)) + 1) * 10

    ax.set_xticks(xticks * FPS)
    ax.set_xticklabels(xticks.astype(int))
    ax.set_xlabel("Time (s)")

    ax.set_yticks([1.6, 0.6])
    ax.set_yticklabels(['Ground Truth', 'Prediction'])
    ax.set_title(f"Action Segmentation")

    # 4. 制作图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=plt.cm.tab20(i % 20), lw=4, label=a) 
                       for a, i in action2id.items()]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)

    plt.tight_layout()

    import os
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"可视化结果已保存至: {output_path}")
    plt.close()