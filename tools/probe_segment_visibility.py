"""动作类的"可见性"审计探针：这一类在当前检测特征里到底能不能被看见？

回答 `docs/features/INPUT_DESIGN_PROPOSAL.md` §1.4 留下的零成本待办
（"确认时间轴边界与可见动作是否一致"）——不需要看视频，用检测特征做一致性审计：

1. **段内 vs idle 可分性（AUC）**：把"真值属于类 X 的帧"与"真值 idle 的帧"当二分类，
   在 **train** 上拟合 LDA，在 train/val/test 上分别量 AUC。
   - AUC(train) 高 + AUC(test) ≈ 0.5 → 记忆化/跨批次退化；
   - **AUC(train) 也 ≈ 0.5 → 该类在当前检测里根本不可见**，做特征工程救不回来（要动检测或标签）。
2. **边界跳变**：对类 X 的每个真值段，比较"段内若干帧"与"段前/段后 idle 帧"的判别分数差；
   与同视频内随机位置的同类差值（零分布）比较，给出"跳变显著"的段占比。
   - 占比低说明**标注边界处检测特征没有变化**——要么事件本身不可见，要么标签时间轴可疑。
3. **逐通道证据**：列出与"段内"最相关的通道（类×区域×统计量），可读性用——例如 flush 是否真的
   伴随 syringe 通道变化。

纯 CPU、只读数据、不训练。用法：

    python tools/probe_segment_visibility.py --contract roi-grid-144 --split test --json tmp/visibility.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from framework.cleansight_eval.core.config import load_config  # noqa: E402
from framework.cleansight_eval.temporal.data import load_split  # noqa: E402

# 契约名 → 实验配置（与 tools/run_strategy_matrix.py 的策略表一致）
CONTRACTS = {
    "roi-grid-144": "gru-actionmixed-auto-roi.yaml",
    "bbox-40": "gru-actionmixed-auto.yaml",
}
# ROI 契约的通道布局：每 18 维一类（6 区域 × 3 统计量，区域行优先）
ROI_STATS = ("presence", "count", "max_area")
DETECTION_CLASSES = ("hand", "scope_control_body", "scope_mid_section", "scope_distal_end",
                     "syringe", "air_gun", "short_brush", "brush_tip_out")
REGIONS = ("left-top", "mid-top", "right-top", "left-bottom", "mid-bottom", "right-bottom")


def fit_lda(x_pos: np.ndarray, x_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """等先验二分类 LDA（与 probe_split_shift/probe_direction 同源，闭式解）。"""

    x = np.concatenate([x_pos, x_neg]).astype(np.float64)
    y = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
    mean_pos, mean_neg = x_pos.mean(axis=0), x_neg.mean(axis=0)
    within = np.zeros((x.shape[1], x.shape[1]))
    for label, mean in ((1.0, mean_pos), (0.0, mean_neg)):
        centered = x[y == label] - mean
        within += centered.T @ centered
    within /= max(len(x) - 2, 1)
    within += np.eye(x.shape[1]) * 1e-6
    weights = np.linalg.solve(within, (mean_pos - mean_neg))
    bias = -0.5 * (mean_pos + mean_neg) @ weights
    return weights, bias


def auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """AUC（Mann-Whitney U 等价式），单类缺失时返回 None。"""

    pos, neg = scores[positive], scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # 平均秩处理并列
    combined = np.concatenate([pos, neg])
    unique, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        rank_sum = np.zeros(len(unique))
        np.add.at(rank_sum, inverse, ranks)
        ranks = (rank_sum / counts)[inverse]
    rank_pos = ranks[: len(pos)].sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def segments_of(labels: np.ndarray, value: int) -> list[tuple[int, int]]:
    """把逐帧标签里连续等于 ``value`` 的区间折成 [start, end) 列表。"""

    spans, start = [], None
    for index, label in enumerate(labels):
        if label == value and start is None:
            start = index
        elif label != value and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def channel_name(contract: str, index: int) -> str:
    """把特征维索引翻成可读通道名（ROI 契约 = 类×区域×统计量）。"""

    if contract == "roi-grid-144":
        cls, rest = divmod(index, 18)
        region, stat = divmod(rest, 3)
        return f"{DETECTION_CLASSES[cls]}.{REGIONS[region]}.{ROI_STATS[stat]}"
    if contract == "bbox-40":
        cls, stat = divmod(index, 5)
        return f"{DETECTION_CLASSES[cls]}.{'presence,cx,cy,w,h'.split(',')[stat]}"
    return f"dim{index}"


def main() -> None:
    parser = argparse.ArgumentParser(description="动作类可见性审计探针")
    parser.add_argument("--contract", default="roi-grid-144", choices=sorted(CONTRACTS))
    parser.add_argument("--split", default="test", help="评估 split（LDA 一律在 train 上拟合）")
    parser.add_argument("--json", default=None, help="把结构化结果写到该路径")
    parser.add_argument("--top-channels", type=int, default=3, help="每类列出多少个最相关通道")
    args = parser.parse_args()

    cfg = load_config(REPO / "framework" / "experiments" / CONTRACTS[args.contract])
    data, schema = cfg["data"], cfg["feature_schema"]
    train_feats, train_truths, id2name = load_split(data, data["split_train"], feature_schema=schema)
    eval_feats, eval_truths, _ = load_split(data, args.split, feature_schema=schema)

    x_train = np.concatenate(train_feats)
    y_train = np.concatenate(train_truths)
    x_eval = np.concatenate(eval_feats)
    y_eval = np.concatenate(eval_truths)
    idle_id = 0
    print(f"契约 {args.contract}（{x_train.shape[1]} 维）| train {len(y_train)} 帧 / {args.split} {len(y_eval)} 帧"
          f" | 类别: {[id2name[i] for i in sorted(id2name)]}")

    records = []
    for action_id in sorted(id2name):
        if action_id == idle_id:
            continue
        name = id2name[action_id]
        pos_tr, neg_tr = x_train[y_train == action_id], x_train[y_train == idle_id]
        pos_ev, neg_ev = x_eval[y_eval == action_id], x_eval[y_eval == idle_id]
        if len(pos_tr) < 20 or len(neg_tr) < 20:
            records.append({"class": name, "support_train": int(len(pos_tr)), "note": "train 样本不足"})
            continue
        weights, bias = fit_lda(pos_tr, neg_tr)
        score_tr = x_train @ weights + bias
        score_ev = x_eval @ weights + bias
        auc_tr = auc(score_tr, np.isin(y_train, [action_id]))
        auc_ev = auc(score_ev, np.isin(y_eval, [action_id])) if len(pos_ev) else None

        # 边界跳变：段内均值 vs 段前/段后 idle 各 5 帧均值
        jumps, nulls = [], []
        for features, truth in zip(eval_feats, eval_truths):
            score = features @ weights + bias
            truth = np.asarray(truth)
            for start, end in segments_of(truth, action_id):
                before = truth[max(0, start - 5):start]
                after = truth[end:end + 5]
                outside = []
                if len(before) and (before == idle_id).all():
                    outside.append(score[max(0, start - 5):start])
                if len(after) and (after == idle_id).all():
                    outside.append(score[end:end + 5])
                if not outside:
                    continue
                inside_mean = float(score[start:end].mean())
                outside_mean = float(np.concatenate(outside).mean())
                jumps.append(inside_mean - outside_mean)
            # 零分布：同视频内随机取 idle 区间，比较前后 5 帧
            idle_idx = np.flatnonzero(truth == idle_id)
            rng = np.random.default_rng(0)
            for _ in range(50):
                if len(idle_idx) < 12:
                    break
                cut = int(rng.choice(idle_idx[6:-6])) if len(idle_idx) > 12 else None
                if cut is None:
                    break
                nulls.append(float(score[cut:cut + 5].mean() - score[cut - 5:cut].mean()))
        significant, effect = None, None
        if jumps and nulls:
            null_scale = float(np.std(nulls)) or 1e-9
            threshold = float(np.percentile(np.abs(nulls), 95))
            significant = float(np.mean([abs(j) > threshold for j in jumps]))
            # 效应量：段内−段外的中位跳变 / 零分布标准差（比"是否超阈值"更能说明可见程度）
            effect = float(np.median(np.abs(jumps)) / null_scale)

        # 逐通道证据：段内均值 − idle 均值（按 train 的逐维标准差归一化）
        std = x_train.std(axis=0) + 1e-6
        delta = (pos_tr.mean(axis=0) - neg_tr.mean(axis=0)) / std
        top = np.argsort(-np.abs(delta))[: args.top_channels]
        records.append({
            "class": name,
            "support_train": int(len(pos_tr)),
            "support_eval": int(len(pos_ev)),
            "segments_eval": int(sum(len(segments_of(np.asarray(t), action_id)) for t in eval_truths)),
            "auc_train": auc_tr,
            "auc_eval": auc_ev,
            "boundary_jump_ratio": significant,
            "boundary_effect_sigma": effect,
            "top_channels": [
                {"channel": channel_name(args.contract, int(i)), "delta_sigma": float(delta[i])} for i in top
            ],
        })

    print(f"\n| 动作类 | support(train) | support({args.split}) | 段数 | AUC(train) | AUC({args.split}) "
          f"| 边界跳变（显著占比 / 效应量） | 最相关通道（Δ/σ） |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for record in records:
        if "auc_train" not in record:
            print(f"| {record['class']} | {record['support_train']} | — | — | — | — | — | {record.get('note','')} |")
            continue
        top = ", ".join(f"{c['channel']} {c['delta_sigma']:+.2f}" for c in record["top_channels"])
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        sig = "n/a" if record["boundary_jump_ratio"] is None else f"{record['boundary_jump_ratio']*100:.0f}%"
        eff = "n/a" if record.get("boundary_effect_sigma") is None else f"{record['boundary_effect_sigma']:.2f}σ"
        print(f"| {record['class']} | {record['support_train']} | {record['support_eval']} | "
              f"{record['segments_eval']} | {fmt(record['auc_train'])} | {fmt(record['auc_eval'])} | "
              f"{sig}（{eff}） | {top} |")

    print("\n读法：AUC(train)≈0.5 → 该类在检测里不可见（特征工程救不回来）；"
          "\n      AUC(train) 高但 AUC(eval)≈0.5 → 记忆化/跨批次退化；"
          "\n      边界跳变占比低 → 标注边界处检测特征没变化（标签时间轴可疑或事件不可见）。")
    if args.json:
        Path(args.json).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结构化结果: {args.json}")


if __name__ == "__main__":
    main()
