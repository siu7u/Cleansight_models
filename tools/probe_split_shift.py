#!/usr/bin/env python3
"""分布漂移与可分性探针：回答"跨批次退化是数据问题还是模型问题"。

纯 CPU、只读 `labels/` + `frames/`（走与训练完全一致的 loader），不训练时序模型、不写数据。
对每个已登记的自动标注特征契约输出三组证据：

1. **特征层批次可分性**（proxy A-distance）：用 LDA 拟合"域"二分类（train vs test / train vs val），
   按视频分组 5 折交叉验证取 held-out AUC。0.5 ≈ 特征不可分（无协变量漂移），1.0 = 完美可分。
2. **类质心映射**（逐帧线性探针）：用 train 拟合等先验多类 LDA，在 val（同批次）与 test
   （跨批次）上给 truth × predicted 混淆矩阵与逐类 recall——线性探针是"特征里到底有多少
   可用信号"的下界参照，可与时序模型指标对照，判断退化发生在特征层还是模型层。
3. **单特征漂移**：test 相对 train 的标准化均值差（SMD）统计（|SMD|>1 的通道占比、中位 |SMD|）。

用法：

    python tools/probe_split_shift.py                        # 全部已登记契约
    python tools/probe_split_shift.py --contract roi-grid-144 --json tmp/probe.json

判读要点（2026-09-18 实测，新 test = project-18 八视频）：域 AUC 仅 0.47~0.61（无明显协变量
漂移），但线性探针在 test 上的 flush recall 0~15%、val 上 48~80% —— 说明退化不在"输入分布
变了"，而在"同一动作在不同批次里长得不一样 / 模型只学到批次内线索"。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.cleansight_eval.core.config import load_config  # noqa: E402
from framework.cleansight_eval.temporal.data import load_split, split_video_names  # noqa: E402

# 契约名 → 实验配置（与 tools/run_strategy_matrix.py 的策略表保持一致）
CONTRACTS: dict[str, str] = {
    "bbox-40-global": "gru-actionmixed-auto.yaml",
    "bbox-40-hand": "gru-actionmixed-auto-hand.yaml",
    "bbox-80-global-hand": "gru-actionmixed-auto-global-hand.yaml",
    "roi-grid-144": "gru-actionmixed-auto-roi.yaml",
    "roi-grid-96-v2": "gru-actionmixed-auto-roi-v2.yaml",
}
DEFAULT_SPLITS = ("train", "val", "test")


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney U 形式的 AUC（positive=1 的分数应更高）。"""

    pos, neg = scores[positive == 1], scores[positive == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(pos.size + neg.size, dtype=np.float64)
    ranks[order] = np.arange(1, pos.size + neg.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def lda_direction(x_fit: np.ndarray, y_fit: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """二分类 LDA 判别方向 ``w = Σ⁻¹(μ₁-μ₀)``（Σ 为带 ridge 的合并协方差）。"""

    mu1, mu0 = x_fit[y_fit == 1].mean(axis=0), x_fit[y_fit == 0].mean(axis=0)
    dim = x_fit.shape[1]
    n1, n0 = int((y_fit == 1).sum()), int((y_fit == 0).sum())
    c1 = np.cov(x_fit[y_fit == 1], rowvar=False) if n1 > 1 else np.zeros((dim, dim))
    c0 = np.cov(x_fit[y_fit == 0], rowvar=False) if n0 > 1 else np.zeros((dim, dim))
    pooled = ((n1 - 1) * c1 + (n0 - 1) * c0) / max(n1 + n0 - 2, 1)
    pooled = pooled + ridge * (float(np.trace(pooled)) / dim or 1.0) * np.eye(dim)
    return np.linalg.solve(pooled, mu1 - mu0)


def domain_auc(feats_a: list[np.ndarray], groups_a: list[str],
               feats_b: list[np.ndarray], groups_b: list[str], folds: int = 5) -> float:
    """按视频分组 K 折交叉验证的域分类 AUC（同视频的帧不跨折，避免相关性抬虚 AUC）。"""

    x = np.concatenate(feats_a + feats_b).astype(np.float64)
    y = np.concatenate([np.zeros(sum(len(f) for f in feats_a), dtype=int),
                        np.ones(sum(len(f) for f in feats_b), dtype=int)])
    groups = np.array([g for f, g in zip(feats_a, groups_a) for _ in range(len(f))]
                      + [g for f, g in zip(feats_b, groups_b) for _ in range(len(f))])
    videos = sorted(set(groups))
    fold_of = {name: index % folds for index, name in enumerate(videos)}
    fold = np.array([fold_of[g] for g in groups])

    scores = np.full(len(y), np.nan)
    for k in range(folds):
        fit_mask, eval_mask = fold != k, fold == k
        if not fit_mask.any() or not eval_mask.any():
            continue
        w = lda_direction(x[fit_mask], y[fit_mask])
        scores[eval_mask] = x[eval_mask] @ w
    valid = ~np.isnan(scores)
    return auc(scores[valid], y[valid])


def fit_multiclass_lda(x: np.ndarray, y: np.ndarray, n_classes: int, ridge: float = 1e-3):
    """等先验多类 LDA：返回 ``(w, b)``，判别分数为 ``x @ w.T + b``。"""

    dim = x.shape[1]
    mus, covs, counts = [], [], []
    for c in range(n_classes):
        xc = x[y == c]
        mus.append(xc.mean(axis=0) if len(xc) else np.zeros(dim))
        covs.append(np.cov(xc, rowvar=False) if len(xc) > 1 else np.zeros((dim, dim)))
        counts.append(len(xc))
    mu = np.stack(mus)
    pooled = sum((n - 1) * cov for n, cov in zip(counts, covs)) / max(sum(counts) - n_classes, 1)
    inv = np.linalg.inv(pooled + ridge * (float(np.trace(pooled)) / dim or 1.0) * np.eye(dim))
    return (inv @ mu.T).T, -0.5 * np.einsum("cd,cd->c", mu @ inv, mu)


def confusion(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, n_classes: int) -> np.ndarray:
    """truth × predicted 混淆矩阵（行 = truth，列 = 预测）。"""

    pred = np.argmax(x @ w.T + b, axis=1)
    return np.array([[int(((y == t) & (pred == p)).sum()) for p in range(n_classes)]
                     for t in range(n_classes)])


def segment_stats(truths: list[np.ndarray], id2name: dict[int, str],
                  min_durations: tuple[int, ...] = (1, 5, 25)) -> dict[str, dict]:
    """逐类真实段长分布与"阈值可达性"。

    "阈值可达性" = 该类真实段中长度 ≥ min_duration 的帧占比，等于该平滑阈值下该类
    召回的**理论上限**（段短于阈值的动作在因果平滑输出里不可能出现）。
    """

    lengths: dict[int, list[int]] = {cid: [] for cid in id2name}
    for truth in truths:
        truth = np.asarray(truth)
        start = 0
        for index in range(1, len(truth) + 1):
            if index == len(truth) or truth[index] != truth[start]:
                lengths[int(truth[start])].append(index - start)
                start = index
    out: dict[str, dict] = {}
    for cid, name in id2name.items():
        values = np.array(lengths[cid])
        entry: dict = {"frames": int(values.sum()), "segments": int(values.size)}
        if values.size:
            entry.update({
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "max": int(values.max()),
                "reachable": {str(md): float(values[values >= md].sum() / values.sum())
                              for md in min_durations},
            })
        out[name] = entry
    return out


def probe_contract(name: str, cfg_path: str) -> dict:
    """对单个契约跑完三组探针，返回结构化结果（同时用于 markdown 与 JSON 输出）。"""

    cfg = load_config(ROOT / "framework/experiments" / cfg_path)
    feature_schema, data = cfg["feature_schema"], cfg["data"]
    n_classes = int(cfg["model"]["num_classes"])
    per_split: dict[str, tuple[list[np.ndarray], list[str]]] = {}
    truths_of: dict[str, list[np.ndarray]] = {}
    labels: list[str] = []
    for split in DEFAULT_SPLITS:
        feats, truths, id2name = load_split(data, split, feature_schema=feature_schema)
        per_split[split] = (feats, split_video_names(data, split))
        truths_of[split] = truths
        labels = [id2name[i] for i in range(n_classes)]

    tr_f, tr_g = per_split["train"]
    va_f, va_g = per_split["val"]
    te_f, te_g = per_split["test"]
    x_tr = np.concatenate(tr_f).astype(np.float64)
    y_tr = np.concatenate(truths_of["train"])

    std = x_tr.std(axis=0)
    smd = np.abs(np.concatenate(te_f).mean(axis=0) - x_tr.mean(axis=0)) / np.where(std > 1e-9, std, np.nan)

    w, b = fit_multiclass_lda(x_tr, y_tr, n_classes)
    result = {
        "dim": int(feature_schema["dim"]),
        "labels": labels,
        "domain_auc": {
            "train_vs_test": domain_auc(tr_f, tr_g, te_f, te_g),
            "train_vs_val": domain_auc(tr_f, tr_g, va_f, va_g),
        },
        "smd": {
            "gt1_fraction": float(np.nanmean(smd > 1.0)),
            "median_abs": float(np.nanmedian(smd)),
        },
        "segments": {split: segment_stats(truths_of[split], {i: labels[i] for i in range(n_classes)})
                     for split in DEFAULT_SPLITS},
        "linear_probe": {},
    }
    for split in ("val", "test"):
        result["linear_probe"][split] = confusion(
            np.concatenate(per_split[split][0]).astype(np.float64),
            np.concatenate(truths_of[split]), w, b, n_classes,
        ).tolist()
    return result


def render(report: dict[str, dict], contracts: list[str]) -> str:
    """把探针结果渲染成 markdown。"""

    lines = ["# 分布漂移与可分性探针", "",
             "- 域 AUC：LDA 域分类器 + 按视频分组 5 折 CV（0.5 = 特征不可分）",
             "- 线性探针：train 拟合等先验多类 LDA，逐帧混淆矩阵（行 = truth，列 = 预测）",
             "- 段长可达性：真实段长 ≥ 阈值的帧占比 = 该 `smoothing_min_duration` 下的召回上限", ""]
    lines += ["| 契约 | 维度 | train vs test AUC | train vs val AUC | |SMD|>1 占比 | 中位 |SMD| |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in contracts:
        item = report[name]
        lines.append(f"| {name} | {item['dim']} | **{item['domain_auc']['train_vs_test']:.3f}** | "
                     f"{item['domain_auc']['train_vs_val']:.3f} | "
                     f"{item['smd']['gt1_fraction']:.2%} | {item['smd']['median_abs']:.2f} |")
    for name in contracts:
        item = report[name]
        labels = item["labels"]
        lines += ["", f"### {name}：段长分布与阈值可达性（召回上限）", "",
                  "| split | 类别 | 帧数 | 段数 | 段长中位 | p90 | 最长 | ≥1 | ≥5 | ≥25 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for split in DEFAULT_SPLITS:
            for label in labels:
                entry = item["segments"][split].get(label) or {}
                if not entry.get("frames"):
                    lines.append(f"| {split} | {label} | 0 | 0 | — | — | — | — | — | — |")
                    continue
                reach = entry.get("reachable") or {}
                lines.append(
                    f"| {split} | {label} | {entry['frames']} | {entry['segments']} | "
                    f"{entry['median']:.0f} | {entry['p90']:.0f} | {entry['max']} | "
                    + " | ".join("n/a" if str(md) not in reach else f"{reach[str(md)] * 100:.0f}%"
                                 for md in (1, 5, 25)) + " |")
        lines += ["", f"### {name}（{item['dim']} 维）：逐帧线性探针混淆矩阵", "",
                  "| truth \\ pred | " + " | ".join(labels) + " | support | recall |",
                  "|" + "---|" * (len(labels) + 2)]
        for split in ("val", "test"):
            mat = np.array(item["linear_probe"][split])
            lines += [f"| **{split}** |" + " | ".join([""] * (len(labels) + 2)) + "|"]
            for c, label in enumerate(labels):
                support = int(mat[c].sum())
                recall = "n/a" if support == 0 else f"{mat[c, c] / support * 100:.1f}%"
                lines.append(f"| {label} | " + " | ".join(str(int(v)) for v in mat[c])
                             + f" | {support} | {recall} |")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="分布漂移与可分性探针（跨批次退化诊断）")
    parser.add_argument("--contract", action="append", default=None,
                        help=f"只跑指定契约，可多次；缺省全部（{', '.join(CONTRACTS)}）")
    parser.add_argument("--json", default=None, help="把结构化结果写到该路径")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    contracts = args.contract or list(CONTRACTS)
    unknown = [c for c in contracts if c not in CONTRACTS]
    if unknown:
        raise SystemExit(f"未知契约 {unknown}；可选: {sorted(CONTRACTS)}")

    report = {}
    for name in contracts:
        print(f"== 探针 {name} ==", flush=True)
        report[name] = probe_contract(name, CONTRACTS[name])
        item = report[name]
        print(f"  train-vs-test AUC={item['domain_auc']['train_vs_test']:.3f}  "
              f"train-vs-val AUC={item['domain_auc']['train_vs_val']:.3f}", flush=True)
    text = render(report, contracts)
    print(text)
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结构化结果: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
