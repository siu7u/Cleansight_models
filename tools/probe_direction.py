#!/usr/bin/env python3
"""方向判别探针：某一对动作类在给定特征契约里到底可不可分，以及**上下文是否有帮助**。

用途（2026-09-18 起）：`long_brush_insert` vs `long_brush_withdraw` 是当前模型逐类指标最差的一对。
本工具回答两个问题：

1. 这对类在特征空间里可分吗？（二分类 LDA 在 train 的该类帧上拟合，在 val/test 上评估 AUC）
2. 把它们前后若干帧**拼起来**（因果过去 k 帧 / 居中 ±k 帧）会提升可分性吗？
   —— 若拼接上下文不提升，说明"加长滑窗"救不了这对类，方向信息不在帧间关系里。

判读：AUC 0.5 = 不可分；0.5~0.6 = 弱信号；>0.7 才算真的可分。
注意 AUC≈0.5 时多分类里的 recall 仍可能非零——那来自"刷子在镜身内"这一**组级**可分性
（insert+withdraw 一起 vs idle），不是方向判别；报告逐类指标时应同时给 precision。

用法：

    python tools/probe_direction.py                      # 全部已登记契约，默认 insert/withdraw
    python tools/probe_direction.py --pair idle,flush --json tmp/dir.json
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
from framework.cleansight_eval.temporal.data import load_split  # noqa: E402

CONTRACTS: dict[str, str] = {
    "bbox-40-global": "gru-actionmixed-auto.yaml",
    "bbox-40-hand": "gru-actionmixed-auto-hand.yaml",
    "bbox-80-global-hand": "gru-actionmixed-auto-global-hand.yaml",
    "roi-grid-144": "gru-actionmixed-auto-roi.yaml",
    "roi-grid-96-v2": "gru-actionmixed-auto-roi-v2.yaml",
}
DEFAULT_PAIR = ("long_brush_insert", "long_brush_withdraw")
# 上下文口径：名称 → (模式, k)。单帧 k=0；因果取过去 k 帧；居中取前后 k 帧。
CONTEXTS: dict[str, tuple[str, int]] = {
    "单帧": ("none", 0),
    "因果-8": ("causal", 8),
    "居中-8": ("centered", 8),
    "居中-24": ("centered", 24),
}


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney U 形式的 AUC（positive=1 的分数应更高）。"""

    pos, neg = scores[positive == 1], scores[positive == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(pos.size + neg.size, dtype=np.float64)
    ranks[order] = np.arange(1, pos.size + neg.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def build_pairs(feats: list[np.ndarray], truths: list[np.ndarray],
                positive_id: int, negative_id: int, mode: str, k: int):
    """构造 (X, y)：只取 truth ∈ {positive, negative} 的帧，按上下文模式拼接相邻帧特征。"""

    rows, labels = [], []
    for sequence, truth in zip(feats, truths):
        sequence = np.asarray(sequence, dtype=np.float64)
        truth = np.asarray(truth)
        for index in range(len(truth)):
            label = truth[index]
            if label not in (positive_id, negative_id):
                continue
            if mode == "none":
                window = [index]
            elif mode == "causal":
                window = range(max(0, index - k), index + 1)
            else:
                window = range(max(0, index - k), min(len(truth), index + k + 1))
            rows.append(np.concatenate([sequence[i] for i in window]))
            labels.append(1 if label == positive_id else 0)
    if not rows:
        return None, None
    width = max(len(row) for row in rows)
    x = np.zeros((len(rows), width), dtype=np.float64)
    for i, row in enumerate(rows):
        x[i, : len(row)] = row
    return x, np.asarray(labels)


def fit_binary_lda(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    """二分类 LDA 方向 ``w = Σ⁻¹(μ₁-μ₀)``（Σ 为带 ridge 的合并协方差）。"""

    mu1, mu0 = x[y == 1].mean(axis=0), x[y == 0].mean(axis=0)
    dim = x.shape[1]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    c1 = np.cov(x[y == 1], rowvar=False) if n1 > 1 else np.zeros((dim, dim))
    c0 = np.cov(x[y == 0], rowvar=False) if n0 > 1 else np.zeros((dim, dim))
    pooled = ((n1 - 1) * c1 + (n0 - 1) * c0) / max(n1 + n0 - 2, 1)
    pooled = pooled + ridge * (float(np.trace(pooled)) / dim or 1.0) * np.eye(dim)
    return np.linalg.solve(pooled, mu1 - mu0)


def probe_contract(name: str, cfg_path: str, pair: tuple[str, str]) -> dict:
    """对单个契约跑"单帧 vs 拼接上下文"的方向可分性对照。"""

    cfg = load_config(ROOT / "framework/experiments" / cfg_path)
    data, feature_schema = cfg["data"], cfg["feature_schema"]
    _f, _t, id2name = load_split(data, "train", feature_schema=feature_schema)
    name2id = {value: key for key, value in id2name.items()}
    missing = [label for label in pair if label not in name2id]
    if missing:
        raise SystemExit(f"契约 {name} 的类别表里没有 {missing}；可用: {sorted(name2id)}")
    positive_id, negative_id = name2id[pair[0]], name2id[pair[1]]

    splits = {}
    for split in ("train", "val", "test"):
        feats, truths, _ = load_split(data, split, feature_schema=feature_schema)
        splits[split] = (feats, truths)

    result: dict[str, dict] = {}
    for ctx_name, (mode, k) in CONTEXTS.items():
        x_tr, y_tr = build_pairs(*splits["train"], positive_id, negative_id, mode, k)
        x_va, y_va = build_pairs(*splits["val"], positive_id, negative_id, mode, k)
        x_te, y_te = build_pairs(*splits["test"], positive_id, negative_id, mode, k)
        if x_tr is None or x_tr.shape[1] == 0:
            continue
        w = fit_binary_lda(x_tr, y_tr)
        entry = {
            "dim": int(x_tr.shape[1]),
            "auc_train": auc(x_tr @ w, y_tr),
            "auc_val": float("nan") if x_va is None else auc(x_va @ w, y_va),
            "auc_test": float("nan") if x_te is None else auc(x_te @ w, y_te),
            "n_train": int(len(y_tr)),
            "n_test": 0 if y_te is None else int(len(y_te)),
        }
        result[ctx_name] = entry
        print(f"  {name:18} {ctx_name:7} dim={entry['dim']:5} "
              f"AUC(val)={entry['auc_val']:.3f} AUC(test)={entry['auc_test']:.3f}", flush=True)
    return {"pair": list(pair), "contexts": result}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="方向判别探针（类对可分性 + 上下文效应）")
    parser.add_argument("--pair", default=",".join(DEFAULT_PAIR),
                        help=f"两个类别名，逗号分隔（默认 {','.join(DEFAULT_PAIR)}）")
    parser.add_argument("--contract", action="append", default=None,
                        help=f"只跑指定契约，可多次；缺省全部（{', '.join(CONTRACTS)}）")
    parser.add_argument("--json", default=None, help="结构化结果输出路径")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pair = tuple(part.strip() for part in args.pair.split(",") if part.strip())
    if len(pair) != 2:
        raise SystemExit("--pair 需要两个类别名，如 --pair long_brush_insert,long_brush_withdraw")
    contracts = args.contract or list(CONTRACTS)
    unknown = [name for name in contracts if name not in CONTRACTS]
    if unknown:
        raise SystemExit(f"未知契约 {unknown}；可选: {sorted(CONTRACTS)}")

    report = {}
    for name in contracts:
        print(f"== 方向判别 {name}：{pair[0]} vs {pair[1]} ==", flush=True)
        report[name] = probe_contract(name, CONTRACTS[name], pair)

    lines = [f"# 方向判别探针：{pair[0]} vs {pair[1]}", "",
             "AUC 0.5 = 不可分；0.5~0.6 = 弱信号；>0.7 才算可分。"
             "`居中-k`/`因果-k` 为拼接 2k+1 / k+1 帧后的可分性。", "",
             "| 契约 | 上下文 | 维度 | AUC(train) | AUC(val) | AUC(test) |", "|---|---|---:|---:|---:|---:|"]
    for name in contracts:
        for ctx_name, entry in report[name]["contexts"].items():
            lines.append(f"| {name} | {ctx_name} | {entry['dim']} | {entry['auc_train']:.3f} | "
                         f"{entry['auc_val']:.3f} | **{entry['auc_test']:.3f}** |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结构化结果: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
