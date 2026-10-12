"""选点口径的 val→test 迁移体检：候选 best_metric 到底能不能预测 test 表现（读产物，不训练）。

动机：`train.best_metric` 决定 best.pt 取哪一轮，而它只用 val 指标选——**如果该 val 指标与 test 表现
不相关，选点就近乎随机**。本工具在全部已有 run 上量化这件事：val 峰值 vs 该 run 的 test 指标，
给出偏移（val−test 中位差）与两种相关系数（Pearson / Spearman；后者对秩更稳健）。

配套事实（同一批量统计里可见）：`val_loss` 的最小值几乎总在第 3~9 轮出现（之后一路回升），
因此**用 val_loss 选点等于选"最早的最低点"**，与段级目标无关。

用法：
    python tools/probe_selection_transfer.py                       # 扫 runs/**
    python tools/probe_selection_transfer.py --runs-dir runs/mstcn2-cap-s4l10h128 --json tmp/sel.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PAIRS = (("edit", "val_edit", "edit"),
         ("f1@0.1", "val_f1_0.1", "f1@0.1"),
         ("f1@0.25", "val_f1_0.25", "f1@0.25"),
         ("f1@0.5", "val_f1_0.5", "f1@0.5"),
         ("acc", "val_acc", "acc"),
         ("loss", "val_loss", "edit"))  # loss 越小越好，test 侧取 edit 仅作对照


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="选点口径的 val→test 迁移体检（不训练）")
    parser.add_argument("--runs-dir", default="runs", help="扫描根目录（默认 runs）")
    parser.add_argument("--json", default=None, help="结果写到此 JSON")
    return parser.parse_args(argv)


def best_of(values: list[float], key: str) -> tuple[int, float]:
    """→ (1-based 最优轮次, 最优值)；val_loss 取最小，其余取最大。"""
    if key == "val_loss":
        index = min(range(len(values)), key=lambda i: values[i])
    else:
        index = max(range(len(values)), key=lambda i: values[i])
    return index + 1, values[index]


def collect(root: Path) -> tuple[dict, list[dict]]:
    """→ ({指标: [(val 峰值, test 值), ...]}, 逐 run 明细)"""
    buckets: dict[str, list[tuple[float, float]]] = {tag: [] for tag, _v, _t in PAIRS}
    details: list[dict] = []
    for history in sorted(root.rglob("history.csv")):
        run = history.parent
        evals = sorted((run / "evals").glob("*.evaluation.json"))
        if not evals:
            continue
        rows = list(csv.DictReader(history.open(encoding="utf-8")))
        if not rows:
            continue
        summary = json.loads(evals[-1].read_text())["metrics"]["summary"]
        record = {"run": str(run)}
        for tag, val_key, test_key in PAIRS:
            if val_key not in rows[0]:
                continue
            if summary.get(test_key, {}).get("state") != "computed":
                continue
            values = [float(row[val_key]) for row in rows]
            epoch, best = best_of(values, val_key)
            record[tag] = {"epoch": epoch, "val": best, "test": summary[test_key]["value"]}
            if tag != "loss":
                buckets[tag].append((best, summary[test_key]["value"]))
        if len(record) > 1:  # 至少配上一项可评估指标才算有效 run（否则打印的 run 数会虚高）
            details.append(record)
    return buckets, details


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    buckets, details = collect(Path(args.runs_dir))
    if not details:
        print("[probe] 未找到含 history.csv + evaluation.json 的 run")
        return 2

    print(f"扫描 {len(details)} 个 run（{args.runs_dir}）")
    print(f"\n{'指标':10} {'n':>4} {'val 中位':>9} {'test 中位':>10} {'val−test':>9} {'Pearson r':>10} {'Spearman ρ':>11}")
    results = {}
    try:
        from scipy.stats import spearmanr
    except ImportError:  # pragma: no cover - scipy 是本仓库既有依赖
        spearmanr = None
    for tag, val_key, test_key in PAIRS:
        pairs = buckets.get(tag) or []
        if len(pairs) < 5 or tag == "loss":
            continue
        val = np.array([p[0] for p in pairs])
        test = np.array([p[1] for p in pairs])
        rho = float(spearmanr(val, test).statistic) if spearmanr else float("nan")
        r = float(np.corrcoef(val, test)[0, 1])
        results[tag] = {"n": len(pairs), "val_median": statistics.median(val),
                        "test_median": statistics.median(test),
                        "gap_median": statistics.median(val - test),
                        "pearson": r, "spearman": rho}
        print(f"{tag:10} {len(pairs):4} {statistics.median(val):9.2f} {statistics.median(test):10.2f} "
              f"{statistics.median(val - test):9.2f} {r:10.3f} {rho:11.3f}")

    epochs = {tag: [d[tag]["epoch"] for d in details if tag in d] for tag in ("edit", "f1@0.25", "f1@0.5", "loss")}
    print("\n选点轮次分布（跨 run 摆幅越大 = 该口径越不稳）：")
    for tag, values in epochs.items():
        if values:
            print(f"  {tag:10} 中位={statistics.median(values):4.0f}  最小={min(values):3} 最大={max(values):3} "
                  f"摆幅={max(values) - min(values):3}")

    print("\n判读：Spearman ρ 高 + val−test 偏移稳定，才说明该口径能用来选点；")
    print("      `loss` 行的 test 列固定为 edit，只用于对照'早停轮次'与段级目标无关。")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"runs": len(details), "transfer": results, "selection_epochs": epochs,
                                    "details": details}, ensure_ascii=False, indent=1))
        print(f"\n[probe] 结果已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
