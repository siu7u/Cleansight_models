"""多种子预测集成探针：把同一配置多个 seed 的逐帧预测做**多数投票**，再同口径评估（读产物，不训练）。

动机：本仓库所有结论都建立在"单 seed 训练 + 逐 seed 对比"上，但**集成**（ensemble）是工业界最常用的
免费提分手段之一，此前从未测过。逐帧标签的天然集成方式是**多数投票**（同一视频、同一帧位置、
N 个 seed 的预测里取多数类）；段级指标（edit / f1@IoU）是否因此改善是可验证的。

口径与公平性：
- 输入是 `artifacts/*.predictions.json`（与正式评测同源）；集成是**确定性**的，无超参可调（除成员集合）；
- 对照必须是**成员自身的平均**（而不是"最好的单 seed"——那属于用 test 选点，是泄漏）；
- 统计用**逐视频配对** Wilcoxon（n = 视频数），报中位差与胜/负；
- 投票平局规则显式声明：票数相同时取**标签表顺序靠前者**（`labels` 由产物给出，顺序即数据契约）。

用法：
    python tools/probe_seed_ensemble.py --run "runs/mstcn2-cap-s4l10h128/*/mstcn2-*" \
        --run "runs/round22-seedext-default/h128/mstcn2-*" --label "旗舰 8 seed"
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CLASSES = ("flush", "long_brush_insert", "long_brush_withdraw", "short_brush_cleaning")
ENSEMBLE_SIZES = (2, 3, 4, 8)


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="多种子预测集成的逐帧多数投票评估（不训练）")
    parser.add_argument("--run", action="append", required=True, help="run glob（可多次，合并为一个成员池）")
    parser.add_argument("--label", default="runs")
    parser.add_argument("--max-combos", type=int, default=70, help="每个 k 最多枚举多少种成员组合")
    parser.add_argument("--merge-d", type=int, default=0,
                        help="投票后再做最小时长合并（帧数，0=不做）；取值应**先验**给定（本仓库惯例 md=5）")
    parser.add_argument("--json", default=None)
    return parser.parse_args(argv)


def majority_vote(stack: np.ndarray, n_labels: int) -> np.ndarray:
    """``[N, T]`` 逐帧标签 → ``[T]`` 多数类（平局取标签 id 较小者）。"""
    votes = np.zeros((len(stack[0]), n_labels), dtype=np.int32)
    for row in stack:
        votes[np.arange(len(row)), row] += 1
    return votes.argmax(axis=1)  # argmax 平局取最小 id → 与文档声明的规则一致


def load_members(patterns: Sequence[str]) -> tuple[dict[str, dict[int, np.ndarray]], list[str], dict[int, str]]:
    """→ ({video: {seed: 标签序列}}, labels, {seed: run 名})"""
    members: dict[str, dict[int, np.ndarray]] = {}
    seed_of_run: dict[int, str] = {}
    labels: list[str] = []
    for pattern in patterns:
        for run in sorted(REPO.glob(pattern)):
            arts = sorted((run / "artifacts").glob("*.predictions.json"))
            if not arts:
                continue
            doc = json.loads(arts[-1].read_text())
            labels = [entry["name"] for entry in doc["labels"]]
            index = {name: position for position, name in enumerate(labels)}
            seed = json.loads((run / "env.json").read_text())["seed"]
            if seed in seed_of_run:
                print(f"[probe] 跳过 seed={seed} 的重复 run：{run.name}（已有 {seed_of_run[seed]}）")
                continue
            seed_of_run[seed] = run.name
            for video, item in doc["items"].items():
                members.setdefault(video, {})[seed] = np.array([index[v] for v in item["predicted_labels"]])
    return members, labels, seed_of_run


def evaluate(predictions: dict[str, np.ndarray], truth: dict[str, np.ndarray], labels: list[str]) -> dict:
    """逐视频计算指标（返回 {video: {...}}）。"""
    from framework.cleansight_eval.core.metrics import temporal_metrics

    out = {}
    for video, predicted in predictions.items():
        metrics = temporal_metrics({video: [labels[int(v)] for v in predicted]},
                                   {video: [labels[int(v)] for v in truth[video]]}, labels)["segment"]
        record = {"edit": metrics["edit"] * 100,
                  "f1_01": metrics["f1_at_iou"]["0.10"] * 100,
                  "f1_025": metrics["f1_at_iou"]["0.25"] * 100}
        truth_labels = truth[video]
        for cls in CLASSES:
            position = labels.index(cls) if cls in labels else -1
            mask = truth_labels == position
            record[cls] = float((predicted[mask] == position).mean() * 100) if position >= 0 and mask.any() else None
        out[video] = record
    return out


def paired(base: dict, candidate: dict, key: str) -> dict | None:
    diffs = [candidate[v][key] - base[v][key] for v in base
             if v in candidate and base[v].get(key) is not None and candidate[v].get(key) is not None
             and abs(candidate[v][key] - base[v][key]) > 1e-9]
    if len(diffs) < 6:
        return None
    wins = sum(1 for d in diffs if d > 0)
    return {"n": len(diffs), "win": wins, "lose": len(diffs) - wins,
            "median": statistics.median(diffs), "p": float(wilcoxon(diffs).pvalue)}


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    merge = None
    if args.merge_d and args.merge_d > 1:
        sys.path.insert(0, str(REPO / "tools"))
        from probe_offline_postprocess import merge_short
        merge = lambda sequence: merge_short(sequence, args.merge_d)
    members, labels, seed_of_run = load_members(args.run)
    if not members:
        print("[probe] 未匹配到含 predictions 的 run")
        return 2
    seeds = sorted(next(iter(members.values())))
    videos = sorted(members)
    truth = {}
    for pattern in args.run:  # 真值随产物给出
        for run in sorted(REPO.glob(pattern)):
            arts = sorted((run / "artifacts").glob("*.predictions.json"))
            if not arts:
                continue
            doc = json.loads(arts[-1].read_text())
            index = {entry["name"]: position for position, entry in enumerate(doc["labels"])}
            for video, item in doc["items"].items():
                truth[video] = np.array([index[v] for v in item["truth_labels"]])
            break
        if truth:
            break

    suffix = f"；投票后合并 d={args.merge_d}" if merge else ""
    print(f"[probe] {args.label}：{len(seeds)} 个 seed（{seeds}）× {len(videos)} 个视频{suffix}")
    singles = evaluate({v: members[v][s] for v in videos for s in seeds[:1]}, truth, labels) if seeds else {}
    per_seed = {}
    for seed in seeds:
        per_seed[seed] = evaluate({v: members[v][seed] for v in videos}, truth, labels)

    header = (f"{'k':>3} {'组合':>5} | {'edit':>7} {'F1@.1':>7} {'F1@.25':>7} | "
              f"{'成员 edit':>9} {'成员 F1.25':>10} | {'Δedit':>7} {'p':>7} {'ΔF1.25':>7} {'p':>7}")
    print("\n" + header)
    print("-" * len(header))
    results = {}
    for k in ENSEMBLE_SIZES:
        if k > len(seeds):
            continue
        combos = list(itertools.combinations(seeds, k))
        if len(combos) > args.max_combos:
            step = len(combos) / args.max_combos
            combos = [combos[int(i * step)] for i in range(args.max_combos)]
        medians = {key: [] for key in ("edit", "f1_01", "f1_025")}
        member_medians = {key: [] for key in ("edit", "f1_01", "f1_025")}
        deltas = []
        for combo in combos:
            ensemble = {}
            for v in videos:
                voted = majority_vote(np.stack([members[v][s] for s in combo]), len(labels))
                ensemble[v] = merge(voted) if merge else voted
            scored = evaluate(ensemble, truth, labels)
            member_scored = {v: {key: statistics.mean(per_seed[s][v][key] for s in combo)
                                 for key in ("edit", "f1_01", "f1_025")} for v in videos}
            for key in medians:
                medians[key].append(statistics.median(x[key] for x in scored.values()))
                member_medians[key].append(statistics.median(x[key] for x in member_scored.values()))
            deltas.append({key: paired(member_scored, scored, key) for key in ("edit", "f1_01", "f1_025")})
        pooled = {key: statistics.median(values) for key, values in medians.items()}
        member = {key: statistics.median(member_medians[key]) for key in member_medians}
        summary = {}
        for key in ("edit", "f1_01", "f1_025"):
            usable = [d[key] for d in deltas if d.get(key)]
            summary[key] = {"delta": statistics.median(d["median"] for d in usable) if usable else float("nan"),
                            "p": statistics.median(d["p"] for d in usable) if usable else float("nan"),
                            "wins": sum(d["win"] for d in usable), "loses": sum(d["lose"] for d in usable)}
        print(f"{k:>3} {len(combos):>5} | {pooled['edit']:>7.2f} {pooled['f1_01']:>7.2f} {pooled['f1_025']:>7.2f} | "
              f"{member['edit']:>9.2f} {member['f1_025']:>10.2f} | {summary['edit']['delta']:>+7.2f} "
              f"{summary['edit']['p']:>7.4f} {summary['f1_025']['delta']:>+7.2f} {summary['f1_025']['p']:>7.4f}")
        results[k] = {"combos": len(combos), **{key: pooled[key] for key in pooled},
                      "member": member, **{f"{key}_{k2}": v for key, s in summary.items() for k2, v in s.items()}}

    single_median = {key: statistics.median(x[key] for x in singles.values()) for key in ("edit", "f1_01", "f1_025")} \
        if singles else {}
    print(f"\n单 seed（第 1 个成员）中位：edit={single_median.get('edit', float('nan')):.2f} "
          f"F1@.1={single_median.get('f1_01', float('nan')):.2f} F1@.25={single_median.get('f1_025', float('nan')):.2f}")
    print("（对照口径：'成员均值' 是同一批成员的**平均**，不是'最好的单 seed'——后者用 test 选点属泄漏）")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"label": args.label, "seeds": seeds, "videos": len(videos),
                                    "single_seed": single_median, "ensembles": results},
                                   ensure_ascii=False, indent=1))
        print(f"\n[probe] 结果已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
