"""两组 run 的官方指标汇总 + 逐 (seed, 视频) × 类配对检验（读 run 产物，不训练）。

用途：任何"A 方案 vs B 方案"的判读都必须落在**逐 (seed, 视频)** 的配对检验上，否则会把
seed / 视频混合的假象当成方案效应（本仓库第十三~十五轮的反例）。本工具把该口径固化成一条命令。

口径：
- **汇总表**读 `runs/*/*/evals/*.evaluation.json` 的 `metrics.summary`（官方 percent，2 位小数），
  取 seed 中位数（与 `docs/EVAL.md` §3.4 的注册表口径一致）；
- **配对检验**从 `artifacts/*.predictions.json` 重算逐视频值，指标走同一个
  `framework.cleansight_eval.core.metrics.temporal_metrics`；配对键 = `(seed, 视频)`（× 类），
  用 Wilcoxon 符号秩检验，零差值的对剔除；
- **段数比** = 每 run 预测总段数 / 真值总段数，再取 seed 中位（与第十四~十六轮同法）;
- **非 idle 帧** = 该 run 全部视频的预测非 idle 帧数之和（防止"靠压掉非 idle 换 edit"的假增益）。

用法：
    # 单 glob（引号防 shell 展开；支持多个 glob 合并成一组）
    python tools/compare_runs.py --left "runs/seedext-h32/h32/mstcn-*" \\
        --left "runs/capacity-lr0005-e60/h32/mstcn-*" --left-label "mstcn h32" \\
        --right "runs/mstcn2-cap-s2l5h32/h32/mstcn2-*" --right-label "mstcn2 s2l5 h32"
    # 只看部分指标 / 落盘 JSON
    python tools/compare_runs.py ... --metrics edit,f1_01,insert --json tmp/cmp.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # 仓库根：import framework（与本目录其他 run 分析工具一致）

# 指标名 → (官方 summary 键, 逐视频重算方式)
SUMMARY_KEYS = {
    "acc": "acc",
    "edit": "edit",
    "f1_01": "f1@0.1",
    "f1_025": "f1@0.25",
    "f1_05": "f1@0.5",
    "tp": "tp@0.5",
    "fp": "fp@0.5",
    "fn": "fn@0.5",
}
CLASS_KEYS = ("flush", "long_brush_insert", "long_brush_withdraw", "short_brush_cleaning")
CLASS_ALIASES = {"insert": "long_brush_insert", "withdraw": "long_brush_withdraw", "sbc": "short_brush_cleaning"}
PRECISION_SUFFIX = ":precision"
DEFAULT_METRICS = ("edit", "f1_01", "f1_025", "insert", "insert:precision", "withdraw", "flush", "sbc")


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="两组 run 的官方汇总 + 逐 (seed,视频)×类配对检验")
    parser.add_argument("--left", action="append", required=True, help="左侧 run glob（可多次，合并为一组）")
    parser.add_argument("--right", action="append", required=True, help="右侧 run glob（可多次，合并为一组）")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help=f"逗号分隔；可选 {','.join(list(SUMMARY_KEYS) + list(CLASS_ALIASES))}")
    parser.add_argument("--json", default=None, help="把汇总与配对结果写到此 JSON")
    parser.add_argument("--left-eval-dir", action="append", default=None, metavar="GLOB",
                        help="左侧评估产物的口径变体目录（如 runs/X/_eval_md5/*）；预测按同名 stem 从 run 取")
    parser.add_argument("--right-eval-dir", action="append", default=None, metavar="GLOB",
                        help="右侧评估产物的口径变体目录")
    parser.add_argument("--per-class", action="store_true",
                        help="额外打印逐类帧级 precision/recall/F1（官方 micro-pool 口径，各 run 中位）")
    return parser.parse_args(argv)


def segment_count(sequence) -> int:
    array = np.asarray(sequence)
    return 0 if array.size == 0 else int(1 + (array[1:] != array[:-1]).sum())


def collect(patterns: list[str], eval_dirs: list[str] | None = None) -> tuple[list[dict], dict]:
    """→ (每个 run 的官方指标行, {seed: {video: 逐视频重算指标}})

    ``eval_dirs`` 给定时，评估产物从**口径变体目录**读（`tools/run_strategy_matrix.py` 产出的
    ``_eval_md<N>/<run-name>/*.evaluation.json``）：预测按**同名 stem** 从对应 run 的
    ``artifacts/`` 取，保证官方口径与逐视频重算同源（否则配对检验会与汇总表口径打架）。
    """
    from framework.cleansight_eval.core.metrics import temporal_metrics

    rows, per_video = [], {}
    seen = set()

    def handle(run: Path, eval_path: Path, artifact_path: Path | None) -> None:
        if run in seen or artifact_path is None:
            return
        seen.add(run)
        seed = json.loads((run / "env.json").read_text())["seed"]
        document = json.loads(eval_path.read_text())
        summary = document["metrics"]["summary"]
        row = {"seed": seed, "run": str(run)}
        for key_name, key in SUMMARY_KEYS.items():
            entry = summary.get(key)
            row[key_name] = entry["value"] if entry and entry.get("state") == "computed" else None
        details = (document["metrics"].get("details") or {}).get("temporal") or {}
        pooled = ((details.get("frame") or {}).get("per_class") or {})
        row["_pooled"] = {
            cls: {"precision": (values.get("precision") or 0.0) * 100,
                  "recall": (values.get("recall") or 0.0) * 100,
                  "f1": (values.get("f1") or 0.0) * 100,
                  "support": values.get("support")}
            for cls, values in pooled.items()
        }
        doc = json.loads(artifact_path.read_text())
        labels = [entry["name"] for entry in doc["labels"]]
        nonidle = predicted_segments = truth_segments = 0
        per_video.setdefault(seed, {})
        for name, item in doc["items"].items():
            pred = np.array(item["predicted_labels"])
            truth = np.array(item["truth_labels"])
            metrics = temporal_metrics({name: pred.tolist()}, {name: truth.tolist()}, labels)["segment"]
            nonidle += int((pred != "idle").sum())
            predicted_segments += segment_count(pred)
            truth_segments += segment_count(truth)
            record = {"edit": metrics["edit"] * 100,
                      "f1_01": metrics["f1_at_iou"]["0.10"] * 100,
                      "f1_025": metrics["f1_at_iou"]["0.25"] * 100,
                      "f1_05": metrics["f1_at_iou"]["0.50"] * 100}
            for cls in CLASS_KEYS:
                mask = truth == cls
                predicted = pred == cls
                record[cls] = float(predicted[mask].mean() * 100) if mask.any() else None
                record[f"{cls}{PRECISION_SUFFIX}"] = \
                    float((truth[predicted] == cls).mean() * 100) if predicted.any() else None
            per_video[seed][name] = record
        row["nonidle"] = nonidle
        row["seg_ratio"] = predicted_segments / max(truth_segments, 1)
        rows.append(row)

    if eval_dirs:
        for pattern in eval_dirs:
            if Path(pattern).is_absolute():
                raise SystemExit(f"--left-eval-dir/--right-eval-dir 请用相对仓库根的 glob：{pattern}")
            for variant in sorted(REPO.glob(pattern)):
                run = variant.parent.parent / variant.name
                if not (run / "env.json").is_file():
                    continue
                for evaluation in sorted(variant.glob("*.evaluation.json")):
                    stem = evaluation.name.replace(".evaluation.json", "")
                    artifact = run / "artifacts" / f"{stem}.predictions.json"
                    handle(run, evaluation, artifact if artifact.is_file() else None)
        return rows, per_video

    for pattern in patterns:
        if Path(pattern).is_absolute():
            raise SystemExit(f"--left/--right 请用相对仓库根的 glob（便于复现）：{pattern}")
        for run in sorted(REPO.glob(pattern)):
            evals = sorted((run / "evals").glob("*.evaluation.json"))
            arts = sorted((run / "artifacts").glob("*.predictions.json"))
            if not evals or not arts:
                continue
            handle(run, evals[-1], arts[-1])
    return rows, per_video


def median_of(rows: list[dict], key: str):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(values) if values else None


def paired(left: dict, right: dict, key: str) -> dict | None:
    """逐 (seed, 视频) 配对：left − right。"""
    diffs = [left[s][n][key] - right[s][n][key]
             for s in left if s in right for n in left[s]
             if left[s][n].get(key) is not None and right[s].get(n, {}).get(key) is not None
             and abs(left[s][n][key] - right[s][n][key]) > 1e-9]
    if len(diffs) < 6:
        return {"n": len(diffs), "insufficient": True}
    wins = sum(1 for d in diffs if d > 0)
    return {"n": len(diffs), "win": wins, "lose": len(diffs) - wins,
            "median": statistics.median(diffs), "p": float(wilcoxon(diffs).pvalue)}


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    left_rows, left_pairs = collect(args.left, args.left_eval_dir)
    right_rows, right_pairs = collect(args.right, args.right_eval_dir)
    if not left_rows or not right_rows:
        print(f"[compare] 数据不足：left {len(left_rows)} run / right {len(right_rows)} run")
        return 2

    print(f"=== {args.left_label}（{len(left_rows)} run）===")
    print(f"=== {args.right_label}（{len(right_rows)} run）===")
    print()
    header = f"{'指标':10} {args.left_label[:26]:>26} {args.right_label[:26]:>26}"
    print(header)
    print("-" * len(header))
    for key in ("acc", "edit", "f1_01", "f1_025", "f1_05", "tp", "fp", "fn", "seg_ratio", "nonidle"):
        lv, rv = median_of(left_rows, key), median_of(right_rows, key)
        if lv is None and rv is None:
            continue
        fmt = lambda v: "—" if v is None else (f"{v:.2f}" if isinstance(v, float) and v < 1e4 else f"{v:.0f}")
        print(f"{key:10} {fmt(lv):>26} {fmt(rv):>26}")

    print(f"\n=== 配对检验（逐 (seed,视频)×类 Wilcoxon；差值 = {args.left_label} − {args.right_label}）===")
    requested = [m.strip() for m in args.metrics.split(",") if m.strip()]
    results = {}
    for metric in requested:
        base_metric, _, qualifier = metric.partition(":")
        key = CLASS_ALIASES.get(base_metric, base_metric)
        if qualifier == "precision":
            key = f"{key}{PRECISION_SUFFIX}"
        elif qualifier:
            print(f"  {metric:22} 未知限定符 :{qualifier}（只支持 :precision）")
            continue
        valid = key in SUMMARY_KEYS or key in CLASS_KEYS or key.endswith(PRECISION_SUFFIX)
        if not valid:
            print(f"  {metric:22} 未知指标，跳过")
            continue
        outcome = paired(left_pairs, right_pairs, key)
        results[metric] = outcome
        if outcome is None or outcome.get("insufficient"):
            print(f"  {metric:22} n={outcome['n'] if outcome else 0:2} 样本不足")
            continue
        flag = "  <<< 显著" if outcome["p"] < 0.05 else ""
        print(f"  {metric:22} n={outcome['n']:2} 胜/负={outcome['win']:3}/{outcome['lose']:<3} "
              f"中位差={outcome['median']:+6.2f}pp p={outcome['p']:.4f}{flag}")

    if args.per_class:
        print(f"\n=== 逐类帧级指标（官方 micro-pool 口径，各 run 中位；支持帧数见 support）===")
        print(f"{'类':24} {args.left_label[:18]:>20} {args.right_label[:18]:>20}")
        print(f"{'':24} {'P / R / F1':>20} {'P / R / F1':>20}")
        labels = list(left_rows[0].get("_pooled", {})) or list(right_rows[0].get("_pooled", {}))
        for cls in labels:
            cells = []
            for rows in (left_rows, right_rows):
                values = [r["_pooled"][cls] for r in rows if r.get("_pooled", {}).get(cls)]
                if not values:
                    cells.append("—")
                    continue
                med = lambda field: statistics.median([v[field] for v in values])
                cells.append(f"{med('precision'):5.1f}/{med('recall'):5.1f}/{med('f1'):5.1f}")
            support = next((r["_pooled"][cls].get("support") for r in left_rows
                            if r.get("_pooled", {}).get(cls)), None)
            print(f"{cls + (f' (n={support})' if support else ''):24} {cells[0]:>20} {cells[1]:>20}")
        per_class_results = {cls: {
            "left": {field: statistics.median([r["_pooled"][cls][field] for r in left_rows
                                               if r.get("_pooled", {}).get(cls)]) for field in ("precision", "recall", "f1")},
            "right": {field: statistics.median([r["_pooled"][cls][field] for r in right_rows
                                                if r.get("_pooled", {}).get(cls)]) for field in ("precision", "recall", "f1")},
        } for cls in labels}
    else:
        per_class_results = None

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "left": {"label": args.left_label, "patterns": args.left,
                     "n_runs": len(left_rows), "seeds": sorted(r["seed"] for r in left_rows),
                     "medians": {k: median_of(left_rows, k) for k in (*SUMMARY_KEYS, "seg_ratio", "nonidle")}},
            "right": {"label": args.right_label, "patterns": args.right,
                      "n_runs": len(right_rows), "seeds": sorted(r["seed"] for r in right_rows),
                      "medians": {k: median_of(right_rows, k) for k in (*SUMMARY_KEYS, "seg_ratio", "nonidle")}},
            "paired": results,
        }, ensure_ascii=False, indent=1))
        print(f"\n[compare] 结果已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
