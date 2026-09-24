"""离线后处理探针：对已存盘 predictions 做平滑 / 最小时长合并，同口径重算指标（不训练）。

**动机**：`evaluation.smoothing_min_duration` 只实现在 `sliding_window_pipeline`（因果 GRU）里；
离线 `full_sequence_temporal`（mstcn/mstcn2）的正式评测是**逐帧 argmax、零平滑**。第八轮那组
"线性探针 edit 50.63"就是在**因果口径**（窗口冷启动 + md=5 迟滞平滑）下算出的，与离线数字不可比。
本工具回答的是：**如果给离线模型补上后处理平滑，能拿多少分**——并同时给出逐帧 LDA 探针在同一套
后处理下的对照（"平滑"与"时序建模"各自值多少）。

口径声明（结果必须随此声明一起引用）：
- `offline-argmax` = 现行正式口径（`metrics.summary`）；
- `offline+median{k}` / `offline+mindur{d}` = **本工具新定义**的后处理口径，指标仍走同一个
  `framework.cleansight_eval.core.metrics.temporal_metrics`，只是输入标签流被后处理过。
- 窗口 `k` / 阈值 `d` 的单位是**采样帧**（本数据集 stride=4，≈7.5 fps，故 d=5 ≈ 0.67 s）。
  在 test 上挑最优 k/d 属于**用测试集选参**；要落地必须先在 val 上选，本工具只做机制判定。

用法：
    python tools/probe_offline_postprocess.py --run "runs/mstcn2-cap-s4l10h128/*/mstcn2-*"
    # 同时给探针对照（需要一个能解析 data/feature_schema 的 run 目录来定位特征）
    python tools/probe_offline_postprocess.py --run "runs/round16-tmse-h32-w060/*/mstcn2-*" --probe
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
sys.path.insert(0, str(REPO))

MEDIAN_K = (1, 3, 5, 7, 9, 15)
MINDUR_D = (2, 3, 5, 9, 15)


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="离线 predictions 的后处理平滑扫描（含 LDA 探针对照）")
    parser.add_argument("--run", action="append", required=True, help="run glob（可多次；相对仓库根）")
    parser.add_argument("--label", default="runs")
    parser.add_argument("--probe", action="store_true", help="额外算逐帧 LDA 探针（离线同口径）做对照")
    parser.add_argument("--probe-config-from", default=None,
                        help="用于解析 data/feature_schema 的 run glob（默认取第一个 --run 的首个 run）")
    parser.add_argument("--median-k", default=",".join(str(k) for k in MEDIAN_K))
    parser.add_argument("--mindur-d", default=",".join(str(d) for d in MINDUR_D))
    parser.add_argument("--compare-run", default=None,
                        help="模型 run glob：报『同后处理下 --run 组 vs 此 run 组』的按视频配对检验（前者通常是探针）")
    parser.add_argument("--compare-label", default="模型")
    parser.add_argument("--select-on-val", action="store_true",
                        help="在 val 上挑后处理参数、再在 test 上评（避免用 test 选参）")
    return parser.parse_args(argv)


def median_filter(sequence: np.ndarray, k: int) -> np.ndarray:
    """逐帧取 ±k//2 邻域内的多数类（奇数窗；边界夹取）。"""
    if k <= 1:
        return sequence
    half = k // 2
    padded = np.pad(sequence, (half, half), mode="edge")
    out = np.empty_like(sequence)
    for index in range(len(sequence)):
        values, counts = np.unique(padded[index:index + k], return_counts=True)
        out[index] = values[np.argmax(counts)]
    return out


def merge_short(sequence: np.ndarray, minimum: int) -> np.ndarray:
    """把长度 < minimum 的段反复并入较长的邻段，直到所有段 ≥ minimum 或只剩一段。"""
    if minimum <= 1:
        return sequence
    out = sequence.copy()
    for _ in range(256):
        segments, start = [], 0
        for index in range(1, len(out) + 1):
            if index == len(out) or out[index] != out[start]:
                segments.append((start, index, out[start]))
                start = index
        merged = False
        for position, (low, high, _label) in enumerate(segments):
            if high - low >= minimum or len(segments) == 1:
                continue
            left = segments[position - 1] if position > 0 else None
            right = segments[position + 1] if position + 1 < len(segments) else None
            if left is None:
                target = right[2]
            elif right is None:
                target = left[2]
            else:
                target = left[2] if (left[1] - left[0]) >= (right[1] - right[0]) else right[2]
            out[low:high] = target
            merged = True
            break
        if not merged:
            break
    return out


def load_runs(patterns: list[str]) -> tuple[dict, list[str]]:
    runs, labels = {}, None
    for pattern in patterns:
        for run in sorted(REPO.glob(pattern)):
            arts = sorted((run / "artifacts").glob("*.predictions.json"))
            if not arts:
                continue
            doc = json.loads(arts[-1].read_text())
            labels = [entry["name"] for entry in doc["labels"]]
            index = {name: position for position, name in enumerate(labels)}
            seed = json.loads((run / "env.json").read_text())["seed"]
            runs[seed] = {name: (np.array([index[v] for v in item["predicted_labels"]]),
                                 np.array([index[v] for v in item["truth_labels"]]))
                          for name, item in doc["items"].items()}
    return runs, labels or []


def probe_on_split(config_glob: str, split_key: str) -> tuple[dict, list[str]]:
    """在指定 split 上用 train 拟合的 LDA 逐帧预测（离线协议：无平滑、无冷启动）。"""
    from framework.cleansight_eval.core.config import load_config
    from framework.cleansight_eval.temporal.data import load_split, split_video_names

    sys.path.insert(0, str(REPO / "tools"))
    from probe_split_shift import fit_multiclass_lda

    run_dir = sorted(REPO.glob(config_glob))[0]
    cfg = load_config(run_dir / "config.resolved.json")
    data, feature_schema = cfg["data"], cfg["feature_schema"]
    n_classes = int(cfg["model"]["num_classes"])
    train_feats, train_truths, id2name = load_split(data, data["split_train"], feature_schema=feature_schema)
    weights, bias = fit_multiclass_lda(np.concatenate(train_feats).astype(np.float64),
                                       np.concatenate(train_truths), n_classes)
    feats, truths, _ = load_split(data, data[split_key], feature_schema=feature_schema)
    names = split_video_names(data, data[split_key])
    labels = [id2name[i] for i in range(n_classes)]
    out = {}
    for name, frames, truth in zip(names, feats, truths):
        scores = frames.astype(np.float64) @ weights.T + bias
        out[name] = (scores.argmax(axis=1), np.asarray(truth))
    return {0: out}, labels


def select_threshold_on_val(config_glob: str, median_ks, mindur_ds, split_key: str = "split_val"):
    """在 **val** 上挑后处理参数（避免用 test 选参），返回 (kind, value, val_edit) 与候选曲线。

    与模型侧对称：模型也是按 val 指标选 checkpoint（`train.best_metric=val_f1_0.5`）。
    """
    runs, labels = probe_on_split(config_glob, split_key)
    if not runs:
        return None, []
    base = score(runs, labels, lambda s: s)
    curve = [("argmax", 0, statistics.median([v["edit"] for v in base.values()]))]
    for k in median_ks:
        if k <= 1:
            continue
        scored = score(runs, labels, lambda s, k=k: median_filter(s, k))
        curve.append((f"median k={k}", k, statistics.median([v["edit"] for v in scored.values()])))
    for d in mindur_ds:
        scored = score(runs, labels, lambda s, d=d: merge_short(s, d))
        curve.append((f"mindur d={d}", d, statistics.median([v["edit"] for v in scored.values()])))
    kind, value, val_edit = max(curve, key=lambda item: item[2])
    return (kind, value, val_edit, curve), runs


def score(runs: dict, labels: list[str], transform) -> dict:
    from framework.cleansight_eval.core.metrics import temporal_metrics

    out = {}
    for seed, items in runs.items():
        for name, (pred, truth) in items.items():
            transformed = transform(pred)
            metrics = temporal_metrics({name: [labels[int(v)] for v in transformed]},
                                       {name: [labels[int(v)] for v in truth]}, labels)["segment"]
            out[(seed, name)] = {
                "edit": metrics["edit"] * 100,
                "f1_01": metrics["f1_at_iou"]["0.10"] * 100,
                "f1_025": metrics["f1_at_iou"]["0.25"] * 100,
                "f1_05": metrics["f1_at_iou"]["0.50"] * 100,
                "tp": metrics["details_at_iou"]["0.50"]["tp"],
                "fp": metrics["details_at_iou"]["0.50"]["fp"],
                "segments": 1 + int((transformed[1:] != transformed[:-1]).sum()) if len(transformed) else 0,
            }
    return out


def aggregate(scored: dict) -> dict:
    median = lambda key: statistics.median([v[key] for v in scored.values()])
    return {key: median(key) for key in ("edit", "f1_01", "f1_025", "f1_05", "fp", "tp", "segments")}


def paired(base: dict, candidate: dict, key: str) -> dict | None:
    diffs = [candidate[k][key] - base[k][key] for k in base
             if k in candidate and abs(candidate[k][key] - base[k][key]) > 1e-9]
    if len(diffs) < 6:
        return None
    wins = sum(1 for d in diffs if d > 0)
    return {"n": len(diffs), "win": wins, "lose": len(diffs) - wins,
            "median": statistics.median(diffs), "p": float(wilcoxon(diffs).pvalue)}


def paired_by_video(probe_scored: dict, model_scored: dict, key: str) -> dict | None:
    """探针（确定性单点）vs 模型（多 seed）：按**视频**配对，模型每个 seed 各算一次差值。"""
    probe_by_video = {video: values for (_seed, video), values in probe_scored.items()}
    diffs = [probe_by_video[video][key] - values[key]
             for (_seed, video), values in model_scored.items()
             if video in probe_by_video and abs(probe_by_video[video][key] - values[key]) > 1e-9]
    if len(diffs) < 6:
        return None
    wins = sum(1 for d in diffs if d > 0)
    return {"n": len(diffs), "win": wins, "lose": len(diffs) - wins,
            "median": statistics.median(diffs), "p": float(wilcoxon(diffs).pvalue)}


def sweep(tag: str, runs: dict, labels: list[str], median_ks, mindur_ds,
          model_scored: dict | None = None, model_label: str = "模型") -> dict:
    """跑一遍后处理扫描；给了 ``model_scored`` 时额外报"同后处理下探针 vs 模型"的配对检验。"""
    base = score(runs, labels, lambda s: s)
    stats = aggregate(base)
    print(f"\n=== {tag}（{len(runs)} seed；基线 = 离线逐帧 argmax、零平滑）===")
    print(f"  基线: edit={stats['edit']:5.2f} F1@.1={stats['f1_01']:5.2f} F1@.25={stats['f1_025']:5.2f} "
          f"F1@.5={stats['f1_05']:5.2f} tp={stats['tp']:.0f} fp={stats['fp']:.0f} 段数={stats['segments']:.0f}")
    suffix = f"   {'探针 vs ' + model_label if model_scored is not None else ''}"
    print(f"  {'后处理':14} {'edit':>7} {'F1@.1':>7} {'F1@.25':>7} {'F1@.5':>7} {'fp':>5} {'段数':>6}   "
          f"edit 配对(vs 基线){suffix}")

    def line(name: str, candidate: dict) -> None:
        stats = aggregate(candidate)
        text = (f"  {name:14} {stats['edit']:7.2f} {stats['f1_01']:7.2f} {stats['f1_025']:7.2f} "
                f"{stats['f1_05']:7.2f} {stats['fp']:5.0f} {stats['segments']:6.0f}   "
                + format_paired(paired(base, candidate, "edit")))
        if model_scored is not None:
            text += "   " + format_paired(paired_by_video(candidate, model_scored, "edit"))
        print(text)

    for k in median_ks:
        if k > 1:
            line(f"median k={k}", score(runs, labels, lambda s, k=k: median_filter(s, k)))
    for d in mindur_ds:
        line(f"mindur d={d}", score(runs, labels, lambda s, d=d: merge_short(s, d)))
    return base


def format_paired(outcome: dict | None) -> str:
    if not outcome:
        return "样本不足"
    flag = "  <<< 显著" if outcome["p"] < 0.05 else ""
    return (f"n={outcome['n']:2} 胜/负={outcome['win']}/{outcome['lose']} "
            f"中位={outcome['median']:+6.2f} p={outcome['p']:.4f}{flag}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    median_ks = [int(v) for v in args.median_k.split(",") if v.strip()]
    mindur_ds = [int(v) for v in args.mindur_d.split(",") if v.strip()]

    runs, labels = load_runs(args.run)
    if not runs:
        print("[probe] 未匹配到含 predictions 的 run")
        return 2
    model_scored = None
    if args.compare_run:
        model_runs, _ = load_runs([args.compare_run])
        model_scored = score(model_runs, labels, lambda s: s)
    sweep(args.label, runs, labels, median_ks, mindur_ds,
          model_scored=model_scored, model_label=args.compare_label)

    if args.probe:
        config_glob = args.probe_config_from or args.run[0]
        probe, probe_labels = probe_on_split(config_glob, "split_eval")
        sweep("逐帧 LDA 探针（离线同口径）", probe, probe_labels, median_ks, mindur_ds,
              model_scored=model_scored, model_label=args.compare_label)

    if args.select_on_val:
        config_glob = args.probe_config_from or args.run[0]
        selection, _ = select_threshold_on_val(config_glob, median_ks, mindur_ds)
        if selection is None:
            print("\n[probe] val 选参失败（无数据）")
            return 2
        kind, value, val_edit, curve = selection
        print(f"\n=== 正当协议：后处理参数在 **val** 上选（与模型按 val 选 checkpoint 对称）===")
        print(f"  val 上的候选曲线（逐视频 edit 中位）: " +
              "、".join(f"{name}={edit:.2f}" for name, _v, edit in curve))
        print(f"  → val 最优：{kind}（val edit {val_edit:.2f}）")
        probe, probe_labels = probe_on_split(config_glob, "split_eval")
        transform = (lambda s, k=value: median_filter(s, k)) if kind.startswith("median") \
            else (lambda s, d=value: merge_short(s, d))
        chosen = score(probe, probe_labels, transform)
        if model_scored is not None:
            for key in ("edit", "f1_01", "f1_025"):
                outcome = paired_by_video(chosen, model_scored, key)
                stats = statistics.median([v[key] for v in chosen.values()])
                model_median = statistics.median([v[key] for v in model_scored.values()])
                print(f"  test 指标 {key:8} 探针={stats:6.2f} vs {args.compare_label}={model_median:6.2f}   "
                      + format_paired(outcome))
        else:
            stats = aggregate(chosen)
            print(f"  test: edit={stats['edit']:.2f} F1@.1={stats['f1_01']:.2f} F1@.25={stats['f1_025']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
