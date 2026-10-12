"""段级误差分解探针：把"漏掉的段"拆成 **标签错** 与 **边界错**（读已存盘 predictions，不训练）。

动机：`metrics.details.temporal.segment.details_at_iou[*].boundary_mae` **只在已匹配（tp）的段上统计**，
是被 IoU 阈值筛选过的有偏子集——它不能回答"模型典型边界误差有多大"，更不能回答
"段级 F1 上不去，是标签认错了还是边界框歪了"。本探针在**全部真值段**上做无偏分解：

对每个真值段（同一标签的极大连续帧段）：
1. 该段内预测的**多数标签** == 真值标签 → 记为「标签正确」；否则「标签错」（无论边界多准都不可能匹配）；
2. 标签正确的段里，取**同类预测段**中与该真值段重叠最大者，算 IoU 与首尾偏移；
3. 于是 ``fn@IoU`` = 标签错的段 + 标签对但 IoU 不达标的段（**边界错**）。

同时给出真值段**时长分布**（段级匹配的结构上限：时长 L 的段在 IoU t 下最多容忍
``e = L(1−t)/2(1+t)`` 帧的边界误差），以及按类明细——用于判断"该修标签/边界，还是该修检出"。

口径：输入是 `artifacts/*.predictions.json` 的逐帧标签（与正式评测同源），帧单位同采样序列。
用法：
    python tools/probe_boundary_error.py --run "runs/mstcn2-cap-s4l10h128/*/mstcn2-*"
    python tools/probe_boundary_error.py --run "<glob>" --json tmp/boundary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

THRESHOLDS = (0.25, 0.50)


BOUNDARY_BUCKETS = ((0, 1), (2, 3), (4, 7), (8, 15), (16, None))


def boundary_distance_profile(pred: np.ndarray, truth: np.ndarray) -> list[tuple[int, bool]]:
    """→ 每帧 ``(到最近真值标签切换点的距离, 该帧是否预测正确)``。

    距离只针对**标签切换点**（相邻帧标签不同处，距离 0）；视频首尾不是标签边界，
    因此整段无切换时距离取序列长度（落进最远的桶）。
    """
    transitions = [index for index in range(1, len(truth)) if truth[index] != truth[index - 1]]
    profile = []
    for index in range(len(truth)):
        if transitions:
            position = int(np.searchsorted(transitions, index))
            candidates = []
            if position > 0:
                candidates.append(index - transitions[position - 1])
            if position < len(transitions):
                candidates.append(transitions[position] - index)
            distance = min(candidates)
        else:
            distance = len(truth)
        profile.append((distance, bool(pred[index] == truth[index])))
    return profile


def bucket_of(distance: int) -> str:
    for low, high in BOUNDARY_BUCKETS:
        if distance >= low and (high is None or distance <= high):
            return f"{low}-{high}" if high is not None else f"{low}+"
    return f"{BOUNDARY_BUCKETS[-1][0]}+"


def profile_summary(profile: list[tuple[int, bool]]) -> dict:
    """按距离分桶统计：帧数、错误数、错误率、以及"占总错误的比例"。"""
    buckets: dict[str, dict] = {}
    total_errors = sum(1 for _d, correct in profile if not correct)
    for distance, correct in profile:
        name = bucket_of(distance)
        entry = buckets.setdefault(name, {"frames": 0, "errors": 0})
        entry["frames"] += 1
        entry["errors"] += 0 if correct else 1
    for entry in buckets.values():
        entry["error_rate"] = entry["errors"] / entry["frames"] * 100 if entry["frames"] else 0.0
        entry["share_of_errors"] = entry["errors"] / total_errors * 100 if total_errors else 0.0
    return {"buckets": buckets, "total_frames": len(profile), "total_errors": total_errors}


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="段级误差分解：标签错 vs 边界错（无偏，全部真值段）")
    parser.add_argument("--run", action="append", required=True, help="run glob（可多次；相对仓库根）")
    parser.add_argument("--label", default="runs")
    parser.add_argument("--json", default=None)
    return parser.parse_args(argv)


def segments(sequence: np.ndarray) -> list[tuple[int, int, int]]:
    """→ [(label, start, end_exclusive)]，按时间顺序。"""
    out, start = [], 0
    for index in range(1, len(sequence) + 1):
        if index == len(sequence) or sequence[index] != sequence[start]:
            out.append((int(sequence[start]), start, index))
            start = index
    return out


def analyse_video(pred: np.ndarray, truth: np.ndarray, labels: list[str]) -> list[dict]:
    """对一条视频的全部真值段做分解，返回逐段记录。"""
    predicted_segments = segments(pred)
    records = []
    for label_id, start, end in segments(truth):
        span = pred[start:end]
        dominant = int(np.bincount(span, minlength=len(labels)).argmax())
        record = {
            "class": labels[label_id],
            "length": end - start,
            "label_correct": dominant == label_id,
            "iou": 0.0,
            "start_offset": None,
            "end_offset": None,
        }
        if record["label_correct"]:
            best = 0.0
            for _plabel, pstart, pend in predicted_segments:
                if _plabel != label_id:
                    continue
                overlap = min(end, pend) - max(start, pstart)
                if overlap <= 0:
                    continue
                union = max(end, pend) - min(start, pstart)
                iou = overlap / union
                if iou > best:
                    best = iou
                    record["start_offset"] = pstart - start
                    record["end_offset"] = pend - end
            record["iou"] = best
        records.append(record)
    return records


def collect(patterns: list[str]) -> tuple[list[dict], dict, list[tuple[int, bool]]]:
    """→ (逐段记录汇总, 运行统计, 逐帧边界距离画像)"""
    all_records, seen = [], {"runs": 0, "videos": 0}
    profile: list[tuple[int, bool]] = []
    for pattern in patterns:
        for run in sorted(REPO.glob(pattern)):
            arts = sorted((run / "artifacts").glob("*.predictions.json"))
            if not arts:
                continue
            doc = json.loads(arts[-1].read_text())
            labels = [entry["name"] for entry in doc["labels"]]
            index = {name: position for position, name in enumerate(labels)}
            seen["runs"] += 1
            for item in doc["items"].values():
                pred = np.array([index[v] for v in item["predicted_labels"]])
                truth = np.array([index[v] for v in item["truth_labels"]])
                all_records.extend(analyse_video(pred, truth, labels))
                profile.extend(boundary_distance_profile(pred, truth))
                seen["videos"] += 1
    return all_records, seen, profile


def summarise(records: list[dict], labels: list[str]) -> dict:
    total = len(records)
    label_wrong = [r for r in records if not r["label_correct"]]
    label_ok = [r for r in records if r["label_correct"]]
    summary = {
        "truth_segments": total,
        "label_wrong": len(label_wrong),
        "label_ok": len(label_ok),
    }
    for threshold in THRESHOLDS:
        matched = [r for r in label_ok if r["iou"] >= threshold]
        key = f"{threshold:.2f}"  # 与指标注册表的 "0.50" 风格一致
        summary[f"matched@{key}"] = len(matched)
        summary[f"boundary_miss@{key}"] = len(label_ok) - len(matched)
        summary[f"fn@{key}"] = total - len(matched)
    durations = [r["length"] for r in records]
    boundary_offsets = [abs(r["start_offset"]) for r in label_ok if r["start_offset"] is not None]
    end_offsets = [abs(r["end_offset"]) for r in label_ok if r["end_offset"] is not None]
    summary["duration"] = {
        "median": statistics.median(durations) if durations else 0,
        "p25": float(np.percentile(durations, 25)) if durations else 0,
        "p75": float(np.percentile(durations, 75)) if durations else 0,
        "le5_frames": sum(1 for d in durations if d <= 5),
        "le10_frames": sum(1 for d in durations if d <= 10),
    }
    summary["boundary_offset_frames"] = {
        "median_abs_start": statistics.median(boundary_offsets) if boundary_offsets else None,
        "median_abs_end": statistics.median(end_offsets) if end_offsets else None,
        "p75_abs_start": float(np.percentile(boundary_offsets, 75)) if boundary_offsets else None,
    }
    summary["per_class"] = {}
    for name in labels:
        subset = [r for r in records if r["class"] == name]
        if not subset:
            continue
        ok = [r for r in subset if r["label_correct"]]
        summary["per_class"][name] = {
            "truth_segments": len(subset),
            "label_wrong": len(subset) - len(ok),
            "matched@0.50": sum(1 for r in ok if r["iou"] >= 0.50),
            "median_duration": statistics.median([r["length"] for r in subset]),
        }
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    records, seen, profile = collect(args.run)
    if not records:
        print("[probe] 未匹配到含 predictions 的 run")
        return 2
    labels: list[str] = []
    for pattern in args.run:  # 类别表从任一产物取
        for run in sorted(REPO.glob(pattern)):
            arts = sorted((run / "artifacts").glob("*.predictions.json"))
            if arts:
                labels = [e["name"] for e in json.loads(arts[-1].read_text())["labels"]]
                break
        if labels:
            break

    summary = summarise(records, labels)
    print(f"=== {args.label}：{seen['runs']} run / {seen['videos']} 视频-次，"
          f"共 {summary['truth_segments']} 个真值段 ===")
    print(f"  标签错（段内多数预测 ≠ 真值标签）：{summary['label_wrong']}"
          f"（{summary['label_wrong'] / max(summary['truth_segments'], 1) * 100:.1f}%）")
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        print(f"  IoU@{key}：匹配 {summary[f'matched@{key}']} / "
              f"标签对但边界不达标 {summary[f'boundary_miss@{key}']} / "
              f"漏检(fn) {summary[f'fn@{key}']}")
    duration = summary["duration"]
    print(f"  真值段时长（采样帧）：中位 {duration['median']:.0f}，p25 {duration['p25']:.0f}，"
          f"p75 {duration['p75']:.0f}，≤5 帧 {duration['le5_frames']} 段，≤10 帧 {duration['le10_frames']} 段")
    offset = summary["boundary_offset_frames"]
    print(f"  标签对段的边界偏移 |Δ|（中位/ p75，帧）：首 {offset['median_abs_start']} / {offset['p75_abs_start']}，"
          f"尾 {offset['median_abs_end']}")

    print(f"\n{'类':24} {'真值段':>7} {'标签错':>7} {'匹配@0.5':>9} {'段时长中位':>10}")
    for name, stats in summary["per_class"].items():
        print(f"{name:24} {stats['truth_segments']:7} {stats['label_wrong']:7} "
              f"{stats['matched@0.50']:9} {stats['median_duration']:10.0f}")

    distance = profile_summary(profile)
    print(f"\n逐帧错误 × 到最近标签切换点的距离（共 {distance['total_frames']} 帧，"
          f"{distance['total_errors']} 个错误帧）：")
    print(f"  {'距离(帧)':>10} {'帧数':>8} {'错误帧':>7} {'错误率':>8} {'占总错误':>9}")
    for name in ("0-1", "2-3", "4-7", "8-15", "16+"):
        entry = distance["buckets"].get(name)
        if not entry:
            continue
        print(f"  {name:>10} {entry['frames']:8} {entry['errors']:7} "
              f"{entry['error_rate']:7.1f}% {entry['share_of_errors']:8.1f}%")
    near = sum(distance["buckets"].get(name, {}).get("errors", 0) for name in ("0-1", "2-3"))
    print(f"  → 距边界 ≤3 帧的帧只占 {sum(distance['buckets'].get(n, {}).get('frames', 0) for n in ('0-1','2-3')) / max(distance['total_frames'],1) * 100:.1f}% 的帧，"
          f"却承载 {near / max(distance['total_errors'], 1) * 100:.1f}% 的错误")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"label": args.label, "runs": seen["runs"],
                                    "boundary_distance": distance, **summary},
                                   ensure_ascii=False, indent=1))
        print(f"\n[probe] 结果已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
