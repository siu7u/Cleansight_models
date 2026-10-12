"""输入特征体检探针：在改特征/加通道之前，先量化"当前输入到底携带多少信息"。

动机（见 docs/features/INPUT_DESIGN_PROPOSAL.md）：特征工程的人天很贵，而"某两个易混
动作在当前检测覆盖下是否可分"这类问题可以用几分钟的统计直接回答。本脚本把提案里的
全部数字做成可复跑的一条命令，避免把"感觉有信号"当成"有信号"。

四段体检（全部只读 labels/ 与 frames/，纯 CPU，不训练）：

1. **检测覆盖**：逐检测类出现率、每帧框数分布 —— 特征维度的天花板由它决定；
2. **动作签名**：每个动作类别下各检测类的出现率 —— 当前特征"看得见"什么；
3. **通道体检**：按 ROI recipe 逐帧算特征，统计恒零通道数、逐通道量纲、
   以及 MS-TCN 归一化口径（``std < 1e-4 → 1.0``）下 z-score 的幅度；
4. **可分性**：对指定类别对（默认 insert vs withdraw），枚举若干"bbox 能推出的非线性量"，
   报原始与离线平滑后的帧级 AUC —— AUC≈0.5 说明该方向不值得投入。

用法：

    python tools/probe_input_features.py --root datasets/cleansight-ActionMixed-auto-lhh
    python tools/probe_input_features.py --root <数据集> --pair 3,4 --json out.json

判据口径：AUC 只度量**单变量**可分性，AUC≈0.5 不等于"模型一定学不出"，但足以否掉
"把该量显式加进输入"的提案；反过来 AUC 明显偏离 0.5 才值得升格为新的 feature mapping。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.cleansight_eval.temporal.features.roi_bbox import (  # noqa: E402
    ROI_GRID_COLS,
    ROI_GRID_ROWS,
    build_roi_frame_features,
)

# 检测类顺序与 frames/data.yaml 一致（8 类）。
DETECTION_CLASSES = [
    "hand",
    "scope_control_body",
    "scope_mid_section",
    "scope_distal_end",
    "syringe",
    "air_gun",
    "short_brush",
    "brush_tip_out",
]
ROI_BLOCK = ROI_GRID_ROWS * ROI_GRID_COLS * 3  # 每类 18 维（6 区域 × 3 通道）
SMOOTH_WINDOW = 31  # 离线居中平滑帧数（约 4s @7.5fps），用作"因果版可达上界"的近似


def _read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    """读一帧 bbox 文本，返回 ``[(class, cx, cy, w, h)]``（跳过非法行）。"""

    boxes: list[tuple[int, float, float, float, float]] = []
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            boxes.append((int(float(parts[0])), *(float(v) for v in parts[1:])))
    return boxes


def _iter_labeled_frames(root: Path, split: str):
    """按 labels manifest 顺序产出 ``(split, stem, frame_id, action_id)``。"""

    labels_dir = root / "labels" / split
    if not labels_dir.is_dir():
        return
    for label_file in sorted(labels_dir.glob("*.txt")):
        stem = label_file.name[:-4]
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                yield split, stem, int(parts[0]), int(parts[1])


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC（并列取平均秩）；正负样本任一为空时返回 nan。"""

    finite = np.isfinite(scores)
    scores, positive = scores[finite], positive[finite]
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    index = 0
    while index < len(scores):  # 并列分数取平均秩
        stop = index
        while stop + 1 < len(scores) and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        ranks[order[index : stop + 1]] = 0.5 * (index + stop) + 1.0
        index = stop + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _centered_mean(values: np.ndarray, window: int) -> np.ndarray:
    """离线居中滑窗均值（仅用于估上界，不可用于在线推理）。"""

    valid = np.isfinite(values).astype(np.float64)
    filled = np.where(np.isfinite(values), values, 0.0)
    kernel = np.ones(window)
    count = np.convolve(valid, kernel, mode="same")
    total = np.convolve(filled, kernel, mode="same")
    return np.where(valid > 0, total / np.maximum(count, 1.0), np.nan)


def probe_coverage(root: Path, records: list) -> dict:
    """第 1、2 段：检测覆盖 + 动作 × 检测类签名。"""

    present = collections.defaultdict(int)
    frames_total = 0
    boxes_per_frame = collections.Counter()
    per_action = collections.defaultdict(collections.Counter)
    action_total = collections.Counter()
    for split, stem, frame_id, action_id in records:
        boxes = _read_boxes(root / "frames" / split / f"{stem}-{frame_id:06d}.txt")
        classes = {box[0] for box in boxes}
        frames_total += 1
        boxes_per_frame[min(len(boxes), 8)] += 1
        action_total[action_id] += 1
        for class_id in range(len(DETECTION_CLASSES)):
            if class_id in classes:
                present[class_id] += 1
                per_action[action_id][class_id] += 1
    return {
        "frames": frames_total,
        "presence_rate": {
            name: round(present[index] / frames_total, 4)
            for index, name in enumerate(DETECTION_CLASSES)
        },
        "boxes_per_frame": dict(sorted(boxes_per_frame.items())),
        "action_presence": {
            action: {
                DETECTION_CLASSES[index]: round(count / action_total[action], 4)
                for index, count in sorted(per_action[action].items())
            }
            for action in sorted(action_total)
        },
        "action_frames": dict(sorted(action_total.items())),
    }


def probe_channels(root: Path, records: list, channel_split: str) -> dict:
    """第 3 段：ROI 逐通道体检（恒零通道、量纲、MS-TCN 口径 z-score 幅度）。"""

    rows = []
    for split, stem, frame_id, _action in records:
        if split != channel_split:
            continue
        rows.append(
            build_roi_frame_features(root / "frames" / split / f"{stem}-{frame_id:06d}.txt")
        )
    if not rows:
        return {}
    matrix = np.stack(rows)
    std = matrix.std(axis=0)
    alive = std > 1e-9
    per_class = {}
    for index, name in enumerate(DETECTION_CLASSES):
        block = matrix[:, index * ROI_BLOCK : (index + 1) * ROI_BLOCK]
        per_class[name] = {
            "alive_dims": int((block.std(axis=0) > 1e-9).sum()),
            "block_dims": ROI_BLOCK,
        }
    # MS-TCN 的 fit_normalization 口径：std < 1e-4 → 1.0（models/mstcn.py）
    guarded = np.where(std < 1e-4, 1.0, std)
    z = np.abs((matrix - matrix.mean(axis=0)) / guarded)
    return {
        "split": channel_split,
        "frames": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "dead_dims": int((~alive).sum()),
        "dead_ratio": round(float((~alive).mean()), 4),
        "alive_std": {
            "p25": round(float(np.percentile(std[alive], 25)), 6),
            "median": round(float(np.median(std[alive])), 6),
            "max": round(float(std[alive].max()), 6),
            "max_over_p25": round(float(std[alive].max() / np.percentile(std[alive], 25)), 1),
        },
        "per_class_alive": per_class,
        "zscore_max": round(float(z.max()), 1),
        "zscore_cells_over_20": int((z > 20).sum()),
    }


def probe_separability(root: Path, records: list, pair: tuple[int, int]) -> dict:
    """第 4 段：类别对的候选非线性量可分性（帧级 AUC）。"""

    candidates = {
        "d_hand_scope_mid": lambda b: _distance(b, 0, 2, multi_left=True),
        "d_hand_scope_control": lambda b: _distance(b, 0, 1, multi_left=True),
        "scope_mid_cy": lambda b: _center(b, 2)[1] if 2 in b else np.nan,
        "scope_length_control_mid": lambda b: _pair_distance(b, 1, 2),
        "hand_area": lambda b: max(
            (entry[2] * entry[3] for entry in b.get(0, [])), default=np.nan
        ),
        "scope_mid_cy_delta1": None,  # 由 scope_mid_cy 的一阶差分填入
        "hand_axis_projection": lambda b: _axis_projection(b),
    }
    scores = {name: [] for name in candidates}
    labels = []
    for split, stem, frame_id, action_id in records:
        if action_id not in pair:
            continue
        boxes = collections.defaultdict(list)
        for class_id, cx, cy, w, h in _read_boxes(
            root / "frames" / split / f"{stem}-{frame_id:06d}.txt"
        ):
            boxes[class_id].append((cx, cy, w, h))
        for name, fn in candidates.items():
            if fn is not None:
                scores[name].append(fn(boxes))
        labels.append(action_id)
    labels = np.asarray(labels)
    previous = None
    delta = []
    for value in scores["scope_mid_cy"]:
        delta.append(np.nan if previous is None or not np.isfinite(value) or not np.isfinite(previous) else value - previous)
        previous = value if np.isfinite(value) else None
    scores["scope_mid_cy_delta1"] = delta

    positive = labels == pair[0]
    report = {}
    for name, values in scores.items():
        array = np.asarray(values, dtype=np.float64)
        report[name] = {
            "auc_raw": round(_auc(array, positive), 3),
            "auc_smoothed": round(_auc(_centered_mean(array, SMOOTH_WINDOW), positive), 3),
            "median_first": None if not np.isfinite(array[positive]).any() else round(float(np.nanmedian(array[positive])), 5),
            "median_second": None if not np.isfinite(array[~positive]).any() else round(float(np.nanmedian(array[~positive])), 5),
        }
    report["_meta"] = {
        "pair": [int(pair[0]), int(pair[1])],
        "frames_first": int(positive.sum()),
        "frames_second": int((~positive).sum()),
        "smooth_window": SMOOTH_WINDOW,
    }
    return report


def _center(boxes: dict, class_id: int):
    entries = boxes.get(class_id, [])
    if not entries:
        return np.array([np.nan, np.nan])
    return np.mean([entry[:2] for entry in entries], axis=0)


def _distance(boxes: dict, left: int, right: int, multi_left: bool = False) -> float:
    left_entries, right_entries = boxes.get(left, []), boxes.get(right, [])
    if not left_entries or not right_entries:
        return float("nan")
    right_center = np.mean([entry[:2] for entry in right_entries], axis=0)
    distances = [float(np.linalg.norm(np.asarray(entry[:2]) - right_center)) for entry in left_entries]
    return min(distances) if multi_left else float(np.mean(distances))


def _pair_distance(boxes: dict, left: int, right: int) -> float:
    a, b = _center(boxes, left), _center(boxes, right)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    return float(np.linalg.norm(a - b))


def _axis_projection(boxes: dict) -> float:
    """hand 在 (control_body → mid_section) 轴上的有符号投影（scope 坐标系内的位置）。"""

    control, mid = _center(boxes, 1), _center(boxes, 2)
    hand = _center(boxes, 0)
    if not (np.isfinite(control).all() and np.isfinite(mid).all() and np.isfinite(hand).all()):
        return float("nan")
    axis = mid - control
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return float("nan")
    return float(np.dot(hand - control, axis / norm))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="输入特征体检探针（检测覆盖/通道量纲/可分性）")
    parser.add_argument("--root", required=True, help="数据集根（含 labels/ 与 frames/）")
    parser.add_argument("--splits", default="train,val,test", help="逗号分隔的 split 列表")
    parser.add_argument("--channel-split", default="train", help="通道体检用的 split")
    parser.add_argument("--pair", default="3,4", help="可分性对照的类别对（默认 3,4 = insert/withdraw）")
    parser.add_argument("--json", default=None, help="把结果额外写入 JSON 文件")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    records = [row for split in splits for row in _iter_labeled_frames(root, split)]
    if not records:
        raise SystemExit(f"没有读到任何标签帧: {root}（检查 labels/<split>/ 是否存在）")
    pair = tuple(int(v) for v in args.pair.split(","))
    result = {
        "root": str(root),
        "splits": splits,
        "coverage": probe_coverage(root, records),
        "channels": probe_channels(root, records, args.channel_split),
        "separability": probe_separability(root, records, pair),
    }

    coverage = result["coverage"]
    print(f"== 检测覆盖（{coverage['frames']} 帧）==")
    print("  每帧框数分布:", coverage["boxes_per_frame"])
    for name, rate in coverage["presence_rate"].items():
        print(f"  {name:20s} {100 * rate:5.1f}%")

    print("\n== 动作 × 检测类出现率 ==")
    header = "  " + f"{'action':>4s} {'frames':>7s} " + " ".join(f"{n[:11]:>12s}" for n in DETECTION_CLASSES)
    print(header)
    for action, frames in coverage["action_frames"].items():
        rates = coverage["action_presence"].get(action, {})
        cells = " ".join(f"{100 * rates.get(n, 0.0):11.1f}%" for n in DETECTION_CLASSES)
        print(f"  {action:>4d} {frames:>7d} {cells}")

    channels = result["channels"]
    if channels:
        print(f"\n== 通道体检（split={channels['split']}，{channels['frames']} 帧）==")
        print(f"  恒零通道 {channels['dead_dims']}/{channels['dim']} = {100 * channels['dead_ratio']:.0f}%")
        print(f"  存活通道 std: p25={channels['alive_std']['p25']} 中位={channels['alive_std']['median']} "
              f"max={channels['alive_std']['max']}（max/p25 = {channels['alive_std']['max_over_p25']}x）")
        print(f"  MS-TCN 口径 z-score: max|z|={channels['zscore_max']}，超 20 的格子 {channels['zscore_cells_over_20']}")
        for name, info in channels["per_class_alive"].items():
            print(f"    {name:20s} 活跃维 {info['alive_dims']}/{info['block_dims']}")

    sep = result["separability"]
    meta = sep["_meta"]
    print(f"\n== 可分性：class {meta['pair'][0]}（正，n={meta['frames_first']}）"
          f" vs class {meta['pair'][1]}（负，n={meta['frames_second']}）==")
    print(f"  {'candidate':26s} {'AUC':>7s} {'AUC(平滑)':>10s}  {'中位(正/负)':>22s}")
    for name, info in sep.items():
        if name == "_meta":
            continue
        medians = f"{info['median_first']} / {info['median_second']}"
        print(f"  {name:26s} {info['auc_raw']:>7.3f} {info['auc_smoothed']:>10.3f}  {medians:>22s}")
    print("\n  判据：|AUC-0.5|≈0 = 该单变量无判别力，显式加进输入无意义；"
          "明显偏离 0.5（含远小于 0.5 的反向信号）才值得升格为新 feature mapping。"
          "样本数过小的类别对（n 很小）结论不可信。")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n结果已写入: {args.json}")


if __name__ == "__main__":
    sys.exit(main())
