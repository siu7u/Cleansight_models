"""按等价类标签（单维度）分组评估检测结果。

用法:
    python scripts/per_tag_eval.py \
      --predictions framework/runs/<run>/artifacts/detection-yolo-*.predictions.json \
      --labels datasets/cleansight-yolo/<dataset>/labels/val/ \
      --tag-meta datasets/cleansight-yolo/tag_metadata.json

输出每个 tag 值上的 mAP / Precision / Recall，以及该 tag 的样本量。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


# --------------- 图像 → clip 前缀匹配 ---------------

def _extract_clip_prefix(image_name: str) -> str | None:
    """从图片文件名中提取 clip 前缀（8 位 hex）。

    支持两种命名模式：
      ActionMixed:  65d70028-clip_1781661552468_1781661702909.mp4-000005.jpg
                    → prefix=65d70028
      group2_small: 10_65d70028-cli_000005.jpg
                    → prefix=65d70028
    """
    name = Path(image_name).stem  # 去掉扩展名
    # 模式 1: {prefix}-clip_{ts1}_{ts2}.mp4-{frame}
    dash_idx = name.find("-")
    if dash_idx > 0:
        candidate = name[:dash_idx]
        if len(candidate) >= 8 and all(c in "0123456789abcdef" for c in candidate[:8].lower()):
            return candidate[:8]
    # 模式 2: {n}_{prefix}-cli_{frame}
    parts = name.split("_")
    if len(parts) >= 2:
        clip_part = parts[1]  # e.g. "65d70028-cli"
        candidate = clip_part.split("-")[0]
        if len(candidate) >= 8:
            return candidate[:8]
    return None


# --------------- 标签匹配----------------

def load_tag_index(tag_meta_path: str) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """加载 tag_metadata.json，返回 (clip→tags, tag→clips)。

    clip→tags: {"65d70028": {"viewpoint": "cam1", "ec_tags": ["dark", "fast_blur"]}}
    tag→clips: {"dark": ["65d70028", ...], "cam1": ["65d70028", ...]}
    """
    with open(tag_meta_path) as f:
        meta = json.load(f)

    clip_info: dict[str, dict[str, Any]] = {}
    tag_to_clips: dict[str, list[str]] = defaultdict(list)

    for task in meta["tasks"]:
        clip_prefix = task["video"].split("-")[0][:8]
        info = {
            "viewpoint": task["viewpoint"],
            "ec_tags": task["ec_tags"],
            "frames_count": task.get("frames_count"),
            "task_id": task["task_id"],
        }
        clip_info[clip_prefix] = info

        # 登记 viewpoint
        vp = task["viewpoint"]
        tag_to_clips[f"viewpoint:{vp}"].append(clip_prefix)

        # 登记 ec_tags
        for t in task["ec_tags"]:
            tag_to_clips[f"ec_tag:{t}"].append(clip_prefix)

    return clip_info, dict(tag_to_clips)


# --------------- 指标计算 -----------------

def _compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """101-point interpolated AP (COCO 风格)。"""
    if len(precision) == 0:
        return 0.0
    # 在 0:0.01:1 上做线性插值
    r_interp = np.linspace(0, 1, 101)
    p_interp = np.interp(r_interp, recall[::-1], precision[::-1])
    return float(np.mean(p_interp))


def _match_predictions(
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[list[float], list[float], int, int]:
    """对一张图匹配 GT 和预测，返回 (precisions, recalls, tp, fp)。

    返回的 precision/recall 按置信度降序排列，直接用于 PR 曲线。
    简化版：不考虑多类别（调用方按类别分组后传入）。
    """
    if not gt_boxes and not pred_boxes:
        return [], [], 0, 0

    gt_used = [False] * len(gt_boxes)
    preds_sorted = sorted(enumerate(pred_boxes), key=lambda x: -x[1]["confidence"])

    tp = np.zeros(len(pred_boxes), dtype=bool)
    fp = np.zeros(len(pred_boxes), dtype=bool)

    for rank, (idx, pred) in enumerate(preds_sorted):
        px, py, pw, ph = pred["xywhn"]
        matched = False
        best_iou = 0.0
        best_gt = -1

        for j, gt in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            gx, gy, gw, gh = gt["xywhn"]

            # IoU 计算 (xywh → x1y1x2y2)
            px1, py1 = px - pw / 2, py - ph / 2
            px2, py2 = px + pw / 2, py + ph / 2
            gx1, gy1 = gx - gw / 2, gy - gh / 2
            gx2, gy2 = gx + gw / 2, gy + gh / 2

            ix1 = max(px1, gx1)
            iy1 = max(py1, gy1)
            ix2 = min(px2, gx2)
            iy2 = min(py2, gy2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            inter = (ix2 - ix1) * (iy2 - iy1)
            union = pw * ph + gw * gh - inter
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_threshold and best_gt >= 0:
            matched = True
            gt_used[best_gt] = True

        tp[rank] = matched
        fp[rank] = not matched

    # 累积 TP/FP → precision/recall 序列
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / len(gt_boxes) if gt_boxes else np.zeros_like(tp_cum, dtype=float)
    precisions = tp_cum / (tp_cum + fp_cum)
    precisions[np.isnan(precisions)] = 0.0

    return precisions.tolist(), recalls.tolist(), int(tp_cum[-1]), int(fp_cum[-1])


def evaluate_subset(
    items: dict[str, dict],
    gt_items: dict[str, list[dict]],
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
) -> dict:
    """对一组图片计算检测指标。

    items: {image_name: {"predictions": [...]}}
    gt_items: {image_name: [{"class_id": ..., "xywhn": [...]}, ...]}
    """
    class_ids = sorted(class_names.keys())
    per_class: dict[str, dict] = {}

    # 按类别收集所有 PR 数据
    all_precisions = defaultdict(list)
    all_recalls = defaultdict(list)

    for img_name, pred_data in items.items():
        gt_list = gt_items.get(img_name, [])
        preds = pred_data.get("predictions", [])

        # 按类别分组
        gt_by_class = defaultdict(list)
        for gt in gt_list:
            gt_by_class[gt["class_id"]].append(gt)

        pred_by_class = defaultdict(list)
        for pred in preds:
            pred_by_class[pred["class_id"]].append(pred)

        for cid in class_ids:
            gt_c = gt_by_class.get(cid, [])
            pred_c = pred_by_class.get(cid, [])

            if not gt_c and not pred_c:
                continue

            precs, recs, tp_count, fp_count = _match_predictions(gt_c, pred_c, iou_threshold)
            all_precisions[cid].extend(precs)
            all_recalls[cid].extend(recs)

    # 逐类 AP
    for cid in class_ids:
        name = class_names[cid]
        precs = all_precisions.get(cid, [])
        recs = all_recalls.get(cid, [])
        if precs and recs:
            # 按 recall 排序
            pairs = sorted(zip(recs, precs), key=lambda x: x[0])
            recs_sorted = np.array([r for r, _ in pairs])
            precs_sorted = np.array([p for _, p in pairs])
            ap = _compute_ap(precs_sorted, recs_sorted)
            # 在 recall=0.5 处取 precision
            precision_at_recall = float(np.interp(0.5, recs_sorted, precs_sorted)) if len(recs_sorted) > 1 else 0.0
            recall_at_conf = float(recs_sorted[-1]) if len(recs_sorted) > 0 else 0.0
        else:
            ap = 0.0
            precision_at_recall = 0.0
            recall_at_conf = 0.0

        per_class[name] = {
            "ap": round(ap, 4),
            "precision": round(precision_at_recall, 4),
            "recall": round(recall_at_conf, 4),
        }

    # mAP = 各类 AP 的均值
    aps = [v["ap"] for v in per_class.values()]
    map50 = round(float(np.mean(aps)) if aps else 0.0, 4)
    mean_p = round(float(np.mean([v["precision"] for v in per_class.values()])) if per_class else 0.0, 4)
    mean_r = round(float(np.mean([v["recall"] for v in per_class.values()])) if per_class else 0.0, 4)

    return {
        "mAP@0.5": map50,
        "precision": mean_p,
        "recall": mean_r,
        "per_class": per_class,
    }


# --------------- 主流程 -----------------

def load_ground_truth(labels_dir: str) -> dict[str, list[dict]]:
    """从 YOLO labels 目录加载真值。

    返回: {image_name: [{"class_id": int, "xywhn": [float,...]}, ...]}
    """
    gt: dict[str, list[dict]] = {}
    label_dir = Path(labels_dir)
    for label_file in sorted(label_dir.glob("*.txt")):
        img_name = label_file.stem + ".jpg"  # 假设图片是 .jpg
        boxes = []
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xywhn = [float(x) for x in parts[1:5]]
                boxes.append({"class_id": cls_id, "xywhn": xywhn})
        if boxes:
            gt[img_name] = boxes
    return gt


def build_image_to_clip(
    image_names: list[str],
    clip_info: dict[str, dict],
) -> dict[str, str]:
    """构建 image_name → clip_prefix 的映射。

    匹配规则：图片名中包含 clip_prefix 即视为匹配。
    """
    mapping: dict[str, str] = {}
    for img in image_names:
        clip_prefix = _extract_clip_prefix(img)
        if clip_prefix and clip_prefix in clip_info:
            mapping[img] = clip_prefix
    return mapping


def main():
    p = argparse.ArgumentParser(description="按等价类标签分组评估检测结果")
    p.add_argument("--predictions", required=True, help="predictions.json 路径")
    p.add_argument("--labels", required=True, help="YOLO labels 目录（如 labels/val/）")
    p.add_argument("--tag-meta", required=True, help="tag_metadata.json 路径")
    p.add_argument("--iou", type=float, default=0.5, help="IoU 阈值（默认 0.5）")
    p.add_argument("--min-samples", type=int, default=50,
                   help="最少样本数，低于此的 tag 跳过不评估（默认 50）")
    args = p.parse_args()

    # 1. 加载数据
    print("加载数据...")
    with open(args.predictions) as f:
        pred_data = json.load(f)

    items = pred_data["items"]
    labels = pred_data["labels"]
    class_names = {int(k): v for k, v in labels.items()}

    gt_items = load_ground_truth(args.labels)
    clip_info, tag_to_clips = load_tag_index(args.tag_meta)

    # 2. 匹配图片到 clip → tag
    image_to_clip = build_image_to_clip(list(items.keys()), clip_info)
    matched = len(image_to_clip)
    total = len(items)
    print(f"图片→clip 匹配: {matched}/{total} ({100 * matched / max(total, 1):.1f}%)")

    if matched == 0:
        # 诊断信息
        img_prefixes = set()
        for img_name in list(items.keys())[:30]:
            pfx = _extract_clip_prefix(img_name)
            if pfx:
                img_prefixes.add(pfx)
        tag_prefixes = set(clip_info.keys())

        print("\n⚠️  没有图片能匹配到 tag_metadata 中的 clip。")
        print(f"  图片中的 clip 前缀（前 30 张）: {sorted(img_prefixes)[:10]}")
        print(f"  tag_metadata 中的 clip 前缀: {sorted(tag_prefixes)[:10]}")
        common = img_prefixes & tag_prefixes
        if common:
            print(f"  共有前缀: {common} — 但仍然匹配 0，请检查 _extract_clip_prefix 逻辑")
        else:
            print("  两者无交集 → tag_metadata 和当前数据集不是同一批 clip。")
            print("  等新数据集（从这 41 个 clip 构建）产出后再运行本脚本。")
        return

    # 3. 按 tag 分组评估
    # 收集每个 tag 对应的图片集
    tag_images: dict[str, set] = defaultdict(set)
    for img_name, clip_prefix in image_to_clip.items():
        info = clip_info.get(clip_prefix)
        if info is None:
            continue
        vp = info["viewpoint"]
        tag_images[f"viewpoint:{vp}"].add(img_name)
        for ec_tag in info.get("ec_tags", []):
            tag_images[f"ec_tag:{ec_tag}"].add(img_name)

    # 4. 逐 tag 评估
    print(f"\n按 tag 分组评估（IoU={args.iou}, min_samples={args.min_samples}）...\n")

    # 先评估全集作为基线
    all_metrics = evaluate_subset(items, gt_items, class_names, args.iou)

    # 逐 tag
    results = []
    for tag_full, img_set in sorted(tag_images.items()):
        if len(img_set) < args.min_samples:
            continue

        subset_items = {img: items[img] for img in img_set if img in items}
        subset_gt = {img: gt_items[img] for img in img_set if img in gt_items}

        if len(subset_items) == 0:
            continue

        metrics = evaluate_subset(subset_items, subset_gt, class_names, args.iou)
        results.append((tag_full, len(img_set), metrics))

    # 5. 打印报告
    _print_report(all_metrics, class_names, results, tag_to_clips, clip_info)


def _print_report(all_metrics, class_names, results, tag_to_clips, clip_info):
    """打印格式化报告。"""

    # 维度标签
    dim_label = {
        "viewpoint": "机位",
        "ec_tag": "成像条件",
    }

    # --- 概要 ---
    print("=" * 80)
    print("                              等价类评估报告")
    print("=" * 80)
    print()
    print(f"  全集基线: mAP@0.5={all_metrics['mAP@0.5']:.4f}  "
          f"P={all_metrics['precision']:.4f}  R={all_metrics['recall']:.4f}")
    print()

    # 按维度分组展示
    for dimension, dim_name in [("viewpoint", "机位"), ("ec_tag", "成像条件")]:
        dim_results = [(tag, n, m) for tag, n, m in results if tag.startswith(f"{dimension}:")]

        if not dim_results:
            continue

        print(f"--- {dim_name} ---")
        print(f"  {'标签':<25} {'样本':>8} {'mAP@0.5':>10} {'Precision':>10} {'Recall':>10}  ΔmAP")
        print(f"  {'-' * 25} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}  {'-' * 7}")

        for tag_full, n_images, m in sorted(dim_results, key=lambda x: -x[2]["mAP@0.5"]):
            tag_name = tag_full.split(":", 1)[1]
            delta = m["mAP@0.5"] - all_metrics["mAP@0.5"]
            sign = "+" if delta >= 0 else ""
            print(f"  {tag_name:<25} {n_images:>8} {m['mAP@0.5']:>10.4f} "
                  f"{m['precision']:>10.4f} {m['recall']:>10.4f}  {sign}{delta:.4f}")

            # 如果该 tag 包含多类零检出，给出提示
            zero_classes = [
                name for name, v in m["per_class"].items()
                if v["recall"] == 0.0
            ]
            if zero_classes:
                print(f"    ⚠️ 零检出: {', '.join(zero_classes)}")

        print()

    # --- 逐类详情（重点类）---
    print("--- 逐类 recall 分解（按 tag） ---")
    print(f"  {'标签':<25}", end="")
    for cid in sorted(class_names.keys()):
        name = class_names[cid]
        print(f" {name:>12}", end="")
    print(f"  {'样本':>8}")
    print(f"  {'-' * 25}", end="")
    for _ in class_names:
        print(f" {'-' * 12}", end="")
    print(f"  {'-' * 8}")

    for tag_full, n_images, m in sorted(results, key=lambda x: -x[2]["mAP@0.5"]):
        tag_name = tag_full.split(":", 1)[1]
        print(f"  {tag_name:<25}", end="")
        for cid in sorted(class_names.keys()):
            name = class_names[cid]
            r = m["per_class"].get(name, {}).get("recall", 0)
            print(f" {r:>12.4f}", end="")
        print(f"  {n_images:>8}")

    print()

    # --- tag 覆盖统计 ---
    print("--- tag 覆盖统计 ---")
    for tag_full in sorted(tag_to_clips.keys()):
        n_clips = len(tag_to_clips[tag_full])
        tag_name = tag_full.split(":", 1)[1]
        # 统计图片数
        matched_counted = sum(
            1 for r in results if r[0] == tag_full
        )
        img_count = results[[r[0] for r in results].index(tag_full)][1] if tag_full in [r[0] for r in results] else 0
        print(f"  {tag_name:<25} {n_clips:>3} clips, {img_count:>5} 张图（匹配）")


if __name__ == "__main__":
    main()
