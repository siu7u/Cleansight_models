"""对已保存的检测 prediction artifact 做离线后处理对比。

脚本不重新运行 YOLO，只对同一预测池比较阈值、NMS、Soft-NMS、WBF、top-k、
max-det 和面积过滤，并统一复算整体/逐类指标及 split CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


Box = tuple[float, float, float, float]


# ============================ 集中实验参数区 ============================
PER_CLASS_THRESHOLD_GRID = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
GLOBAL_THRESHOLD_GRID = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
REFERENCE_CONF = 0.30
REFERENCE_NMS_IOU = 0.55
REFERENCE_SOFT_NMS_SIGMA = 0.5
REFERENCE_TOPK = {0: 2, 1: 1, 2: 1}


@dataclass(frozen=True)
class Detection:
    image: str
    cls: int
    conf: float
    box: Box


def xywh_to_xyxy(box: Box) -> Box:
    x, y, w, h = box
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def area_xyxy(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xywh(first: Box, second: Box) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(first)
    bx1, by1, bx2, by2 = xywh_to_xyxy(second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area_xyxy((ax1, ay1, ax2, ay2)) + area_xyxy((bx1, by1, bx2, by2)) - inter
    return inter / union if union > 0 else 0.0


def load_predictions(path: Path) -> tuple[list[Detection], dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = {int(k): str(v) for k, v in data["labels"].items()}
    detections: list[Detection] = []
    for image, payload in data["items"].items():
        for item in payload.get("predictions", []):
            detections.append(
                Detection(
                    image=image,
                    cls=int(item["class_id"]),
                    conf=float(item["confidence"]),
                    box=tuple(float(v) for v in item["xywhn"]),  # type: ignore[arg-type]
                )
            )
    return detections, labels


def load_ground_truth(labels_dir: Path) -> dict[str, list[Detection]]:
    truth: dict[str, list[Detection]] = {}
    for label_path in labels_dir.glob("*.txt"):
        image = label_path.with_suffix(".jpg").name
        items: list[Detection] = []
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            items.append(
                Detection(
                    image=image,
                    cls=int(parts[0]),
                    conf=1.0,
                    box=tuple(float(v) for v in parts[1:5]),  # type: ignore[arg-type]
                )
            )
        truth[image] = items
    return truth


def class_aware_nms(items: Iterable[Detection], iou_threshold: float, *, agnostic: bool = False) -> list[Detection]:
    groups: dict[tuple[str, int], list[Detection]] = defaultdict(list)
    for det in items:
        key = (det.image, -1 if agnostic else det.cls)
        groups[key].append(det)

    kept: list[Detection] = []
    for group in groups.values():
        selected: list[Detection] = []
        for det in sorted(group, key=lambda item: item.conf, reverse=True):
            if all(iou_xywh(det.box, prev.box) < iou_threshold for prev in selected):
                selected.append(det)
        kept.extend(selected)
    return kept


def topk_per_class(items: Iterable[Detection], limits: dict[int, int]) -> list[Detection]:
    groups: dict[tuple[str, int], list[Detection]] = defaultdict(list)
    for det in items:
        groups[(det.image, det.cls)].append(det)
    out: list[Detection] = []
    for (_, cls), group in groups.items():
        limit = limits.get(cls)
        ranked = sorted(group, key=lambda item: item.conf, reverse=True)
        out.extend(ranked[:limit] if limit is not None else ranked)
    return out


def max_det_per_image(items: Iterable[Detection], limit: int) -> list[Detection]:
    groups: dict[str, list[Detection]] = defaultdict(list)
    for det in items:
        groups[det.image].append(det)
    out: list[Detection] = []
    for group in groups.values():
        out.extend(sorted(group, key=lambda item: item.conf, reverse=True)[:limit])
    return out


def soft_nms(
    items: Iterable[Detection],
    iou_threshold: float,
    *,
    score_threshold: float,
    sigma: float = 0.5,
    method: str = "linear",
) -> list[Detection]:
    groups: dict[tuple[str, int], list[Detection]] = defaultdict(list)
    for det in items:
        groups[(det.image, det.cls)].append(det)

    kept: list[Detection] = []
    for group in groups.values():
        candidates = list(group)
        while candidates:
            best_index = max(range(len(candidates)), key=lambda idx: candidates[idx].conf)
            best = candidates.pop(best_index)
            if best.conf >= score_threshold:
                kept.append(best)

            decayed: list[Detection] = []
            for det in candidates:
                overlap = iou_xywh(best.box, det.box)
                if method == "gaussian":
                    conf = det.conf * math.exp(-((overlap * overlap) / sigma))
                elif overlap > iou_threshold:
                    conf = det.conf * (1.0 - overlap)
                else:
                    conf = det.conf
                if conf >= score_threshold:
                    decayed.append(Detection(det.image, det.cls, conf, det.box))
            candidates = decayed
    return kept


def weighted_box_fusion(items: Iterable[Detection], iou_threshold: float) -> list[Detection]:
    groups: dict[tuple[str, int], list[Detection]] = defaultdict(list)
    for det in items:
        groups[(det.image, det.cls)].append(det)

    fused: list[Detection] = []
    for group in groups.values():
        remaining = sorted(group, key=lambda item: item.conf, reverse=True)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            kept = []
            for det in remaining:
                if iou_xywh(seed.box, det.box) >= iou_threshold:
                    cluster.append(det)
                else:
                    kept.append(det)
            remaining = kept
            weight_sum = sum(item.conf for item in cluster)
            if weight_sum <= 0:
                fused.append(seed)
                continue
            box = tuple(
                sum(item.box[idx] * item.conf for item in cluster) / weight_sum
                for idx in range(4)
            )
            fused.append(Detection(seed.image, seed.cls, max(item.conf for item in cluster), box))  # type: ignore[arg-type]
    return fused


def score_counts(predictions: list[Detection], truth: dict[str, list[Detection]], classes: Iterable[int], iou_thr: float) -> dict:
    per_class = {}
    total_tp = total_fp = total_fn = 0
    for cls in classes:
        gt_by_image: dict[str, list[Detection]] = {
            image: [item for item in items if item.cls == cls] for image, items in truth.items()
        }
        used: dict[str, set[int]] = defaultdict(set)
        tp = fp = 0
        for pred in sorted((p for p in predictions if p.cls == cls), key=lambda item: item.conf, reverse=True):
            candidates = gt_by_image.get(pred.image, [])
            best_iou, best_idx = 0.0, None
            for idx, gt in enumerate(candidates):
                if idx in used[pred.image]:
                    continue
                value = iou_xywh(pred.box, gt.box)
                if value > best_iou:
                    best_iou, best_idx = value, idx
            if best_idx is not None and best_iou >= iou_thr:
                tp += 1
                used[pred.image].add(best_idx)
            else:
                fp += 1
        gt_count = sum(len(items) for items in gt_by_image.values())
        fn = gt_count - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt_count if gt_count else 0.0
        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": per_class,
    }


def average_precision(predictions: list[Detection], truth: dict[str, list[Detection]], cls: int, iou_thr: float) -> float:
    gt_by_image: dict[str, list[Detection]] = {
        image: [item for item in items if item.cls == cls] for image, items in truth.items()
    }
    total_gt = sum(len(items) for items in gt_by_image.values())
    if total_gt == 0:
        return 0.0
    used: dict[str, set[int]] = defaultdict(set)
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    for pred in sorted((p for p in predictions if p.cls == cls), key=lambda item: item.conf, reverse=True):
        candidates = gt_by_image.get(pred.image, [])
        best_iou, best_idx = 0.0, None
        for idx, gt in enumerate(candidates):
            if idx in used[pred.image]:
                continue
            value = iou_xywh(pred.box, gt.box)
            if value > best_iou:
                best_iou, best_idx = value, idx
        if best_idx is not None and best_iou >= iou_thr:
            used[pred.image].add(best_idx)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    if not tp_flags:
        return 0.0

    cum_tp = 0
    cum_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for tp, fp in zip(tp_flags, fp_flags):
        cum_tp += tp
        cum_fp += fp
        recalls.append(cum_tp / total_gt)
        precisions.append(cum_tp / max(cum_tp + cum_fp, 1))

    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    ap = 0.0
    for idx in range(1, len(mrec)):
        if mrec[idx] != mrec[idx - 1]:
            ap += (mrec[idx] - mrec[idx - 1]) * mpre[idx]
    return ap


def score(predictions: list[Detection], truth: dict[str, list[Detection]], labels: dict[int, str]) -> dict:
    classes = sorted(labels)
    counts = score_counts(predictions, truth, classes, 0.5)
    ap50 = {cls: average_precision(predictions, truth, cls, 0.5) for cls in classes}
    ap5095 = {
        cls: sum(average_precision(predictions, truth, cls, thr / 100) for thr in range(50, 100, 5)) / 10
        for cls in classes
    }
    counts["mAP50"] = sum(ap50.values()) / len(classes)
    counts["mAP50_95"] = sum(ap5095.values()) / len(classes)
    for cls in classes:
        counts["per_class"][cls]["ap50"] = ap50[cls]
        counts["per_class"][cls]["ap50_95"] = ap5095[cls]
    counts["num_predictions"] = len(predictions)
    return counts


def apply_thresholds(predictions: Iterable[Detection], thresholds: dict[int, float]) -> list[Detection]:
    return [det for det in predictions if det.conf >= thresholds.get(det.cls, 0.0)]


def apply_global_threshold(predictions: Iterable[Detection], threshold: float) -> list[Detection]:
    return [det for det in predictions if det.conf >= threshold]


def filter_area(predictions: Iterable[Detection], min_area: float) -> list[Detection]:
    return [det for det in predictions if det.box[2] * det.box[3] >= min_area]


def split_name(image: str) -> str:
    stem = Path(image).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def filter_by_images(items: Iterable[Detection], images: set[str]) -> list[Detection]:
    return [item for item in items if item.image in images]


def write_split_csv(path: Path, rows: list[dict], predictions: list[Detection], truth: dict[str, list[Detection]], labels: dict[int, str]) -> None:
    groups: dict[str, set[str]] = defaultdict(set)
    for image in truth:
        groups[split_name(image)].add(image)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "split",
                "num_images",
                "num_predictions",
                "mAP50",
                "mAP50_95",
                "precision",
                "recall",
                "f1",
                "hand_precision",
                "hand_recall",
                "scope_control_body_precision",
                "scope_control_body_recall",
                "scope_mid_section_precision",
                "scope_mid_section_recall",
            ],
        )
        writer.writeheader()
        by_strategy = {row["name"]: row["fn"] for row in rows}
        for strategy, fn in by_strategy.items():
            processed = fn(predictions)
            for group_name, images in sorted(groups.items()):
                split_truth = {image: truth[image] for image in images}
                split_predictions = filter_by_images(processed, images)
                metrics = rounded_metrics(score(split_predictions, split_truth, labels), labels)
                writer.writerow(
                    {
                        "strategy": strategy,
                        "split": group_name,
                        "num_images": len(images),
                        "num_predictions": metrics["num_predictions"],
                        "mAP50": metrics["mAP50"],
                        "mAP50_95": metrics["mAP50_95"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "hand_precision": metrics["per_class"]["hand"]["precision"],
                        "hand_recall": metrics["per_class"]["hand"]["recall"],
                        "scope_control_body_precision": metrics["per_class"]["scope_control_body"]["precision"],
                        "scope_control_body_recall": metrics["per_class"]["scope_control_body"]["recall"],
                        "scope_mid_section_precision": metrics["per_class"]["scope_mid_section"]["precision"],
                        "scope_mid_section_recall": metrics["per_class"]["scope_mid_section"]["recall"],
                    }
                )


def find_per_class_thresholds(
    predictions: list[Detection],
    truth: dict[str, list[Detection]],
    labels: dict[int, str],
    grid: list[float],
) -> dict[int, float]:
    thresholds = {}
    for cls in sorted(labels):
        best_threshold = grid[0]
        best_f1 = -1.0
        cls_predictions = [item for item in predictions if item.cls == cls]
        for threshold in grid:
            metrics = score_counts(
                [item for item in cls_predictions if item.conf >= threshold],
                truth,
                [cls],
                0.5,
            )["per_class"][cls]
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold
        thresholds[cls] = best_threshold
    return thresholds


def rounded_metrics(metrics: dict, labels: dict[int, str]) -> dict:
    out = {
        key: round(float(metrics[key]), 4)
        for key in ("mAP50", "mAP50_95", "precision", "recall", "f1")
    }
    out["num_predictions"] = int(metrics["num_predictions"])
    out["per_class"] = {
        labels[cls]: {
            key: round(float(metrics["per_class"][cls][key]), 4)
            for key in ("precision", "recall", "f1", "ap50", "ap50_95")
        }
        for cls in sorted(labels)
    }
    return out


def write_report(path: Path, rows: list[dict], labels: dict[int, str], official: dict | None) -> None:
    lines = [
        "# YOLO Post-processing Comparison",
        "",
        "Offline metrics are recomputed from the saved prediction artifact and YOLO test labels.",
        "They are intended for relative post-processing comparison; the official baseline remains the benchmark JSON.",
        "",
    ]
    if official:
        summary = official["metrics"]["summary"]
        lines.extend(
            [
                "## Official Baseline",
                "",
                f"- mAP@0.5: `{summary['mAP@0.5']['value']}`",
                f"- mAP@0.5:0.95: `{summary['mAP@0.5:0.95']['value']}`",
                f"- precision: `{summary['precision']['value']}`",
                f"- recall: `{summary['recall']['value']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Offline Comparison",
            "",
            "| Strategy | mAP50 | mAP50-95 | Precision | Recall | F1 | Predictions |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['name']} | {m['mAP50']:.4f} | {m['mAP50_95']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['num_predictions']} |"
        )
    lines.extend(["", "## Per-class Metrics", ""])
    for row in rows:
        lines.extend([f"### {row['name']}", "", "| Class | AP50 | AP50-95 | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for cls in sorted(labels):
            item = row["metrics"]["per_class"][labels[cls]]
            lines.append(
                f"| {labels[cls]} | {item['ap50']:.4f} | {item['ap50_95']:.4f} | "
                f"{item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compare detection post-processing methods offline.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--official-eval", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-splits-csv", default=None)
    args = parser.parse_args(argv)

    predictions, labels = load_predictions(Path(args.predictions))
    truth = load_ground_truth(Path(args.labels_dir))
    official = json.loads(Path(args.official_eval).read_text(encoding="utf-8")) if args.official_eval else None

    per_class_thresholds = find_per_class_thresholds(predictions, truth, labels, PER_CLASS_THRESHOLD_GRID)
    # 策略清单是本实验的唯一配置入口；新增策略时同时写入名称和参数，保证报告可追溯。
    strategies: list[tuple[str, Callable[[list[Detection]], list[Detection]], dict]] = [
        ("baseline_saved_predictions", lambda items: list(items), {}),
        *[
            (
                f"global_conf_{threshold:g}",
                lambda items, threshold=threshold: apply_global_threshold(items, threshold),
                {"conf": threshold},
            )
            for threshold in GLOBAL_THRESHOLD_GRID
        ],
        (
            "per_class_conf_best_f1_exploratory",
            lambda items: apply_thresholds(items, per_class_thresholds),
            {"thresholds": {labels[k]: v for k, v in per_class_thresholds.items()}},
        ),
        (
            "per_class_conf_then_nms_0.55",
            lambda items: class_aware_nms(apply_thresholds(items, per_class_thresholds), REFERENCE_NMS_IOU),
            {"thresholds": {labels[k]: v for k, v in per_class_thresholds.items()}, "iou": REFERENCE_NMS_IOU},
        ),
        (
            "per_class_conf_then_topk_hand2_body1_mid1",
            lambda items: topk_per_class(apply_thresholds(items, per_class_thresholds), REFERENCE_TOPK),
            {
                "thresholds": {labels[k]: v for k, v in per_class_thresholds.items()},
                "topk": {"hand": 2, "scope_control_body": 1, "scope_mid_section": 1},
            },
        ),
        (
            "global_conf_0.30_then_nms_0.55",
            lambda items: class_aware_nms(apply_global_threshold(items, REFERENCE_CONF), REFERENCE_NMS_IOU),
            {"conf": REFERENCE_CONF, "iou": REFERENCE_NMS_IOU},
        ),
        (
            "global_conf_0.30_then_softnms_linear_0.55",
            lambda items: soft_nms(apply_global_threshold(items, REFERENCE_CONF), REFERENCE_NMS_IOU, score_threshold=REFERENCE_CONF, method="linear"),
            {"conf": REFERENCE_CONF, "iou": REFERENCE_NMS_IOU, "method": "linear"},
        ),
        (
            "global_conf_0.30_then_softnms_gaussian",
            lambda items: soft_nms(apply_global_threshold(items, REFERENCE_CONF), REFERENCE_NMS_IOU, score_threshold=REFERENCE_CONF, method="gaussian", sigma=REFERENCE_SOFT_NMS_SIGMA),
            {"conf": REFERENCE_CONF, "iou": REFERENCE_NMS_IOU, "sigma": REFERENCE_SOFT_NMS_SIGMA, "method": "gaussian"},
        ),
        (
            "global_conf_0.30_then_wbf_0.55",
            lambda items: weighted_box_fusion(apply_global_threshold(items, REFERENCE_CONF), REFERENCE_NMS_IOU),
            {"conf": REFERENCE_CONF, "iou": REFERENCE_NMS_IOU},
        ),
        ("class_aware_nms_0.45", lambda items: class_aware_nms(items, 0.45), {"iou": 0.45}),
        ("class_aware_nms_0.55", lambda items: class_aware_nms(items, 0.55), {"iou": 0.55}),
        ("class_agnostic_nms_0.70", lambda items: class_aware_nms(items, 0.70, agnostic=True), {"iou": 0.70}),
        ("max_det_10", lambda items: max_det_per_image(items, 10), {"max_det": 10}),
        ("max_det_20", lambda items: max_det_per_image(items, 20), {"max_det": 20}),
        ("max_det_50", lambda items: max_det_per_image(items, 50), {"max_det": 50}),
        (
            "global_conf_0.30_then_max_det_10",
            lambda items: max_det_per_image(apply_global_threshold(items, REFERENCE_CONF), 10),
            {"conf": REFERENCE_CONF, "max_det": 10},
        ),
        (
            "topk_hand2_body1_mid1",
            lambda items: topk_per_class(items, REFERENCE_TOPK),
            {"topk": {"hand": 2, "scope_control_body": 1, "scope_mid_section": 1}},
        ),
        ("min_area_0.001", lambda items: filter_area(items, 0.001), {"min_area": 0.001}),
        ("min_area_0.002", lambda items: filter_area(items, 0.002), {"min_area": 0.002}),
    ]

    rows = []
    for name, fn, params in strategies:
        processed = fn(predictions)
        rows.append({"name": name, "params": params, "metrics": rounded_metrics(score(processed, truth, labels), labels), "fn": fn})
    rows.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["mAP50_95"]), reverse=True)

    output = {
        "prediction_artifact": str(Path(args.predictions)),
        "labels_dir": str(Path(args.labels_dir)),
        "official_eval": str(Path(args.official_eval)) if args.official_eval else None,
        "labels": labels,
        "note": "Per-class threshold search is exploratory because it is selected on this evaluation split.",
        "results": [{key: value for key, value in row.items() if key != "fn"} for row in rows],
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_md, rows, labels, official)
    split_csv = Path(args.out_splits_csv) if args.out_splits_csv else out_json.with_suffix(".splits.csv")
    write_split_csv(split_csv, rows[:5], predictions, truth, labels)
    print(f"[postprocess] json: {out_json}")
    print(f"[postprocess] report: {out_md}")
    print(f"[postprocess] split_csv: {split_csv}")
    print(f"[postprocess] best_by_f1: {rows[0]['name']} {rows[0]['metrics']}")


if __name__ == "__main__":
    main()
