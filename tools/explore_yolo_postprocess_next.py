"""在 val 选 YOLO 后处理参数，再把同一策略用于 test。

脚本先生成低置信度/高 NMS IoU 的高召回 prediction artifact，再比较阈值、NMS、
Soft-NMS、WBF 等策略。部署视频可能出现额外人员或多只手，因此不使用固定 top-k。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.cli.postprocess_detection import (
    Detection,
    apply_global_threshold,
    apply_thresholds,
    class_aware_nms,
    filter_area,
    load_ground_truth,
    load_predictions,
    rounded_metrics,
    score,
    score_counts,
    soft_nms,
    weighted_box_fusion,
)
from benchmark.core.artifacts import build_detection_prediction_artifact


StrategyFn = Callable[[list[Detection]], list[Detection]]


# ============================ 集中参数区 ============================
DEFAULT_WEIGHTS = ROOT / "runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt"
DEFAULT_DATA_YAML = ROOT / "datasets/cleansight-yolo/group1_large/data.yaml"
DEFAULT_IMAGE_SIZE = 768
DEFAULT_DEVICE = "0"
DEFAULT_BATCH = 16
DEFAULT_ARTIFACT_CONF = 0.01
DEFAULT_ARTIFACT_IOU = 0.95
DEFAULT_MAX_DET = 300
DEFAULT_OUTPUT_DIR = ROOT / "runs/yolo_postprocess_next"
THRESHOLD_SEARCH_GRID = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]


def read_data_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def split_dir(data_yaml: Path, split: str) -> Path:
    payload = read_data_yaml(data_yaml)
    root = Path(str(payload.get("path") or data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    configured = payload.get(split)
    if configured is None:
        raise ValueError(f"data.yaml 未声明 split={split!r}")
    if isinstance(configured, list):
        if len(configured) != 1:
            raise ValueError("当前脚本只支持单个 split 图片目录")
        configured = configured[0]
    value = Path(str(configured)).expanduser()
    return value if value.is_absolute() else (root / value).resolve()


def labels_from_yaml(data_yaml: Path) -> dict[int, str]:
    names = read_data_yaml(data_yaml).get("names") or {}
    return {int(key): str(value) for key, value in dict(names).items()}


def _device_arg(device: str) -> str:
    return device


def generate_prediction_artifact(
    *,
    weights: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    batch: int,
    half: bool,
    out_path: Path,
    reuse: bool,
) -> Path:
    if reuse and out_path.exists():
        print(f"[reuse] {out_path}")
        return out_path

    from ultralytics import YOLO

    source = split_dir(data_yaml, split)
    labels = labels_from_yaml(data_yaml)
    model = YOLO(str(weights))
    items: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for result in model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=_device_arg(device),
        batch=batch,
        half=half,
        stream=True,
        verbose=False,
    ):
        boxes: list[dict[str, Any]] = []
        if result.boxes is not None:
            xywhn = result.boxes.xywhn.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            confidences = result.boxes.conf.detach().cpu().tolist()
            boxes = [
                {
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "xywhn": [float(value) for value in coords],
                }
                for class_id, confidence, coords in zip(classes, confidences, xywhn)
            ]
        items[Path(result.path).name] = {"predictions": boxes}

    artifact = build_detection_prediction_artifact(
        items=items,
        labels=labels,
        split=split,
        prediction_format="class_confidence_xywhn",
    )
    artifact["source"] = {
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "note": "High-recall artifact for offline post-processing exploration.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    print(f"[artifact] {split}: {out_path} images={len(items)}")
    return out_path


def class_specific_nms(items: Iterable[Detection], ious: dict[int, float]) -> list[Detection]:
    out: list[Detection] = []
    for cls, iou in sorted(ious.items()):
        out.extend(class_aware_nms((item for item in items if item.cls == cls), iou))
    known = set(ious)
    out.extend(item for item in items if item.cls not in known)
    return out


def find_thresholds(
    predictions: list[Detection],
    truth: dict[str, list[Detection]],
    labels: dict[int, str],
    grid: list[float],
) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for cls in sorted(labels):
        cls_predictions = [item for item in predictions if item.cls == cls]
        best_threshold = grid[0]
        best_f1 = -1.0
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


def make_strategies(labels: dict[int, str], thresholds: dict[int, float]) -> list[dict[str, Any]]:
    hand = next(cls for cls, name in labels.items() if name == "hand")
    body = next(cls for cls, name in labels.items() if name == "scope_control_body")
    mid = next(cls for cls, name in labels.items() if name == "scope_mid_section")

    strategies: list[dict[str, Any]] = []

    def add(name: str, params: dict[str, Any], fn: StrategyFn) -> None:
        strategies.append({"name": name, "params": params, "fn": fn})

    add("baseline_high_recall_artifact", {}, lambda items: list(items))

    for conf in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        add(
            f"global_conf_{conf:.2f}",
            {"conf": conf},
            lambda items, conf=conf: apply_global_threshold(items, conf),
        )
        for iou in [0.55, 0.65]:
            add(
                f"global_conf_{conf:.2f}_nms_{iou:.2f}",
                {"conf": conf, "nms_iou": iou},
                lambda items, conf=conf, iou=iou: class_aware_nms(apply_global_threshold(items, conf), iou),
            )

    threshold_names = {labels[cls]: value for cls, value in thresholds.items()}
    add(
        "val_per_class_conf",
        {"thresholds": threshold_names},
        lambda items: apply_thresholds(items, thresholds),
    )

    for iou in [0.55, 0.65, 0.75]:
        add(
            f"val_per_class_conf_nms_{iou:.2f}",
            {"thresholds": threshold_names, "nms_iou": iou},
            lambda items, iou=iou: class_aware_nms(apply_thresholds(items, thresholds), iou),
        )

    for hand_iou in [0.70, 0.80, 0.90]:
        for other_iou in [0.55, 0.65]:
            ious = {hand: hand_iou, body: other_iou, mid: other_iou}
            add(
                f"val_per_class_conf_class_nms_hand{hand_iou:.2f}_other{other_iou:.2f}",
                {
                    "thresholds": threshold_names,
                    "class_nms_iou": {labels[cls]: value for cls, value in ious.items()},
                },
                lambda items, ious=ious: class_specific_nms(apply_thresholds(items, thresholds), ious),
            )

    for min_area in [0.0005, 0.0010]:
        add(
            f"val_per_class_conf_min_area_{min_area:g}",
            {"thresholds": threshold_names, "min_area": min_area},
            lambda items, min_area=min_area: filter_area(apply_thresholds(items, thresholds), min_area),
        )

    for iou in [0.55, 0.65]:
        add(
            f"global_conf_0.30_softnms_gaussian_{iou:.2f}",
            {"conf": 0.30, "softnms_iou": iou, "method": "gaussian", "sigma": 0.5},
            lambda items, iou=iou: soft_nms(
                apply_global_threshold(items, 0.30),
                iou,
                score_threshold=0.30,
                method="gaussian",
                sigma=0.5,
            ),
        )
        add(
            f"global_conf_0.30_wbf_{iou:.2f}",
            {"conf": 0.30, "wbf_iou": iou},
            lambda items, iou=iou: weighted_box_fusion(apply_global_threshold(items, 0.30), iou),
        )

    return strategies


def evaluate_strategies(
    *,
    artifact_path: Path,
    labels_dir: Path,
    strategies: list[dict[str, Any]],
    labels: dict[int, str],
    full_top_k: int = 8,
    force_full_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    predictions, artifact_labels = load_predictions(artifact_path)
    if artifact_labels != labels:
        raise ValueError(f"artifact labels 不一致: {artifact_labels} != {labels}")
    truth = load_ground_truth(labels_dir)

    prelim: list[dict[str, Any]] = []
    for strategy in strategies:
        processed = strategy["fn"](predictions)
        metrics = score_counts(processed, truth, sorted(labels), 0.5)
        prelim.append(
            {
                "name": strategy["name"],
                "params": strategy["params"],
                "quick_metrics": {
                    "precision": round(float(metrics["precision"]), 4),
                    "recall": round(float(metrics["recall"]), 4),
                    "f1": round(float(metrics["f1"]), 4),
                    "num_predictions": len(processed),
                },
            }
        )
    prelim.sort(key=lambda row: (row["quick_metrics"]["f1"], row["quick_metrics"]["precision"]), reverse=True)
    full_names = {row["name"] for row in prelim[:full_top_k]}
    if force_full_names:
        full_names.update(force_full_names)

    by_name = {strategy["name"]: strategy for strategy in strategies}
    rows: list[dict[str, Any]] = []
    for name in full_names:
        strategy = by_name[name]
        processed = strategy["fn"](predictions)
        rows.append(
            {
                "name": strategy["name"],
                "params": strategy["params"],
                "metrics": rounded_metrics(score(processed, truth, labels), labels),
            }
        )
    rows.sort(key=lambda row: (row["metrics"]["f1"], row["metrics"]["precision"]), reverse=True)
    return rows


def write_report(path: Path, payload: dict[str, Any]) -> None:
    labels = payload["labels"]
    best_val = payload["val_results"][0]
    best_test = next(row for row in payload["test_results"] if row["name"] == best_val["name"])
    lines = [
        "# YOLO 后处理下一轮探索实验",
        "",
        "## 1. 实验目标",
        "",
        "本轮实验目标是继续优化 YOLO 检测输出质量，同时避免固定 top-k 这类强业务假设。实验采用 val 选择后处理参数，再把同一参数应用到 test，尽量接近后续离线视频推理上线口径。",
        "",
        "## 2. 固定条件",
        "",
        f"- 模型权重：`{payload['weights']}`",
        f"- 数据集：`{payload['data_yaml']}`",
        f"- 输入尺寸：`imgsz={payload['imgsz']}`",
        f"- 高召回预测 artifact 生成：`conf={payload['artifact_conf']}`，`iou={payload['artifact_iou']}`，`max_det={payload['max_det']}`",
        f"- 评测类别：`{', '.join(labels.values())}`",
        "",
        "## 3. 方法",
        "",
        "- 全局置信度阈值搜索。",
        "- 全局阈值 + class-aware NMS 搜索。",
        "- val 上选择逐类置信度阈值，再在 test 上复用。",
        "- 逐类阈值 + 统一 NMS IoU 搜索。",
        "- 逐类阈值 + 类别专属 NMS，重点让 `hand` 使用更宽松的 NMS IoU，减少贴近手部被误删。",
        "- 面积过滤、Soft-NMS、Weighted Box Fusion 作为对照。",
        "- 不使用固定 top-k。",
        "",
        "## 4. Val 选出的逐类阈值",
        "",
        "| 类别 | 阈值 |",
        "| --- | ---: |",
    ]
    for cls, name in labels.items():
        lines.append(f"| `{name}` | {payload['val_thresholds'][str(cls)]:.2f} |")

    lines.extend(
        [
            "",
            "## 5. Val 排名前 10",
            "",
            "| 排名 | 策略 | Precision | Recall | F1 | mAP50 | mAP50-95 | Predictions |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(payload["val_results"][:10], start=1):
        m = row["metrics"]
        lines.append(
            f"| {idx} | `{row['name']}` | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['mAP50']:.4f} | {m['mAP50_95']:.4f} | {m['num_predictions']} |"
        )

    lines.extend(
        [
            "",
            "## 6. Test 排名前 10",
            "",
            "| 排名 | 策略 | Precision | Recall | F1 | mAP50 | mAP50-95 | Predictions |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(payload["test_results"][:10], start=1):
        m = row["metrics"]
        lines.append(
            f"| {idx} | `{row['name']}` | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['mAP50']:.4f} | {m['mAP50_95']:.4f} | {m['num_predictions']} |"
        )

    lines.extend(
        [
            "",
            "## 7. 按 Val 最优策略在 Test 上的结果",
            "",
            f"- Val 最优策略：`{best_val['name']}`",
            f"- 参数：`{json.dumps(best_val['params'], ensure_ascii=False)}`",
            "",
            "| Split | Precision | Recall | F1 | mAP50 | mAP50-95 | Predictions |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, row in [("val", best_val), ("test", best_test)]:
        m = row["metrics"]
        lines.append(
            f"| {split} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['mAP50']:.4f} | {m['mAP50_95']:.4f} | {m['num_predictions']} |"
        )

    lines.extend(["", "### Test 逐类结果", "", "| 类别 | Precision | Recall | F1 | AP50 | AP50-95 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name, item in best_test["metrics"]["per_class"].items():
        lines.append(
            f"| `{name}` | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | "
            f"{item['ap50']:.4f} | {item['ap50_95']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 8. 初步结论",
            "",
            "1. 若 test 排名靠前策略与 val 最优策略一致或接近，说明该后处理参数有较好的泛化价值。",
            "2. 如果类别专属 NMS 优于统一 NMS，说明手部贴近场景确实需要和镜体类别分开处理。",
            "3. 如果面积过滤提升有限，说明误检主要不是极小框造成，而更可能来自相邻目标、遮挡或类别混淆。",
            "4. 本轮仍然只优化单帧检测输出，不直接评估 track id 连续性；接入视频离线推理时建议与 `BoT-SORT high-clean` 组合验证。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore deployable YOLO post-processing.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--data-yaml", default=str(DEFAULT_DATA_YAML))
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--artifact-conf", type=float, default=DEFAULT_ARTIFACT_CONF)
    parser.add_argument("--artifact-iou", type=float, default=DEFAULT_ARTIFACT_IOU)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    data_yaml = Path(args.data_yaml).resolve()
    out_dir = Path(args.out_dir).resolve()
    labels = labels_from_yaml(data_yaml)

    artifact_paths = {}
    for split in ["val", "test"]:
        artifact_paths[split] = generate_prediction_artifact(
            weights=weights,
            data_yaml=data_yaml,
            split=split,
            imgsz=args.imgsz,
            conf=args.artifact_conf,
            iou=args.artifact_iou,
            max_det=args.max_det,
            device=args.device,
            batch=args.batch,
            half=args.half,
            out_path=out_dir / f"artifacts/yolo11s_imgsz{args.imgsz}_{split}_conf{args.artifact_conf:g}_iou{args.artifact_iou:g}.predictions.json",
            reuse=args.reuse,
        )

    val_predictions, _ = load_predictions(artifact_paths["val"])
    val_truth = load_ground_truth(split_dir(data_yaml, "val").parents[1] / "labels" / "val")
    thresholds = find_thresholds(val_predictions, val_truth, labels, THRESHOLD_SEARCH_GRID)
    strategies = make_strategies(labels, thresholds)
    always_full_names = {
        "val_per_class_conf",
        "val_per_class_conf_nms_0.55",
        "val_per_class_conf_nms_0.65",
        "val_per_class_conf_nms_0.75",
        "val_per_class_conf_class_nms_hand0.70_other0.55",
        "val_per_class_conf_class_nms_hand0.80_other0.55",
        "val_per_class_conf_class_nms_hand0.90_other0.55",
    }

    val_rows = evaluate_strategies(
        artifact_path=artifact_paths["val"],
        labels_dir=split_dir(data_yaml, "val").parents[1] / "labels" / "val",
        strategies=strategies,
        labels=labels,
        force_full_names=always_full_names,
    )
    val_full_names = {row["name"] for row in val_rows}
    test_rows = evaluate_strategies(
        artifact_path=artifact_paths["test"],
        labels_dir=split_dir(data_yaml, "test").parents[1] / "labels" / "test",
        strategies=strategies,
        labels=labels,
        force_full_names=val_full_names | always_full_names,
    )

    payload = {
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "imgsz": args.imgsz,
        "artifact_conf": args.artifact_conf,
        "artifact_iou": args.artifact_iou,
        "max_det": args.max_det,
        "labels": labels,
        "val_thresholds": {str(cls): value for cls, value in thresholds.items()},
        "artifacts": {split: str(path) for split, path in artifact_paths.items()},
        "val_results": val_rows,
        "test_results": test_rows,
        "note": "Strategies are selected on val and reported on test. Fixed top-k is intentionally excluded.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"yolo11s_imgsz{args.imgsz}_postprocess_next.json"
    out_md = out_dir / f"yolo11s_imgsz{args.imgsz}_postprocess_next.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_md, payload)

    best_val = val_rows[0]
    best_test = next(row for row in test_rows if row["name"] == best_val["name"])
    print(f"[report] {out_md}")
    print(f"[json] {out_json}")
    print(f"[best-val] {best_val['name']} {best_val['metrics']}")
    print(f"[best-val-on-test] {best_test['name']} {best_test['metrics']}")


if __name__ == "__main__":
    main()
