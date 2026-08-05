"""ROI 数据集构建：从 YOLO 数据集 GT 框裁剪正样本 + 背景负样本。

输出 ``X: [N, roi_size, roi_size, 3]``（BGR）与多标签 ``y: [N, num_classes]``，
保存为 ``runs/feature_fusion/datasets/<classes>/`` 下的 npy + meta.json。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def build_roi_dataset(
    group_dir: Path,
    classes: List[str],
    roi_size: int = 224,
    neg_ratio: float = 1.0,
) -> Tuple["object", "object", List[str], dict]:
    """
    从 YOLO 数据集中为指定类别提取 ROI 区域。

    返回 (X, y, class_names, stats)；X/y 为 numpy 数组（BGR 图像块与多标签）。
    """

    import cv2
    import numpy as np

    group_dir = Path(group_dir)
    data_yaml = group_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml 缺失: {data_yaml}")

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    all_names = cfg["names"]
    name_to_id = {name: i for i, name in enumerate(all_names)}
    target_ids = {name_to_id[c] for c in classes if c in name_to_id}
    if not target_ids:
        raise ValueError(f"指定类别 {classes} 不在 data.yaml 中: {list(all_names)}")
    target_names = [c for c in classes if c in name_to_id]
    print(f"[build] 目标类别: {target_names} (IDs: {target_ids})")

    samples = []
    stats = {
        "total_frames": 0,
        "per_class_pos": {c: 0 for c in target_names},
        "neg_frames": 0,
        "neg_rois": 0,
    }

    for split in ("train", "val"):
        img_dir = group_dir / "images" / split
        lbl_dir = group_dir / "labels" / split
        if not img_dir.is_dir():
            continue

        img_files = sorted([f for f in img_dir.iterdir()
                           if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        print(f"[build] 处理 {split}: {len(img_files)} 张图片")

        for img_path in img_files:
            stats["total_frames"] += 1
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            frame_boxes = []
            if lbl_path.is_file():
                for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    frame_boxes.append((cid, cx, cy, bw, bh))

            frame_has_target = False
            class_vector = np.zeros(len(target_names), dtype=np.float32)
            for cid, cx, cy, bw, bh in frame_boxes:
                if cid not in target_ids:
                    continue
                frame_has_target = True
                cls_idx = target_names.index(all_names[cid])
                class_vector[cls_idx] = 1.0
                stats["per_class_pos"][all_names[cid]] += 1

                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                pad_w = int((x2 - x1) * 0.2)
                pad_h = int((y2 - y1) * 0.2)
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(w, x2 + pad_w)
                y2 = min(h, y2 + pad_h)
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = img[y1:y2, x1:x2]
                roi = cv2.resize(roi, (roi_size, roi_size))
                samples.append((roi, class_vector.copy()))

            if not frame_has_target and neg_ratio > 0:
                stats["neg_frames"] += 1
                num_negs = max(1, int(neg_ratio))
                for _ in range(num_negs):
                    crop_w = np.random.randint(roi_size, max(roi_size + 1, w // 3))
                    crop_h = np.random.randint(roi_size, max(roi_size + 1, h // 3))
                    x1 = np.random.randint(0, max(1, w - crop_w))
                    y1 = np.random.randint(0, max(1, h - crop_h))
                    roi = img[y1:y1 + crop_h, x1:x1 + crop_w]
                    roi = cv2.resize(roi, (roi_size, roi_size))
                    samples.append((roi, np.zeros(len(target_names), dtype=np.float32)))
                    stats["neg_rois"] += 1

    if not samples:
        raise RuntimeError("未生成任何样本！请检查数据集路径和类别名")

    X = np.stack([s[0] for s in samples], axis=0)
    y = np.stack([s[1] for s in samples], axis=0)

    print(f"[build] 完成: X={X.shape} y={y.shape}")
    for c, n in stats["per_class_pos"].items():
        print(f"    {c}: {n}")
    print(f"  负样本: {stats['neg_rois']}")

    return X, y, target_names, stats


def save_dataset(X, y, classes: List[str], stats: dict, base_dir: Path) -> Path:
    """保存 ROI 数据集到磁盘。"""

    import numpy as np

    ds_dir = base_dir / "-".join(classes)
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / "X.npy", X)
    np.save(ds_dir / "y.npy", y)
    (ds_dir / "meta.json").write_text(
        json.dumps({"classes": classes, "stats": stats,
                    "X_shape": list(X.shape), "y_shape": list(y.shape)},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[save] 数据集已保存到: {ds_dir}")
    return ds_dir


def load_dataset(classes: List[str], base_dir: Path):
    """加载已保存的 ROI 数据集，返回 (X, y, classes)。"""

    import numpy as np

    ds_dir = base_dir / "-".join(classes)
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"数据集不存在: {ds_dir}\n请先构建（训练时自动构建）")
    X = np.load(ds_dir / "X.npy")
    y = np.load(ds_dir / "y.npy")
    meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
    print(f"[load] X={X.shape} y={y.shape} classes={meta['classes']}")
    return X, y, meta["classes"]
