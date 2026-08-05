#!/usr/bin/env python3
"""把 ModelScope 的 YOLO 风格检测数据导入当前分组数据集。

下载后的 ActionMixed 数据集目录语义如下:
  images/{train,val,test}/*.jpg  -> 帧图片
  frames/{train,val,test}/*.txt  -> 检测框标签,使用全局类别 id
  labels/{train,val,test}/*.txt  -> 动作阶段标签,不是 YOLO 检测框

本脚本会把 frames/data.yaml 中的全局检测类别映射到 config.yaml 里的
分组类别,再把匹配的图片和标签追加到 datasets/<group> 下。训练仍使用
train/val; test 只保留给 holdout 评估。
"""
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from utils.common import ROOT, load_config


DEFAULT_SOURCE = ROOT / "raw" / "modelscope" / "cleansight-ActionMixed"
OUT_ROOT = ROOT / "datasets"
DATASET_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把下载好的 ModelScope 检测样本追加到分组 YOLO 数据集。"
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Downloaded ModelScope dataset root.",
    )
    parser.add_argument(
        "--prefix",
        default="ms_",
        help="Filename prefix for imported samples.",
    )
    return parser.parse_args()


def load_names(data_yaml: Path) -> dict[int, str]:
    """从 frames/data.yaml 读取全局检测类别 id 到名称的映射。"""
    if not data_yaml.exists():
        raise FileNotFoundError(f"missing ModelScope detection data.yaml: {data_yaml}")
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = raw.get("names") or {}
    return {int(k): str(v) for k, v in names.items()}


def write_group_data_yaml(group: str, labels: list[str]) -> None:
    """确保每个分组数据集都有 Ultralytics 需要的 data.yaml。"""
    names = "\n".join(f"  {i}: {lab}" for i, lab in enumerate(labels))
    out = OUT_ROOT / group
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(labels)}\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )


def parse_yolo_lines(label_path: Path) -> list[tuple[int, str]]:
    """从 YOLO 标签文件中读取 (class_id, box_suffix) 行。"""
    rows = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append((int(parts[0]), " ".join(parts[1:5])))
    return rows


def import_split(
    source: Path,
    split: str,
    global_names: dict[int, str],
    group_class_ids: dict[str, dict[str, int]],
    prefix: str,
) -> dict[str, int]:
    """导入一个 split,并返回每个分组的样本数量。"""
    counts = defaultdict(int)
    image_dir = source / "images" / split
    frame_label_dir = source / "frames" / split
    if not image_dir.exists() or not frame_label_dir.exists():
        return counts

    for label_path in sorted(frame_label_dir.glob("*.txt")):
        rows = parse_yolo_lines(label_path)
        if not rows:
            continue

        grouped_lines = defaultdict(list)
        for global_cid, box in rows:
            class_name = global_names.get(global_cid)
            if class_name is None:
                continue
            for group, class_ids in group_class_ids.items():
                local_cid = class_ids.get(class_name)
                if local_cid is not None:
                    grouped_lines[group].append(f"{local_cid} {box}")

        if not grouped_lines:
            continue

        image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            print(f"  [warn] missing image for {label_path.relative_to(source)}")
            continue

        out_name = f"{prefix}{image_path.name}"
        for group, lines in grouped_lines.items():
            out_img_dir = OUT_ROOT / group / "images" / split
            out_lab_dir = OUT_ROOT / group / "labels" / split
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lab_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, out_img_dir / out_name)
            (out_lab_dir / f"{Path(out_name).stem}.txt").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            counts[group] += 1
    return counts


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        raise SystemExit(f"ModelScope dataset not found: {source}")

    cfg = load_config()
    groups = cfg["groups"]
    global_names = load_names(source / "frames" / "data.yaml")
    group_class_ids = {
        group: {name: idx for idx, name in enumerate(class_names)}
        for group, class_names in groups.items()
    }

    for group, class_names in groups.items():
        write_group_data_yaml(group, class_names)

    total = defaultdict(int)
    for split in DATASET_SPLITS:
        counts = import_split(source, split, global_names, group_class_ids, args.prefix)
        for group, count in counts.items():
            total[(group, split)] += count

    print(f"ModelScope source: {source}")
    for group in groups:
        train_count = total[(group, "train")]
        val_count = total[(group, "val")]
        test_count = total[(group, "test")]
        print(f"{group}: imported train={train_count}, val={val_count}, test={test_count}")
    print("下一步: 03_train.py / 04_validate.py")


if __name__ == "__main__":
    main()
