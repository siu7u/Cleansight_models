#!/usr/bin/env python3
"""为 CPU 增强对比实验准备子数据集。

统计全量数据集的 split 规模，并从 datasets/cleansight-yolo 确定性采样
train/val 子集到 WSL ext4 目录（默认 ~/cleansight-yolo-sub），附带 data.yaml。
样本选择：按文件名排序后等间隔抽样，保证覆盖均匀且可复现。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

REPO_DATASET = Path("/mnt/e/曦源/Cleansight_models/datasets/cleansight-yolo")
GROUPS = ("group1_large", "group2_small")
SPLITS = ("train", "val")


def stats() -> None:
    for group in GROUPS:
        for split in SPLITS:
            img_dir = REPO_DATASET / group / "images" / split
            lbl_dir = REPO_DATASET / group / "labels" / split
            n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.is_dir() else -1
            n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.is_dir() else -1
            print(f"{group}/{split}: {n_img} img / {n_lbl} lbl")


def sample(src: Path, dst: Path, n: int) -> int:
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    total = len(files)
    if total == 0:
        return 0
    if n >= total:
        chosen = files
    else:
        step = total / n
        chosen = [files[int(i * step)] for i in range(n)]
    for f in chosen:
        shutil.copy2(f, dst / f.name)
    return len(chosen)


def build(out_dir: Path, n_train: int, n_val: int) -> None:
    for group in GROUPS:
        src_g = REPO_DATASET / group
        dst_g = out_dir / group
        for split in SPLITS:
            n = n_train if split == "train" else n_val
            for sub in ("images", "labels"):
                (dst_g / sub / split).mkdir(parents=True, exist_ok=True)
            n_img = sample(src_g / "images" / split, dst_g / "images" / split, n)
            # 标签按图片基名对应复制（.jpg/.png -> .txt）
            n_lbl = 0
            src_lbl = src_g / "labels" / split
            dst_lbl = dst_g / "labels" / split
            for f in sorted((dst_g / "images" / split).glob("*.jpg")):
                label = src_lbl / (f.stem + ".txt")
                if label.is_file():
                    shutil.copy2(label, dst_lbl / label.name)
                    n_lbl += 1
            print(f"{group}/{split}: 采样 {n_img} 图 / {n_lbl} 标签")
        # data.yaml（path 用绝对路径，保留原 nc/names，避免相对 cwd 解析问题）
        src_yaml = src_g / "data.yaml"
        src_cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) if src_yaml.is_file() else {}
        dst_yaml = dst_g / "data.yaml"
        lines = [
            f"path: {dst_g}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {src_cfg.get('nc', 0)}",
            "names:",
        ]
        for k, v in (src_cfg.get("names") or {}).items():
            lines.append(f"  {k}: {v}")
        dst_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"data.yaml -> {dst_yaml}")


def main() -> None:
    global REPO_DATASET  # noqa: PLW0603 —— 允许 --src 覆盖模块级源路径

    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="只统计不采样")
    ap.add_argument("--src", default=None,
                    help="源数据集根目录（含 group1_large/group2_small 分组；默认内置路径）")
    ap.add_argument("--out", default=str(Path.home() / "cleansight-yolo-sub"),
                    help="子集输出目录（默认 ~/cleansight-yolo-sub）")
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=500)
    args = ap.parse_args()

    if args.src:
        REPO_DATASET = Path(args.src)
    if args.stats:
        stats()
        return
    build(Path(args.out), args.train, args.val)
    print("完成。训练时 data.yaml 用: ", args.out)


if __name__ == "__main__":
    main()
