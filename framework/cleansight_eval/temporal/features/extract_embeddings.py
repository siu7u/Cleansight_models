"""形态 B（像素特征进时序）第一步：逐帧整图 CNN embedding 离线预计算。

把 ActionMixed 式数据（``labels/<split>/<video>.mp4.txt`` + ``images/<split>/``
帧图）的每个标签帧过一遍 CNN backbone，产出与标签行**一一对齐**的逐视频 embedding：

    <out>/<split>/<video>.mp4.npy      # [T, D] float32，T = 标签行数
    <out>/meta.json                    # 方案/backbone/输入尺寸/逐视频统计

设计口径（与 IMAGE_FEATURE_TRAINING.md §4 一致）：
- **因果性**：每帧 embedding 只依赖当前帧图像，无跨帧聚合；
- **确定性**：eval + no_grad + 无随机，同权重同设备结果一致；缺图帧补零并记录
  （语义与 bbox 契约"空帧全零"一致），不静默跳过导致错位；
- 预计算产物作为训练输入（load_split 读 npy），训练侧不再依赖图像与 GPU。

用法示例：

    python -m framework.cleansight_eval.temporal.features.extract_embeddings \
        --root datasets/cleansight-ActionMixed --splits train val test \
        --backbone resnet18 --out-dir runs/image_embeddings/actionmixed-resnet18-v1

torch/torchvision/cv2 为重依赖，全部在函数内 import（仓库惯例），纯配置/校验
场景无需安装。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# backbone 名 → (torchvision 加载函数名, 输入尺寸, 特征维度)
BACKBONES: dict[str, tuple[str, int, int]] = {
    "resnet18": ("resnet18", 224, 512),
    "resnet34": ("resnet34", 224, 512),
    "resnet50": ("resnet50", 224, 2048),
    "mobilenet_v3_small": ("mobilenet_v3_small", 224, 576),
    "efficientnet_b0": ("efficientnet_b0", 224, 1280),
}

# 与 CLEAN/队友工具一致的 ImageNet 归一化口径（预处理写入 meta 供契约追溯）
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def build_backbone(name: str, device: str):
    """加载预训练 backbone 并去掉分类头，返回 (model, input_size, feat_dim)。

    权重枚举名解析与 ``classification/model.py`` 的 FeatureFusionModel 同规
    （resnet / efficientnet / mobilenet 三族），保证两处可互换。
    """

    import torch
    import torch.nn as nn
    import torchvision.models as tv_models

    loader_name, input_size, feat_dim = BACKBONES[name]
    model_fn = getattr(tv_models, loader_name)
    if loader_name.startswith("resnet"):
        weights = getattr(tv_models, f"ResNet{loader_name[6:]}_Weights").DEFAULT
        model = model_fn(weights=weights)
        model.fc = nn.Identity()
    elif loader_name.startswith("efficientnet"):
        key = f"EfficientNet_{loader_name.split('_')[1].upper()}_Weights"
        model = model_fn(weights=getattr(tv_models, key).DEFAULT)
        model.classifier = nn.Identity()
    elif loader_name.startswith("mobilenet"):
        key = f"MobileNet_V3_{'Small' if 'small' in loader_name else 'Large'}_Weights"
        model = model_fn(weights=getattr(tv_models, key).DEFAULT)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"不支持的 backbone: {name}")
    model.eval().to(torch.device(device))
    return model, input_size, feat_dim


def _frame_image_path(images_dir: Path, stem: str, frame_id: int) -> Path:
    """标签帧号 → 图像路径（帧号 6 位补零，与 frames/ txt 命名同规）。"""

    return images_dir / f"{stem}-{frame_id:06d}.jpg"


def load_frame_rgb(path: Path, size: int):
    """读一帧图（cv2.imdecode，支持非 ASCII 路径），返回 RGB uint8 [size, size, 3]。"""

    import cv2
    import numpy as np

    if not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def _preprocess_tensor(images: list, size: int, device: str):
    """RGB 帧列表 → [B, 3, size, size] 归一化张量。"""

    import numpy as np
    import torch

    array = np.stack(images, axis=0).astype(np.float32) / 255.0  # [B, H, W, 3]
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to(device)
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def extract_split(
    root: Path,
    split: str,
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    backbone_name: str,
    device: str,
    batch_size: int = 32,
    max_videos: int | None = None,
    max_frames: int | None = None,
) -> dict:
    """提取一个 split 的全部视频 embedding；返回统计 dict。"""

    import numpy as np
    import torch

    model, input_size, feat_dim = build_backbone(backbone_name, device)
    out_split = out_dir / split
    out_split.mkdir(parents=True, exist_ok=True)

    stats: dict = {"videos": 0, "frames": 0, "missing_images": []}
    t0 = time.perf_counter()
    for label_file in sorted(labels_dir.glob("*.mp4.txt")):
        if max_videos is not None and stats["videos"] >= max_videos:
            break
        stem = label_file.name[:-4]  # "<video>.mp4"
        rows = [line.split() for line in label_file.read_text().splitlines() if len(line.split()) == 2]
        if max_frames is not None:
            rows = rows[:max_frames]
        if not rows:
            continue
        embeddings = np.zeros((len(rows), feat_dim), dtype=np.float32)
        pending_idx: list[int] = []
        pending_frames: list = []
        missing: list[int] = []
        for index, (frame_text, _action) in enumerate(rows):
            frame_id = int(frame_text)
            path = _frame_image_path(images_dir, stem, frame_id)
            frame = load_frame_rgb(path, input_size)
            if frame is None:
                missing.append(frame_id)
                continue
            pending_idx.append(index)
            pending_frames.append(frame)
            if len(pending_frames) >= batch_size:
                with torch.no_grad():
                    feats = model(_preprocess_tensor(pending_frames, input_size, device))
                for idx, feat in zip(pending_idx, feats.detach().cpu().numpy()):
                    embeddings[idx] = feat
                pending_idx, pending_frames = [], []
        if pending_frames:
            with torch.no_grad():
                feats = model(_preprocess_tensor(pending_frames, input_size, device))
            for idx, feat in zip(pending_idx, feats.detach().cpu().numpy()):
                embeddings[idx] = feat
        np.save(out_split / f"{stem}.npy", embeddings)
        stats["videos"] += 1
        stats["frames"] += len(rows)
        for frame_id in missing:
            stats["missing_images"].append(f"{stem}:{frame_id}")
        if stats["videos"] <= 2 or max_videos == stats["videos"]:
            print(f"  [{split}] {stem}: {len(rows)} 帧 → {out_split / (stem + '.npy')}"
                  f"{'（缺图 ' + str(len(missing)) + ' 帧补零）' if missing else ''}")

    stats["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="逐帧整图 CNN embedding 离线预计算（形态 B 第一步）")
    parser.add_argument("--root", required=True, help="数据集根目录（含 labels/ 与 images/）")
    parser.add_argument("--splits", default="train,val,test", help="逗号分隔的 split 列表")
    parser.add_argument("--backbone", default="resnet18", choices=sorted(BACKBONES),
                        help="CNN backbone（torchvision 预训练）")
    parser.add_argument("--out-dir", required=True, help="产物根目录（<out>/<split>/<video>.mp4.npy + meta.json）")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0")
    parser.add_argument("--max-videos", type=int, default=None, help="每 split 最多处理视频数（smoke）")
    parser.add_argument("--max-frames", type=int, default=None, help="每视频最多处理帧数（smoke）")
    args = parser.parse_args(argv)

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(args.root).resolve()
    images_dir = root / "images"
    labels_root = root / "labels"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _, input_size, feat_dim = BACKBONES[args.backbone]

    all_stats: dict = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        labels_dir = labels_root / split
        if not labels_dir.is_dir():
            print(f"[skip] labels split 目录不存在: {labels_dir}")
            continue
        split_images = images_dir / split
        print(f"== {split} ==")
        all_stats[split] = extract_split(
            root, split, split_images, labels_dir, out_dir,
            args.backbone, device,
            batch_size=args.batch_size,
            max_videos=args.max_videos,
            max_frames=args.max_frames,
        )

    meta = {
        "scheme": "frame_embedding",
        "backbone": args.backbone,
        "input_size": input_size,
        "feat_dim": feat_dim,
        "device": device,
        "preprocess": {
            "resize": f"{input_size}x{input_size}",
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
            "color": "BGR->RGB",
        },
        "semantics": "每帧整图 embedding，与标签行一一对齐；缺图帧补零；因果、无状态",
        "stats": all_stats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"完成: {out_dir}/meta.json")


if __name__ == "__main__":
    sys.exit(main())
