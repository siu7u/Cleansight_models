"""在视频上绘制检测框，生成带标注的预览视频。

模型执行走 framework（``framework.cleansight_eval.detection.inference``），本工具只做
CLI 编排与可视化。

用法:
    # 对视频文件
    python tools/visualize_detections.py \
      --ckpt framework/runs/clean-small-v0.2/checkpoints/clean-small-v0.2.pt \
      --source /path/to/video.mp4 \
      --output output_video.mp4

    # 对图片目录（会拼成视频）
    python tools/visualize_detections.py \
      --ckpt framework/runs/clean-large-v0.2/checkpoints/clean-large-v0.2.pt \
      --source datasets/cleansight-ActionMixed/images/val/ \
      --output output.mp4 \
      --fps 24

    # 只看高置信度检测
    python tools/visualize_detections.py \
      --ckpt ... --source ... --output ... \
      --conf 0.25 --show-labels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.cleansight_eval.detection.inference import load_predictor, predict_frame


# 每个类别的显示颜色（BGR）
CLASS_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),    # 0: 绿色
    (255, 0, 0),    # 1: 蓝色
    (0, 0, 255),    # 2: 红色
    (255, 255, 0),  # 3: 青色
    (255, 0, 255),  # 4: 品红
    (0, 255, 255),  # 5: 黄色
    (128, 0, 128),  # 6: 紫色
    (0, 128, 128),  # 7: 深青
]


def _get_color(class_id: int) -> tuple[int, int, int]:
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def draw_boxes(
    frame: np.ndarray,
    detections: list[dict],
    names: dict[int, str],
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """在帧上绘制检测框。

    detections: [{"class_id": int, "confidence": float, "xywhn": [cx,cy,w,h]}, ...]
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for d in detections:
        conf = d["confidence"]
        if conf < conf_threshold:
            continue

        cls_id = d["class_id"]
        cx, cy, bw, bh = d["xywhn"]

        # 归一化坐标 → 像素坐标
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = _get_color(cls_id)
        name = names.get(cls_id, f"cls_{cls_id}")

        # 框
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # 标签背景
        label = f"{name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            overlay,
            (x1, y1 - label_h - 8),
            (x1 + label_w + 4, y1),
            color,
            -1,
        )
        # 标签文字
        cv2.putText(
            overlay,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return overlay


def run_on_video(
    model,
    source: str,
    output: str,
    conf: float,
    imgsz: int,
    names: dict[int, str],
    fps: int,
    max_frames: int,
) -> None:
    """对视频逐帧推理并写入带标注的输出视频。"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {source}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = fps if fps > 0 else (in_fps if in_fps > 0 else 24)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"输入: {source}")
    print(f"分辨率: {width}x{height}, FPS: {in_fps:.1f}, 总帧数: {total_frames}")
    print(f"输出: {output} @ {out_fps} FPS")
    if max_frames and max_frames > 0:
        print(f"最多处理: {max_frames} 帧")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, out_fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = predict_frame(model, frame, imgsz=imgsz, conf=conf)
        annotated = draw_boxes(frame, detections, names, conf)
        writer.write(annotated)
        frame_idx += 1
        if max_frames and max_frames > 0 and frame_idx >= max_frames:
            break

        if frame_idx % 100 == 0:
            det_count = len(detections)
            print(f"  帧 {frame_idx}/{total_frames} ({100*frame_idx//max(total_frames,1)}%), "
                  f"检出 {det_count} 个目标")

    cap.release()
    writer.release()
    print(f"完成: {frame_idx} 帧 → {output}")


def _collect_images(source: str) -> list[Path]:
    """收集图片目录下所有图片文件。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    path = Path(source)
    if path.is_file():
        return [path]
    images = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
    if not images:
        # 检查子目录（YOLO 数据集结构 images/val/）
        if (path / "images" / "val").is_dir():
            images = sorted(p for p in (path / "images" / "val").iterdir()
                           if p.suffix.lower() in exts)
        elif (path / "val").is_dir():
            images = sorted(p for p in (path / "val").iterdir()
                           if p.suffix.lower() in exts)
    return images


def run_on_images(
    model,
    source: str,
    output: str,
    conf: float,
    imgsz: int,
    names: dict[int, str],
    fps: int,
    max_frames: int,
) -> None:
    """对图片目录逐张推理，拼成视频输出。"""
    images = _collect_images(source)
    if not images:
        raise FileNotFoundError(f"未找到图片: {source}")

    if max_frames and max_frames > 0:
        images = images[:max_frames]

    print(f"图片数量: {len(images)}")
    print(f"输出: {output} @ {fps} FPS")

    if len(images) == 0:
        print("无图片可处理")
        return

    # 先读第一张确定尺寸
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"无法读取图片: {images[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    total_detections = 0
    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        detections = predict_frame(model, frame, imgsz=imgsz, conf=conf)
        total_detections += len(detections)
        annotated = draw_boxes(frame, detections, names, conf)
        writer.write(annotated)

        if (i + 1) % 100 == 0:
            print(f"  图片 {i+1}/{len(images)}, 累计检出 {total_detections} 个目标")

    writer.release()
    print(f"完成: {len(images)} 张图片 → {output}")
    print(f"总检出数: {total_detections} ({total_detections/max(len(images),1):.1f} 个/图)")


def _extract_dataset_name(source_path: Path, is_video: bool) -> str:
    """从 source 路径提取可读的数据集名。

    datasets/cleansight-ActionMixed/images/val/ → ActionMixed
    datasets/cleansight-yolo/group2_small/images/val/ → group2_small
    /path/to/surgery.mp4 → surgery
    """
    if is_video:
        return source_path.stem

    # 目录: 向上找到 datasets/ 或 cleansight- 开头的目录
    parts = source_path.resolve().parts
    for i, part in enumerate(parts):
        if part.startswith("cleansight-") or part == "datasets":
            if i + 1 < len(parts) and parts[i + 1].startswith("cleansight-"):
                return parts[i + 1]
            # 再往下找一层 (如 datasets/cleansight-yolo/group2_small)
            for j in range(i + 1, len(parts)):
                if parts[j] in ("images", "labels", "frames"):
                    continue
                if j > i + 1:
                    return parts[j]
            if i + 1 < len(parts):
                return parts[i + 1]
    # 兜底
    return source_path.name.rstrip("/") or "unknown"


def main():
    p = argparse.ArgumentParser(description="在视频/图片上可视化检测框")
    p.add_argument("--ckpt", required=True, help="checkpoint 路径 (.pt)")
    p.add_argument("--source", required=True,
                   help="输入视频路径 或 图片目录")
    p.add_argument("--output", default=None,
                   help="输出视频完整路径。不传则自动生成: {out_dir}/{ckpt_name}_{source_name}.mp4")
    p.add_argument("--out-dir", default="outputs",
                   help="默认输出目录 (默认 outputs/)")
    p.add_argument("--conf", type=float, default=0.001,
                   help="置信度阈值 (默认 0.001，几乎显示所有)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="推理分辨率 (默认 640)")
    p.add_argument("--fps", type=int, default=24,
                   help="输出视频帧率 (默认 24)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="最多处理帧数，0=全部 (默认 0)")
    p.add_argument("--show-labels", action="store_true", default=True,
                   help="显示类别标签和置信度 (默认开启)")
    args = p.parse_args()

    # 判断输入类型
    source_path = Path(args.source)
    is_video = source_path.is_file() and source_path.suffix.lower() in {
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"
    }

    # 生成默认输出路径: {out_dir}/{checkpoint}_{dataset}_conf{conf}.mp4
    if args.output:
        output_path = args.output
    else:
        ckpt_name = Path(args.ckpt).stem
        # 从 source 路径提取数据集名
        dataset_name = _extract_dataset_name(source_path, is_video)
        # 拼接文件名
        conf_str = str(args.conf).rstrip("0").rstrip(".") if args.conf != int(args.conf) else str(int(args.conf))
        filename = f"{ckpt_name}_{dataset_name}_conf{conf_str}.mp4"
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / filename)

    print("加载模型...")
    model, names = load_predictor(args.ckpt, args.imgsz)
    print(f"类别: {names}")
    print(f"置信度阈值: {args.conf}")

    if is_video:
        run_on_video(
            model, args.source, output_path,
            conf=args.conf, imgsz=args.imgsz, names=names, fps=args.fps,
            max_frames=args.max_frames,
        )
    else:
        run_on_images(
            model, args.source, output_path,
            conf=args.conf, imgsz=args.imgsz, names=names, fps=args.fps,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
