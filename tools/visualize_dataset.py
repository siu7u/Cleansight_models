"""持久化训练数据可视化：frames/ bbox + labels/ 动作 → 帧图叠加 → 预览视频。

用途：数据集链（切分帧 → YOLO 自动标注 → 产出时序训练数据并持久化进数据集）的
**训练产物检查**——把 ``temporal/data.py`` 实际消费的 ``frames/<split>/`` 检测框
与 ``labels/<split>/`` 动作标签直接画回帧图，人工核对框位置/类别/动作阶段是否
符合预期。全程只读持久化产物，不加载模型、不做动态推理（与数据集链
"持久化进数据集，不每次动态推理"的原则一致）。

输入布局（与 convert / run-dataset 产出同构，如 datasets/cleansight-ActionMixed-auto）：
    <dataset>/labels/<split>/<序列>.txt                每行 "frame_id action_id"
    <dataset>/frames/<split>/<序列>-<帧号:06d>.txt     每行 "class_id cx cy w h"
    <dataset>/labels/data.yaml + frames/data.yaml      类别映射（缺失时按内置表兜底）

像素源二选一（--images 优先）：
    --images <目录>   图片帧序列（<序列>-<帧号:06d>.jpg，如 datasets/.../images/train）
    --video <路径>    原视频（按真实帧号抽取，如数据集构建时使用的视频目录）

用法：
    # 图片帧源（run-dataset 通道）
    python tools/visualize_dataset.py \
        --dataset datasets/cleansight-ActionMixed-auto --split train \
        --sequence 05ba4406-clip_....mp4 \
        --images datasets/cleansight-ActionMixed/images/train \
        --output outputs/visualizations/<序列>_dataset_preview.mp4

    # 视频源（数据集未持久化帧图时；--sequence 可省略，默认取 split 第一个序列）
    python tools/visualize_dataset.py --dataset datasets/cleansight-ActionMixed-auto \
        --split train \
        --video path/to/<视频名>.mp4 \
        --output outputs/visualizations/<序列>_dataset_preview.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

# 每类别的显示颜色（BGR），按类别 id 轮转（与 tools/visualize_annotations.py 同款色板）
CLASS_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),    # 绿
    (255, 0, 0),    # 蓝
    (0, 0, 255),    # 红
    (255, 255, 0),  # 青
    (255, 0, 255),  # 品红
    (0, 255, 255),  # 黄
    (128, 0, 128),  # 紫
    (0, 128, 128),  # 深青
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# data.yaml 缺失时的兜底类别表（与 auto_annotate._constants 顺序一致）
DETECTION_CLASSES = [
    "hand", "scope_control_body", "scope_mid_section", "scope_distal_end",
    "syringe", "air_gun", "short_brush", "brush_tip_out",
]
ACTION_CLASSES = ["idle", "air_gun_cleaning", "brush_cleaning", "scope_cleaning",
                  "syringe_cleaning", "short_brush_cleaning"]


def load_class_names(yaml_path: Path, fallback: list[str]) -> dict[int, str]:
    """读 ``frames/data.yaml``（或 ``labels/data.yaml``）的 names，返回 ``{id: 名称}``。

    文件不存在或没有 names 字段时返回 ``{i: fallback[i]}`` 兜底表（仅显示用途，
    不校验类别数与训练配置的一致性）。
    """

    if yaml_path.is_file():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {i: str(name) for i, name in enumerate(names)}
    return {i: name for i, name in enumerate(fallback)}


def load_label_frames(label_path: Path) -> list[tuple[int, int]]:
    """读动作标签 ``"frame_id action_id"``，返回 ``[(真实帧号, 动作 id), ...]``。

    行格式非法（非两列）时跳过，与 ``temporal/data.py`` 的解析口径一致。
    """

    frames: list[tuple[int, int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                frames.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue  # 非整数行跳过，与 temporal/data.py 的解析口径一致
    return frames


def load_frame_boxes(box_path: Path) -> list[tuple[int, float, float, float, float]]:
    """读单帧 bbox ``"class_id cx cy w h"``（YOLO 归一化中心点），返回解析后的列表。"""

    boxes: list[tuple[int, float, float, float, float]] = []
    if not box_path.is_file():
        return boxes
    for line in box_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append(tuple(float(v) for v in parts))  # type: ignore[arg-type]
    return boxes


def find_frame_image(images_dir: Path, sequence: str, frame: int) -> Path | None:
    """在图片帧序列目录中查找 ``<序列>-<帧号:06d>.<ext>``，找不到返回 None。"""

    stem = f"{sequence}-{frame:06d}"
    for ext in sorted(IMAGE_EXTENSIONS):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def render_dataset_preview(
    dataset_root: Path,
    split: str,
    sequence: str,
    *,
    images_dir: Path | None = None,
    video_path: Path | None = None,
    output_path: Path,
    max_frames: int = 0,
    fps: float = 0.0,
) -> Path:
    """把持久化训练数据画回帧图，写出预览视频，返回输出路径。

    每个动作标签帧（``labels/<split>/<序列>.txt``）从像素源取图（``--images``
    优先，其次 ``--video`` 按真实帧号 ``frame-1`` 抽取），叠加 ``frames/`` 中的
    检测框（类别名 + 色板）与左上角动作标签、右上角帧号。缺失 bbox 文件时
    该帧只画动作标签并告警（统计数量）；缺失像素帧时跳过该帧并告警。
    全程不加载模型。``max_frames<=0`` 处理全部标签帧；``fps<=0`` 时视频源取
    视频帧率、图片源取 7.5（与训练数据抽样率一致）。
    """

    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    if images_dir is None and video_path is None:
        raise ValueError("必须提供 --images 或 --video 之一作为帧图像素源")

    label_path = dataset_root / "labels" / split / f"{sequence}.txt"
    if not label_path.is_file():
        available = sorted(
            p.name[:-4] for p in (dataset_root / "labels" / split).glob("*.txt")
        )
        raise FileNotFoundError(
            f"序列 {sequence} 在 {split} 无动作标签: {label_path}"
            + (f"；可用: {', '.join(available[:10])}" if available else "")
        )
    label_frames = load_label_frames(label_path)
    if not label_frames:
        raise ValueError(f"动作标签为空: {label_path}")

    detection_names = load_class_names(dataset_root / "frames" / "data.yaml", DETECTION_CLASSES)
    action_names = load_class_names(dataset_root / "labels" / "data.yaml", ACTION_CLASSES)
    frames_dir = dataset_root / "frames" / split

    # 像素源：图片目录按帧号找图；视频按帧号 seek（帧号 1-based → 索引 frame-1）
    cap: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    try:
        if video_path is not None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            video_fps = 0.0
            probe = None
            for label_frame, _action in label_frames:
                probe = find_frame_image(Path(images_dir), sequence, label_frame)
                if probe is not None:
                    break
            if probe is None:
                raise FileNotFoundError(
                    f"images 目录中找不到序列 {sequence} 的帧图: {images_dir}"
                )
            probe_image = cv2.imread(str(probe))
            if probe_image is None:
                raise ValueError(f"无法读取帧图: {probe}")
            height, width = probe_image.shape[:2]

        out_fps = fps if fps > 0 else (video_fps if video_fps > 0 else 7.5)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
        )

        missing_boxes = 0
        missing_frames = 0
        drawn_frames = 0
        for frame_id, action_id in label_frames:
            if 0 < max_frames < drawn_frames + 1:
                break
            if images_dir is not None:
                image_path = find_frame_image(Path(images_dir), sequence, frame_id)
                frame = cv2.imread(str(image_path)) if image_path is not None else None
            else:
                assert cap is not None
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
                ok, frame = cap.read()
                if not ok:
                    frame = None
            if frame is None:
                missing_frames += 1
                print(f"[visualize-dataset] 跳过（缺像素帧 {frame_id}）: {sequence}")
                continue

            box_path = frames_dir / f"{sequence}-{frame_id:06d}.txt"
            if not box_path.is_file():
                missing_boxes += 1
                print(
                    f"[visualize-dataset] 告警: 帧 {frame_id} 无 bbox 文件（只画动作标签）: {box_path.name}"
                )
            for class_id, cx, cy, w, h in load_frame_boxes(box_path):
                color = CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]
                name = detection_names.get(int(class_id), f"cls_{int(class_id)}")
                x1 = int((cx - w / 2) * width)
                y1 = int((cy - h / 2) * height)
                x2 = int((cx + w / 2) * width)
                y2 = int((cy + h / 2) * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            action_name = action_names.get(action_id, f"action_{action_id}")
            action_color = CLASS_COLORS[action_id % len(CLASS_COLORS)]
            (label_w, label_h), _baseline = cv2.getTextSize(
                action_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (4, 4), (4 + label_w + 6, 4 + label_h + 8), action_color, -1)
            cv2.putText(frame, action_name, (7, 4 + label_h + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"frame {frame_id}", (width - 130, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            drawn_frames += 1
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()

    print(
        f"[visualize-dataset] {sequence}: {drawn_frames} 帧 -> {output_path} "
        f"({width}x{height} @ {out_fps:.1f}fps)"
    )
    if missing_boxes:
        print(f"[visualize-dataset] 缺 bbox 文件帧数: {missing_boxes}")
    if missing_frames:
        print(f"[visualize-dataset] 缺像素帧数: {missing_frames}")
    return output_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="持久化训练数据 → 画框预览视频（只读 frames/ + labels/，不加载模型）"
    )
    p.add_argument("--dataset", required=True, help="数据集根（labels/<split>/ + frames/<split>/ 布局）")
    p.add_argument("--split", default="train", help="split 名（默认 train）")
    p.add_argument("--sequence", default=None, help="序列名（labels/<split>/ 下 .txt 文件名，缺省取第一个）")
    p.add_argument("--images", default=None, help="图片帧序列目录（<序列>-<帧号:06d>.jpg）")
    p.add_argument("--video", default=None, help="原视频路径（按真实帧号抽取帧图）")
    p.add_argument("--output", required=True, help="预览视频输出路径（mp4）")
    p.add_argument("--max-frames", type=int, default=0, help="最多处理的标签帧数（0=全部）")
    p.add_argument("--fps", type=float, default=0.0, help="输出帧率（0=自动：视频源取视频帧率，图片源 7.5）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sequence = args.sequence
    if sequence is None:
        split_dir = Path(args.dataset) / "labels" / args.split
        candidates = sorted(split_dir.glob("*.txt")) if split_dir.is_dir() else []
        if not candidates:
            raise FileNotFoundError(f"{split_dir} 下没有序列可渲染")
        sequence = candidates[0].name[:-4]
        print(f"[visualize-dataset] --sequence 未指定，取第一个序列: {sequence}")
    render_dataset_preview(
        Path(args.dataset),
        args.split,
        sequence,
        images_dir=Path(args.images) if args.images else None,
        video_path=Path(args.video) if args.video else None,
        output_path=Path(args.output),
        max_frames=args.max_frames,
        fps=args.fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
