"""时序模型预测视频：把持久化预测产物叠加到视频帧上，显示当前动作阶段。

用途：时序模型训练 + 评测完成后，把评测时持久化的预测产物
（``runs/<run>/artifacts/*.predictions.json``，prediction-artifact-v1 schema，
**不重新加载 checkpoint、不重新推理**）与训练数据集（labels/ GT 动作 +
frames/ 检测框）一起渲染成预览视频：顶部横幅显示当前抽样帧的**预测动作阶段**
（Pred，动作色整条填充），其下小字显示人工真值（GT），帧内叠加 YOLO 检测框
（``--no-boxes`` 关闭）。每帧对应数据集 ``labels/<split>/<seq>.txt`` 的一个
抽样帧，预测序列与标签行严格对齐（长度不一致直接报错，防静默错渲）。

与 ``tools/visualize_dataset.py``（只渲染数据集 GT 产物）互补：本工具渲染的是
模型在测试集上的预测结果，用于看"模型认为当前处于哪个动作阶段"。

输入：
    --artifact  预测产物 JSON（runs/*/artifacts/*.predictions.json）
    --dataset   数据集根（labels/<split>/ + frames/<split>/ 布局）
    --images    图片帧序列目录（<序列>-<帧号:06d>.jpg）或 --video 原视频
输出：--out-dir/<序列名>_pred.mp4（每个视频一个文件）

用法：
    python tools/visualize_predictions.py \
        --artifact runs/mstcn-20260817-165320/artifacts/full_sequence_temporal-mstcn-20260817-165622.predictions.json \
        --dataset datasets/cleansight-ActionMixed \
        --images datasets/cleansight-ActionMixed/images \
        --out-dir outputs/visualizations

    # 只渲染指定视频 / 视频源像素 / 快速预览
    python tools/visualize_predictions.py --artifact ... --dataset ... \
        --sequence 1fcfcdea-clip_....mp4 \
        --video path/to/<视频名>.mp4 --max-frames 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

try:
    from tools.visualize_dataset import (
        CLASS_COLORS,
        find_frame_image,
        load_class_names,
        load_frame_boxes,
    )
except ImportError:  # 以脚本方式运行（python tools/xxx.py）时 tools/ 在 sys.path
    from visualize_dataset import CLASS_COLORS, find_frame_image, load_class_names, load_frame_boxes

DETECTION_CLASSES = [
    "hand", "scope_control_body", "scope_mid_section", "scope_distal_end",
    "syringe", "air_gun", "short_brush", "brush_tip_out",
]

BANNER_HEIGHT = 30   # 顶部预测横幅高度
GT_STRIP_HEIGHT = 20  # 预测横幅下方的 GT 小字条高度


def load_artifact(path: Path) -> dict:
    """读预测产物 JSON，校验必需字段（labels / items）。"""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or "items" not in artifact or "labels" not in artifact:
        raise ValueError(f"预测产物缺少必需字段（labels/items）: {path}")
    return artifact


def _find_split(dataset_root: Path, sequence: str, prefer_split: str | None) -> str | None:
    """在数据集 labels/ 下定位序列所属 split：显式 --split > 唯一命中 > test > 排序首个。"""

    labels_root = dataset_root / "labels"
    if not labels_root.is_dir():
        return None
    candidates = sorted(
        split_dir.name
        for split_dir in labels_root.iterdir()
        if split_dir.is_dir() and (split_dir / f"{sequence}.txt").is_file()
    )
    if prefer_split:
        if prefer_split in candidates:
            return prefer_split
        raise FileNotFoundError(
            f"--split {prefer_split} 下没有序列 {sequence} 的动作标签"
        )
    if len(candidates) == 1:
        return candidates[0]
    if "test" in candidates:
        return "test"
    return candidates[0] if candidates else None


def render_prediction_video(
    artifact: dict,
    dataset_root: Path,
    sequence: str,
    *,
    images_dir: Path | None = None,
    video_path: Path | None = None,
    out_path: Path,
    split: str | None = None,
    max_frames: int = 0,
    fps: float = 0.0,
    draw_boxes: bool = True,
) -> Path:
    """把单个视频的预测序列叠加到帧图，写出预览视频，返回输出路径。

    ``artifact["items"][sequence]`` 的 ``predicted_label_ids`` / ``truth_label_ids``
    与数据集 ``labels/<split>/<seq>.txt`` 逐行对齐（长度不一致报错）。顶部横幅
    用预测动作色整条填充（白字 "Pred: <动作名>"，右上角帧号），横幅下方深色小字条
    显示真值（"GT: <动作名>"，预测不符时追加 " MISMATCH"）。像素源与缺帧跳过
    语义同 ``tools/visualize_dataset.py``。返回输出路径。
    """

    dataset_root = Path(dataset_root)
    out_path = Path(out_path)
    if images_dir is None and video_path is None:
        raise ValueError("必须提供 --images 或 --video 之一作为帧图像素源")

    items = artifact["items"]
    if sequence not in items:
        raise KeyError(f"预测产物中没有序列 {sequence}；可用: {', '.join(list(items)[:10])}")
    item = items[sequence]
    predicted_ids = list(item.get("predicted_label_ids", []))
    truth_ids = list(item.get("truth_label_ids", []))

    split = _find_split(dataset_root, sequence, split)
    if split is None:
        raise FileNotFoundError(f"数据集 labels/ 下找不到序列 {sequence} 的动作标签")
    label_path = dataset_root / "labels" / split / f"{sequence}.txt"
    label_frames = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                label_frames.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue
    if not label_frames:
        raise ValueError(f"动作标签为空: {label_path}")
    if len(predicted_ids) != len(label_frames):
        raise ValueError(
            f"{sequence}: 预测序列长度 {len(predicted_ids)} 与标签帧数 "
            f"{len(label_frames)} 不一致（预测产物与数据集不匹配，拒绝渲染）"
        )
    if truth_ids and len(truth_ids) != len(label_frames):
        raise ValueError(
            f"{sequence}: 真值序列长度 {len(truth_ids)} 与标签帧数 {len(label_frames)} 不一致"
        )
    label_id_to_name = {int(l["id"]): str(l["name"]) for l in artifact["labels"]}
    detection_names = load_class_names(dataset_root / "frames" / "data.yaml", DETECTION_CLASSES)
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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
        )

        missing_frames = 0
        drawn_frames = 0
        correct = 0
        for index, (frame_id, _gt_id) in enumerate(label_frames):
            if 0 < max_frames < drawn_frames + 1:
                break
            pred_id = predicted_ids[index]
            truth_id = truth_ids[index] if index < len(truth_ids) else None
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
                print(f"[visualize-predictions] 跳过（缺像素帧 {frame_id}）: {sequence}")
                continue

            if draw_boxes:
                box_path = frames_dir / f"{sequence}-{frame_id:06d}.txt"
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

            # 顶部预测横幅：动作色整条填充 + 白字动作名 + 帧号
            pred_color = CLASS_COLORS[pred_id % len(CLASS_COLORS)]
            cv2.rectangle(frame, (0, 0), (width, BANNER_HEIGHT), pred_color, -1)
            pred_name = label_id_to_name.get(pred_id, f"cls_{pred_id}")
            cv2.putText(frame, f"Pred: {pred_name}", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frame {frame_id}", (width - 140, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # 真值小字条：深色底 + 白字；预测不符时标注 MISMATCH
            cv2.rectangle(frame, (0, BANNER_HEIGHT), (width, BANNER_HEIGHT + GT_STRIP_HEIGHT),
                          (60, 60, 60), -1)
            if truth_id is not None:
                truth_name = label_id_to_name.get(truth_id, f"cls_{truth_id}")
                mismatch = " MISMATCH" if pred_id != truth_id else ""
                cv2.putText(frame, f"GT: {truth_name}{mismatch}", (8, BANNER_HEIGHT + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if pred_id == truth_id:
                    correct += 1

            writer.write(frame)
            drawn_frames += 1

        total = len(label_frames[:max_frames]) if max_frames > 0 else len(label_frames)
        acc = correct / total * 100 if total else 0.0
        print(
            f"[visualize-predictions] {sequence}: {drawn_frames}/{total} 帧 "
            f"frame-acc={acc:.1f}% -> {out_path} ({width}x{height} @ {out_fps:.1f}fps)"
        )
        if missing_frames:
            print(f"[visualize-predictions] 缺像素帧数: {missing_frames}")
        return out_path
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def render_artifact_videos(
    artifact: dict,
    dataset_root: Path,
    *,
    images_dir: Path | None = None,
    video_path: Path | None = None,
    out_dir: Path,
    sequence: str | None = None,
    split: str | None = None,
    max_frames: int = 0,
    fps: float = 0.0,
    draw_boxes: bool = True,
) -> list[Path]:
    """批量渲染产物内全部（或 --sequence 指定）视频，每个写 ``<out_dir>/<序列名>_pred.mp4``。

    数据集 labels/ 中找不到的视频跳过并告警（不中断其余视频）；返回产出文件列表。
    """

    out_dir = Path(out_dir)
    outputs: list[Path] = []
    names = [sequence] if sequence else sorted(artifact["items"])
    for name in names:
        if name not in artifact["items"]:
            raise KeyError(f"预测产物中没有序列 {name}；可用: {', '.join(list(artifact['items'])[:10])}")
        try:
            split_of = _find_split(dataset_root, name, split)
            if split_of is None:
                print(f"[visualize-predictions] 跳过（数据集 labels/ 中无此序列）: {name}")
                continue
            output = render_prediction_video(
                artifact,
                dataset_root,
                name,
                images_dir=images_dir,
                video_path=video_path,
                out_path=out_dir / f"{Path(name).stem}_pred.mp4",
                split=split_of,
                max_frames=max_frames,
                fps=fps,
                draw_boxes=draw_boxes,
            )
            outputs.append(output)
        except FileNotFoundError as exc:
            print(f"[visualize-predictions] 跳过（{exc}）: {name}")
    return outputs


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="时序预测产物 → 叠加预测动作阶段的预览视频（不重新推理）"
    )
    p.add_argument("--artifact", required=True, help="预测产物 JSON（runs/*/artifacts/*.predictions.json）")
    p.add_argument("--dataset", required=True, help="数据集根（labels/<split>/ + frames/<split>/ 布局）")
    p.add_argument("--images", default=None, help="图片帧序列目录（<序列>-<帧号:06d>.jpg）")
    p.add_argument("--video", default=None, help="原视频路径（按真实帧号抽取帧图）")
    p.add_argument("--out-dir", default="outputs/visualizations", help="输出目录（每视频一个 <序列名>_pred.mp4）")
    p.add_argument("--sequence", default=None, help="只渲染指定序列（缺省渲染产物内全部视频）")
    p.add_argument("--split", default=None, help="数据集 split 名（缺省自动探测：唯一命中 > test > 排序首个）")
    p.add_argument("--max-frames", type=int, default=0, help="每视频最多处理的标签帧数（0=全部）")
    p.add_argument("--fps", type=float, default=0.0, help="输出帧率（0=自动：视频源取视频帧率，图片源 7.5）")
    p.add_argument("--no-boxes", action="store_true", help="不叠加 YOLO 检测框（只显示预测/真值）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = load_artifact(Path(args.artifact))
    render_artifact_videos(
        artifact,
        Path(args.dataset),
        images_dir=Path(args.images) if args.images else None,
        video_path=Path(args.video) if args.video else None,
        out_dir=Path(args.out_dir),
        sequence=args.sequence,
        split=args.split,
        max_frames=args.max_frames,
        fps=args.fps,
        draw_boxes=not args.no_boxes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
