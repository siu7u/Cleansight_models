"""把 auto-annotate 检测 JSON 的检测框画回视频帧，生成预览视频。

用途：人工检查 YOLO 检测质量（框位置、漏检/误检、置信度），是
``cli.annotate run`` 产出 JSON 后的直观检查手段；也是
``docs/YOLO_REVIEW_FLOW.md`` 人工审核闭环前的快速预览。

用法：
    # 基本用法：JSON + 原视频 → 预览视频
    python tools/visualize_annotations.py \
        --json outputs/annotations/<视频名>.json \
        --video path/to/<视频名>.mp4 \
        --output outputs/visualizations/<视频名>_preview.mp4

    # 只画置信度 >= 0.4 的框 / 只处理前 300 帧（长视频快速预览）
    python tools/visualize_annotations.py --json ... --video ... --output ... \
        --conf 0.4 --max-frames 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

# 每类别的显示颜色（BGR），按轨迹顺序轮转
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


def load_tracks(json_path: Path) -> list[tuple[str, dict[int, tuple[float, float, float, float, float]]]]:
    """读 auto-annotate JSON，按轨迹返回 ``(类别名, {帧号: (x, y, w, h, conf)})``。

    x/y/w/h 为左上角百分比（0-100，与 legacy 标注同口径）；conf 为检测置信度。
    """

    task = json.loads(json_path.read_text(encoding="utf-8"))[0]
    tracks: list[tuple[str, dict]] = []
    for ann in task.get("annotations", []):
        for result in ann.get("result", []):
            if result.get("type") != "videorectangle":
                continue
            value = result.get("value") or {}
            label = (value.get("labels") or [""])[0]
            if not label:
                continue
            by_frame: dict[int, tuple[float, float, float, float, float]] = {}
            for entry in value.get("sequence", []):
                if entry.get("enabled"):
                    by_frame[int(entry["frame"])] = (
                        float(entry["x"]), float(entry["y"]),
                        float(entry["width"]), float(entry["height"]),
                        float(entry.get("conf", 0.0)),
                    )
            tracks.append((label, by_frame))
    return tracks


def render_preview(
    json_path: Path,
    video_path: Path,
    output_path: Path,
    *,
    conf_threshold: float = 0.0,
    max_frames: int = 0,
) -> Path:
    """把检测框画到视频帧上，写出预览视频，返回输出路径。

    帧号按 1-based 与 JSON sequence 对齐；``max_frames<=0`` 表示处理全部帧。
    ``output_path`` 的父目录不存在时自动创建。
    """

    tracks = load_tracks(json_path)
    color_by_label = {label: CLASS_COLORS[i % len(CLASS_COLORS)] for i, (label, _) in enumerate(tracks)}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if 0 < max_frames < frame_idx:
                break
            for label, by_frame in tracks:
                det = by_frame.get(frame_idx)
                if det is None:
                    continue
                x, y, w, h, conf = det
                if conf < conf_threshold:
                    continue
                x1 = int(x / 100 * width)
                y1 = int(y / 100 * height)
                x2 = int((x + w) / 100 * width)
                y2 = int((y + h) / 100 * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_by_label[label], 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_by_label[label], 1)
            writer.write(frame)
    finally:
        cap.release()
    print(f"[visualize-annotations] {frame_idx} 帧 -> {output_path}（{width}x{height} @ {fps:.0f}fps）")
    return output_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="auto-annotate 检测 JSON → 画框预览视频")
    p.add_argument("--json", required=True, help="auto-annotate 检测 JSON（cli.annotate run 产出）")
    p.add_argument("--video", required=True, help="原视频文件（帧号与 JSON sequence 对齐）")
    p.add_argument("--output", required=True, help="预览视频输出路径（mp4）")
    p.add_argument("--conf", type=float, default=0.0, help="只画置信度 >= 此值的框（默认全部）")
    p.add_argument("--max-frames", type=int, default=0, help="最多处理帧数（0=全部，长视频快速预览用）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    render_preview(
        Path(args.json),
        Path(args.video),
        Path(args.output),
        conf_threshold=args.conf,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
