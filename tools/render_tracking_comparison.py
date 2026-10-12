"""渲染 YOLO 检测与 tracker 轨迹的双栏/三栏对比视频。

左栏是逐帧 YOLO 框；右栏是带持久 track id 和短轨迹尾迹的结果；传入
``--tracker-before`` 时增加默认 tracker 中栏，用于默认/优化配置对照。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CLASS_NAMES = {0: "hand", 1: "scope_control_body", 2: "scope_mid_section"}
CLASS_COLORS = {
    0: (0, 220, 0),
    1: (255, 120, 0),
    2: (0, 80, 255),
}


# ============================ 集中参数区 ============================
DEFAULT_CHECKPOINT = Path("runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt")
DEFAULT_IMAGE_DIR = Path("datasets/cleansight-yolo/group1_large/images/test")
DEFAULT_TRACKER = "botsort.yaml"
DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.70
DEFAULT_IMAGE_SIZE = 640
DEFAULT_MAX_DET = 20
DEFAULT_FPS = 6.0
DEFAULT_DEVICE = "0"


def frame_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def collect_frames(image_dir: Path, task_id: int) -> list[Path]:
    return sorted(image_dir.glob(f"t{task_id}_*.jpg"), key=frame_number)


def color_for_track(track_id: int | None, class_id: int) -> tuple[int, int, int]:
    if track_id is None:
        return (160, 160, 160)
    base = CLASS_COLORS.get(class_id, (220, 220, 220))
    shift = (track_id * 41) % 90
    return tuple(min(255, channel + shift) for channel in base)


def boxes_from_result(result) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    cls = result.boxes.cls.detach().cpu().tolist()
    conf = result.boxes.conf.detach().cpu().tolist()
    ids = result.boxes.id.detach().cpu().tolist() if result.boxes.id is not None else [None] * len(xyxy)
    rows = []
    for coords, class_id, score, track_id in zip(xyxy, cls, conf, ids):
        cid = int(class_id)
        if cid not in CLASS_NAMES:
            continue
        rows.append(
            {
                "xyxy": tuple(int(round(v)) for v in coords),
                "class_id": cid,
                "confidence": float(score),
                "track_id": None if track_id is None else int(track_id),
            }
        )
    return rows


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - 8)
    cv2.rectangle(frame, (x, y0), (min(frame.shape[1] - 1, x + tw + 7), y0 + th + 8), color, -1)
    cv2.putText(frame, text, (x + 3, y0 + th + 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_predict_panel(frame, boxes: list[dict]) -> np.ndarray:
    out = frame.copy()
    per_class_index = defaultdict(int)
    for box in boxes:
        cid = box["class_id"]
        per_class_index[cid] += 1
        x1, y1, x2, y2 = box["xyxy"]
        color = CLASS_COLORS.get(cid, (220, 220, 220))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cid]} #{per_class_index[cid]} {box['confidence']:.2f}"
        draw_label(out, label, x1, y1, color)
    return out


def draw_track_panel(frame, boxes: list[dict], trails: dict[int, deque]) -> np.ndarray:
    out = frame.copy()
    active_ids = set()
    for box in boxes:
        cid = box["class_id"]
        tid = box["track_id"]
        x1, y1, x2, y2 = box["xyxy"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = color_for_track(tid, cid)
        if tid is not None:
            active_ids.add(tid)
            trails[tid].append((cx, cy, color))
            while len(trails[tid]) > 18:
                trails[tid].popleft()

    for tid, points in trails.items():
        pts = list(points)
        for i in range(1, len(pts)):
            x0, y0, color = pts[i - 1]
            x1, y1, _ = pts[i]
            alpha = i / max(1, len(pts) - 1)
            line_color = tuple(int(channel * (0.35 + 0.65 * alpha)) for channel in color)
            cv2.line(out, (x0, y0), (x1, y1), line_color, 2, cv2.LINE_AA)
        if tid not in active_ids and len(points) > 0:
            # Keep a short fading trail for temporarily lost tracks.
            points.popleft()

    for box in boxes:
        cid = box["class_id"]
        tid = box["track_id"]
        x1, y1, x2, y2 = box["xyxy"]
        color = color_for_track(tid, cid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        id_text = "no-id" if tid is None else f"ID {tid}"
        label = f"{CLASS_NAMES[cid]} {id_text} {box['confidence']:.2f}"
        draw_label(out, label, x1, y1, color)
    return out


def add_header(panel: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    h, w = panel.shape[:2]
    header = 62
    out = np.zeros((h + header, w, 3), dtype=np.uint8)
    out[:header] = (25, 25, 25)
    out[header:] = panel
    cv2.putText(out, title, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, subtitle, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Render before/after tracker comparison with trails")
    parser.add_argument("--ckpt", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, type=Path)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--tracker", default=DEFAULT_TRACKER)
    parser.add_argument("--tracker-before", default=None, help="Optional tracker for a three-panel default-vs-optimized comparison")
    parser.add_argument("--title-before", default="Default BoT-SORT track")
    parser.add_argument("--title-after", default="After: BoT-SORT track")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from ultralytics import YOLO

    frames = collect_frames(args.image_dir, args.task_id)
    if not frames:
        raise FileNotFoundError(f"No frames for task {args.task_id}")
    sample = cv2.imread(str(frames[0]))
    if sample is None:
        raise RuntimeError(f"Cannot read {frames[0]}")
    h, w = sample.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel_count = 3 if args.tracker_before else 2
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w * panel_count, h + 62))

    predict_model = YOLO(str(args.ckpt))
    before_track_model = YOLO(str(args.ckpt)) if args.tracker_before else None
    track_model = YOLO(str(args.ckpt))
    before_trails: dict[int, deque] = defaultdict(deque)
    trails: dict[int, deque] = defaultdict(deque)

    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        pred_result = predict_model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]
        before_track_result = None
        if args.tracker_before and before_track_model is not None:
            before_track_result = before_track_model.track(
                frame,
                tracker=args.tracker_before,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                max_det=args.max_det,
                device=args.device,
                persist=True,
                verbose=False,
            )[0]
        track_result = track_model.track(
            frame,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=args.device,
            persist=True,
            verbose=False,
        )[0]
        pred_boxes = boxes_from_result(pred_result)
        before_track_boxes = boxes_from_result(before_track_result) if before_track_result is not None else []
        track_boxes = boxes_from_result(track_result)
        left = draw_predict_panel(frame, pred_boxes)
        middle = draw_track_panel(frame, before_track_boxes, before_trails) if args.tracker_before else None
        right = draw_track_panel(frame, track_boxes, trails)
        left = add_header(left, "Before: YOLO predict", "Per-frame boxes only; numbers reset every frame")
        if middle is not None:
            middle = add_header(middle, args.title_before, "Persistent IDs with default tracker settings")
        right = add_header(right, args.title_after, "Persistent IDs and short trajectory trails")
        panels = [left, right] if middle is None else [left, middle, right]
        combo = np.concatenate(panels, axis=1)
        for split in range(1, panel_count):
            cv2.line(combo, (w * split, 0), (w * split, combo.shape[0]), (230, 230, 230), 2)
        writer.write(combo)

    writer.release()
    cap = cv2.VideoCapture(str(args.output))
    print({"output": str(args.output), "opened": cap.isOpened(), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS), "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "size": args.output.stat().st_size})
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
