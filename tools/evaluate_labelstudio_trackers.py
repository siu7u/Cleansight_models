"""用 Label Studio 原生轨迹真值评估 YOLO tracker。

YOLO txt 只有逐帧框，没有 track id；Label Studio 的 ``videorectangle.sequence``
保留对象时间线，因此本脚本用 result.id 恢复 GT track，并计算检测 P/R/F1、IDF1、
ID switches、fragments 等指标。参数默认值集中在文件顶部，命令行可覆盖。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CLASS_NAMES = {0: "hand", 1: "scope_control_body", 2: "scope_mid_section"}
LABEL_TO_CLASS = {name: idx for idx, name in CLASS_NAMES.items()}
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 215, 255), (255, 0, 255), (255, 255, 0)]


# ============================ 集中参数区 ============================
DEFAULT_LABELSTUDIO = Path("datasets/labelstudio-yolo-test")
DEFAULT_IMAGE_DIR = Path("datasets/cleansight-yolo/group1_large/images/test")
DEFAULT_CHECKPOINT = Path("runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt")
DEFAULT_OUTPUT_DIR = Path("runs/labelstudio_track_eval")
DEFAULT_IMAGE_SIZE = 640
DEFAULT_MAX_DET = 20
DEFAULT_DEVICE = "0"
DEFAULT_IOU_MATCH = 0.5
DEFAULT_TRACKERS = "bytetrack.yaml,botsort.yaml,ocsort.yaml,deepocsort.yaml,tracktrack.yaml,fasttrack.yaml"
DEFAULT_CONFS = "0.15,0.25,0.30,0.40"
DEFAULT_NMS_IOUS = "0.50,0.70"


@dataclass(frozen=True)
class Box:
    task_id: int
    frame: int
    class_id: int
    track_id: str | None
    conf: float
    xyxy: tuple[float, float, float, float]


def load_labelstudio_tasks(path: Path) -> list[dict]:
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"没有找到 Label Studio JSON: {path}")
    data = json.loads(files[0].read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"暂只支持 Label Studio list export: {files[0]}")
    return data


def build_segments(seq: list[dict]) -> list[tuple[int, dict, int, dict]]:
    seq = sorted(seq, key=lambda item: int(item["frame"]))
    segs = []
    for a, b in zip(seq, seq[1:]):
        if a.get("enabled", True):
            segs.append((int(a["frame"]), a, int(b["frame"]), b))
    if seq and seq[-1].get("enabled", True):
        last = seq[-1]
        segs.append((int(last["frame"]), last, int(last["frame"]), last))
    return segs


def box_at(segs: list[tuple[int, dict, int, dict]], frame: float) -> tuple[float, float, float, float] | None:
    for f0, b0, f1, b1 in segs:
        if f0 <= frame <= f1:
            t = 0.0 if f1 == f0 else (frame - f0) / (f1 - f0)
            return (
                float(b0["x"]) + (float(b1["x"]) - float(b0["x"])) * t,
                float(b0["y"]) + (float(b1["y"]) - float(b0["y"])) * t,
                float(b0["width"]) + (float(b1["width"]) - float(b0["width"])) * t,
                float(b0["height"]) + (float(b1["height"]) - float(b0["height"])) * t,
            )
    return None


def ls_percent_to_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (
        max(0.0, x / 100.0 * width),
        max(0.0, y / 100.0 * height),
        min(float(width), (x + w) / 100.0 * width),
        min(float(height), (y + h) / 100.0 * height),
    )


def frame_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def task_frames(image_dir: Path, task_id: int) -> list[Path]:
    return sorted(image_dir.glob(f"t{task_id}_*.jpg"), key=frame_number)


def infer_real_fps(frames: list[Path], duration: float | None) -> float:
    if not frames or not duration:
        return 30.0
    max_frame = max(frame_number(path) for path in frames)
    if max_frame <= 0:
        return 30.0
    fps = max_frame / float(duration)
    if 20 <= fps <= 35:
        return fps
    return 30.0


def task_tracks(task: dict, image_dir: Path) -> tuple[list[dict], dict]:
    task_id = int(task["id"])
    frames = task_frames(image_dir, task_id)
    rects = []
    for ann_idx, ann in enumerate(task.get("annotations", []) or []):
        for result_idx, result in enumerate(ann.get("result", []) or []):
            if result.get("type") != "videorectangle":
                continue
            value = result.get("value") or {}
            labels = value.get("labels") or []
            if not labels or labels[0] not in LABEL_TO_CLASS:
                continue
            seq = value.get("sequence") or []
            if not seq:
                continue
            rects.append(
                {
                    "track_id": f"t{task_id}:{result.get('id') or ann_idx}_{result_idx}",
                    "class_id": LABEL_TO_CLASS[labels[0]],
                    "label": labels[0],
                    "segments": build_segments(seq),
                    "frames_count": value.get("framesCount"),
                    "duration": value.get("duration"),
                }
            )
    duration = next((item.get("duration") for item in rects if item.get("duration")), None)
    frames_count = next((item.get("frames_count") for item in rects if item.get("frames_count")), None)
    real_fps = infer_real_fps(frames, duration)
    ls_fps = (float(frames_count) / float(duration)) if frames_count and duration else real_fps
    scale = ls_fps / real_fps if real_fps else 1.0
    meta = {"task_id": task_id, "frames": len(frames), "duration": duration, "frames_count": frames_count, "real_fps_est": real_fps, "ls_fps": ls_fps, "scale": scale}
    return rects, meta


def gt_for_frame(task_id: int, real_frame: int, image_shape: tuple[int, int], tracks: list[dict], scale: float) -> list[Box]:
    h, w = image_shape[:2]
    ls_frame = real_frame * scale
    boxes: list[Box] = []
    for track in tracks:
        raw = box_at(track["segments"], ls_frame)
        if raw is None:
            continue
        x1, y1, x2, y2 = ls_percent_to_xyxy(raw, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(Box(task_id, real_frame, int(track["class_id"]), str(track["track_id"]), 1.0, (x1, y1, x2, y2)))
    return boxes


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def greedy_match(gt: list[Box], pred: list[Box], iou_thr: float) -> list[tuple[int, int, float]]:
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g.class_id != p.class_id:
                continue
            value = iou(g.xyxy, p.xyxy)
            if value >= iou_thr:
                candidates.append((value, gi, pi))
    matches = []
    used_g, used_p = set(), set()
    for value, gi, pi in sorted(candidates, reverse=True):
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, value))
    return matches


def boxes_from_result(result: Any, task_id: int, frame: int) -> list[Box]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    cls = result.boxes.cls.detach().cpu().tolist()
    confs = result.boxes.conf.detach().cpu().tolist()
    ids = result.boxes.id.detach().cpu().tolist() if result.boxes.id is not None else [None] * len(xyxy)
    boxes = []
    for coords, class_id, conf, track_id in zip(xyxy, cls, confs, ids):
        cid = int(class_id)
        if cid not in CLASS_NAMES:
            continue
        scoped_id = None if track_id is None else f"t{task_id}:p{int(track_id)}"
        boxes.append(Box(task_id, frame, cid, scoped_id, float(conf), tuple(float(v) for v in coords)))
    return boxes


def reset_tracker_state(model) -> None:
    trackers = getattr(model, "trackers", None)
    if not trackers:
        return
    for tracker in trackers:
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()


def inference_options(method: dict) -> dict:
    options = {
        "conf": method["conf"],
        "iou": method["iou"],
        "imgsz": method["imgsz"],
        "max_det": method["max_det"],
        "device": method["device"],
        "verbose": False,
    }
    if method.get("half"):
        options["half"] = True
    return options


def evaluate_matches(gt_all: list[Box], pred_all: list[Box], matches_by_frame: dict[tuple[int, int], list[tuple[Box, Box, float]]]) -> dict:
    matched_pairs = [pair for pairs in matches_by_frame.values() for pair in pairs]
    tp = len(matched_pairs)
    fp = len(pred_all) - tp
    fn = len(gt_all) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    pair_counts: Counter[tuple[str, str]] = Counter()
    gt_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    timeline: dict[str, list[tuple[int, int, str | None]]] = defaultdict(list)
    for gt in gt_all:
        if gt.track_id is not None:
            gt_counts[gt.track_id] += 1
    for pred in pred_all:
        if pred.track_id is not None:
            pred_counts[pred.track_id] += 1
    for (task_id, frame), pairs in matches_by_frame.items():
        matched_gt = set()
        for gt, pred, _value in pairs:
            matched_gt.add(gt.track_id)
            if gt.track_id is not None and pred.track_id is not None:
                pair_counts[(gt.track_id, pred.track_id)] += 1
                timeline[gt.track_id].append((task_id, frame, pred.track_id))
        for gt in [item for item in gt_all if item.task_id == task_id and item.frame == frame]:
            if gt.track_id not in matched_gt:
                timeline[gt.track_id].append((task_id, frame, None))

    used_gt, used_pred = set(), set()
    idtp = 0
    for (gt_id, pred_id), count in sorted(pair_counts.items(), key=lambda item: item[1], reverse=True):
        if gt_id in used_gt or pred_id in used_pred:
            continue
        used_gt.add(gt_id)
        used_pred.add(pred_id)
        idtp += count
    idfp = len([p for p in pred_all if p.track_id is not None]) - idtp
    idfn = len(gt_all) - idtp
    id_precision = idtp / (idtp + idfp) if idtp + idfp else 0.0
    id_recall = idtp / (idtp + idfn) if idtp + idfn else 0.0
    idf1 = 2 * id_precision * id_recall / (id_precision + id_recall) if id_precision + id_recall else 0.0

    switches = 0
    fragments = 0
    track_coverages = []
    for gt_id, entries in timeline.items():
        entries = sorted(entries)
        prev_pred = None
        was_matched = False
        matched_count = 0
        for _, _, pred_id in entries:
            if pred_id is not None:
                matched_count += 1
                if prev_pred is not None and pred_id != prev_pred:
                    switches += 1
                if not was_matched:
                    fragments += 1
                was_matched = True
                prev_pred = pred_id
            else:
                was_matched = False
        total = gt_counts.get(gt_id, len(entries))
        track_coverages.append(matched_count / total if total else 0.0)
    mostly_tracked = sum(1 for value in track_coverages if value >= 0.8)
    mostly_lost = sum(1 for value in track_coverages if value <= 0.2)
    return {
        "gt_boxes": len(gt_all),
        "pred_boxes": len(pred_all),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "id_precision": id_precision,
        "id_recall": id_recall,
        "idf1": idf1,
        "id_switches": switches,
        "fragments": max(0, fragments - len(gt_counts)),
        "gt_tracks": len(gt_counts),
        "pred_tracks": len(pred_counts),
        "mostly_tracked": mostly_tracked,
        "mostly_lost": mostly_lost,
        "mean_gt_track_coverage": statistics.mean(track_coverages) if track_coverages else 0.0,
    }


def run_method(model, tasks: list[dict], image_dir: Path, method: dict, iou_thr: float, max_tasks: int = 0) -> tuple[dict, list[dict]]:
    gt_all: list[Box] = []
    pred_all: list[Box] = []
    matches_by_frame: dict[tuple[int, int], list[tuple[Box, Box, float]]] = {}
    per_task = []
    selected = tasks[:max_tasks] if max_tasks and max_tasks > 0 else tasks
    for idx, task in enumerate(selected, start=1):
        task_id = int(task["id"])
        if method["mode"] == "track":
            reset_tracker_state(model)
        frames = task_frames(image_dir, task_id)
        if not frames:
            continue
        tracks, meta = task_tracks(task, image_dir)
        if not tracks:
            continue
        task_gt: list[Box] = []
        task_pred: list[Box] = []
        task_matches: dict[tuple[int, int], list[tuple[Box, Box, float]]] = {}
        source = [str(path) for path in frames]
        options = inference_options(method)
        if method["mode"] == "predict":
            result_iter = model.predict(
                source=source,
                batch=1,
                stream=True,
                **options,
            )
        else:
            result_iter = model.track(
                source=source,
                tracker=method["tracker"],
                batch=1,
                persist=True,
                stream=True,
                **options,
            )
        for frame_path, result in zip(frames, result_iter):
            real_frame = frame_number(frame_path)
            shape = result.orig_img.shape if getattr(result, "orig_img", None) is not None else cv2.imread(str(frame_path)).shape
            gt = gt_for_frame(task_id, real_frame, shape, tracks, float(meta["scale"]))
            pred = boxes_from_result(result, task_id, real_frame)
            pairs = [(gt[gi], pred[pi], value) for gi, pi, value in greedy_match(gt, pred, iou_thr)]
            key = (task_id, real_frame)
            task_matches[key] = pairs
            matches_by_frame[key] = pairs
            task_gt.extend(gt)
            task_pred.extend(pred)
        gt_all.extend(task_gt)
        pred_all.extend(task_pred)
        metrics = evaluate_matches(task_gt, task_pred, task_matches)
        metrics.update({"task_id": task_id, "frames": len(frames), "duration": meta.get("duration"), "video": Path((task.get("data") or {}).get("video", "")).name})
        per_task.append(metrics)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if idx % 5 == 0:
            print(f"  {method['name']}: tasks {idx}/{len(selected)}", flush=True)
    return evaluate_matches(gt_all, pred_all, matches_by_frame), per_task


def summarize_tasks(tasks: list[dict], image_dir: Path) -> list[dict]:
    rows = []
    for task in tasks:
        task_id = int(task["id"])
        frames = task_frames(image_dir, task_id)
        rects, meta = task_tracks(task, image_dir)
        if not frames or not rects:
            continue
        rows.append(
            {
                "task_id": task_id,
                "video": Path((task.get("data") or {}).get("video", "")).name,
                "frames": len(frames),
                "duration": meta.get("duration"),
                "gt_tracks": len(rects),
                "classes": dict(Counter(item["label"] for item in rects)),
                "real_fps_est": meta.get("real_fps_est"),
                "scale": meta.get("scale"),
            }
        )
    return rows


def draw_boxes(frame: np.ndarray, boxes: list[Box], names: dict[int, str], *, with_id: bool) -> np.ndarray:
    out = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy)
        color = COLORS[box.class_id % len(COLORS)]
        label = names.get(box.class_id, str(box.class_id))
        tid = "" if not with_id or box.track_id is None else f" id={box.track_id}"
        text = f"{label}{tid} {box.conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y0 = max(0, y1 - th - 7)
        cv2.rectangle(out, (x1, y0), (x1 + tw + 4, y0 + th + 6), color, -1)
        cv2.putText(out, text, (x1 + 2, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_video(model, task_id: int, image_dir: Path, out_path: Path, method: dict, fps: float) -> None:
    frames = task_frames(image_dir, task_id)
    if not frames:
        raise FileNotFoundError(f"没有找到 task {task_id} 的图片帧")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"无法读取图片: {frames[0]}")
    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    names = {int(k): v for k, v in dict(model.names).items()}
    if method["mode"] == "track":
        reset_tracker_state(model)
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        real_frame = frame_number(frame_path)
        options = inference_options(method)
        if method["mode"] == "predict":
            results = model.predict(frame, **options)
        else:
            results = model.track(frame, tracker=method["tracker"], persist=True, **options)
        boxes = boxes_from_result(results[0], task_id, real_frame) if results else []
        writer.write(draw_boxes(frame, boxes, names, with_id=(method["mode"] == "track")))
    writer.release()


def rounded(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        else:
            out[key] = value
    return out


def write_report(path: Path, payload: dict) -> None:
    methods = payload["methods"]
    task_summary = payload["task_summary"]
    best = methods[0] if methods else None
    lines = [
        "# 基于 Label Studio GT Track 的 YOLO Tracker 对比报告",
        "",
        "## 1. 数据和对应关系",
        "",
        f"- Label Studio 导出：`{payload['labelstudio_path']}`",
        f"- YOLO 测试帧目录：`{payload['image_dir']}`",
        f"- 成功对应 task 数：`{len(task_summary)}`",
        f"- 成功对应测试帧数：`{sum(item['frames'] for item in task_summary)}`",
        f"- 本次实际评测 task 数：`{payload.get('evaluated_task_count', 'unknown')}`",
        f"- 本次实际评测帧数：`{payload.get('evaluated_frame_count', 'unknown')}`",
        "- 对应方式：Label Studio `task.id` ↔ YOLO 测试图片名前缀 `t{task_id}_xxxxxx.jpg`",
        "- GT track id：使用每个 `videorectangle` result 的 `result.id`，即一个检测物的一条时间轴。",
        "",
        "当前 YOLO txt 标签仍然只是逐帧 `class_id cx cy w h`，不包含 track id；本次评测没有使用 YOLO txt 恢复轨迹，而是直接从 Label Studio 原生 `sequence` 恢复 GT 轨迹。",
        "",
        "## 2. 评测指标",
        "",
        "- `Precision/Recall/F1`：逐帧检测框指标，同类别且 IoU 达到阈值才算匹配。",
        "- `IDF1`：轨迹身份匹配综合指标，越高说明同一 GT 目标更稳定地对应同一个预测 track id。",
        "- `ID Switches`：同一 GT 轨迹在时间上匹配到不同预测 id 的次数，越低越好。",
        "- `Fragments`：GT 轨迹从未匹配变为匹配的额外片段数，越低说明轨迹越连续。",
        "- `Mostly Tracked`：覆盖率不低于 80% 的 GT 轨迹数量。",
        "- `Mostly Lost`：覆盖率不高于 20% 的 GT 轨迹数量。",
        "",
        f"本次匹配 IoU 阈值：`{payload['iou_match']}`。",
        "",
        "## 3. 全量结果",
        "",
        "| 排名 | 方法 | Tracker | conf | NMS IoU | Precision | Recall | F1 | IDF1 | ID Switches | Fragments | MT | ML | Pred Boxes |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(methods, start=1):
        m = item["metrics"]
        lines.append(
            f"| {rank} | `{item['name']}` | `{item.get('tracker', '-')}` | {item['conf']:.2f} | {item['iou']:.2f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['idf1']:.4f} | "
            f"{m['id_switches']} | {m['fragments']} | {m['mostly_tracked']} | {m['mostly_lost']} | {m['pred_boxes']} |"
        )
    if best:
        lines += [
            "",
            "## 4. 当前最佳方法",
            "",
            f"- 最佳方法：`{best['name']}`",
            f"- Tracker：`{best.get('tracker', '-')}`",
            f"- 参数：`conf={best['conf']:.2f}`，`iou={best['iou']:.2f}`，`imgsz={best['imgsz']}`，`max_det={best['max_det']}`",
            f"- IDF1：`{best['metrics']['idf1']:.4f}`",
            f"- F1：`{best['metrics']['f1']:.4f}`",
            "",
        ]
    lines += [
        "## 5. 视频片段级结果 Top 10",
        "",
        "下面列出最佳方法下按 IDF1 排名前 10 的视频片段，用于选择可视化样例。",
        "",
        "| Task | Video | Frames | GT Tracks | Precision | Recall | F1 | IDF1 | ID Switches | Fragments |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("best_per_task_top", []):
        lines.append(
            f"| {row['task_id']} | `{row['video']}` | {row['frames']} | {row['gt_tracks']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['idf1']:.4f} | "
            f"{row['id_switches']} | {row['fragments']} |"
        )
    lines += [
        "",
        "## 6. 可视化输出",
        "",
        f"- Track 前视频：`{payload.get('pre_track_video')}`",
        f"- Track 后视频：`{payload.get('post_track_video')}`",
        f"- 可视化 task：`{payload.get('visualized_task_id')}`",
        "",
        "Track 前视频是普通 YOLO predict 的逐帧检测框；Track 后视频是最佳 tracker 的输出框和预测 track id。两者使用同一个 task、同一批测试帧。",
        "",
        "## 7. 结论和后续改进",
        "",
        "本次评测已经从无 GT 的代理指标切换到 Label Studio 原生轨迹 GT。后续优化应优先围绕两个问题：一是提高 `scope_control_body` 的检测召回，否则 tracker 没有框可跟；二是降低手部贴近时的合并框问题，可尝试更高输入分辨率、更高 NMS IoU、调整置信度阈值、补充贴近/遮挡样本，必要时引入实例分割模型。",
        "",
        "固定 top-k 不建议作为时序分割前的强约束，因为工作人员路过或多手同时出现时会删除真实目标。更合适的策略是保留较高召回检测，再由 tracker 和后续时序模型利用轨迹稳定性、空间位置和动作上下文过滤无关目标。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trackers with Label Studio GT track ids")
    parser.add_argument("--labelstudio", default=DEFAULT_LABELSTUDIO, type=Path)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, type=Path)
    parser.add_argument("--ckpt", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--imgsz", default=DEFAULT_IMAGE_SIZE, type=int)
    parser.add_argument("--max-det", default=DEFAULT_MAX_DET, type=int)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--iou-match", default=DEFAULT_IOU_MATCH, type=float)
    parser.add_argument("--trackers", default=DEFAULT_TRACKERS)
    parser.add_argument("--confs", default=DEFAULT_CONFS)
    parser.add_argument("--nms-ious", default=DEFAULT_NMS_IOUS)
    parser.add_argument("--max-tasks", default=0, type=int)
    args = parser.parse_args()

    from ultralytics import YOLO

    tasks = load_labelstudio_tasks(args.labelstudio)
    task_summary = summarize_tasks(tasks, args.image_dir)
    matched_ids = {item["task_id"] for item in task_summary}
    tasks = [task for task in tasks if int(task["id"]) in matched_ids]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trackers = [item.strip() for item in args.trackers.split(",") if item.strip()]
    confs = [float(item) for item in args.confs.split(",") if item.strip()]
    nms_ious = [float(item) for item in args.nms_ious.split(",") if item.strip()]
    methods = [
        {"name": f"predict_conf{conf:.2f}_iou{iou_value:.2f}", "mode": "predict", "tracker": "-", "conf": conf, "iou": iou_value, "imgsz": args.imgsz, "max_det": args.max_det, "device": args.device, "half": args.half}
        for conf in confs
        for iou_value in nms_ious
    ]
    methods += [
        {"name": f"{Path(tracker).stem}_conf{conf:.2f}_iou{iou_value:.2f}", "mode": "track", "tracker": tracker, "conf": conf, "iou": iou_value, "imgsz": args.imgsz, "max_det": args.max_det, "device": args.device, "half": args.half}
        for tracker in trackers
        for conf in confs
        for iou_value in nms_ious
    ]

    evaluated = []
    for idx, method in enumerate(methods, start=1):
        print(f"[{idx}/{len(methods)}] {method['name']}", flush=True)
        model = YOLO(str(args.ckpt))
        metrics, per_task = run_method(model, tasks, args.image_dir, method, args.iou_match, args.max_tasks)
        item = dict(method)
        item["metrics"] = rounded(metrics)
        item["per_task"] = [rounded(row) for row in per_task]
        evaluated.append(item)
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    evaluated.sort(key=lambda item: (item["metrics"]["idf1"], item["metrics"]["f1"]), reverse=True)
    best = evaluated[0] if evaluated else None
    selected_task = None
    if best:
        candidates = [
            row for row in best["per_task"]
            if row.get("gt_tracks", 0) >= 3 and row.get("frames", 0) >= 40
        ]
        candidates.sort(key=lambda row: (row["idf1"], row["f1"], row["frames"]), reverse=True)
        selected_task = candidates[0] if candidates else (best["per_task"][0] if best["per_task"] else None)

    pre_video = post_video = None
    if best and selected_task:
        task_id = int(selected_task["task_id"])
        fps = 6.0
        pre_method = {"name": "predict_pre_track", "mode": "predict", "tracker": "-", "conf": best["conf"], "iou": best["iou"], "imgsz": best["imgsz"], "max_det": best["max_det"], "device": best["device"], "half": best["half"]}
        pre_video = args.out_dir / f"task{task_id}_before_track_predict_conf{best['conf']:.2f}.mp4"
        post_video = args.out_dir / f"task{task_id}_after_track_{Path(best['tracker']).stem}_conf{best['conf']:.2f}_iou{best['iou']:.2f}.mp4"
        print(f"Rendering {pre_video}", flush=True)
        model = YOLO(str(args.ckpt))
        render_video(model, task_id, args.image_dir, pre_video, pre_method, fps)
        del model
        print(f"Rendering {post_video}", flush=True)
        model = YOLO(str(args.ckpt))
        render_video(model, task_id, args.image_dir, post_video, best, fps)
        del model

    payload = {
        "labelstudio_path": str(args.labelstudio),
        "image_dir": str(args.image_dir),
        "ckpt": str(args.ckpt),
        "iou_match": args.iou_match,
        "task_summary": task_summary,
        "evaluated_task_count": len(evaluated[0]["per_task"]) if evaluated else 0,
        "evaluated_frame_count": sum(row["frames"] for row in evaluated[0]["per_task"]) if evaluated else 0,
        "methods": evaluated,
        "best_per_task_top": (best["per_task"][:0] if not best else sorted(best["per_task"], key=lambda row: (row["idf1"], row["f1"]), reverse=True)[:10]),
        "visualized_task_id": selected_task["task_id"] if selected_task else None,
        "pre_track_video": str(pre_video) if pre_video else None,
        "post_track_video": str(post_video) if post_video else None,
    }
    json_path = args.out_dir / "labelstudio_gt_track_eval.json"
    md_path = args.out_dir / "labelstudio_gt_track_eval.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(md_path, payload)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
