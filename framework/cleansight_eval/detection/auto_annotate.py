"""YOLO 自动标注：视频 → legacy 时序标注 JSON。

用已训练 YOLO checkpoint 对视频逐帧检测，产出与历史 Label Studio 导出同构的
JSON（``legacy/yolo-detection/pipeline/raw/exports/*.json`` 的 videorectangle
结构），可直接被 ``legacy/temporal-*/lab.py::load_data_json`` 消费，用于给无
标注新视频自动生成检测标注（时序模型特征输入）。

格式约定（与人工标注对齐，用户已确认）：
- 轨迹划分：每类别按框面积取 top-K（``hand`` 默认 2 条、其他类别 1 条），
  与 ``temporal/features/clean_bbox_v2`` 的 slot 语义一致；YOLO 无实例跟踪，
  帧间不关联实例。
- ``sequence`` 全帧写入（覆盖 ``[1, framesCount]``，保证 legacy 消费端
  ``interpolate_sequence`` 得到等长轨迹）：有效帧 ``enabled=true`` + 左上角
  百分比坐标 + 非标准 ``conf`` 字段（legacy 解析器忽略未知字段，兼容）；
  缺席帧 ``enabled=false`` + 外推上一有效框坐标（legacy 无 presence 表达，
  坐标外推与人工标注的离场点语义一致）。
- 坐标：YOLO 归一化中心点 ``[cx,cy,w,h]`` → 左上角百分比（与 Label Studio
  导出一致），裁剪到 [0,100]。
- 帧号从 1 开始，``time=(frame-1)/fps``；``duration=framesCount/fps``。
- 动作标签（timelinelabels）由 YOLO 无法产出，result 中不包含。

模型执行只发生在 framework 检测域：ultralytics 的加载与逐帧推理复用
``detection/inference.py``，本模块不直接 import ultralytics。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from . import inference

# 默认每类别轨迹数（slot 数）；hand 双实例与 clean_bbox_v2 的 hand slots 一致。
DEFAULT_TOP_K: dict[str, int] = {"hand": 2}


def _ls_box(xywhn) -> tuple[float, float, float, float]:
    """归一化中心点 [cx,cy,w,h] → 左上角百分比 (x,y,width,height)，裁剪到 [0,100]。"""

    cx, cy, w, h = (float(v) for v in xywhn)
    clamp = lambda v: max(0.0, min(100.0, v))
    return (
        clamp((cx - w / 2.0) * 100.0),
        clamp((cy - h / 2.0) * 100.0),
        clamp(w * 100.0),
        clamp(h * 100.0),
    )


def _top_k_per_frame(frame_detections: list[list[dict]], top_k: dict[str, int]) -> dict:
    """把逐帧检测按类别分组，每帧每类别保留面积最大的前 K 个框（面积降序）。"""

    per_class: dict[str, dict[int, list[dict]]] = {}
    for frame_index, detections in enumerate(frame_detections):
        for detection in detections:
            per_class.setdefault(detection["class"], {}).setdefault(frame_index, []).append(
                detection
            )
    for class_name, by_frame in per_class.items():
        limit = int(top_k.get(class_name, 1))
        for frame_index in by_frame:
            by_frame[frame_index].sort(
                key=lambda d: d["xywhn"][2] * d["xywhn"][3], reverse=True
            )
            by_frame[frame_index] = by_frame[frame_index][:limit]
    return per_class


def build_track_sequences(
    frame_detections: list[list[dict]],
    frames_count: int,
    fps: float,
    top_k: dict[str, int] | None = None,
) -> list[tuple[str, list[dict]]]:
    """逐帧检测 → legacy 轨迹列表 ``[(类别名, 全帧 sequence), ...]``。

    ``frame_detections`` 每帧元素为 ``{"class", "confidence", "xywhn"}``（已合并
    多个 checkpoint 并映射为全局类名）。返回的 sequence 覆盖 ``[1, frames_count]``
    全部帧：有效帧 ``enabled=true`` 并附坐标与 ``conf``；缺席帧 ``enabled=false``
    坐标外推上一有效框（从未出现则全 0），保证 legacy 消费端轨迹等长。
    """

    top_k = top_k or DEFAULT_TOP_K
    per_class = _top_k_per_frame(frame_detections, top_k)
    tracks: list[tuple[str, list[dict]]] = []
    for class_name in sorted(per_class):  # 确定性类别顺序
        by_frame = per_class[class_name]
        slot_count = max(len(frames) for frames in by_frame.values())
        for slot in range(slot_count):
            sequence: list[dict] = []
            last_box = (0.0, 0.0, 0.0, 0.0)
            for frame_index in range(frames_count):
                candidates = by_frame.get(frame_index) or []
                entry: dict = {
                    "frame": frame_index + 1,
                    "rotation": 0,
                    "time": frame_index / fps if fps > 0 else 0.0,
                }
                if slot < len(candidates):
                    detection = candidates[slot]
                    last_box = tuple(float(v) for v in detection["xywhn"])
                    x, y, width, height = _ls_box(last_box)
                    entry.update(
                        {
                            "enabled": True,
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "conf": float(detection["confidence"]),
                        }
                    )
                else:
                    x, y, width, height = _ls_box(last_box)
                    entry.update(
                        {
                            "enabled": False,
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                        }
                    )
                sequence.append(entry)
            tracks.append((class_name, sequence))
    return tracks


def build_task(
    video_name: str,
    tracks: list[tuple[str, list[dict]]],
    frames_count: int,
    fps: float,
    task_id: int,
) -> list[dict]:
    """轨迹列表 → legacy Label Studio 导出同构的 task JSON（单 task 数组）。

    结构与 ``legacy/yolo-detection/pipeline/raw/exports/*.json`` 的
    ``annotations[].result[].videorectangle`` 逐字段一致；不产出
    ``timelinelabels``（YOLO 无法生成动作标签）。
    """

    results = []
    for class_name, sequence in tracks:
        value: dict = {
            "labels": [class_name],
            "framesCount": frames_count,
            "sequence": sequence,
        }
        if fps > 0:
            value["duration"] = round(frames_count / fps, 6)
        results.append({"type": "videorectangle", "value": value})
    return [
        {
            "id": task_id,
            "data": {"video": video_name},
            "annotations": [{"result": results}],
        }
    ]


def detect_video(
    video_path: Path,
    models: list,
    *,
    imgsz: int,
    conf: float,
    max_frames: int | None = None,
) -> tuple[int, float, list[list[dict]]]:
    """对视频逐帧推理，合并多个 checkpoint 的检测并映射为全局类名。

    ``models`` 为 ``[(model, class_map), ...]``，``class_map`` 是本地类别 id →
    全局类名映射。返回 ``(frames_count, fps, frame_detections)``；未映射类别的
    检测被丢弃。
    """

    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_detections: list[list[dict]] = []
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break
            merged: list[dict] = []
            for model, class_map in models:
                for detection in inference.predict_frame(
                    model, frame, imgsz=imgsz, conf=conf
                ):
                    class_name = class_map.get(detection["class_id"])
                    if class_name is None:
                        continue
                    merged.append(
                        {
                            "class": class_name,
                            "confidence": detection["confidence"],
                            "xywhn": detection["xywhn"],
                        }
                    )
            frame_detections.append(merged)
            frame_index += 1
    finally:
        cap.release()
    if total <= 0:
        total = frame_index
    return total, fps, frame_detections


def run_auto_annotate(
    videos: list[Path],
    checkpoint_specs: list[dict],
    out_dir: Path,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    top_k: dict[str, int] | None = None,
    max_frames: int | None = None,
    progress_every: int = 100,
    runs_dir: Path | None = None,
) -> list[Path]:
    """主入口：视频列表 + checkpoint 配置 → legacy 标注 JSON 文件。

    ``checkpoint_specs`` 每项为 ``{"path": <权重路径>, "class_map": {本地id: 全局类名}}``；
    每个 checkpoint 的本地类别 id 必须存在于权重 ``names`` 中，否则报错（防止
    类别表与权重不一致导致静默错标）。``runs_dir`` 非空时把 ultralytics 中间
    产物重定向到该目录。产出文件为 ``out_dir/<视频名>.json``，返回产出文件列表。
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for spec in checkpoint_specs:
        ckpt = Path(spec["path"])
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint 权重不存在: {ckpt}")
        class_map = {int(k): str(v) for k, v in spec["class_map"].items()}
        model, names = inference.load_predictor(
            str(ckpt), imgsz=imgsz, runs_dir=str(runs_dir) if runs_dir is not None else None
        )
        unknown = sorted(set(class_map) - set(names))
        if unknown:
            raise ValueError(
                f"{ckpt.name} 的 class_map 含权重中不存在的类别 id: {unknown}；"
                f"可用: {sorted(names)}"
            )
        models.append((model, class_map))

    outputs: list[Path] = []
    for task_id, video in enumerate(videos):
        video = Path(video)
        total, fps, frame_detections = detect_video(
            video, models, imgsz=imgsz, conf=conf, max_frames=max_frames
        )
        if not frame_detections:
            raise ValueError(f"无法解码视频: {video}")
        tracks = build_track_sequences(frame_detections, total, fps, top_k)
        task = build_task(video.name, tracks, total, fps, task_id)
        output = out_dir / f"{video.stem}.json"
        output.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(output)
        print(
            f"[auto-annotate] {video.name}: {total} 帧 @ {fps:.2f} fps, "
            f"{len(tracks)} 条轨迹 → {output.relative_to(out_dir.parent)}"
        )
    return outputs
