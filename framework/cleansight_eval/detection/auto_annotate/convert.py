"""自动标注 → 时序训练数据布局：legacy 标注 JSON + 人工动作标签 → 训练数据。

把自动标注 JSON（``auto_annotate.run`` 产出）+ 人工 Label Studio 导出
（timelinelabels 动作标签）合并成 framework 时序训练数据布局
（``labels/<split>/`` + ``frames/<split>/``），供 ``temporal/data.py`` 直接消费。
legacy 格式的读写与坐标转换统一走 ``auto_annotate.legacy_format``。
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from . import legacy_format
from ._constants import ACTION_CLASSES, DETECTION_CLASSES, TARGET_LABEL_FPS


def _load_manual_labels(labels_export: Path) -> dict[str, dict]:
    """读人工 Label Studio 导出，按视频文件名索引 timelinelabels 与帧率信息。

    返回 ``{视频文件名: {"ls_fps": float, "ranges": [(start, end, 动作名), ...]}}``；
    ``ls_fps`` 由 videorectangle 的 ``framesCount/duration`` 推导（LS 标注端帧率）。
    """

    tasks = json.loads(labels_export.read_text(encoding="utf-8"))
    by_video: dict[str, dict] = {}
    for task in tasks:
        video_name = Path(task.get("data", {}).get("video", "")).name
        if not video_name:
            continue
        ranges: list[tuple[int, int, str]] = []
        ls_fps = None
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                value = result.get("value", {})
                if result.get("type") == "videorectangle":
                    frames_count = value.get("framesCount")
                    duration = value.get("duration")
                    if frames_count and duration:
                        ls_fps = frames_count / duration
                elif result.get("type") == "timelinelabels":
                    label = value.get("timelinelabels", [""])[0]
                    for span in value.get("ranges", []):
                        ranges.append((int(span["start"]), int(span["end"]), label))
        by_video[video_name] = {"ls_fps": ls_fps, "ranges": ranges}
    return by_video


def _action_at(ranges: list[tuple[int, int, str]], frame: int) -> str:
    """帧号（真实解码帧号）命中的动作名；未命中返回 idle。"""

    for start, end, action in ranges:
        if start <= frame <= end:
            return action
    return "idle"


def convert_annotations(
    annotation_dir: Path,
    labels_export: Path,
    out_root: Path,
    *,
    split: str = "train",
) -> list[Path]:
    """自动标注 JSON + 人工 LS 导出 → 时序训练数据布局。

    产出（与 ``temporal/data.py`` 消费契约一致）：
    - ``labels/<split>/<video>.mp4.txt``：抽样帧 ``"frame_id action_id"``
      （1-based 真实解码帧号，按 ~7.5fps 抽样；动作标签来自人工
      timelinelabels，经 LS 帧率 → 真实帧率换算，未命中区间为 idle）
    - ``frames/<split>/<video>.mp4-<f:06d>.txt``：逐帧 bbox
      ``"class_id cx cy w h"``（8 类全局顺序，5 列兼容 40 维 v1 特征）
    - ``labels/data.yaml`` 与 ``frames/data.yaml``：类别映射

    每个自动标注 JSON 必须有可用的同名人工 task（动作标签 + LS 帧率）才能转换；
    人工导出中缺失或没有有效标注的视频会被**跳过并告警**（不中断其余视频），
    汇总在结尾打印。返回产出文件列表。
    """

    annotation_dir = Path(annotation_dir)
    out_root = Path(out_root)
    manual = _load_manual_labels(Path(labels_export))
    labels_dir = out_root / "labels" / split
    frames_dir = out_root / "frames" / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: cid for cid, name in enumerate(DETECTION_CLASSES)}
    action_to_id = {name: aid for aid, name in enumerate(ACTION_CLASSES)}

    outputs: list[Path] = []
    skipped: list[str] = []
    for json_path in sorted(annotation_dir.glob("*.json")):
        task = json.loads(json_path.read_text(encoding="utf-8"))
        legacy = legacy_format.parse_legacy_task(task)
        video_name = legacy.video_name
        manual_info = manual.get(video_name)
        if manual_info is None:
            print(f"[convert] 跳过（人工导出中无此视频，无动作标签）: {video_name}")
            skipped.append(video_name)
            continue
        ls_fps = manual_info["ls_fps"]
        if not ls_fps:
            print(
                f"[convert] 跳过（人工标注无 framesCount/duration，无法换算帧号）: {video_name}"
            )
            skipped.append(video_name)
            continue
        frames_count = legacy.frames_count
        duration = legacy.duration
        real_fps = frames_count / duration if duration else ls_fps
        # LS 标注帧号 → 真实解码帧号：ls = real × (ls_fps / real_fps)
        scale = ls_fps / real_fps
        real_ranges = [
            (max(1, round(start / scale)), min(frames_count, round(end / scale)), action)
            for start, end, action in manual_info["ranges"]
        ]
        stride = max(1, round(real_fps / TARGET_LABEL_FPS))
        label_frames = list(range(1, frames_count + 1, stride))

        # sequence 按帧序排列（frame 1..framesCount），建 帧号 → [(类名, 有效检测)] 索引
        sequences: dict[int, list[tuple[str, dict]]] = {}
        for class_name, sequence in legacy.tracks:
            for entry in sequence:
                if entry.get("enabled"):
                    sequences.setdefault(int(entry["frame"]), []).append((class_name, entry))

        label_lines: list[str] = []
        for frame in label_frames:
            action = _action_at(real_ranges, frame)
            label_lines.append(f"{frame} {action_to_id[action]}")
        label_path = labels_dir / f"{video_name}.txt"
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        outputs.append(label_path)

        for frame in label_frames:
            lines = []
            for class_name, entry in sequences.get(frame, []):
                class_id = class_to_id.get(class_name)
                if class_id is None:
                    continue  # 自动标注未覆盖类别（short_brush 等）该帧恒零
                # 左上角百分比 → YOLO 归一化中心点（与 legacy lsexport.to_yolo 同口径）
                cx, cy, nw, nh = legacy_format.xywhn_from_ls_box(
                    entry["x"], entry["y"], entry["width"], entry["height"]
                )
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            frame_path = frames_dir / f"{video_name}-{frame:06d}.txt"
            frame_path.write_text("".join(lines), encoding="utf-8")

        print(
            f"[convert] {video_name}: {len(label_frames)} 个标签帧（stride={stride}, "
            f"LS {ls_fps:.1f}fps → 真实 {real_fps:.1f}fps）→ {split}/"
        )

    # 类别映射（首次生成；已有文件不覆盖，避免与登记数据冲突）
    labels_yaml = out_root / "labels" / "data.yaml"
    if not labels_yaml.is_file():
        labels_yaml.write_text(
            yaml.safe_dump(
                {"nc": len(ACTION_CLASSES), "names": {i: n for i, n in enumerate(ACTION_CLASSES)}},
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    frames_yaml = out_root / "frames" / "data.yaml"
    if not frames_yaml.is_file():
        frames_yaml.write_text(
            yaml.safe_dump(
                {"nc": len(DETECTION_CLASSES), "names": {i: n for i, n in enumerate(DETECTION_CLASSES)}},
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    if skipped:
        print(
            f"[convert] 跳过 {len(skipped)} 个无人工标签的视频（不产出训练数据）: "
            + ", ".join(skipped)
        )
    return outputs
