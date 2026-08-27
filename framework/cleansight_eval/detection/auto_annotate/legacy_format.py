"""legacy Label Studio 导出格式适配层：读写与坐标转换。

自动标注的产出/消费契约是历史 Label Studio 导出的 ``videorectangle`` 结构
（``legacy/yolo-detection/pipeline/raw/exports/*.json``）。本模块是唯一读写
该格式的地方：

- ``build_task``：轨迹列表 → legacy task JSON（写入端，供 ``run`` 使用）
- ``parse_legacy_task``：legacy task JSON → 结构化表示（读取端，供 ``convert`` 使用）
- ``ls_box_from_xywhn`` / ``xywhn_from_ls_box``：左上角百分比 ↔ 归一化中心点
  互逆转换（与 legacy ``lsexport.to_yolo`` 同口径）

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
"""

from __future__ import annotations

from dataclasses import dataclass


def ls_box_from_xywhn(xywhn) -> tuple[float, float, float, float]:
    """归一化中心点 [cx,cy,w,h] → 左上角百分比 (x,y,width,height)，裁剪到 [0,100]。"""

    cx, cy, w, h = (float(v) for v in xywhn)
    clamp = lambda v: max(0.0, min(100.0, v))
    return (
        clamp((cx - w / 2.0) * 100.0),
        clamp((cy - h / 2.0) * 100.0),
        clamp(w * 100.0),
        clamp(h * 100.0),
    )


def xywhn_from_ls_box(
    x: float, y: float, width: float, height: float
) -> tuple[float, float, float, float]:
    """左上角百分比 (x,y,width,height) → 归一化中心点 [cx,cy,w,h]（逆转换）。"""

    return (
        (x + width / 2.0) / 100.0,
        (y + height / 2.0) / 100.0,
        width / 100.0,
        height / 100.0,
    )


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


@dataclass(frozen=True)
class LegacyTask:
    """legacy task JSON 的结构化表示（``parse_legacy_task`` 的产物）。

    ``tracks`` 为 ``[(类别名, 全帧 sequence), ...]``，与
    ``auto_annotate.run.build_track_sequences`` 的输出形状一致。
    """

    video_name: str
    frames_count: int
    duration: float | None
    tracks: list[tuple[str, list[dict]]]


def parse_legacy_task(task: list) -> LegacyTask:
    """legacy task JSON（``build_task`` 产出的数组）→ 结构化表示。

    兼容宽泛结构：只取第一个 task，收集其中全部 ``videorectangle`` 结果；
    ``framesCount`` / ``duration`` 取首个 videorectangle 的值。
    """

    if not isinstance(task, list) or not task:
        raise ValueError("自动标注 task 必须是非空数组")
    item = task[0]
    data = item.get("data") or {}
    video_name = data.get("video")
    if not video_name:
        raise ValueError("task 缺少 data.video")
    annotations = item.get("annotations") or []
    if not annotations:
        raise ValueError(f"{video_name} 的 task 缺少 annotations")

    tracks: list[tuple[str, list[dict]]] = []
    frames_count = None
    duration = None
    for result in annotations[0].get("result") or []:
        if result.get("type") != "videorectangle":
            continue
        value = result.get("value") or {}
        labels = value.get("labels") or []
        if not labels:
            continue
        if frames_count is None:
            frames_count = value.get("framesCount")
            duration = value.get("duration")
        tracks.append((labels[0], list(value.get("sequence") or [])))
    if frames_count is None:
        raise ValueError(f"{video_name} 的 task 没有 videorectangle 结果（无法确定帧数）")
    return LegacyTask(
        video_name=video_name,
        frames_count=int(frames_count),
        duration=float(duration) if duration else None,
        tracks=tracks,
    )
