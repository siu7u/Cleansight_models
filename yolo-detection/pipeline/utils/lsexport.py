#!/usr/bin/env python3
"""
Label Studio 视频导出解析 —— 共享核心逻辑,供各脚本复用。

已验证正确、**不可退化**的几处:
  - fps 对齐:LS 标注帧号按标注端 fps(常 24)计,真实视频 fps 可能不同,
    用 scale = ls_fps/real_fps 把真实解码帧号映射回 LS 帧号,消除漂移/尾部丢失。
  - 关键帧线性插值:sequence 只存关键帧,中间帧插值;enabled=False = 目标离场。
  - 坐标转换:LS 左上角百分比 -> YOLO 归一化中心点 (cx,cy,w,h),裁剪到 [0,1]。

只依赖标准库(路径/JSON)。cv2 由调用方使用,这里不导入,便于纯解析场景。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # yolo_pipeline/(自包含)
EXPORT_DIR = ROOT / "raw" / "exports"
VIDEO_DIR = ROOT / "raw" / "videos"


def latest_export(export_dir: Path = EXPORT_DIR) -> Path:
    """取 raw/exports/ 下文件名排序最后一个 JSON。"""
    files = sorted(export_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"raw/exports/ 下没有导出 JSON: {export_dir}")
    return files[-1]


def load_tasks(json_path: Path):
    return json.load(open(json_path, encoding="utf-8"))


def task_video_name(task) -> str:
    """task 引用的视频文件名(不含目录)。"""
    rel = task.get("data", {}).get("video", "") or ""
    return Path(rel).name


def iter_results(task, rtype=None):
    """遍历 task 所有 annotation 的 result;rtype 非空时只出该 type。"""
    for ann in task.get("annotations", []):
        for r in ann.get("result", []):
            if rtype is None or r.get("type") == rtype:
                yield r


def clip_meta(task):
    """从第一个 videorectangle 读 (framesCount, duration);取不到返回 (None, None)。"""
    for r in iter_results(task, "videorectangle"):
        v = r["value"]
        return v.get("framesCount"), v.get("duration")
    return None, None


def build_label_index(groups: dict):
    """LS 类别名 -> (组名, class_id) 的反查表。"""
    label2group = {}
    for g, labels in groups.items():
        for cid, lab in enumerate(labels):
            label2group[lab] = (g, cid)
    return label2group


def build_segments(seq):
    """把关键帧序列拆成可见插值区间 [(f0, box0, f1, box1), ...]。"""
    seq = sorted(seq, key=lambda s: s["frame"])
    segs = []
    for a, b in zip(seq, seq[1:]):
        if a.get("enabled", True):
            segs.append((a["frame"], a, b["frame"], b))
    if seq and seq[-1].get("enabled", True):  # 末关键帧若在场,补单帧区间
        last = seq[-1]
        segs.append((last["frame"], last, last["frame"], last))
    return segs


def box_at(segs, frame):
    """该帧插值后的框 (x,y,w,h) 百分比;不可见返回 None。"""
    for f0, b0, f1, b1 in segs:
        if f0 <= frame <= f1:
            t = 0.0 if f1 == f0 else (frame - f0) / (f1 - f0)
            return (
                b0["x"] + (b1["x"] - b0["x"]) * t,
                b0["y"] + (b1["y"] - b0["y"]) * t,
                b0["width"] + (b1["width"] - b0["width"]) * t,
                b0["height"] + (b1["height"] - b0["height"]) * t,
            )
    return None


def to_yolo(x, y, w, h):
    """LS 左上角百分比 -> YOLO 归一化中心点,裁剪到 [0,1]。"""
    cx = (x + w / 2) / 100.0
    cy = (y + h / 2) / 100.0
    nw = w / 100.0
    nh = h / 100.0
    clamp = lambda v: max(0.0, min(1.0, v))
    return clamp(cx), clamp(cy), clamp(nw), clamp(nh)


def collect_tracks(task, label2group):
    """收集 task 内所有目标轨迹: [(group, class_id, segments), ...](仅分组内类别)。"""
    tracks = []
    for r in iter_results(task, "videorectangle"):
        v = r["value"]
        labs = v.get("labels") or []
        if not labs or labs[0] not in label2group:
            continue
        g, cid = label2group[labs[0]]
        segs = build_segments(v.get("sequence", []))
        if segs:
            tracks.append((g, cid, segs))
    return tracks


def fps_scale(cap_fps, framesCount, duration):
    """真实帧号 -> LS 帧号 的比例 scale = ls_fps / real_fps。"""
    ls_fps = (framesCount / duration) if (framesCount and duration) else cap_fps
    return (ls_fps / cap_fps) if cap_fps else 1.0, ls_fps
