"""

本模块负责从标注 JSON 中解析视频目标轨迹与行为标签，
并将其转换为模型可用的特征矩阵与真值序列。

主要流程：
1. 解析 videorectangle → 目标轨迹（关键帧插值）
2. 解析 timelinelabels → 行为时间段
3. 将轨迹转换为 (cx, cy, w, h, conf) 特征
4. 输出：
   - features: np.ndarray, shape = (T, N * 5)
   - truths: List[str], 每一帧的行为标签
"""


import json
import numpy as np
from typing import List, Tuple


def interpolate_sequence(sequence, total):
    """
    将稀疏的关键帧序列插值为完整帧序列。

    参数
    ----
    sequence : List[Dict]
        关键帧列表
    total : int
        视频总帧数

    返回
    ----
    frames : List[Dict]
        长度为 total 的完整帧序列，
    """
    if len(sequence) == 0:
        return []

    # 确保按 frame 升序
    seq = sorted(sequence, key=lambda kf: kf["frame"])
    start, end = seq[0]["frame"], seq[-1]["frame"]
    frames = []

    collected = set()

    first = seq[0]
    for f in range(1, first["frame"] + 1):
        if f in collected:
            continue
        collected.add(f)
        frames.append({
            "frame": f,
            "x": first["x"],
            "y": first["y"],
            "width": first["width"],
            "height": first["height"],
        })

    for i in range(len(seq) - 1):
        kf0, kf1 = seq[i], seq[i + 1]

        f0, f1 = kf0["frame"], kf1["frame"]

        # 只处理在 [start, end] 内的区间
        if f1 < start or f0 > end:
            continue

        for f in range(f0, f1 + 1):
            if f in collected:
                continue
            collected.add(f)
            if f == f0:
                # 关键帧
                frames.append({
                    "frame": f,
                    "x": kf0["x"],
                    "y": kf0["y"],
                    "width": kf0["width"],
                    "height": kf0["height"],
                })
            else:
                alpha = (f - f0) / (f1 - f0)
                frames.append({
                    "frame": f,
                    "x": kf0["x"] + alpha * (kf1["x"] - kf0["x"]),
                    "y": kf0["y"] + alpha * (kf1["y"] - kf0["y"]),
                    "width": kf0["width"] + alpha * (kf1["width"] - kf0["width"]),
                    "height": kf0["height"] + alpha * (kf1["height"] - kf0["height"]),
                })

    last = seq[-1]
    for f in range(last["frame"], total + 1):
        if f in collected:
            continue
        collected.add(f)
        frames.append({
            "frame": f,
            "x": last["x"],
            "y": last["y"],
            "width": last["width"],
            "height": last["height"],
        })

    assert len(frames) == total, (
        f"expected {total}, actual {len(frames)}\n"
        f"current {collected}\n"
    )

    return frames


def to_features(objects) -> np.ndarray:
    """
    将多个目标的轨迹数据转换为统一的特征矩阵。

    参数
    ----
    objects : Dict[str, List[Dict]]
        key   : 目标名称（如 hand0, scope1）
        value : 每一帧的 bbox 信息，包含：
                x, y, width, height

    返回
    ----
    features : np.ndarray
        形状为 (T, N * 5) 的特征矩阵：
        - T : 帧数
        - N : 目标数
        - 5 : [cx, cy, w, h, conf]

    说明
    ----
    - cx, cy 为中心点坐标
    - conf 暂时固定为 1.0
    - 最终 reshape 是为了适配时序模型输入
    """

    features = []
    F = len(objects)
    for _, frames in objects.items():
        feats = []
        for frame in frames:
            x, y, w, h = frame['x'], frame['y'], frame['width'], frame['height']
            cx, cy = x + w / 2, y + h / 2
            # Label Studio 返回的是归一化后的数据
            conf = 1.0  # TODO
            feats.append([cx, cy, w, h, conf])
        features.append(feats)
    features = np.array(features).transpose(1, 0, 2).reshape(-1, F * 5)
    return features


def to_truths(ranges: List[Tuple[int, int, str]], total: int, default: str = "Idle") -> List[str]:
    """
    将时间段标签转换为逐帧行为标签。

    参数
    ----
    ranges : List[Tuple[int, int, str]]
        (start_frame, end_frame, label)
    total : int
        视频总帧数
    default : str
        默认标签（无行为时）

    返回
    ----
    truths : List[str]
        每一帧对应的行为标签，长度 = total
    """

    truths = [default] * total

    for start, end, label in ranges:
        for f in range(start, end + 1):
            if 1 <= f <= total:
                truths[f - 1] = label

    return truths


def load_data_json(file_path: str, id: int) -> Tuple[np.ndarray, List[str]]:
    """
    从 Label Studio 导出的 JSON 格式文件中加载指定视频 ID 的数据。

    支持：
    - videorectangle ：目标轨迹
    - timelinelabels ：行为时间段

    参数
    ----
    file_path : str
        JSON 文件路径
    id : int
        视频 ID（Label Studio 中的 task id）

    返回
    ----
    features : np.ndarray
        特征矩阵，shape = (T, N * 5)
    truths : List[str]
        每一帧的行为标签
    """

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    objects = {}
    action_ranges = []
    total = 0

    for item in data:
        if not int(item["id"] == id):
            continue

        for i, anno in enumerate(item["annotations"]):
            for j, res in enumerate(anno["result"]):
                value, type = res["value"], res["type"]

                if type == "videorectangle":
                    label = f'{value["labels"][0]}{i}{j}'  # no multiple labels
                    total = value["framesCount"]
                    frames = interpolate_sequence(value["sequence"], total)
                    if len(frames) > 0:
                        objects.setdefault(label, []).extend(frames)

                if type == "timelinelabels":
                    ranges = value["ranges"]
                    for range in ranges:
                        label = f'{value["timelinelabels"][0]}'  # no multiple labels  
                        start, end = range["start"], range["end"]
                        action_ranges.append((start, end, label))

        features = to_features(objects)
        truths = to_truths(action_ranges, total)
        return features, truths

    raise ValueError(f"Id {id} not found")


def load_data_json_min(file_path: str, id: int) -> Tuple[np.ndarray, List[str]]:
    """
    从 Label Studio 导出的 JSON-MIN 格式文件中加载指定视频 ID 的数据。

    支持：
    - objects ：目标轨迹
    - actions ：行为时间段

    参数
    ----
    file_path : str
        JSON 文件路径
    id : int
        视频 ID（Label Studio 中的 task id）

    返回
    ----
    features : np.ndarray
        特征矩阵，shape = (T, N * 5)
    truths : List[str]
        每一帧的行为标签
    """

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    objects = {}
    action_ranges = []
    total = 0

    for item in data:
        if not int(item["id"] == id):
            continue

        for idx, object in enumerate(item.get("objects", [])):
            label = f'{object["labels"][0]}{idx}'  # no multiple labels
            total = object["framesCount"]
            objects.setdefault(label, []).extend(
                interpolate_sequence(object["sequence"], total)
            )

        for action in item.get("actions", []):
            ranges = action["ranges"]
            for range in ranges:
                label = f'{action["timelinelabels"][0]}'  # no multiple labels
                start, end = range["start"], range["end"]
                action_ranges.append((start, end, label))

        features = to_features(objects)
        truths = to_truths(action_ranges, total)
        return features, truths
    
    raise ValueError(f"Id {id} not found")
