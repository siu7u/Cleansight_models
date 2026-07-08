"""YOLO 检测结果到因果时序特征序列的映射。

本模块是时序模型离线训练和在线推理共享的特征契约：离线特征生成与在线推理
都必须调用同一个 `step()`。如果特征顺序、维度、阈值或类别别名发生变化，
就需要登记新的 feature_mapping 版本，并重训依赖它的时序模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FEATURE_CLASSES = [
    "hand",
    "short_brush_visible",
    "syringe",
    "air_gun",
    "scope_control_body",
    "scope_mid_section",
    "scope_distal_end",
    "brush_tip_out",
]

PER_CLASS_DIM = 8
FEATURE_DIM = len(FEATURE_CLASSES) * PER_CLASS_DIM


@dataclass
class Detection:
    """供时序特征映射器消费的一条归一化 YOLO 检测结果。

    框坐标已经按帧宽高归一化到 `[0, 1]`。`cls` 必须匹配 `FEATURE_CLASSES`；
    未知类别会被特征提取器忽略。
    """

    cls: str
    cx: float
    cy: float
    w: float
    h: float
    conf: float


class DetectionFeatureExtractor:
    """因果、有状态的逐帧特征提取器。

    提取器为每个类别保存上一帧中心点，因此 `step()` 可以输出位移通道。
    流式推理时应复用同一个实例；开始新视频前需要调用 `reset()`。
    """

    def __init__(self, conf_threshold: float = 0.25):
        """创建提取器，并忽略置信度低于 `conf_threshold` 的检测结果。"""

        self.conf_threshold = conf_threshold
        self.reset()

    def reset(self) -> None:
        """在处理新序列前清空每个类别的上一帧中心点。"""

        self._prev_center = {c: None for c in FEATURE_CLASSES}

    def step(self, detections: Iterable[Detection]) -> np.ndarray:
        """将单帧检测结果转换为一个 `FEATURE_DIM` 特征向量。

        输出布局为每个 `FEATURE_CLASSES` 类别重复
        `[present, cx, cy, w, h, conf, dcx, dcy]`。该方法是因果的，应由
        离线特征生成和在线推理共同使用。
        """

        best: dict[str, Detection] = {}
        for det in detections:
            if det.cls not in self._prev_center or det.conf < self.conf_threshold:
                continue
            if det.cls not in best or det.conf > best[det.cls].conf:
                best[det.cls] = det

        feat = np.zeros(FEATURE_DIM, dtype=np.float32)
        for i, cls_name in enumerate(FEATURE_CLASSES):
            base = i * PER_CLASS_DIM
            det = best.get(cls_name)
            if det is None:
                self._prev_center[cls_name] = None
                continue

            feat[base : base + 6] = (1.0, det.cx, det.cy, det.w, det.h, det.conf)
            prev = self._prev_center[cls_name]
            if prev is not None:
                feat[base + 6] = det.cx - prev[0]
                feat[base + 7] = det.cy - prev[1]
            self._prev_center[cls_name] = (det.cx, det.cy)
        return feat


def extract_sequence(
    frames: list[list[Detection]], conf_threshold: float = 0.25
) -> np.ndarray:
    """将按帧组织的整段视频检测结果转换为 `[T, F]` 特征矩阵。"""

    extractor = DetectionFeatureExtractor(conf_threshold=conf_threshold)
    return np.stack([extractor.step(frame) for frame in frames], axis=0)


def yolo_result_to_detections(result, img_w: int, img_h: int) -> list[Detection]:
    """将一个 Ultralytics result 对象转换为归一化的 `Detection` 列表。"""

    out: list[Detection] = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append(
            Detection(
                cls=result.names[int(box.cls)],
                cx=(x1 + x2) / 2 / img_w,
                cy=(y1 + y2) / 2 / img_h,
                w=(x2 - x1) / img_w,
                h=(y2 - y1) / img_h,
                conf=float(box.conf),
            )
        )
    return out
