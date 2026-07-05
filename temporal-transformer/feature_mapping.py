"""YOLO detection to causal feature sequence mapping.

This module is the shared offline/online contract for temporal models:
offline feature generation and online inference must both call step().
Changing the feature order, dimensions, thresholds, or class aliases creates
a new feature_mapping version and requires retraining dependent models.
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
    cls: str
    cx: float
    cy: float
    w: float
    h: float
    conf: float


class DetectionFeatureExtractor:
    """Causal, stateful per-frame feature extractor."""

    def __init__(self, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.reset()

    def reset(self) -> None:
        self._prev_center = {c: None for c in FEATURE_CLASSES}

    def step(self, detections: Iterable[Detection]) -> np.ndarray:
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
    extractor = DetectionFeatureExtractor(conf_threshold=conf_threshold)
    return np.stack([extractor.step(frame) for frame in frames], axis=0)


def yolo_result_to_detections(result, img_w: int, img_h: int) -> list[Detection]:
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
