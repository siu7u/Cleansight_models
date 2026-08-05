"""检测推理辅助：加载 checkpoint 并对任意帧/视频做检测（framework 检测域）。

模型执行只允许发生在 framework 内部：本模块封装 Ultralytics YOLO 的加载与逐帧推理，
``tools/visualize_detections.py`` 等工具只做 CLI 编排与可视化，不直接 import ultralytics。
"""

from __future__ import annotations

from typing import Any


def load_predictor(ckpt_path: str, imgsz: int = 640):
    """加载 YOLO 模型，返回 ``(model, names_dict)``。"""

    from ultralytics import YOLO

    model = YOLO(str(ckpt_path))
    names = {int(k): v for k, v in dict(model.names).items()}
    return model, names


def predict_frame(model, frame_bgr, *, imgsz: int, conf: float) -> list[dict]:
    """对单帧 BGR ndarray 推理，返回 ``[{"class_id", "confidence", "xywhn"}, ...]``。"""

    results = model(frame_bgr, imgsz=imgsz, conf=conf, verbose=False)
    if not results or results[0].boxes is None:
        return []
    r = results[0]
    boxes_xywhn = r.boxes.xywhn.cpu().tolist()
    classes = r.boxes.cls.cpu().tolist()
    confs = r.boxes.conf.cpu().tolist()
    return [
        {"class_id": int(cls), "confidence": float(c), "xywhn": xywhn}
        for cls, c, xywhn in zip(classes, confs, boxes_xywhn)
    ]


def predict_media(
    model,
    frame_iterator,
    *,
    imgsz: int,
    conf: float,
    progress_every: int = 100,
    on_progress=None,
) -> list[list[dict]]:
    """对帧迭代器逐帧推理，返回与输入对齐的检测列表（与调用方解耦）。"""

    all_detections: list[list[dict]] = []
    for idx, frame in enumerate(frame_iterator):
        detections = predict_frame(model, frame, imgsz=imgsz, conf=conf)
        all_detections.append(detections)
        if (idx + 1) % progress_every == 0 and on_progress is not None:
            on_progress(idx + 1, len(detections))
    return all_detections
