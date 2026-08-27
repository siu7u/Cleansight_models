"""检测推理辅助：加载 checkpoint 并对任意帧/视频做检测（framework 检测域）。

模型执行只允许发生在 framework 内部：本模块封装 Ultralytics YOLO 的加载与逐帧推理，
``tools/visualize_detections.py`` 等工具只做 CLI 编排与可视化，不直接 import ultralytics。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_predictor(ckpt_path: str, imgsz: int = 640, runs_dir: str | None = None):
    """加载 YOLO 模型，返回 ``(model, names_dict)``。

    ``runs_dir`` 非空时把 ultralytics 中间产物目录重定向到该目录；兼容
    ultralytics 8.3（``ultralytics.cfg.RUNS_DIR``）与 8.4+（runs_dir 由
    ``settings`` 管理，``utils.RUNS_DIR`` 派生自 settings）。ultralytics 的
    默认 runs 目录可能落在只读或无关位置（如按安装位置推断的 git 仓库根）。
    不传则保持 ultralytics 默认行为。
    """

    from ultralytics import YOLO

    if runs_dir is not None:
        import ultralytics.cfg as ucfg
        import ultralytics.utils as uutils
        from ultralytics import settings

        if hasattr(ucfg, "RUNS_DIR"):  # 8.3：cfg 模块级导出
            ucfg.RUNS_DIR = Path(runs_dir)
        uutils.RUNS_DIR = Path(runs_dir)  # 8.4+：模块级常量
        settings.update({"runs_dir": str(runs_dir)})  # 8.4+：settings 为唯一真源
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


def predict_frames(
    model,
    frames_bgr: list,
    *,
    imgsz: int,
    conf: float,
    track: bool = False,
) -> list[list[dict]]:
    """对一帧列表批量推理（GPU 利用率更高），返回与输入对齐的检测列表。

    ``track=True`` 时启用 ByteTrack 实例跟踪（ultralytics ``model.track``，
    连续调用保持跨帧 ID），每项检测额外含 ``"track_id"``（未跟踪到时为 None）。
    """

    kwargs: dict = dict(imgsz=imgsz, conf=conf, verbose=False)
    if track:
        kwargs["persist"] = True
        results = model.track(frames_bgr, **kwargs)
    else:
        results = model(frames_bgr, **kwargs)
    if not results:
        return [[] for _ in frames_bgr]
    outputs: list[list[dict]] = []
    for r in results:
        if r.boxes is None:
            outputs.append([])
            continue
        boxes_xywhn = r.boxes.xywhn.cpu().tolist()
        classes = r.boxes.cls.cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()
        track_ids = r.boxes.id.cpu().tolist() if r.boxes.id is not None else [None] * len(classes)
        outputs.append(
            [
                {
                    "class_id": int(cls),
                    "confidence": float(c),
                    "xywhn": xywhn,
                    "track_id": int(tid) if tid is not None else None,
                }
                for cls, c, xywhn, tid in zip(classes, confs, boxes_xywhn, track_ids)
            ]
        )
    return outputs


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
