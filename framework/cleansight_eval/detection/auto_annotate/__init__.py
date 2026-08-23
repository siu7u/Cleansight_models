"""YOLO 自动标注：视频 → legacy 标注 JSON（run）；图片帧序列数据集 → 时序训练数据（dataset）；
合并人工动作标签 → 时序训练数据（convert）。

- ``run``：用已训练 YOLO checkpoint 对视频逐帧检测，产出与历史 Label Studio
  导出同构的 legacy 标注 JSON（``auto_annotate.legacy_format`` 定义格式读写）。
- ``dataset``：用已训练 YOLO checkpoint 对图片帧序列数据集（images/ + 动作
  标签 labels/）逐帧检测，产出与 ``convert`` 同构的时序训练数据
  （frames/ + labels/），供 ``temporal/data.py`` 消费。
- ``convert``：自动标注 JSON + 人工 Label Studio 导出（timelinelabels）→
  framework 时序训练数据布局（labels/ + frames/），供 ``temporal/data.py`` 消费。

公共 API 在此再导出（CLI 与测试经由 ``cleansight_eval.detection.auto_annotate``
访问）；内部实现按职责拆分在 ``run`` / ``dataset`` / ``convert`` / ``legacy_format``。
"""

from .. import inference  # noqa: F401  （保持 auto_annotate.inference 可引用；测试 monkeypatch 依赖）
from . import convert, dataset, run  # noqa: F401  （auto_annotate.run / auto_annotate.convert 供 monkeypatch 定位）
from ._constants import ACTION_CLASSES, DEFAULT_TOP_K, DETECTION_CLASSES, TARGET_LABEL_FPS
from .convert import convert_annotations
from .dataset import run_dataset_annotate
from .legacy_format import (
    LegacyTask,
    build_task,
    ls_box_from_xywhn,
    parse_legacy_task,
    xywhn_from_ls_box,
)
from .run import build_track_sequences, detect_video, run_auto_annotate

__all__ = [
    "ACTION_CLASSES",
    "DEFAULT_TOP_K",
    "DETECTION_CLASSES",
    "LegacyTask",
    "TARGET_LABEL_FPS",
    "build_task",
    "build_track_sequences",
    "convert_annotations",
    "detect_video",
    "ls_box_from_xywhn",
    "parse_legacy_task",
    "run_auto_annotate",
    "run_dataset_annotate",
    "xywhn_from_ls_box",
]
