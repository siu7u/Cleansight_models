"""任务注册表。新增任务在此登记，配置以 ``task`` 名字引用。

对齐 ``feeding/__init__.py`` 与 ``families/__init__.py`` 的写法：CLI 通过
``get_task(cfg["task"])`` 分派，不 import 具体任务实现。
"""

from __future__ import annotations

from .detection.task import DetectionTask
from .temporal.task import TemporalTask

_TASKS = {
    TemporalTask.task_id: TemporalTask,
    DetectionTask.task_id: DetectionTask,
}


def get_task(name: str):
    if name not in _TASKS:
        raise KeyError(f"未注册的任务: {name}；已注册: {sorted(_TASKS)}")
    return _TASKS[name]()
