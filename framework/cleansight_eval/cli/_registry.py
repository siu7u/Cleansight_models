"""task → 纵编排器 的分派表（唯一同时 import 两纵的地方）。

CLI 的组合根（composition root）：core 不知道纵的存在、两纵互不 import，唯有此处
按 ``cfg["task"]`` 把请求分派到对应纵的编排器。编排器靠**同名方法约定**被 duck-type
调用（``validate_config`` / ``train`` / ``evaluate``）——这是编排（脊柱关切）的约定，
不是模型语义 Protocol，故无需 ``tasks/base.py``。
"""

from __future__ import annotations

from ..detection.orchestration import DetectionOrchestrator
from ..temporal.orchestration import TemporalOrchestrator

_VERTICALS = {
    TemporalOrchestrator.task_id: TemporalOrchestrator,
    DetectionOrchestrator.task_id: DetectionOrchestrator,
}


def get_vertical(task: str):
    if task not in _VERTICALS:
        raise KeyError(f"未注册的任务/纵: {task}；已注册: {sorted(_VERTICALS)}")
    return _VERTICALS[task]()
