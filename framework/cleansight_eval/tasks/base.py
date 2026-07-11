"""任务层接口（需求 §5.1 / §6.2）。

任务类型描述模型解决的问题（时序分类/分割、目标检测），决定真值语义、
可用指标族，以及训练/评估的主体流程。框架层只负责运行组织、配置、信封与
矩阵等**与模型语义无关**的能力（§4.2）；训练循环、指标口径、推理协议这些
**语义**由各任务自行实现。

与执行模式、模型族一样，任务以注册表方式登记（见 ``tasks/__init__.py``），
配置用 ``task`` 字段引用。CLI 只做分派，不理解具体任务。
"""

from __future__ import annotations

from typing import Protocol

from ..core.envelope import EvalEnvelope


class Task(Protocol):
    """一类问题的可复用语义契约。"""

    task_id: str

    def validate_config(self, cfg: dict) -> None:
        """校验该任务专属的配置字段（框架层只校验通用字段）。"""
        ...

    def train(self, cfg: dict, runs_dir: str, seed: int, device) -> str:
        """执行一次训练，返回 checkpoint 路径（附带可重建的 sidecar 元信息）。"""
        ...

    def evaluate(self, cfg: dict, ckpt: str, feeding_name: str, device) -> EvalEnvelope:
        """在指定喂入模式下评估一个 checkpoint，返回三态信封（只出事实）。"""
        ...
