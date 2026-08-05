"""训练与预测流水线的最小公共接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .execution import PredictionOutput


class Pipeline(ABC):
    """约束 CLI 可调度流水线的必要能力，不共享具体训练或评估实现。

    各流水线自行拥有数据组织、监督方式、训练循环和 checkpoint 策略；这里仅提供
    注册与调用所需的稳定接口。可视化属于可选能力，因此不纳入该抽象契约。
    """

    pipeline_name: ClassVar[str]

    @abstractmethod
    def validate_config(self, cfg: dict) -> None:
        """校验当前流水线需要的配置，不修改配置内容。"""

    @abstractmethod
    def train(self, cfg: dict, runs_dir: str, seed: int, device) -> str:
        """执行流水线自己的训练流程，返回可供评估使用的 checkpoint 路径。"""

    @abstractmethod
    def predict(self, cfg: dict, ckpt: str, device) -> PredictionOutput:
        """加载 checkpoint 执行预测，返回不包含统一指标判分的事实结果。"""
