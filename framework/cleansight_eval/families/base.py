"""模型族层接口（需求 §6.3）。

模型族只实现真正专属的能力：构建网络、forward、loss、输出转换、checkpoint
兼容检查、必要时的状态管理。同架构的不同规模通过模型配置表达，不复制实现
（§4.3 / §7.2）。

注意：本 ``ModelFamily`` 协议是**时序族契约**（forward 返回 ``[B,T,C]``
logits，供 TemporalTask 使用）。检测族 ``YoloFamily`` 由 ultralytics 封装
训练/验证，方法集不同 —— §4.2 明确允许不同模型的 forward 过程不统一。二者
都经 ``get_family(family_id)`` 取用，由所属 Task 决定调用哪套方法。
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class ModelFamily(Protocol):
    """一套可复用模型结构实现的契约。"""

    family_id: str

    def build_network(self, model_cfg: dict) -> nn.Module:
        """根据模型配置构建网络（同族不同规模只改配置）。"""
        ...

    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """返回逐帧 logits ``[B, T, num_classes]``。"""
        ...

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor, criterion) -> torch.Tensor:
        """计算训练损失（模型可定义自己的因果契约）。"""
        ...

    def predict_frame_logits(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """给定窗口 ``[1, T, F]``，返回最后一帧 logits ``[num_classes]``。"""
        ...

    def checkpoint_meta(self, model_cfg: dict, feature_schema: dict, extra: dict) -> dict:
        """生成写入 checkpoint 的重建元信息。"""
        ...
