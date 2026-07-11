"""GRU 模型族（参照实现，需求验收 §13.1 的样板）。

演示同架构不同规模（hidden/num_layers）只靠模型配置表达，训练/评估复用 temporal
任务与执行模式，模型族只提供网络与因果 loss 契约。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .network import GRUClassifier


class GruFamily:
    family_id = "gru"

    def build_network(self, model_cfg: dict) -> nn.Module:
        return GRUClassifier(
            input_dim=model_cfg["input_dim"],
            num_classes=model_cfg["num_classes"],
            hidden=model_cfg.get("hidden", 128),
            num_layers=model_cfg.get("num_layers", 3),
        )

    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return model(x)

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor, criterion) -> torch.Tensor:
        # 因果契约：只对窗口最后一帧计算损失（与原 temporal_main 一致）。
        last_logits = logits[:, -1, :]
        return criterion(last_logits, y)

    def predict_frame_logits(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        logits = model(x)  # [1, T, C]
        return logits[0, -1]

    def checkpoint_meta(self, model_cfg: dict, feature_schema: dict, extra: dict) -> dict:
        meta = {
            "family": self.family_id,
            "input_dim": model_cfg["input_dim"],
            "num_classes": model_cfg["num_classes"],
            "model": model_cfg,
            "feature_schema": feature_schema,
        }
        meta.update(extra)
        return meta


def get_family() -> GruFamily:
    return GruFamily()
