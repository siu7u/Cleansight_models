"""GRU 模型族（参照实现，需求验收 §13.1 的样板）。

演示同架构不同规模（hidden/num_layers）只靠模型配置表达，训练/评估复用 temporal
任务与执行模式，模型族只提供网络与因果 loss 契约。

每个架构自足于单文件：网络定义（``GRUClassifier``）与族契约（``GruFamily``）同处此处。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """因果 GRU 时序分类器，输出逐帧动作 logits。

    输入张量形状为 ``[B, T, F]``，其中 ``F`` 必须与 checkpoint 的特征维度一致。
    单向 GRU 保证每帧输出只依赖当前帧与历史帧，可用于流式滑窗推理。
    """

    def __init__(self, input_dim, num_classes, hidden=128, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)  # (B, T, H)
        return self.head(out)  # (B, T, num_classes)


class GruFamily:
    family_id = "gru"

    def build_network(self, model_cfg: dict) -> nn.Module:
        return GRUClassifier(
            input_dim=model_cfg["input_dim"],
            num_classes=model_cfg["num_classes"],
            hidden=model_cfg.get("hidden", 128),
            num_layers=model_cfg.get("num_layers", 3),
        )

    def prepare(self, model: nn.Module, features: list) -> None:
        """训练前钩子（统一契约）。GRU 不需要输入归一化，空操作。"""

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
