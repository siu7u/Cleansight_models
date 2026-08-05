"""全序列 Transformer 动作分类器。

模型只负责网络结构：输入 ``[B, T, F]``，输出逐帧 logits ``[B, T, C]``。使用完整的
Transformer Encoder 上下文，因此是**非因果**模型，只进入 ``full_sequence_temporal``
流水线；训练/评估监督口径仍由流水线统一处理。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    """带可学习位置编码的逐帧 Transformer 分类器。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"Transformer d_model 必须能被 nhead 整除: d_model={d_model}, nhead={nhead}")
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行逐帧分类；``x`` 为 ``[B, T, F]``，返回 ``[B, T, C]``。"""

        if x.ndim != 3:
            raise ValueError(f"Transformer 输入必须是 [B, T, F]，实际为 {tuple(x.shape)}")
        if x.shape[1] > self.position.shape[1]:
            raise ValueError(
                f"序列长度 T={x.shape[1]} 超过 max_len={self.position.shape[1]}，请增大模型配置 max_len"
            )
        z = self.input_projection(x)
        z = z + self.position[:, : x.shape[1]]
        z = self.encoder(z)
        return self.classifier(self.norm(z))
