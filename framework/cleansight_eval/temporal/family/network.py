"""GRU 网络定义（模型族层，迁移自 temporal-gru/model/gru.py）。"""

from __future__ import annotations

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
