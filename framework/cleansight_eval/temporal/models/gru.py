"""GRU 时序模型（纯 nn.Module，无监督/喂入语义）。

模型只提供网络结构：输入 ``[B, T, F]``、输出逐帧 logits ``[B, T, C]``。监督口径（末帧
vs 逐帧）与推理方式（滑窗 vs 全序列）由流水线决定，不写在模型里。

单向 GRU 每帧输出只依赖当前帧与历史帧，因此**因果**，可用于滑窗流式推理，也可用于全
序列离线推理。规模（hidden/num_layers）由模型配置表达。
"""

from __future__ import annotations

import torch.nn as nn


class GRUClassifier(nn.Module):
    """因果 GRU 时序分类器，输出逐帧动作 logits。"""

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
