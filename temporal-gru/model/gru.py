import torch.nn as nn

class GRUClassifier(nn.Module):
    """因果 GRU 时序分类器，输出逐帧动作 logits。

    输入张量形状为 `[B, T, F]`，其中 `F` 必须与 checkpoint 的特征维度一致。
    该模型使用单向 GRU，每一帧输出只依赖当前帧和历史帧，因此可用于流式滑窗推理。
    """

    def __init__(self, input_dim, num_classes, hidden=128, num_layers=3):
        """创建 GRU 编码器和逐帧分类头。"""

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
        """返回形状为 `[B, T, num_classes]` 的逐帧 logits。"""

        # x: (B, T, F)
        out, _ = self.rnn(x)        # (B, T, H)
        return self.head(out)       # (B, T, num_classes)
