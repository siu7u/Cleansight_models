from pytorch_tcn import TCN
import torch.nn as nn

class TCNClassifier(nn.Module):
    """因果 TCN 时序分类器，输出逐帧动作 logits。

    对外输入契约为 `[B, T, F]`；内部会转置为 `[B, F, T]` 以适配 Conv1d 风格
    的 TCN 层。`causal=True` 保证模型可用于在线滑窗推理。
    """

    def __init__(self, input_dim, num_classes, hidden_dims=(64, 64, 64)):
        """创建因果 TCN 堆叠和逐帧分类头。"""

        super().__init__()
        self.tcn = TCN(
            num_inputs=input_dim,
            num_channels=list(hidden_dims),
            kernel_size=3,
            dropout=0.2,
            causal=True,
            use_norm='weight_norm',
            activation='relu',
        )
        self.head = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        """返回形状为 `[B, T, num_classes]` 的逐帧 logits。"""

        # x: (B, T, F)  batch_first
        x = x.transpose(1, 2)     # (B, F, T)，供 Conv1d 使用
        feat = self.tcn(x)        # (B, C, T)
        feat = feat.transpose(1, 2)  # (B, T, C)
        return self.head(feat)    # 逐帧 logits (B, T, num_classes)
