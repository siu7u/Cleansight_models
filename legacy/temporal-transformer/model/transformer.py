import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    """因果 Transformer 时序分类器，输出逐帧动作 logits。

    输入张量形状为 `[B, T, F]`。模型使用因果 attention mask，因此每一帧只能
    关注自己和历史帧。当前 v1 checkpoint 按固定窗口训练，生产中优先使用
    流式 `[1, window, F]` 推理，而不是直接整段视频推理。
    """

    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=3):
        """创建输入投影层、因果编码器和分类头。"""

        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, padding_mask=None):
        """返回形状为 `[B, T, num_classes]` 的逐帧 logits。

        如果传入 `padding_mask`，它遵循 PyTorch Transformer 语义，用于标记
        不应参与 attention 的 padding 时间步。
        """

        # x: (B, T, F)
        B, T, _ = x.shape
        x = self.proj(x)

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device) * float('-inf'),
            diagonal=1
        )
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        return self.head(out)
